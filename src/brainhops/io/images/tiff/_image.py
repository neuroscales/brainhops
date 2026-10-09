import math
from io import BytesIO
from numbers import Number

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic

from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Scaling,
    Sequence,
    Transformation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.images.base import _utils_raster as raster
from brainhops.io.images.tiff import _utils as backend

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

_DIALECTS = ("ome", "imagej", "plain")

# Required by ImageJ, and accepted by OME-TIFF and plain TIFF.
_STORAGE_ORDER = "TZCYXS"

# Letters for the other axes, in the order they are handed out. OME-TIFF stores
# Q (and the others that tifffile lists) as modulo dimensions, and plain TIFF
# stores any letter.
_OTHER_CODES = "QIEHAPRLVMBFGJKNOUW"

# ImageJ metadata that the writer computes itself instead of copying.
_IMAGEJ_COMPUTED = frozenset(
    {
        "ImageJ",
        "images",
        "channels",
        "slices",
        "frames",
        "hyperstack",
        "axes",
        "unit",
        "yunit",
        "zunit",
        "spacing",
        "finterval",
        "tunit",
    }
)
# ImageJ spellings of the units it knows.
_IMAGEJ_UNITS = {"um": "micron", "angstrom": "Å"}
_IMAGEJ_DTYPES = (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"))

# Tags written back as they were read: (code, name).
_ROUND_TRIP_TAGS = (
    (269, "DocumentName"),
    (271, "Make"),
    (272, "Model"),
    (315, "Artist"),
    (316, "HostComputer"),
    (33432, "Copyright"),
)

# Data that does not fit in a classic TIFF, as tifffile decides: 4 GiB minus a
# margin for the metadata.
_BIGTIFF_THRESHOLD = 2**32 - 2**25

_OME_SUFFIXES = (".ome.tif", ".ome.tiff", ".ome.btf", ".ome.tf2", ".ome.tf8")

# OME symbols of the length units (as `length_unit` normalises them) and time
# units (as `time_unit` does) that both brainhops and OME know.
_OME_LENGTHS = {
    "km": "km",
    "m": "m",
    "dm": "dm",
    "cm": "cm",
    "mm": "mm",
    "um": "µm",
    "nm": "nm",
    "pm": "pm",
    "angstrom": "Å",
    "inch": "in",
}
_OME_TIMES = {
    "h": "h",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "µs",
    "ns": "ns",
}


# ----------------------------------------------------------------------
#   READING: GEOMETRY
# ----------------------------------------------------------------------


def _overridden(pixel_size: tx.Any, axes: tx.Sequence[Axis]) -> tx.Set[str]:
    """Return the names of the spatial axes whose size `pixel_size` sets."""
    if pixel_size is None:
        return set()
    if isinstance(pixel_size, tx.Mapping):
        return set(pixel_size)
    return {axis.name for axis in raster.spatial_axes(axes)}


def _origin(
    axes: tx.Sequence[Axis],
    scales: tx.Mapping[str, tx.Tuple[float, tx.Any]],
    metadata: backend.TiffMetadata,
    origin: tx.Any,
) -> tx.Dict[str, float]:
    """Return the position of the first sample along each spatial axis, by
    name, in the unit of the axis.

    `origin` gives positions in the unit of the pixel size, or ignores those of
    the file if it is `False`. Otherwise, the OME plane positions of the file
    are used for the axes whose size and unit are known. A position without a
    unit, or in the OME "reference frame", is taken in the unit of the pixel
    size of the file.
    """
    names = [axis.name for axis in raster.spatial_axes(axes)]
    if origin is False:
        return {}
    if origin is not None:
        if isinstance(origin, tx.Mapping):
            out = {name: float(value) for name, value in origin.items()}
        elif isinstance(origin, (Number, np.number)):
            out = {name: float(origin) for name in names}
        else:
            values = [float(v) for v in np.ravel(origin)]
            if len(values) != len(names):
                raise ValueError(
                    f"origin gives {len(values)} positions, but the image "
                    f"has {len(names)} spatial axes {names}."
                )
            out = dict(zip(names, values))
        unknown = set(out) - set(names)
        if unknown:
            raise ValueError(
                f"origin names axes that are not spatial axes of this "
                f"image: {sorted(unknown)}."
            )
        return out
    out = {}
    for name, (value, unit) in metadata.origin.items():
        if name not in names or name not in scales:
            continue
        target = scales[name][1]
        if target is None:
            continue
        source = backend.length_unit(unit)
        if source is None:
            if name not in metadata.scales:
                continue
            source = metadata.scales[name][1]
        try:
            out[name] = raster.convert_length(value, source, target)
        except ValueError:
            continue
    return out


def _geometry(
    codes: str,
    base_shape: tx.Sequence[int],
    level_shape: tx.Sequence[int],
    metadata: backend.TiffMetadata,
    pixel_size: tx.Any = None,
    unit: tx.Any = None,
    origin: tx.Any = None,
) -> Transformation:
    """Return the transformation of one level of a series, from the metadata of
    the series and the overrides of the caller.
    """
    axes = raster.storage_axes(codes)
    perm = raster.canonical_permutation(axes)
    axes = [axes[p] for p in perm]
    scales = raster.resolve_pixel_size(
        axes, metadata.scales, pixel_size=pixel_size, unit=unit
    )
    mine = _overridden(pixel_size, axes)
    for name, sign in metadata.signs.items():
        if name in scales and name not in mine:
            size, u = scales[name]
            scales[name] = (sign * size, u)
    factors = backend.level_factors(
        [base_shape[p] for p in perm], [level_shape[p] for p in perm]
    )
    for axis, factor in zip(axes, factors):
        if factor != 1 and raster.axis_group(axis) != "space":
            raise ParserContentError(
                f"A pyramid level is downsampled along the non-spatial "
                f"axis {axis.name!r}, which cannot be placed in space."
            )
    return raster.level_transformation(
        axes, scales, factors, _origin(axes, scales, metadata, origin)
    )


# ----------------------------------------------------------------------
#   WRITING: GEOMETRY
# ----------------------------------------------------------------------


def _affine_parts(
    xform: tx.Optional[Transformation], ndim: int
) -> tx.Optional[tx.Tuple[np.ndarray, np.ndarray]]:
    """Return the diagonal and translation of a transformation that scales and
    translates each axis separately, or `None` for any other.
    """
    if xform is None or isinstance(xform, Identity):
        return np.ones(ndim), np.zeros(ndim)
    if isinstance(xform, Sequence):
        try:
            xform = xform.compute()
        except Exception:
            return None
    if isinstance(xform, Scaling):
        scale = getattr(xform, "scale", None)
        if scale is None:
            return np.ones(ndim), np.zeros(ndim)
        scale = np.atleast_1d(np.asarray(scale, dtype=float))
        if scale.size != ndim:
            return None
        return scale, np.zeros(ndim)
    affine = xform
    if not isinstance(affine, Affine):
        try:
            affine = xform.to(Affine)
        except Exception:
            return None
    matrix = getattr(affine, "matrix", None)
    if matrix is None:
        return np.ones(ndim), np.zeros(ndim)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (ndim, ndim + 1):
        return None
    linear = matrix[:, :-1]
    if np.any(linear - np.diag(np.diag(linear))):
        return None
    return np.diag(linear).copy(), matrix[:, -1].copy()


class _Geometry(Magic, frozen=True):
    """What a TIFF writer can store of the geometry of an image."""

    sizes: tx.Dict[str, float]  # positive, in `units`
    units: tx.Dict[str, str]  # normalised by `length_unit`
    signs: tx.Dict[str, float]  # -1 for a flipped axis
    origin: tx.Dict[str, float]  # in `units`
    time: tx.Optional[tx.Tuple[float, str]]  # interval and unit

    def size(self, name: str, unit: str) -> float:
        """Return the size along an axis, converted to another unit."""
        return raster.convert_length(self.sizes[name], self.units[name], unit)


def _write_geometry(
    xform: tx.Optional[Transformation], axes: tx.Sequence[Axis]
) -> _Geometry:
    """Return the pixel size, origin and time interval of the preferred
    transformation, as far as TIFF can store them.

    A spatial axis has a size when the transformation scales it onto an axis
    with a unit of length, and sizes are given only if every spatial axis has
    one. A rotation or a shear gives no size.
    """
    empty = _Geometry(sizes={}, units={}, signs={}, origin={}, time=None)
    parts = _affine_parts(xform, len(axes))
    if parts is None:
        return empty
    diag, shift = parts
    output = getattr(getattr(xform, "output", None), "axes", None)
    try:
        output = list(output) if output is not None else None
    except TypeError:
        output = None
    if output is None or len(output) != len(axes):
        return empty
    sizes, units, signs, origin = {}, {}, {}, {}
    time = None
    for i, axis in enumerate(axes):
        unit = getattr(output[i], "unit", None)
        group = raster.axis_group(axis)
        scale = float(diag[i])
        if not math.isfinite(scale) or scale == 0:
            continue
        if group == "space":
            name = backend.length_unit(None if unit is None else str(unit))
            if name is None:
                # a unit that is not a length gives no axis a size
                return _Geometry(
                    sizes={}, units={}, signs={}, origin={}, time=time
                )
            sizes[axis.name] = abs(scale)
            units[axis.name] = name
            if scale < 0:
                signs[axis.name] = -1.0
            if float(shift[i]):
                origin[axis.name] = float(shift[i])
        elif group == "time" and time is None:
            name = backend.time_unit(None if unit is None else str(unit))
            if name is not None and scale > 0:
                time = (scale, name)
    return _Geometry(
        sizes=sizes, units=units, signs=signs, origin=origin, time=time
    )


def _write_codes(
    axes: tx.Sequence[Axis],
    shape: tx.Sequence[int],
    dtype: np.dtype,
    original: tx.Optional[str],
) -> str:
    """Return the storage codes under which to write the axes, slowest first.

    The order read from the file is kept if it still describes the axes.
    Otherwise, x, y and z become `X`, `Y` and `Z`, time `T`, and a channel axis
    `C`, or `S` for the 3 or 4 samples of an 8-bit RGB(A) image. Other axes get
    letters of their own and come first, followed by the standard axes in the
    order `TZCYXS`.
    """
    names = [axis.name for axis in axes]
    if original:
        stored = [axis.name for axis in raster.storage_axes(original)]
        if sorted(stored) == sorted(names) and len(original) == len(axes):
            return original
    space = [i for i, a in enumerate(axes) if raster.axis_group(a) == "space"]
    time = [i for i, a in enumerate(axes) if raster.axis_group(a) == "time"]
    channel = [
        i for i, a in enumerate(axes) if raster.axis_group(a) == "channel"
    ]
    if len(space) > 3:
        raise WriterError(
            f"A TIFF file stores at most three spatial axes, but this "
            f"image has {len(space)}."
        )
    if len(channel) > 2:
        raise WriterError(
            f"A TIFF file stores at most two channel axes (channels and "
            f"samples), but this image has {len(channel)}."
        )
    codes: tx.Dict[int, str] = {}
    letters = "XYZ"
    named = {"x": "X", "y": "Y", "z": "Z"}
    left = list(letters[: len(space)])
    for i in space:
        code = named.get(axes[i].name)
        if code in left:
            codes[i] = code
            left.remove(code)
    for i in space:
        if i not in codes:
            codes[i] = left.pop(0)
    if time:
        codes[time[0]] = "T"
    if len(channel) == 2:
        codes[channel[0]] = "C"
        codes[channel[1]] = "S"
    elif channel:
        i = channel[0]
        rgb = (
            shape[i] in (3, 4)
            and np.dtype(dtype) == np.uint8
            and len(space) == 2
            and not time
        )
        codes[i] = "S" if rgb else "C"
    used = set(codes.values())
    spare = [c for c in _OTHER_CODES if c not in used]
    others = []
    for i, axis in enumerate(axes):
        if i in codes:
            continue
        letter = str(axis.name or "").upper()
        if len(letter) == 1 and letter.isalpha() and letter not in used:
            if letter in "XYZTCS":
                letter = spare.pop(0)
        else:
            letter = spare.pop(0)
        if letter in spare:
            spare.remove(letter)
        used.add(letter)
        codes[i] = letter
        others.append(letter)
    ordered = sorted(
        codes.values(),
        key=lambda c: (
            _STORAGE_ORDER.index(c) + len(_OTHER_CODES)
            if c in _STORAGE_ORDER
            else others.index(c)
        ),
    )
    return "".join(ordered)


def _materialize(array: tx.Any) -> np.ndarray:
    """Convert any array (NumPy, dask, CuPy) to a C-contiguous NumPy array."""
    if hasattr(array, "get") and not isinstance(array, np.ndarray):
        array = array.get()
    if hasattr(array, "compute") and not isinstance(array, np.ndarray):
        array = array.compute()
    return np.ascontiguousarray(np.asarray(array))


def _is_ome_name(name: tx.Optional[str]) -> bool:
    return bool(name) and str(name).lower().endswith(_OME_SUFFIXES)


def _imagej_compatible(codes: str, dtype: np.dtype) -> bool:
    if any(c not in _STORAGE_ORDER for c in codes):
        return False
    positions = [_STORAGE_ORDER.index(c) for c in codes]
    if positions != sorted(positions):
        return False
    if "S" in codes:
        return np.dtype(dtype) == np.uint8
    return np.dtype(dtype) in _IMAGEJ_DTYPES


def _plane_positions(
    codes: str, shape: tx.Sequence[int], geometry: _Geometry
) -> tx.Dict[str, tx.List[float]]:
    """Return the OME position of every plane, in the order in which tifffile
    writes them: over every axis but Y, X and S, C-ordered.
    """
    names = {"X": "x", "Y": "y", "Z": "z"}
    plane_axes = [i for i, c in enumerate(codes) if c not in "YXS"]
    plane_shape = [shape[i] for i in plane_axes]
    count = int(np.prod(plane_shape)) if plane_shape else 1
    index = (
        np.indices(plane_shape).reshape(len(plane_shape), -1)
        if plane_shape
        else np.zeros((0, 1), dtype=int)
    )
    out: tx.Dict[str, tx.List[float]] = {}
    for code in "XYZ":
        name = names[code]
        if name not in geometry.origin:
            continue
        values = np.full(count, geometry.origin[name])
        if code == "Z" and "Z" in codes:
            k = plane_axes.index(codes.index("Z"))
            step = geometry.sizes.get(name, 0.0)
            step *= geometry.signs.get(name, 1.0)
            values = values + step * index[k]
        out[f"Position{code}"] = [float(v) for v in values]
        unit = _OME_LENGTHS.get(geometry.units[name], "µm")
        out[f"Position{code}Unit"] = [unit] * count
    return out


# ----------------------------------------------------------------------
#   SHARED READER / WRITER MACHINERY
# ----------------------------------------------------------------------


class _TiffMixin:
    """Sniffing and file handling shared by single-scale and multiscale TIFF
    images.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.EXTENSIONS
    HINTS = ("tiff", "tifffile")

    @classmethod
    def _sniff_head(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> bool:
        with preserve_position(file):
            head = file.read(4)
        if backend.is_tiff(head):
            return True
        if error:
            if error is True:
                error = SnifferContentError
            raise error("This content is not a TIFF file.")
        return False

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a TIFF file."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an image from a file.

        A local file is opened again by name when the pixels are read, which
        allows memory-mapping, while a remote file is read into memory. The
        options are those of `from_source`.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        return cls.from_source(
            backend.TiffSource.from_file(filename), **kwargs
        )

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read an image from an open binary file, which is read into memory
        with its position restored. The options are those of `from_source`.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_source(backend.TiffSource(content=content), **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read an image from bytes. The options are those of `from_source`."""
        return cls.from_source(
            backend.TiffSource(content=bytes(content)), **kwargs
        )

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the image to an open binary file, whose name, if it has one,
        chooses the dialect as in `to_bytes`.
        """
        kwargs.setdefault("name", _to_filename(file))
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a file.

        A name ending with `.ome.tif`, `.ome.tiff`, `.ome.btf`, `.ome.tf2` or
        `.ome.tf8` asks for OME-TIFF. The writing plan is computed before the
        file is opened, so that an image that cannot be written leaves no file
        behind. A local file is written by tifffile directly.
        """
        kwargs.setdefault("name", _to_filename(filename))
        writer = self._writer(**kwargs)
        if backend._is_local(filename):
            writer(str(path.Path(filename)))
            return
        buffer = BytesIO()
        writer(buffer)
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("wb") as f:
            f.write(buffer.getvalue())

    def to_bytes(self, **kwargs) -> bytes:
        """Encode the image as TIFF.

        Parameters
        ----------
        dialect : {"ome", "imagej", "plain"}, optional
            The metadata dialect. By default, OME-TIFF if `name` has an OME
            suffix; else ImageJ if the image was read from ImageJ and ImageJ
            can store it; else OME-TIFF if the image was read from OME-TIFF or
            its pixel size or time interval is known, and its axes need no
            letter outside `TZCYXS`; and plain TIFF otherwise.
        name : str, optional
            The file name, which may ask for OME-TIFF.
        bigtiff : bool, optional
            Whether to write BigTIFF. By default, only when the data is too
            large for a classic TIFF (4 GiB).
        **options : Any
            Passed to `TiffWriter.write` of tifffile, such as
            `compression="zlib"` or `tile=(256, 256)`.

        Returns
        -------
        bytes
            The encoded image.
        """
        buffer = BytesIO()
        self._writer(**kwargs)(buffer)
        return buffer.getvalue()

    def _writer(self, **kwargs) -> tx.Callable[[tx.Any], None]:
        raise NotImplementedError  # pragma: no cover

    def _level_storage(
        self,
        data: tx.Any,
        xform: tx.Optional[Transformation],
        original: tx.Optional[str],
        codes: tx.Optional[str] = None,
    ) -> tx.Tuple[np.ndarray, str, tx.List[Axis]]:
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        shape = tuple(int(d) for d in data.shape)
        axes = raster.image_axes(len(shape), getattr(xform, "input", None))
        if axes is None:
            axes = raster.default_axes(len(shape))
        if codes is None:
            codes = _write_codes(axes, shape, data.dtype, original)
        try:
            stored = raster.to_storage(data, axes, codes)
        except ValueError as e:
            raise WriterError(
                f"Cannot store the axes of this image as {codes!r}: {e}"
            ) from e
        return _materialize(stored), codes, axes

    def _options(
        self,
        storage: np.ndarray,
        codes: str,
        geometry: _Geometry,
        name: tx.Optional[str],
        dialect: tx.Optional[str],
        allow_imagej: bool = True,
    ) -> tx.Tuple[str, tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
        """Return the dialect, together with the options of `TiffWriter` and
        `TiffWriter.write` for the base level.
        """
        read = getattr(self, "dialect", None)
        modulo = any(c not in _STORAGE_ORDER for c in codes)
        if dialect is not None:
            dialect = str(dialect).lower()
            if dialect not in _DIALECTS:
                raise WriterError(
                    f"dialect must be one of {_DIALECTS}, not {dialect!r}."
                )
            if dialect == "imagej" and not (
                allow_imagej and _imagej_compatible(codes, storage.dtype)
            ):
                raise WriterError(
                    f"ImageJ cannot store this image: it stores axes in the "
                    f"order TZCYXS, data of type uint8, uint16 or float32, "
                    f"and no pyramid (axes {codes!r}, type {storage.dtype})."
                )
        elif _is_ome_name(name):
            dialect = "ome"
        elif (
            read == "imagej"
            and allow_imagej
            and _imagej_compatible(codes, storage.dtype)
        ):
            dialect = "imagej"
        elif not modulo and (
            read == "ome" or geometry.sizes or geometry.time is not None
        ):
            dialect = "ome"
        else:
            dialect = "plain"

        tags = getattr(self, "tags", None) or {}
        write: tx.Dict[str, tx.Any] = {}
        if "S" in codes and storage.shape[codes.index("S")] in (3, 4):
            write["photometric"] = "rgb"
            write["planarconfig"] = (
                "contig" if codes.endswith("S") else "separate"
            )
        else:
            write["photometric"] = "minisblack"
        if tags.get("Software"):
            write["software"] = tags["Software"]
        if tags.get("DateTime"):
            write["datetime"] = tags["DateTime"]
        extratags = []
        for code, key in _ROUND_TRIP_TAGS:
            value = tags.get(key)
            if isinstance(value, str) and value:
                extratags.append((code, "s", 0, value, True))
        if extratags:
            write["extratags"] = extratags

        sizes = geometry.sizes
        xy = "x" in sizes and "y" in sizes
        if dialect == "ome":
            metadata: tx.Dict[str, tx.Any] = {"axes": codes}
            for key in ("x", "y", "z"):
                if key in sizes:
                    upper = key.upper()
                    unit = _OME_LENGTHS.get(geometry.units[key])
                    size = sizes[key]
                    if unit is None:
                        unit, size = "µm", geometry.size(key, "um")
                    metadata[f"PhysicalSize{upper}"] = size
                    metadata[f"PhysicalSize{upper}Unit"] = unit
            if geometry.time is not None and "T" in codes:
                interval, tunit = geometry.time
                if tunit in _OME_TIMES:
                    metadata["TimeIncrement"] = interval
                    metadata["TimeIncrementUnit"] = _OME_TIMES[tunit]
            planes = _plane_positions(codes, storage.shape, geometry)
            if planes:
                metadata["Plane"] = planes
            metadata.update(self._ome_extras(codes, storage.shape))
            write["metadata"] = metadata
            if xy:
                write.update(_resolution_cm(geometry))
            opener = {"ome": True}
        elif dialect == "imagej":
            metadata = self._imagej_extras(codes, storage.shape)
            metadata["axes"] = codes
            # ImageJ has one length unit for all axes: their common unit, or
            # micrometres if they disagree.
            units = set(geometry.units.values())
            unit = units.pop() if len(units) == 1 else "um"
            if xy:
                metadata["unit"] = _IMAGEJ_UNITS.get(unit, unit)
                write["resolution"] = (
                    1 / geometry.size("x", unit),
                    1 / geometry.size("y", unit),
                )
            if "z" in sizes:
                metadata["unit"] = _IMAGEJ_UNITS.get(unit, unit)
                metadata["spacing"] = geometry.size(
                    "z", unit
                ) * geometry.signs.get("z", 1.0)
            if geometry.time is not None and "T" in codes:
                interval, tunit = geometry.time
                metadata["finterval"] = interval
                if tunit != "s":
                    metadata["tunit"] = tunit
            write["metadata"] = metadata
            opener = {"imagej": True}
        else:
            write["metadata"] = {"axes": codes}
            description = tags.get("ImageDescription")
            if (
                isinstance(description, str)
                and read is None
                and not description.lstrip().startswith(("{", "<"))
            ):
                write["description"] = description
            if xy:
                write.update(_resolution_cm(geometry))
            opener = {}
        return dialect, opener, write

    def _ome_extras(
        self, codes: str, shape: tx.Sequence[int]
    ) -> tx.Dict[str, tx.Any]:
        """Return the image name and channel names of the source OME-XML, when
        they still apply.
        """
        xml = getattr(self, "ome_xml", None)
        if not xml:
            return {}
        index = getattr(self, "_ome_index", None)
        if index is None:
            index = getattr(self, "series", None) or 0
        image, pixels = backend.ome_image(xml, index)
        out: tx.Dict[str, tx.Any] = {}
        if image is not None and image.get("Name"):
            out["Name"] = image.get("Name")
        if pixels is not None:
            count = shape[codes.index("C")] if "C" in codes else 1
            names = backend.ome_channel_names(pixels)
            if len(names) == count and all(names):
                out["Channel"] = {"Name": names}
        return out

    def _imagej_extras(
        self, codes: str, shape: tx.Sequence[int]
    ) -> tx.Dict[str, tx.Any]:
        """Return the ImageJ metadata read with the image that does not
        describe its geometry, when it still applies (display ranges and LUTs
        per channel, labels per plane).
        """
        source = getattr(self, "imagej_metadata", None) or {}
        channels = shape[codes.index("C")] if "C" in codes else 1
        planes = int(
            np.prod([n for c, n in zip(codes, shape) if c not in "YXS"])
        )
        out: tx.Dict[str, tx.Any] = {}
        for key, value in source.items():
            if key in _IMAGEJ_COMPUTED:
                continue
            if key == "Ranges":
                if len(value) != 2 * channels:
                    continue
            elif key == "LUTs":
                if len(value) != channels:
                    continue
            elif key == "Labels":
                if len(value) != planes:
                    continue
            out[key] = value
        return out


def _resolution_cm(geometry: _Geometry) -> tx.Dict[str, tx.Any]:
    """Return the resolution tags, in pixels per centimetre."""
    return {
        "resolution": (
            1 / geometry.size("x", "cm"),
            1 / geometry.size("y", "cm"),
        ),
        "resolutionunit": "CENTIMETER",
    }


def _bigtiff(nbytes: int, bigtiff: tx.Optional[bool]) -> bool:
    if bigtiff is None:
        return nbytes > _BIGTIFF_THRESHOLD
    return bool(bigtiff)


def _wrap_write(function: tx.Callable[[], None]) -> None:
    """Run a write, reporting the refusals of tifffile as `WriterError`."""
    try:
        function()
    except WriterError:
        raise
    except (ValueError, TypeError, KeyError) as e:
        raise WriterError(f"tifffile cannot write this image: {e}") from e
    except Exception as e:
        if type(e).__name__ in ("OmeXmlError", "TiffFileError"):
            raise WriterError(f"tifffile cannot write this image: {e}") from e
        raise


# ----------------------------------------------------------------------
#   SINGLE-SCALE IMAGE
# ----------------------------------------------------------------------


@register_format
class TiffImage(
    _TiffMixin,
    BinaryFileReader,
    BinaryFileWriter,
    WritableFileBasedImage,
    SingleScaleImage,
):
    """One image of a TIFF file, read and written with tifffile.

    The file is a plain TIFF, BigTIFF, OME-TIFF or ImageJ hyperstack. The data
    is one level (the full resolution by default) of one series (the first by
    default), F-ordered as x, y and z, then time, channels and other axes. The
    only transformation scales pixels to `"physical"`, with a translation if
    the file records an origin, or is the identity, with no unit, when the
    pixel size is unknown. The pixels are read on first access, and the
    metadata attributes are written back on save as far as they still apply.
    """

    dialect: tx.Annotated[
        tx.Optional[str],
        tx.Doc(
            "The metadata dialect of the file: 'ome', 'imagej', or `None` "
            "for a plain TIFF file."
        ),
    ] = None

    ome_xml: tx.Annotated[
        tx.Optional[str], tx.Doc("The OME-XML of an OME-TIFF file.")
    ] = None

    imagej_metadata: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc("The ImageJ metadata of an ImageJ hyperstack."),
    ] = None

    tags: tx.Annotated[
        tx.Optional[tx.Dict[str, tx.Any]],
        tx.Doc(
            "The tags of interest of the first page, by name: resolution, "
            "description, software, date, artist, copyright, ..."
        ),
    ] = None

    series: tx.Annotated[
        tx.Optional[int], tx.Doc("The index of the series that was read.")
    ] = None

    level: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The pyramid level that was read (0: full resolution)."),
    ] = None

    n_series: tx.Annotated[
        tx.Optional[int], tx.Doc("The number of series in the file.")
    ] = None

    n_levels: tx.Annotated[
        tx.Optional[int],
        tx.Doc("The number of pyramid levels of the series."),
    ] = None

    storage_axes: tx.Annotated[
        tx.Optional[str],
        tx.Doc(
            "The axes of the series as tifffile stores them, slowest first, "
            "such as 'TZCYX' or 'YXS'."
        ),
    ] = None

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        source = getattr(self, "_source", None)
        if source is None:
            return None
        mmap, lazy = getattr(self, "_read_options", (True, None))
        raw = backend.read_series(
            source,
            series=self.series or 0,
            level=self.level or 0,
            mmap=mmap,
            lazy=lazy,
        )
        data, _ = raster.to_canonical(raw, self.storage_axes)
        return data

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `LIKELY` for any TIFF or BigTIFF header.

        [`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage]
        claims pyramidal series with more confidence, unless a level is
        requested.
        """
        if cls._sniff_head(file, error):
            return Confidence.LIKELY
        return Confidence.NO

    @classmethod
    def from_source(
        cls,
        source: backend.TiffSource,
        series: int = 0,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        origin: tx.Any = None,
        mmap: bool = True,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """Read one level of one series.

        Parameters
        ----------
        source : TiffSource
            Where the file is.
        series : int, optional
            The series, the first by default. A negative index counts from the
            end.
        level : int, optional
            The pyramid level, 0 (the full resolution) by default.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size, overriding the one of the file.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the sizes of the
            file are converted.
        origin : bool, float, Sequence[float] or Mapping[str, float], optional
            The position of the first pixel, in the unit of the pixel size,
            overriding the OME plane positions. `False` ignores those
            positions.
        mmap : bool, optional
            Whether to memory-map a local uncompressed series (the default).
        lazy : bool, optional
            Whether the data is a dask array over the Zarr store of tifffile,
            which needs dask and zarr. By default, only when dask is the array
            backend.

        Raises
        ------
        ParserContentError
            If the file cannot be read as TIFF.
        IndexError
            If there is no such series or level.
        TypeError
            If an unknown option is given.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        index = 0 if level is None else int(level)
        with source.open() as tif:
            chosen = backend._select(tif, int(series), index)
            series = int(series) % len(tif.series)
            base = tif.series[series]
            index %= len(base.levels)
            metadata = backend.series_metadata(tif, series)
            codes = str(chosen.axes)
            xform = _geometry(
                codes,
                tuple(base.shape),
                tuple(chosen.shape),
                metadata,
                pixel_size=pixel_size,
                unit=unit,
                origin=origin,
            )
            image = cls(
                transformations=[xform],
                dialect=metadata.dialect,
                ome_xml=metadata.ome_xml,
                imagej_metadata=metadata.imagej,
                tags=metadata.tags,
                series=series,
                level=index,
                n_series=len(tif.series),
                n_levels=len(base.levels),
                storage_axes=codes,
            )
        image._source = source
        image._read_options = (mmap, lazy)
        image._ome_index = metadata.ome_index
        return image

    def _writer(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
        **options,
    ) -> tx.Callable[[tx.Any], None]:
        tf = backend._require_tifffile()
        xform = self.transformation
        storage, codes, axes = self._level_storage(
            self.data, xform, self.storage_axes
        )
        geometry = _write_geometry(xform, axes)
        _, opener, write = self._options(
            storage, codes, geometry, name, dialect
        )
        write.update(options)
        opener["bigtiff"] = _bigtiff(storage.nbytes, bigtiff)

        def run(target: tx.Any) -> None:
            def go() -> None:
                with tf.TiffWriter(target, **opener) as writer:
                    writer.write(storage, **write)

            _wrap_write(go)

        return run
