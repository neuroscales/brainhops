"""The TIFF writer: what tifffile is asked to write, and the record of it.

A TIFF image or pyramid is written in two steps. The first step plans the
file: the axes of the stored arrays, their shapes, the metadata dialect,
and the options that tifffile receives, from the data, the preferred
transformation and the record of the object. The second step gives the
plan and the pixels to tifffile. The record of a consolidated object is
read from a file of the same plan that tifffile writes without pixels
into a sparse buffer, so that the record holds what the written file
holds, as the reader reads it, without encoding a single pixel.
"""

# stdlib
import io
import shutil

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.path import is_local, replace_file
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base.parsers import WriterError
from brainhops.io.images.base import _utils_raster as raster

# this format
from . import _utils as backend
from ._data import _TiffProxy
from ._geometry import Geometry, write_geometry
from ._raw import TiffRaw, read_record

# ----------------------------------------------------------------------
#   CONSTANTS
# ----------------------------------------------------------------------

_DIALECTS = ("ome", "imagej", "plain")

RECORD_OPTIONS = ("dialect", "name", "bigtiff")
"""The writer options that change the record of the written file."""

ENCODING_OPTIONS = (
    "compression",
    "compressionargs",
    "predictor",
    "tile",
    "rowsperstrip",
)
"""The writer options that tifffile receives to encode the pixels.

They change how the pixels are packed in the file, but nothing that the
record holds, so `consolidate` does not take them and only the writers
do.
"""

# ImageJ requires this axis order, and OME-TIFF and plain TIFF accept it.
_STORAGE_ORDER = "TZCYXS"

# These letters are given to the other axes, in this order. OME-TIFF stores Q,
# and the other letters that tifffile lists, as modulo dimensions, while plain
# TIFF stores any letter.
_OTHER_CODES = "QIEHAPRLVMBFGJKNOUW"

# The writer computes these ImageJ metadata keys itself instead of copying
# them.
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
# ImageJ spells these units differently from brainhops.
_IMAGEJ_UNITS = {"um": "micron", "angstrom": "Å"}
_IMAGEJ_DTYPES = (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"))

# These tags are written back as they were read, as (code, name) pairs.
_ROUND_TRIP_TAGS = (
    (269, "DocumentName"),
    (271, "Make"),
    (272, "Model"),
    (315, "Artist"),
    (316, "HostComputer"),
    (33432, "Copyright"),
)

# Data larger than this does not fit in a classic TIFF, as tifffile decides.
# The threshold is 4 GiB minus a margin for the metadata.
_BIGTIFF_THRESHOLD = 2**32 - 2**25

_OME_SUFFIXES = (".ome.tif", ".ome.tiff", ".ome.btf", ".ome.tf2", ".ome.tf8")

# These tables give the OME symbols of the length and time units that both
# brainhops and OME know. Their keys are the names that `length_unit` and
# `time_unit` return.
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
#   OPTIONS
# ----------------------------------------------------------------------


def split_options(
    kwargs: tx.Mapping[str, tx.Any],
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """Split the options of a writer into those of the record and of tifffile.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        The options given to a writer.

    Returns
    -------
    record : dict[str, Any]
        The options of `consolidate`, each of `RECORD_OPTIONS`, `None`
        when it is not given.
    encoding : dict[str, Any]
        The options of tifffile that are given.

    Raises
    ------
    TypeError
        If an unknown option is given.
    """
    known = RECORD_OPTIONS + ENCODING_OPTIONS
    unknown = sorted(name for name in kwargs if name not in known)
    if unknown:
        raise TypeError(
            f"Unknown TIFF writer option(s): {', '.join(unknown)}. The "
            f"options are {', '.join(known)}."
        )
    record = {name: kwargs.get(name) for name in RECORD_OPTIONS}
    encoding = {
        name: kwargs[name]
        for name in ENCODING_OPTIONS
        if kwargs.get(name) is not None
    }
    return record, encoding


def keeps_record(
    record: tx.Optional[TiffRaw],
    dialect: tx.Optional[str],
    name: tx.Optional[str],
    bigtiff: tx.Optional[bool],
) -> bool:
    """Tell whether the options of `consolidate` leave a record as it is.

    An option that is not given, or that names what the record holds,
    leaves it as it is: a dialect that is the dialect of the record, a
    name that does not ask for OME-TIFF or a record of OME-TIFF, and the
    BigTIFF flag of the record.
    """
    if record is None:
        return False
    held = record.dialect or "plain"
    if dialect is not None and str(dialect).lower() != held:
        return False
    if _is_ome_name(name) and held != "ome":
        return False
    return bigtiff is None or bool(bigtiff) == record.bigtiff


# ----------------------------------------------------------------------
#   PLAN
# ----------------------------------------------------------------------


class Plan(tx.NamedTuple):
    """What tifffile is asked to write for an image or a pyramid."""

    arrays: tx.Tuple[tx.Any, ...]
    """The array of each level in the order of brainhops, finest first."""

    axes: tx.List[Axis]
    """The axes of the arrays."""

    codes: str
    """The storage axes, slowest first, such as `"ZCYX"`."""

    shapes: tx.List[tx.Tuple[int, ...]]
    """The stored shape of each level, in the order of `codes`."""

    dtypes: tx.List[np.dtype]
    """The data type of each level."""

    opener: tx.Dict[str, tx.Any]
    """The options of `tifffile.TiffWriter`."""

    first: tx.Dict[str, tx.Any]
    """The options of `TiffWriter.write` for the finest level."""

    rest: tx.Dict[str, tx.Any]
    """The options of `TiffWriter.write` for the coarser levels."""


def plan_file(
    arrays: tx.Sequence[tx.Any],
    transformation: tx.Optional[Transformation],
    base: tx.Optional[TiffRaw],
    kept: bool,
    dialect: tx.Optional[str] = None,
    name: tx.Optional[str] = None,
    bigtiff: tx.Optional[bool] = None,
) -> Plan:
    """Plan the TIFF file of an image or of a pyramid.

    The axes of the arrays are those of the input of the transformation,
    or else the default axes. They are stored in the order of the record
    when it still describes them, or else in the order `TZCYXS`, as
    [`brainhops.io.images.tiff`][] describes. The geometry that TIFF can
    store is taken from the transformation, and the coarser levels of a
    pyramid store none, because a reader derives it from their shapes.

    The dialect is the one that `dialect` asks for, OME-TIFF for a name
    with an OME suffix, ImageJ for a record of ImageJ that ImageJ can
    still store, and otherwise the dialect of the record, as long as the
    geometry is the geometry of the record. A new geometry, or an image
    without a record, is written in OME-TIFF when its pixel size or time
    interval is known and its axes need no letter outside `TZCYXS`, and
    in plain TIFF otherwise. ImageJ cannot store a pyramid.

    Parameters
    ----------
    arrays : sequence of array
        The array of each level, finest first, in the order of brainhops.
        Only their shapes and their types are read.
    transformation : Transformation or None
        The preferred transformation of the finest level, composed with
        the placement of a pyramid.
    base : TiffRaw or None
        The record that supplies the dialect, the tags, the ImageJ
        metadata and the names of the OME image and channels.
    kept : bool
        Whether the geometry is the geometry of the record.
    dialect : {"ome", "imagej", "plain"}, optional
        The dialect to write.
    name : str, optional
        The name of the file, which may ask for OME-TIFF.
    bigtiff : bool, optional
        Whether to write BigTIFF, which by default is written only when the
        pixels do not fit in a classic TIFF file.

    Returns
    -------
    Plan
        What tifffile is asked to write.

    Raises
    ------
    WriterError
        If there is no level, a level has no data, the axes cannot be
        stored, or the dialect is unknown or cannot store the image.
    """
    if not arrays:
        raise WriterError(
            "This multiscale image has no levels, so there is nothing "
            "to write."
        )
    if any(array is None for array in arrays):
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    shape = tuple(int(n) for n in arrays[0].shape)
    axes = raster.image_axes(
        len(shape), getattr(transformation, "input", None)
    )
    if axes is None:
        axes = raster.default_axes(len(shape))
    dtype = np.dtype(arrays[0].dtype)
    original = None if base is None else base.storage_axes
    codes = _write_codes(axes, shape, dtype, original)
    try:
        perm = raster.storage_permutation(axes, codes)
    except ValueError as e:
        raise WriterError(
            f"Cannot store the axes of this image as {codes!r}: {e}"
        ) from e
    shapes, dtypes = [], []
    for index, array in enumerate(arrays):
        level = tuple(int(n) for n in array.shape)
        if len(level) != len(shape):
            raise WriterError(
                f"Level {index} of this pyramid does not have as many "
                f"axes as its first level."
            )
        shapes.append(tuple(level[p] for p in perm))
        dtypes.append(np.dtype(array.dtype))
    geometry = write_geometry(transformation, axes)
    chosen = _dialect(
        dialect,
        name,
        base,
        kept,
        codes,
        dtype,
        geometry,
        pyramid=len(arrays) > 1,
    )
    opener, first = _options(base, chosen, codes, shapes[0], geometry)
    nbytes = sum(
        int(np.prod(level)) * kind.itemsize
        for level, kind in zip(shapes, dtypes)
    )
    opener["bigtiff"] = _bigtiff(nbytes, bigtiff)
    rest = {
        key: first[key]
        for key in ("photometric", "planarconfig")
        if key in first
    }
    return Plan(
        arrays=tuple(arrays),
        axes=list(axes),
        codes=codes,
        shapes=shapes,
        dtypes=dtypes,
        opener=opener,
        first=first,
        rest=rest,
    )


def planned_record(plan: Plan) -> TiffRaw:
    """Return the record of the file of a plan, without writing a pixel.

    tifffile writes the file of the plan with empty levels into a sparse
    buffer, which keeps only the bytes that are written and skips the
    pixels, and the record is read from that buffer as from a file. The
    record therefore holds what a file of the plan holds, as the reader
    reads it. The OME-XML differs from that of a written file in its
    UUID only, which tifffile makes anew for each file.

    Parameters
    ----------
    plan : Plan
        The plan of the file.

    Returns
    -------
    TiffRaw
        The record of the first series of the file.

    Raises
    ------
    WriterError
        If tifffile cannot write the plan.
    """
    tf = backend._require_tifffile()
    buffer = _SparseBuffer()

    def go() -> None:
        with tf.TiffWriter(buffer, **plan.opener) as writer:
            for index, (shape, dtype) in enumerate(
                zip(plan.shapes, plan.dtypes)
            ):
                options = _level_options(plan, index)
                writer.write(None, shape=shape, dtype=dtype, **options)

    _wrap_write(go)
    buffer.seek(0)
    with tf.TiffFile(buffer) as tif:
        return read_record(tif, 0)


def write_plan(
    plan: Plan, target: tx.Any, encoding: tx.Mapping[str, tx.Any]
) -> None:
    """Write the file of a plan with its pixels.

    Each level is read and converted to its stored order just before it
    is written, so that a pyramid holds one level in memory at a time.

    Parameters
    ----------
    plan : Plan
        The plan of the file.
    target : PathLike or IO
        The local path of the file, or a stream open for writing.
    encoding : Mapping[str, Any]
        The options of tifffile that encode the pixels.

    Raises
    ------
    WriterError
        If tifffile cannot write the file.
    """
    tf = backend._require_tifffile()

    def go() -> None:
        with tf.TiffWriter(target, **plan.opener) as writer:
            for index, array in enumerate(plan.arrays):
                if isinstance(array, _TiffProxy):
                    # The lazy array is read as tifffile reads it, a
                    # memory map or a dask array, and transposed once.
                    array = array.read()
                stored = raster.to_storage(array, plan.axes, plan.codes)
                options = _level_options(plan, index)
                options.update(encoding)
                writer.write(_materialize(stored), **options)

    _wrap_write(go)


def write_target(filename: tx.Any, write: tx.Callable[[tx.Any], None]) -> None:
    """Write a file at once with a function that writes a name or a stream.

    A local file is written next to its target and then moved over it,
    so that a lazy array that reads the old file never reads a file that
    is half written. A remote file is first written into memory.

    Parameters
    ----------
    filename : str or PathLike
        The path of the file.
    write : callable
        The function that writes the file, given a local name or a
        stream.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if is_local(filename):
        replace_file(filename, write)
        return
    buffer = io.BytesIO()
    write(buffer)
    with filename.open("wb") as f:
        f.write(buffer.getvalue())


def copy_source(source: backend.TiffSource, target: tx.Any) -> None:
    """Write the bytes of a source file, as they are, to a name or a stream.

    Parameters
    ----------
    source : TiffSource
        The source, a local file or the bytes of a file.
    target : PathLike or IO
        The local path of the file, or a stream open for writing.
    """
    stream = hasattr(target, "write")
    if not stream and source.filename is not None:
        shutil.copyfile(source.filename, target)
        return
    if source.content is not None:
        content = source.content
    else:
        with open(source.filename, "rb") as file:
            content = file.read()
    if stream:
        target.write(content)
    else:
        with open(target, "xb") as file:
            file.write(content)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _level_options(plan: Plan, index: int) -> tx.Dict[str, tx.Any]:
    """Return the options of `TiffWriter.write` for one level of a plan.

    The finest level holds the coarser levels as its SubIFDs.
    """
    if index == 0:
        options = dict(plan.first)
        if len(plan.arrays) > 1:
            options["subifds"] = len(plan.arrays) - 1
        return options
    options = dict(plan.rest)
    options["subfiletype"] = 1
    return options


def _dialect(
    dialect: tx.Optional[str],
    name: tx.Optional[str],
    base: tx.Optional[TiffRaw],
    kept: bool,
    codes: str,
    dtype: np.dtype,
    geometry: Geometry,
    pyramid: bool,
) -> str:
    """Choose the dialect of a file, as [`plan_file`][] describes.

    Raises
    ------
    WriterError
        If the dialect is unknown, or ImageJ cannot store the image.
    """
    read = None if base is None else base.dialect
    modulo = any(c not in _STORAGE_ORDER for c in codes)
    imagej = not pyramid and _imagej_compatible(codes, dtype)
    if dialect is not None:
        dialect = str(dialect).lower()
        if dialect not in _DIALECTS:
            raise WriterError(
                f"dialect must be one of {_DIALECTS}, not {dialect!r}."
            )
        if dialect == "imagej" and not imagej:
            raise WriterError(
                f"ImageJ cannot store this image: it stores axes in the "
                f"order TZCYXS, data of type uint8, uint16 or float32, "
                f"and no pyramid (axes {codes!r}, type {dtype})."
            )
        return dialect
    if _is_ome_name(name):
        return "ome"
    if read == "imagej" and imagej:
        return "imagej"
    # The geometry of the record is what the record's dialect stores, so a
    # plain file stays plain and an OME-TIFF file stays OME-TIFF.
    if base is not None and kept and (read is None or read == "ome"):
        if read is None or not modulo:
            return read or "plain"
    if not modulo and (
        read == "ome" or geometry.sizes or geometry.time is not None
    ):
        return "ome"
    return "plain"


def _options(
    base: tx.Optional[TiffRaw],
    dialect: str,
    codes: str,
    shape: tx.Sequence[int],
    geometry: Geometry,
) -> tx.Tuple[tx.Dict[str, tx.Any], tx.Dict[str, tx.Any]]:
    """Return the options of `TiffWriter` and of `TiffWriter.write` for the
    finest level.
    """
    tags = {} if base is None else base.tags or {}
    write: tx.Dict[str, tx.Any] = {}
    if "S" in codes and shape[codes.index("S")] in (3, 4):
        write["photometric"] = "rgb"
        write["planarconfig"] = "contig" if codes.endswith("S") else "separate"
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
        planes = _plane_positions(codes, shape, geometry)
        if planes:
            metadata["Plane"] = planes
        metadata.update(_ome_extras(base, codes, shape))
        write["metadata"] = metadata
        if xy:
            write.update(_resolution_cm(geometry))
        opener = {"ome": True}
    elif dialect == "imagej":
        metadata = _imagej_extras(base, codes, shape)
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
            and base is not None
            and base.dialect is None
            and not description.lstrip().startswith(("{", "<"))
        ):
            write["description"] = description
        if xy:
            write.update(_resolution_cm(geometry))
        opener = {}
    return opener, write


def _ome_extras(
    base: tx.Optional[TiffRaw], codes: str, shape: tx.Sequence[int]
) -> tx.Dict[str, tx.Any]:
    """Return the image name and channel names of the OME-XML of a record,
    when they still apply.
    """
    if base is None or not base.ome_xml:
        return {}
    index = base.ome_index
    if index is None:
        index = base.series
    image, pixels = backend.ome_image(base.ome_xml, index)
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
    base: tx.Optional[TiffRaw], codes: str, shape: tx.Sequence[int]
) -> tx.Dict[str, tx.Any]:
    """Return the ImageJ metadata of a record, except the entries that
    describe its geometry.

    The writer computes the geometry entries again. The display ranges and
    LUTs, which are given for each channel, and the labels, which are given
    for each plane, are kept only if their number still matches the image.
    """
    source = {} if base is None else base.imagej or {}
    channels = shape[codes.index("C")] if "C" in codes else 1
    planes = int(np.prod([n for c, n in zip(codes, shape) if c not in "YXS"]))
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


def _write_codes(
    axes: tx.Sequence[Axis],
    shape: tx.Sequence[int],
    dtype: np.dtype,
    original: tx.Optional[str],
) -> str:
    """Return the storage codes under which to write the axes, slowest first.

    The order read from the file is kept if it still describes the axes.
    Otherwise, x, y and z become `X`, `Y` and `Z`, and time becomes `T`. A
    channel axis becomes `C`, or `S` when it holds the 3 or 4 samples of a
    two-dimensional 8-bit RGB or RGBA image. Other axes get letters of their
    own and come first, followed by the standard axes in the order `TZCYXS`.

    Raises
    ------
    WriterError
        If the image has more than three spatial axes or more than two
        channel axes.
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
    """Convert any array (NumPy, dask, CuPy, lazy) to a C-contiguous NumPy
    array."""
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
    codes: str, shape: tx.Sequence[int], geometry: Geometry
) -> tx.Dict[str, tx.List[float]]:
    """Return the OME position of every plane, in the order in which tifffile
    writes the planes.

    The planes run over every axis except Y, X and S, in C order.
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


def _resolution_cm(geometry: Geometry) -> tx.Dict[str, tx.Any]:
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
    """Run a write function, and report the refusals of tifffile as
    `WriterError`.
    """
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


class _SparseBuffer(io.RawIOBase):
    """A stream in memory that keeps only the bytes written to it.

    tifffile writes an empty level by moving past its pixels and writing
    one byte, so the buffer stores the IFDs and the metadata of a file
    without allocating its pixels. A range that was never written reads
    as zeros.
    """

    def __init__(self) -> None:
        super().__init__()
        self._chunks: tx.List[tx.Tuple[int, bytearray]] = []
        self._position = 0
        self._size = 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = int(offset)
        elif whence == io.SEEK_CUR:
            self._position += int(offset)
        else:
            self._position = self._size + int(offset)
        return self._position

    def write(self, content: tx.Any) -> int:
        content = bytes(content)
        last = self._chunks[-1] if self._chunks else None
        if last is not None and last[0] + len(last[1]) == self._position:
            last[1].extend(content)
        else:
            self._chunks.append((self._position, bytearray(content)))
        self._position += len(content)
        self._size = max(self._size, self._position)
        return len(content)

    def readinto(self, buffer: tx.Any) -> int:
        count = max(0, min(len(buffer), self._size - self._position))
        start, stop = self._position, self._position + count
        out = bytearray(count)
        for offset, chunk in self._chunks:
            low, high = max(offset, start), min(offset + len(chunk), stop)
            if low < high:
                out[low - start : high - start] = chunk[
                    low - offset : high - offset
                ]
        buffer[:count] = out
        self._position = stop
        return count
