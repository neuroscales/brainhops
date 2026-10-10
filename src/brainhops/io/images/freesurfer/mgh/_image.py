"""The MGH image format."""

# stdlib
import gzip
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace
from nibabel.freesurfer import mghformat as _mgh

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    ParserExistsError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._geometry import (
    Arrangement,
    arrange_voxel_to_ras,
    declared_axes,
)
from brainhops.io.common.freesurfer import FreesurferFormat
from brainhops.io.common.freesurfer._geometry import fs_geometry_from_vox2ras
from brainhops.io.common.mgh import MghMetadata, MghRaw
from brainhops.io.common.mgh._constants import _MGH_AXES, _MRI_PARAMS
from brainhops.io.common.mgh._raw import (
    _compressed_name,
    encode_mgh,
    read_mgh,
)
from brainhops.io.common.mgh._views import _image_to_disk, _image_to_model
from brainhops.io.common.nifti._geometry import _scale_spatial, _unit_scale
from brainhops.io.images.base import ImageFormat

_SCANNER = "scanner"
"""Name of the scanner RAS space, the preferred world space."""

_TKR = "tkr"
"""Name of the tkr (surface) RAS space."""

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_MGH_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
}


@register_format
class MghImage(
    ImageFormat,
    SingleScaleImage,
    FreesurferFormat,
    BinaryFileReader,
    BinaryFileWriter,
):
    """An image stored in a FreeSurfer MGH or MGZ file.

    An MGH image holds everything that its file stores beside the voxels
    as [`MghMetadata`][], whose record is an [`MghRaw`][], and the voxels
    as stored in `raw`. When the image is read from a file, `raw` is a
    nibabel proxy, which reads the voxels only when they are needed, and
    `data` is decoded from it on first access. The voxels are in Fortran
    order, and the data are indexed `(x, y, z)` or `(x, y, z, frames)`,
    where the frames form a time axis.

    The transformations are decoded from the record, unless other
    transformations are assigned. They are, in order, a [`Scaling`][] to
    `"physical"` by the voxel size in millimeters and the TR in
    milliseconds, an [`Affine`][] to `"tkr"` (`MghRaw.vox2tkr`), and the
    preferred [`Affine`][] to `"scanner"` (`MghRaw.vox2ras`).

    When the image is written, the record of its metadata is the base of
    the new file, as [`to_raw`][] describes, so the acquisition parameters
    of the footer, the trailing tags and the other header fields are kept.
    An image that is read and written again without changes gives the
    same bytes, after decompression for an `.mgz` file. They are lost when
    the image is converted to another format.

    !!! note "`goodRASFlag`"
        When the flag is not positive, FreeSurfer and the reader assume 1
        mm voxels, LIA direction cosines and a zero centre. A file whose
        geometry is kept is written back with its flag, and a file whose
        geometry is encoded from the image sets the flag.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mgh", ".mgz", ".mgh.gz")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mgh", "mgz")

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The voxels as the file stores them, or `None` without data.

    After a read, `raw` is a nibabel proxy. Setting `data` stores the new
    array here.
    """

    _metadata: KwOnly[NoRepr[tx.Optional[MghMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        # The constructor assigns the fields in order, so the default of
        # `raw`, assigned after `data`, erases the array stored by `data=`.
        # The array is therefore stored again. When both are given, `data`
        # takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(cache=True, invalidates=("data",))
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, decoded from `raw` on first access and cached.

        Data decoded from a file is read-only, because changing it in
        place would not change `raw`, which is what the image writes. To
        change the voxels, assign `data`: setting it stores the new array
        in `raw` and drops the cached value.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.raw = None if value is None else _image_to_disk(value)

    @smartproperty(cache=True, unset=(None, "empty"))
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The decoded list is cached, so `img.transformation is
        img.transformation`. Assigning a list replaces the decoded one, and
        assigning other metadata drops it. An image without metadata has
        no transformations.
        """
        if self.metadata is None or self.metadata.raw is None:
            return []
        return _mgh_to_transformations(self.metadata.raw)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the record.

        The axes are `x`, `y` and `z`, and `t` for a file with frames, in
        Fortran order. An image without metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return _mgh_system(self.metadata.raw)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MGH image.

        The stream is tested with [`MghRaw.sniff_fileobj`][].
        """
        return MghRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        data: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an MGH image."""
        return cls.sniff_fileobj(BytesIO(data), error=error, **kwargs)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "MghImage":
        """Read an image from the path of an MGH or MGZ file.

        A local file whose name matches its content is handed to nibabel
        by name, so that nibabel opens it whenever the voxels are read and
        can memory-map them. Any other file is read into memory. The
        keyword arguments `mmap` and `keep_file_open` are passed to the
        nibabel proxy, and the others are ignored.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        raw, proxy = read_mgh(filename, **kwargs)
        return cls(raw=proxy, metadata=MghMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MghImage":
        """Read an image from an open MGH or MGZ stream.

        The record is read at once, and the proxy reads the voxels from
        the stream when they are needed, so the stream must stay open
        while the data may be read. A stream that cannot seek is first
        read into memory.
        """
        raw, proxy = read_mgh(file, **kwargs)
        return cls(raw=proxy, metadata=MghMetadata.from_raw(raw))

    @classmethod
    def from_nibabel(
        cls, mgh: tx.Union[_mgh.MGHImage, _mgh.MGHHeader], **kwargs
    ) -> "MghImage":
        """Build an image from a nibabel MGH image or header.

        The voxels of an image become `raw` as they are, and a copy of its
        header becomes the record of the metadata, without tags. A header
        alone gives an image without data.

        Parameters
        ----------
        mgh : nibabel.freesurfer.mghformat.MGHImage or MGHHeader
            The nibabel image or header.
        **kwargs : Any
            Other fields of the image.

        Returns
        -------
        MghImage
            The image.

        Raises
        ------
        TypeError
            If `mgh` is neither an MGH image nor an MGH header.
        """
        if isinstance(mgh, _mgh.MGHImage):
            raw, header = mgh.dataobj, mgh.header
        elif isinstance(mgh, _mgh.MGHHeader):
            raw, header = None, mgh
        else:
            raise TypeError(
                f"Expected an MGH image or header, got {type(mgh)}"
            )
        metadata = MghMetadata.from_raw(MghRaw(header=header).copy())
        return cls(raw=raw, metadata=metadata, **kwargs)

    # --- writing ------------------------------------------------------

    def to_raw(self, like: tx.Any = None, **overrides) -> MghRaw:
        """Return the [`MghRaw`][] record that writing the image stores.

        The record is encoded in a fixed order. The base is a copy of the
        record of the metadata, or a new record for an image without
        metadata. The dimensions and the voxel type follow the data. The
        geometry comes from the transformations, as described in
        [`to_nibabel`][]: the voxel size, the cosines, the centre and the
        `goodRASFlag` of the base are kept when the voxel-to-scanner matrix
        and the spatial dimensions are those of the base, and they are
        encoded again otherwise. The MRI parameters of `like` are copied
        next, then a TR stated by the transformations, and the `overrides`
        are applied last. The tags and the other fields of the base are
        kept.

        When this encoding changes nothing, the record of the metadata is
        returned as it is, without a copy, so that an image that was read
        and not changed writes the record that it read.

        Parameters
        ----------
        like : path, nibabel MGH image or header, or MGH object, optional
            Template whose MRI parameters override those of this image. A
            TR stated by the transformations overrides both.
        **overrides : Any
            Header fields set last. `dtype` sets the stored voxel type,
            which by default is the type of the data if MGH can store it
            (uint8, int16, int32, float32), or else the nearest one.

        Returns
        -------
        MghRaw
            The record of the file.

        Raises
        ------
        WriterError
            If there is no data, the data have more than four dimensions,
            or the voxel type cannot be stored.
        UnrepresentableTransformationError
            If the transformation has no affine form, has a second
            non-spatial axis, or shifts or scales frames in a way that MGH
            cannot store.
        """
        return _encoded(self, like, overrides)[0]

    def to_nibabel(self, like: tx.Any = None, **overrides) -> _mgh.MGHImage:
        """Build the nibabel MGH image that encodes this image.

        The voxel-to-scanner matrix comes from the `"scanner"`
        transformation, or else from the preferred one unless it maps to
        tkr RAS, which MGH cannot store. Without a transformation, the
        voxels are 1 mm along the RAS axes. The data are transposed to put
        the spatial axes first and at most one frame axis after them (see
        [`plan_axes`][brainhops.io.common._geometry.plan_axes]). The TR of
        a time axis is stored in milliseconds.

        The header of the nibabel image is the header of [`to_raw`][]. A
        nibabel image has no place for the trailing tags, which are left
        out, and nibabel gives a header whose `goodRASFlag` is 0 its own
        default geometry.

        Parameters
        ----------
        like : path, nibabel MGH image or header, or MGH object, optional
            Template whose MRI parameters override those of this image. A
            TR stated by the transformation overrides both.
        **overrides : Any
            Header fields set last, as for [`to_raw`][].

        Returns
        -------
        nibabel.freesurfer.mghformat.MGHImage
            The encoded image.

        Raises
        ------
        WriterError
            If there is no data, the data have more than four dimensions,
            or the voxel type cannot be stored.
        UnrepresentableTransformationError
            If the transformation cannot be stored, as for [`to_raw`][].
        """
        record, data = _encoded(self, like, overrides)
        return _mgh.MGHImage(data, record.vox2ras, header=record.header)

    def to_bytes(self, compress: bool = False, **kwargs) -> bytes:
        """Return the MGH encoding: header, voxels, footer and tags.

        The record is built by [`to_raw`][], which receives the keyword
        arguments. A proxy that reaches the writer unchanged is written as
        the file stores it, so that an untouched image gives the bytes of
        its file.

        Parameters
        ----------
        compress : bool, default=False
            Whether to gzip the result into MGZ.
        **kwargs : Any
            The arguments of [`to_raw`][].

        Returns
        -------
        bytes
            The content of the file.
        """
        record, data = _encoded(self, kwargs.pop("like", None), kwargs)
        content = encode_mgh(record, data)
        return gzip.compress(content) if compress else content

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a path.

        The file is gzipped when its name ends with `.mgz` or `.gz`, unless
        `compress` says otherwise. The other keyword arguments are those of
        [`to_raw`][].
        """
        kwargs.setdefault("compress", _compressed_name(filename))
        content = self.to_bytes(**kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("wb") as f:
            f.write(content)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _mgh_system(record: MghRaw) -> CoordinateSystem:
    """Return the voxel coordinate system of a record."""
    axes = _MGH_AXES[: len(record.shape)]
    return CoordinateSystem(name="voxel", axes=axes, order="F")


def _mgh_to_transformations(record: MghRaw) -> tx.List[Transformation]:
    """Return the scaling, tkr RAS and scanner RAS transformations."""
    voxel_space = _mgh_system(record)
    axes = list(voxel_space.axes)
    tr = record.mri_params["tr"]

    # Frames are in milliseconds only when the TR is known; a unit is
    # never invented.
    units = {"space": "mm", "time": "ms" if tr > 0 else None}
    phys_axes = [replace(axis, unit=units.get(axis.type)) for axis in axes]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)

    ras_axes = [
        replace(
            axis,
            orientation=Orientation(
                type="anatomical", value=_RAS_ORIENTATION[axis.name]
            ),
        )
        if axis.name in _RAS_ORIENTATION
        else axis
        for axis in phys_axes
    ]
    ras_space = CoordinateSystem(name="RAS", axes=ras_axes)

    zooms = list(record.voxel_size)
    if len(axes) > 3:
        zooms.append(tr if tr > 0 else 1.0)
    vox2phys = Scaling(
        input=voxel_space, output=phys_space, scale=zooms[: len(axes)]
    )

    def _affine(matrix: np.ndarray, name: str) -> Affine:
        return Affine(
            input=voxel_space,
            output=replace(ras_space, name=name),
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    return [
        vox2phys,
        _affine(record.vox2tkr, _TKR),
        _affine(record.vox2ras, _SCANNER),
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _encoded(
    image: MghImage, like: tx.Any, overrides: tx.Mapping[str, tx.Any]
) -> tx.Tuple[MghRaw, ArrayProtocol]:
    """Return the record and the voxels that writing an image stores.

    The record is encoded as [`MghImage.to_raw`][] describes, and the
    voxels are `raw`, arranged in the axis order of the file when the
    transformations declare another order.
    """
    if image.raw is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    overrides = dict(overrides)
    dtype = overrides.pop("dtype", None)
    vox2ras, layout, tr = _scanner_geometry(image.transformations, image.raw)
    data = image.raw if layout is None else layout.apply(image.data)
    shape = _mgh_shape(data)

    # The base is a copy of the record, or a new record.
    held = None if image.metadata is None else image.metadata.raw
    record = MghRaw() if held is None else image.metadata.to_raw()
    header = record.header

    # The layout of the voxels and the geometry come from the image.
    header["dims"] = shape + (1,) * (4 - len(shape))
    header.set_data_dtype(_mgh_dtype(data, dtype))
    if held is None or not _same_geometry(held, vox2ras, shape):
        _set_geometry(header, vox2ras, shape)

    # The template comes next, then the TR of the image, then the
    # overrides.
    template = _like_header(like)
    if template is not None:
        for name in _MRI_PARAMS:
            header[name] = template[name]
    if tr is not None:
        header["tr"] = tr
    for name, value in overrides.items():
        header[name] = value

    if record == held:
        return held, data
    return record, data


def _mgh_shape(data: ArrayProtocol) -> tx.Tuple[int, ...]:
    """Return the shape that MGH stores for an array, with at least 3 axes.

    Raises
    ------
    WriterError
        If the array has more than four dimensions, or none.
    """
    shape = tuple(int(d) for d in data.shape)
    if not 1 <= len(shape) <= 4:
        raise WriterError(
            f"MGH stores volumes of up to four dimensions "
            f"(x, y, z, frames), not {len(shape)}."
        )
    return shape + (1,) * max(0, 3 - len(shape))


def _same_geometry(
    record: MghRaw, vox2ras: np.ndarray, shape: tx.Tuple[int, ...]
) -> bool:
    """Tell whether a record already stores a geometry.

    The geometry is the same when the spatial dimensions are those of the
    record and the voxel-to-scanner matrix is the one that the record
    decodes to, exactly. A record read from a file and transformations
    decoded from it therefore keep the stored fields, which encoding them
    again could round differently.
    """
    dims = tuple(int(d) for d in record.header["dims"][:3])
    return dims == shape[:3] and np.array_equal(record.vox2ras, vox2ras)


def _set_geometry(
    header: _mgh.MGHHeader, vox2ras: np.ndarray, shape: tx.Tuple[int, ...]
) -> None:
    """Encode a voxel-to-scanner matrix in the geometry fields of a header.

    The voxel size, the direction cosines and the centre are recovered
    from the matrix, and the `goodRASFlag` is set so that FreeSurfer reads
    them.
    """
    voxel_size, xras, yras, zras, cras = fs_geometry_from_vox2ras(
        vox2ras, shape[:3]
    )
    header["delta"] = voxel_size
    header["Mdc"] = (xras, yras, zras)
    header["Pxyz_c"] = cras
    header["goodRASFlag"] = 1


_MGH_POLICY = dict(fill_space=True, max_nonspatial=1)
"""Placement of axes in MGH files: three spatial axes, then frames."""


def _scanner_transformation(
    transformations: tx.Sequence[Transformation],
) -> tx.Optional[Transformation]:
    """Pick the transformation that gives the voxel-to-scanner matrix.

    A transformation named `"scanner"` wins. Otherwise the last
    transformation that does not map to tkr RAS is used, or `None` if
    there is none.
    """
    transformations = list(transformations or [])
    chosen = None
    for xform in transformations:
        if getattr(getattr(xform, "output", None), "name", None) == _SCANNER:
            chosen = xform
    if chosen is None:
        for xform in reversed(transformations):
            name = getattr(getattr(xform, "output", None), "name", None)
            if name != _TKR:
                chosen = xform
                break
    return chosen


def _scanner_geometry(
    transformations: tx.Sequence[Transformation], data: tx.Any
) -> tx.Tuple[np.ndarray, tx.Any, tx.Optional[float]]:
    """Return the scanner matrix, the axis layout and the TR to store.

    The matrix is in millimeters and the TR in milliseconds (see
    [`MghImage.to_nibabel`][]). Without a transformation, the result is
    the identity, no layout and no TR.
    """
    chosen = _scanner_transformation(transformations)
    if chosen is None:
        return np.eye(4), None, None
    ndim = len(getattr(data, "shape", ()) or ())
    voxel_axes = declared_axes(getattr(chosen, "input", None), ndim)
    arranged = arrange_voxel_to_ras(
        chosen, voxel_axes, "MGH", "scanner", **_MGH_POLICY
    )
    output = getattr(chosen, "output", None)
    matrix = _scale_spatial(arranged.matrix, _unit_scale(output, "mm"))
    return matrix, arranged.layout, _frame_tr(arranged)


def _frame_tr(arranged: Arrangement) -> tx.Optional[float]:
    """Return the TR in milliseconds of the frame axis, or `None`.

    A time axis that still counts frames, with a scale of one and no time
    unit, states no TR. Frames that are not time can be neither scaled
    nor shifted.
    """
    if not arranged.others:
        return None
    scale, offset = arranged.others[0]
    groups = [
        g[3]
        for g in (arranged.voxel_groups, arranged.world_groups)
        if g is not None and len(g) > 3
    ]
    if any(g != "time" for g in groups):
        if (scale, offset) != (1.0, 0.0):
            raise UnrepresentableTransformationError(
                "MGH stores no spacing or origin for frames that are not "
                f"time, so a map that scales or shifts them ({scale}, "
                f"{offset}) cannot be written."
            )
        return None
    if offset != 0:
        raise UnrepresentableTransformationError(
            f"MGH stores no origin for the time axis, so a map that "
            f"shifts it ({offset}) cannot be written."
        )
    if scale <= 0:
        raise UnrepresentableTransformationError(
            f"MGH stores the repetition time as a positive number, so a "
            f"map that scales time by {scale} cannot be written."
        )
    axes = list(getattr(arranged.world, "axes", None) or [])
    unit = getattr(axes[3], "unit", None) if len(axes) > 3 else None
    if is_physicalunit(unit) and is_timeunit(unit):
        return float(scale) * float(unit.scale) * 1000.0
    if scale == 1.0:
        # Still a frame index, as read from a file without a TR.
        return None
    return float(scale)


def _mgh_dtype(data: tx.Any, dtype: tx.Any = None) -> np.dtype:
    """Return the voxel type to store (see [`MghImage.to_nibabel`][])."""
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype not in _MGH_DTYPES:
            raise WriterError(
                f"MGH cannot store voxels of type {dtype}; it stores uint8, "
                f"int16, int32 and float32."
            )
        return dtype
    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if dtype in _MGH_DTYPES:
        return dtype
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype.kind == "f":
        return np.dtype(np.float32)
    if dtype.kind in "iu":
        if np.can_cast(dtype, np.int16):
            return np.dtype(np.int16)
        if np.can_cast(dtype, np.int32):
            return np.dtype(np.int32)
        info = np.iinfo(np.int32)
        values = np.asarray(data)
        if values.size == 0 or (
            values.min() >= info.min and values.max() <= info.max
        ):
            return np.dtype(np.int32)
        raise WriterError(
            f"MGH stores integers in at most 32 bits, and this {dtype} "
            f"array holds values outside the int32 range."
        )
    raise WriterError(
        f"MGH cannot store voxels of type {dtype}; it stores uint8, int16, "
        f"int32 and float32."
    )


def _like_header(like: tx.Any) -> tx.Optional[_mgh.MGHHeader]:
    """Resolve a `like` template to the MGH header to copy fields from.

    The template is a nibabel MGH header or image, an [`MghRaw`][] record,
    MGH metadata, an MGH image, or the path of an MGH file, whose record
    alone is read. Any other template gives `None`.
    """
    if like is None:
        return None
    if isinstance(like, _mgh.MGHHeader):
        return like
    if isinstance(like, _mgh.MGHImage):
        return like.header
    if isinstance(like, MghImage):
        like = like.metadata
    if isinstance(like, MghMetadata):
        like = like.raw
    if isinstance(like, (str, path.PathLike)):
        like = MghRaw.from_filename(like)
    if isinstance(like, MghRaw):
        return like.header
    return None
