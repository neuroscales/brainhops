"""The AFNI image format."""

# stdlib
import math
import os
import time
import uuid
from collections import OrderedDict
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Scaling,
    Transformation,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import local_path as _local_path
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.afni import AfniFormat, AfniMetadata, AfniRaw
from brainhops.io.common.afni._constants import _BRIK_SUFFIXES, AFNI_VIEWS
from brainhops.io.common.afni._data import (
    _BrikProxy,
    _same_file,
    brick_code,
    brick_dtype,
    brik_proxy,
    read_afni,
    write_brik,
)
from brainhops.io.common.afni._files import _brik_suffix, afni_dataset_files
from brainhops.io.common.afni._geometry import (
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_view,
    afni_voxel_to_dicom,
    afni_world,
)
from brainhops.io.common.afni._raw import (
    _declined,
    _format_attributes,
    _looks_like_head,
    _sniffed_raw,
    _sniffed_raw_at,
)
from brainhops.io.common.afni._views import _image_to_disk, _image_to_model
from brainhops.io.images.base import ImageFormat

_INDEX = "index"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_PHYSICAL = "physical"
_CARDINAL = "-cardinal"

# Codes of `TAXIS_NUMS[2]` for the time units the writer may meet.
_TAXIS_CODES = {"millisecond": 77001, "second": 77002, "hertz": 77003}

# Attributes that the writer computes itself, never copied from the
# source.
_GENERATED = (
    "TYPESTRING",
    "IDCODE_STRING",
    "IDCODE_DATE",
    "SCENE_DATA",
    "ORIENT_SPECIFIC",
    "ORIGIN",
    "DELTA",
    "IJK_TO_DICOM",
    "IJK_TO_DICOM_REAL",
    "DATASET_RANK",
    "DATASET_DIMENSIONS",
    "BRICK_TYPES",
    "BRICK_STATS",
    "BRICK_FLOAT_FACS",
    "BYTEORDER_STRING",
)

# Attributes that describe sub-bricks or the time axis, copied from the
# source header only if the number of sub-bricks is unchanged.
_PER_BRICK = (
    "BRICK_LABS",
    "BRICK_KEYWORDS",
    "BRICK_STATAUX",
    "BRICK_STATSYM",
    "STAT_AUX",
    "TAXIS_NUMS",
    "TAXIS_FLOATS",
    "TAXIS_OFFSETS",
)

# Attributes that describe the grid, copied from the source header only
# if the shape and geometry are unchanged.
_PER_GRID = (
    "MARKS_XYZ",
    "MARKS_LAB",
    "MARKS_HELP",
    "MARKS_FLAGS",
    "TAGSET_NUM",
    "TAGSET_FLOATS",
    "TAGSET_LABELS",
)

# TYPESTRING values, indexed by the code in `SCENE_DATA[2]`.
_TYPESTRINGS = (
    "3DIM_HEAD_ANAT",
    "3DIM_HEAD_FUNC",
    "3DIM_GEN_ANAT",
    "3DIM_GEN_FUNC",
)
_ANAT_SPGR, _ANAT_EPI, _ANAT_BUCK = 0, 2, 11


# Attributes that describe how the BRIK stores the voxels. The data and
# the `datatype` option decide them, so they cannot be overridden.
_LAYOUT = (
    "DATASET_RANK",
    "DATASET_DIMENSIONS",
    "BRICK_TYPES",
    "BRICK_STATS",
    "BRICK_FLOAT_FACS",
    "BYTEORDER_STRING",
)

# Attributes that describe the grid of the voxels in DICOM space.
_GEOMETRY = (
    "ORIENT_SPECIFIC",
    "ORIGIN",
    "DELTA",
    "IJK_TO_DICOM",
    "IJK_TO_DICOM_REAL",
)

# Attributes that identify a dataset, renewed when it changes.
_IDENTITY = ("IDCODE_STRING", "IDCODE_DATE")


def _afni_axes(header: AfniRaw) -> tx.List[Axis]:
    """Return the voxel axes of an AFNI dataset.

    Several sub-bricks add a fourth axis: `t` for a time series, and
    `brick`, with no meaning defined by AFNI, otherwise.
    """
    axes = [Axis(name, "space", unit=_INDEX) for name in _SPATIAL]
    if header.nvals > 1:
        if header.taxis is not None:
            axes.append(Axis("t", "time", unit=_INDEX))
        else:
            axes.append(Axis("brick", unit=_INDEX))
    return axes


@register_format
class AfniImage(
    ImageFormat,
    SingleScaleImage,
    AfniFormat,
    BinaryFileReader,
    BinaryFileWriter,
):
    """An image stored as an AFNI dataset (`.HEAD` and `.BRIK` files).

    An AFNI image holds the header of its dataset as [`AfniMetadata`][],
    whose record is an [`AfniRaw`][], and the voxels as stored in `raw`.
    When the image is read from a file, `raw` is a lazy array that reads
    the BRIK only when the voxels are needed, and `data` is decoded from
    it on first access. The data are indexed `[x, y, z]`, or
    `[x, y, z, sub-brick]`, in Fortran order and scaled by
    `BRICK_FLOAT_FACS`.

    The transformations are decoded from the record, unless other
    transformations are assigned. They lead to `physical`,
    `<view>-cardinal` and `<view>`, the last of which is preferred.

    When the image is written, the record of its metadata is the base of
    the new header, as [`to_raw`][] describes, so the attributes that the
    data model has no place for, such as `HISTORY_NOTE` or `BRICK_LABS`,
    are kept. An image that is read and written again without changes
    gives the same bytes in both files, after decompression of the BRIK.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".HEAD",
        ".BRIK",
        ".BRIK.gz",
        ".BRIK.bz2",
        ".head",
        ".brik",
        ".brik.gz",
        ".brik.bz2",
    )
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("brik",)

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The voxels as the BRIK stores them, or `None` without data.

    After a read, `raw` is a lazy array of the BRIK, which applies the
    scale factors of the header when it is read. Setting `data` stores
    the new array here.
    """

    _metadata: KwOnly[NoRepr[tx.Optional[AfniMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the dataset, which holds its record, or `None`.

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
        return _afni_to_transformations(self.metadata.raw)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read from `raw` without reading the voxels.

        The data is `raw` as the lazy array presents it, so the two have
        the same shape, and the lazy array knows its shape from the
        record. An image without data raises as [`SingleScaleImage`][]
        does.
        """
        if self.raw is None:
            return self.data.shape
        return tuple(int(d) for d in self.raw.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the record.

        The axes are `x`, `y` and `z`, in Fortran order, and a fourth axis
        for several sub-bricks: `t` for a time series and `brick`
        otherwise. An image without metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        axes = _afni_axes(self.metadata.raw)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(cls, header: AfniRaw) -> float:
        """Score a valid header as a plain image.

        A dataset with exactly three sub-bricks may be a warp (such as the
        `_WARP` output of 3dQwarp), so it is only a weak match.
        """
        if header.nvals == 3:
            return Confidence.WEAK
        return Confidence.LIKELY

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an AFNI image header.

        The position of the stream is restored, and only the start of a
        stream that does not look like a header is read.
        """
        raw, failure = _sniffed_raw(file)
        if raw is None:
            return _declined(error, failure)
        return cls._score_header(raw)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a path names an AFNI image.

        The `.HEAD` file of the dataset is tested, whichever file of the
        dataset the path names.
        """
        raw, failure = _sniffed_raw_at(filename)
        if raw is None:
            return _declined(error, failure)
        return cls._score_header(raw)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an AFNI image header."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "AfniImage":
        """Read an image from the path of a `.HEAD`, a `.BRIK` or a dataset.

        The record is read from the `.HEAD` file, and the BRIK is checked
        but not read: the lazy array reads it when the voxels are needed.
        Other keyword arguments are ignored.

        Parameters
        ----------
        filename : path
            Any file of the dataset, or the bare dataset name.
        mmap : bool, default=True
            Whether the lazy array memory-maps a local BRIK that is not
            compressed and whose sub-bricks share a type.
        **kwargs : Any
            Ignored.

        Raises
        ------
        ParserExistsError
            If the `.HEAD` or the `.BRIK` file does not exist.
        ParserContentError
            If the header is invalid, or the BRIK cannot be read.
        """
        raw, proxy = read_afni(filename, mmap=mmap)
        return cls(raw=proxy, metadata=AfniMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "AfniImage":
        """Read an image from an open `.HEAD` or `.BRIK` file.

        The other file of the dataset is found from the name of the
        stream, which must have one, and the BRIK is opened by name when
        the voxels are needed, without memory mapping. A stream that does
        not start like a header is taken to be the BRIK, which is not
        read. The position of the stream is restored, and the keyword
        arguments are ignored.

        Raises
        ------
        ParserContentError
            If the stream has no file name.
        """
        name = getattr(file, "name", None)
        named = isinstance(name, (str, bytes, os.PathLike))
        with preserve_position(file):
            start = file.read(256)
            if isinstance(start, str):
                start = start.encode("latin-1")
            head = _looks_like_head(bytes(start).decode("latin-1"))
            rest = file.read() if head and named else b""
        if not named:
            raise ParserContentError(
                "An AFNI dataset is two files (.HEAD and .BRIK), and this "
                "stream has no file name to find the other one from. Read "
                "the dataset from its path instead."
            )
        if not head:
            return cls.from_filename(os.fsdecode(name), mmap=False)
        if isinstance(rest, str):
            rest = rest.encode("latin-1")
        record = AfniRaw.from_bytes(bytes(start) + bytes(rest)).validate()
        _, brik, _ = afni_dataset_files(os.fsdecode(name))
        proxy = brik_proxy(record, brik, mmap=False)
        return cls(raw=proxy, metadata=AfniMetadata.from_raw(record))

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "AfniImage":
        """Refuse to read bytes, which cannot hold a dataset of two files.

        Raises
        ------
        ParserContentError
            Always.
        """
        raise ParserContentError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Read it from its path instead."
        )

    # --- writing ------------------------------------------------------

    def to_raw(
        self,
        view: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        attributes: tx.Optional[tx.Mapping[str, tx.Any]] = None,
    ) -> AfniRaw:
        """Return the [`AfniRaw`][] record that writing the image stores.

        The record is encoded in a fixed order. The base is the record of
        the metadata, or no attribute for an image without metadata.

        1. The attributes that describe how the BRIK stores the voxels
           (`DATASET_RANK`, `DATASET_DIMENSIONS`, `BRICK_TYPES`,
           `BRICK_STATS`, `BRICK_FLOAT_FACS` and `BYTEORDER_STRING`)
           follow `raw`. A lazy array of a BRIK that `datatype` does not
           change is copied as the BRIK stores it, so its attributes are
           those of its own record. Any other array is stored without
           scale factors, in little-endian order, with the type that
           `datatype` names or else the closest type to that of the data.
        2. The geometry comes from the preferred transformation. The
           geometry attributes of the base are kept as they are stored
           when the transformation and the spatial shape are those of the
           base, exactly. When the transformation is only close to that of
           the base, the cardinal grid of the base is kept, and otherwise
           the grid is the cardinal grid closest to the transformation.
           `IJK_TO_DICOM_REAL` is then the transformation itself.
        3. The view is `view`, or else the view named by the world space
           of the preferred transformation, then the view of the base,
           then `"orig"`.
        4. The other attributes of the base are kept, except those of the
           sub-bricks when their number changes, those of the grid when it
           changes, and `TEMPLATE_SPACE` when the view changes. The
           `attributes` are applied last.

        When this encoding gives the attributes of the base, the record of
        the metadata is returned as it is, without a copy, so that an
        image that was read and not changed writes the header that it
        read. Otherwise, the dataset gets a new `IDCODE_STRING` and
        `IDCODE_DATE`, unless `attributes` sets them.

        Parameters
        ----------
        view : {"orig", "acpc", "tlrc"}, optional
            The view of the dataset. Writing to a file whose name carries
            a view, such as `anat+tlrc.HEAD`, gives that view.
        datatype : str or dtype, optional
            AFNI type name (`"byte"`, `"short"`, `"int"`, `"float"`,
            `"double"`, `"complex"`) or NumPy type of the stored values.
            Values are never rescaled.
        attributes : mapping, optional
            Attributes set last. A value of `None` removes an attribute.
            The attributes that describe how the BRIK stores the voxels,
            the grid and the view (`SCENE_DATA`) are refused, because the
            data, `datatype`, the transformations and `view` decide them.
            A string must be Latin-1 text.

        Returns
        -------
        AfniRaw
            The record of the dataset.

        Raises
        ------
        WriterError
            If there is no data, the data have no dimension or more than
            four, the type cannot be stored, the geometry is degenerate,
            the view is unknown, or `attributes` sets an attribute that the
            data, the geometry or the view decide, or a string that is not
            Latin-1 text.
        UnrepresentableTransformationError
            If the preferred transformation is not affine.
        """
        return _encoded(self, view, datatype, attributes)[0]

    def to_filename(
        self,
        filename: path.FilenameLike,
        view: tx.Optional[str] = None,
        datatype: tx.Optional[tx.Any] = None,
        attributes: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        **kwargs,
    ) -> None:
        """Write the `.HEAD` and `.BRIK` files of a dataset.

        The path may name the `.HEAD` file, the `.BRIK` file (a `.gz` or
        `.bz2` suffix compresses it) or the bare dataset. The header is
        the record of [`to_raw`][], whose view defaults to the view in the
        file name. The BRIK is written first, then the header. As AFNI
        does, a sibling BRIK with another suffix is removed, so that it
        cannot shadow the new one.

        The lazy array of an image may read the BRIK that is removed, for
        example when an image read from `x+orig.BRIK` is saved as
        `x+orig.BRIK.gz`. The image then reads the new BRIK instead, and
        its data is decoded again from that file when it is next read. The
        values are the same unless `datatype` changed how they are stored.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        if kwargs:
            raise TypeError(
                f"Unknown AFNI writer option(s): {', '.join(kwargs)}"
            )
        if isinstance(filename, str):
            filename = path.Path(filename)
        head, _, stem = afni_dataset_files(filename)
        brik_ext = ".brik" if head.name.endswith(".head") else ".BRIK"
        suffix = ""
        if ".BRIK" in filename.name.upper():
            suffix = _brik_suffix(filename)
        brik = filename.parent / (stem + brik_ext + suffix)
        view = view or afni_view(filename)
        record, data = _encoded(self, view, datatype, attributes)
        write_brik(record, data, brik)
        with head.open("wb") as f:
            f.write(record.to_bytes())
        for other in _BRIK_SUFFIXES:
            stale = filename.parent / (stem + brik_ext + other)
            if other != suffix and path.exists(stale):
                local = _local_path(stale)
                if local is not None:
                    _leave_brik(self, stale, record, brik)
                    os.remove(local)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path; AFNI datasets cannot be written to a stream."""
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return self.to_fileobj(file, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Refuse a stream, which cannot hold a dataset of two files.

        Raises
        ------
        WriterError
            Always.
        """
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which a "
            "single stream cannot hold. Write it to a path instead."
        )

    def to_bytes(self, **kwargs) -> bytes:
        """Refuse to write bytes, which cannot hold a dataset of two files.

        Raises
        ------
        WriterError
            Always.
        """
        raise WriterError(
            "An AFNI dataset is two files (.HEAD and .BRIK), which bytes "
            "alone cannot hold. Write it to a path instead."
        )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _encoded(
    image: AfniImage,
    view: tx.Optional[str],
    datatype: tx.Optional[tx.Any],
    attributes: tx.Optional[tx.Mapping[str, tx.Any]],
) -> tx.Tuple[AfniRaw, ArrayProtocol]:
    """Return the record and the voxels that writing an image stores.

    The record is encoded as [`AfniImage.to_raw`][] describes. The voxels
    are `raw` when it is a lazy array that is copied as the BRIK stores
    it, and otherwise an array of at least three dimensions, which the
    writer encodes with the types of the record.
    """
    if image.raw is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    overrides = dict(attributes or {})
    refused = sorted(set(_LAYOUT).intersection(overrides))
    if refused:
        raise WriterError(
            f"The AFNI attributes {refused} cannot be set: they describe "
            f"how the BRIK stores the voxels, which the data and "
            f"datatype= decide."
        )
    refused = sorted(set(_GEOMETRY + ("SCENE_DATA",)).intersection(overrides))
    if refused:
        raise WriterError(
            f"The AFNI attributes {refused} cannot be set: they describe "
            f"the grid and the view, which the transformations and view= "
            f"decide."
        )
    for name, value in overrides.items():
        if isinstance(value, str) and not _is_latin1(value):
            raise WriterError(
                f"The AFNI attribute {name} holds characters that a "
                f"header, which is Latin-1 text, cannot store."
            )
    held = None if image.metadata is None else image.metadata.raw
    data = _stored_data(image, datatype)
    shape = tuple(int(d) for d in data.shape)
    nvals = shape[3] if len(shape) == 4 else 1

    matrix = afni_voxel_to_dicom(image.transformation)
    geometry, same_grid = _geometry(held, matrix, shape[:3])
    output = getattr(image.transformation, "output", None)
    chosen = (
        view
        or afni_view(getattr(output, "name", None))
        or (held.view if held is not None else None)
        or "orig"
    )
    if chosen not in AFNI_VIEWS:
        raise WriterError(
            f"Unknown AFNI view {chosen!r}: use one of {AFNI_VIEWS}."
        )
    taxis = _time_axis(image, held, nvals)

    # The generated attributes come first, in the order that AFNI writes
    # them. The identity is that of the base until the encoding is known
    # to change the dataset.
    attrs = OrderedDict()
    typestring, scene = _typestring_and_scene(held, nvals, taxis)
    attrs["TYPESTRING"] = typestring
    for name in _IDENTITY:
        attrs[name] = None if held is None else held.get(name)
    attrs["SCENE_DATA"] = (AFNI_VIEWS.index(chosen),) + scene
    attrs.update(geometry)
    attrs.update(_layout(data, nvals, datatype))

    # The other attributes of the base follow, then the time axis.
    if held is not None:
        for key, value in held.attributes.items():
            if key in attrs or key in _GENERATED:
                continue
            if key in _PER_BRICK and held.nvals != nvals:
                continue
            if key in _PER_GRID and not same_grid:
                continue
            if key == "TEMPLATE_SPACE" and held.view != chosen:
                continue
            attrs[key] = value
    if taxis is not None and "TAXIS_NUMS" not in attrs:
        tr, unit = taxis
        attrs["TAXIS_NUMS"] = (nvals, 0, _TAXIS_CODES.get(unit, 77002))
        attrs["TAXIS_FLOATS"] = (0.0, float(tr), 0.0, 0.0, 0.0)

    for name, value in overrides.items():
        if value is None:
            attrs.pop(name, None)
        else:
            attrs[name] = value

    # The attributes are compared as they are written, so that a NaN, which
    # differs from itself, still compares equal, and the order of the
    # attributes does not matter.
    record = _record(attrs)
    if held is not None and _written(record) == _written(held):
        return held, data
    if "IDCODE_STRING" not in overrides:
        attrs["IDCODE_STRING"] = "AFN_" + uuid.uuid4().hex[:22]
    if "IDCODE_DATE" not in overrides:
        attrs["IDCODE_DATE"] = time.ctime()
    return _record(attrs).validate(), data


def _stored_data(
    image: AfniImage, datatype: tx.Optional[tx.Any]
) -> ArrayProtocol:
    """Return the voxels to write, as a lazy array or as an array.

    A lazy array of a BRIK is kept when `datatype` names the type of
    every sub-brick or is not given, so that the BRIK is copied as it is
    stored. Otherwise, the data are written, with three spatial axes and
    an optional axis of sub-bricks.

    Raises
    ------
    WriterError
        If the data have no dimension or more than four.
    """
    raw = image.raw
    if isinstance(raw, _BrikProxy):
        if datatype is None:
            return raw
        stored = brick_dtype(datatype)
        native = [dtype.newbyteorder("=") for dtype in raw.record.dtypes]
        if all(dtype == stored for dtype in native):
            return raw
        raw = image.data
    ndim = len(np.shape(raw))
    if ndim == 0 or ndim > 4:
        raise WriterError(
            f"AFNI stores three spatial axes and sub-bricks, so a "
            f"{ndim}D array cannot be written."
        )
    if ndim < 3:
        raw = np.reshape(raw, tuple(np.shape(raw)) + (1,) * (3 - ndim))
    return raw


def _layout(
    data: ArrayProtocol, nvals: int, datatype: tx.Optional[tx.Any]
) -> "OrderedDict[str, tx.Any]":
    """Return the attributes that describe how the BRIK stores the voxels.

    A lazy array of a BRIK is copied as it is stored, so its attributes
    are those of its own record, as stored. Any other array is stored in
    little-endian order without scale factors, and its statistics are
    computed from its values.
    """
    if isinstance(data, _BrikProxy):
        return OrderedDict(
            (name, data.record[name])
            for name in _LAYOUT
            if name in data.record
        )
    if datatype is None:
        datatype = getattr(data, "dtype", np.dtype(np.float32))
    stored = brick_dtype(datatype)
    shape = tuple(int(d) for d in np.shape(data))
    layout = OrderedDict()
    layout["DATASET_RANK"] = (3, nvals, 0, 0, 0, 0, 0, 0)
    layout["DATASET_DIMENSIONS"] = shape[:3] + (0, 0)
    layout["BRICK_TYPES"] = (brick_code(stored),) * nvals
    layout["BRICK_STATS"] = _brick_stats(data, nvals)
    layout["BRICK_FLOAT_FACS"] = (0.0,) * nvals
    layout["BYTEORDER_STRING"] = "LSB_FIRST"
    return layout


def _geometry(
    held: tx.Optional[AfniRaw],
    matrix: np.ndarray,
    shape: tx.Tuple[int, int, int],
) -> tx.Tuple["OrderedDict[str, tx.Any]", bool]:
    """Return the geometry attributes to write, and whether the grid is kept.

    The geometry attributes of the base are kept as they are stored when
    the voxel-to-DICOM matrix and the spatial shape are those of the base,
    exactly. A record read from a file and transformations decoded from it
    therefore keep the stored attributes, which encoding them again could
    spell differently. Otherwise, the cardinal grid of the base is kept
    when the matrix is close to that of the base, and the closest cardinal
    grid is used when it is not.

    Raises
    ------
    WriterError
        If the matrix is degenerate or not finite.
    """
    if not np.all(np.isfinite(matrix)) or not np.linalg.det(matrix):
        raise WriterError(
            "The voxel-to-world matrix is degenerate or not finite, so "
            "it cannot be written as AFNI geometry."
        )
    same_shape = held is not None and held.shape == shape
    if same_shape and np.array_equal(held.voxel_to_dicom, matrix):
        stored = OrderedDict(
            (name, held[name]) for name in _GEOMETRY if name in held
        )
        return stored, True
    same_grid = same_shape and np.allclose(
        held.voxel_to_dicom, matrix, atol=1e-5
    )
    if same_grid:
        orient, origin, delta = held.orient, held.origin, held.delta
    else:
        orient, origin, delta = afni_geometry_from_matrix(matrix)
    cardinal = afni_cardinal_matrix(orient, origin, delta)
    geometry = OrderedDict()
    geometry["ORIENT_SPECIFIC"] = tuple(int(o) for o in orient)
    geometry["ORIGIN"] = tuple(float(o) for o in origin)
    geometry["DELTA"] = tuple(float(d) for d in delta)
    geometry["IJK_TO_DICOM"] = tuple(float(v) for v in cardinal[:3].ravel())
    geometry["IJK_TO_DICOM_REAL"] = tuple(float(v) for v in matrix[:3].ravel())
    return geometry, same_grid


def _written(record: AfniRaw) -> str:
    """Return the attributes of a record as written, sorted by name."""
    return _format_attributes(OrderedDict(sorted(record.attributes.items())))


def _is_latin1(value: str) -> bool:
    """Tell whether a string can be written in Latin-1 text."""
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _leave_brik(
    image: AfniImage, stale: tx.Any, record: AfniRaw, brik: tx.Any
) -> None:
    """Stop an image from reading a BRIK that is about to be removed.

    When the lazy array of the image reads that BRIK, it is replaced by a
    lazy array of the new BRIK, which the new record describes, and the
    cached data is dropped, since a lazy Dask array may still read the
    old file.
    """
    raw = image.raw
    if not isinstance(raw, _BrikProxy) or not _same_file(raw.brik, stale):
        return
    image.raw = brik_proxy(record, brik, raw.mmap)
    image.__dict__.pop("_cache_data", None)


def _record(attrs: tx.Mapping[str, tx.Any]) -> AfniRaw:
    """Build a record from attributes, leaving out those set to `None`.

    The values are normalized to strings and flat tuples, as a record
    read from a file holds them.
    """
    return AfniRaw(
        attributes=OrderedDict(
            (key, _as_value(value))
            for key, value in attrs.items()
            if value is not None
        )
    )


def _as_value(value: tx.Any) -> tx.Any:
    """Normalize an attribute value to a string or a flat tuple."""
    if isinstance(value, str):
        return value
    values = np.ravel(np.asarray(value)).tolist()
    return tuple(values)


def _brick_stats(data: tx.Any, nvals: int) -> tx.Tuple[float, ...]:
    """Return `BRICK_STATS`: the finite minimum and maximum of each brick."""
    stats = []
    for i in range(nvals):
        brick = data[..., i] if np.ndim(data) == 4 else data
        brick = np.real(np.asarray(brick))
        if brick.dtype.kind == "f":
            brick = brick[np.isfinite(brick)]
        if brick.size:
            stats += [float(np.min(brick)), float(np.max(brick))]
        else:
            stats += [0.0, 0.0]
    return tuple(stats)


def _typestring_and_scene(
    source: tx.Optional[AfniRaw],
    nvals: int,
    taxis: tx.Optional[tx.Tuple[float, str]],
) -> tx.Tuple[str, tx.Tuple[int, ...]]:
    """Return the TYPESTRING and the `SCENE_DATA` values after the view.

    Valid source values are kept. Otherwise the dataset is anatomical:
    SPGR for one volume, EPI for a time series and bucket otherwise.
    """
    if source is not None:
        typestring = source.get("TYPESTRING")
        scene = source.get("SCENE_DATA")
        if (
            isinstance(typestring, str)
            and typestring in _TYPESTRINGS
            and isinstance(scene, tuple)
            and len(scene) >= 3
        ):
            return typestring, tuple(int(s) for s in scene[1:])
    if nvals == 1:
        func = _ANAT_SPGR
    elif taxis is not None:
        func = _ANAT_EPI
    else:
        func = _ANAT_BUCK
    return _TYPESTRINGS[0], (func, 0, -999, -999, -999, -999, -999)


def _time_axis(
    image: AfniImage, source: tx.Optional[AfniRaw], nvals: int
) -> tx.Optional[tx.Tuple[float, str]]:
    """Return the TR and its unit for a time series, or `None`.

    The source time axis is kept if the number of sub-bricks is unchanged.
    Otherwise the TR of a time axis is read from a [`Scaling`][], and
    defaults to one second.
    """
    if source is not None and source.nvals == nvals and source.taxis:
        return source.taxis
    if nvals < 2:
        return None
    system = getattr(image, "system", None)
    for xform in image.transformations or []:
        if system is not None:
            break
        system = getattr(xform, "input", None)
    axes = list(getattr(system, "axes", None) or [])
    if len(axes) < 4 or getattr(axes[3], "type", None) != "time":
        return None
    for xform in reversed(list(image.transformations or [])):
        if not isinstance(xform, Scaling):
            continue
        scale = np.ravel(np.asarray(getattr(xform, "scale", []), float))
        if scale.size != 4 or not math.isfinite(scale[3]):
            continue
        out_axes = list(getattr(xform.output, "axes", None) or []) + [None]
        unit = getattr(out_axes[3], "unit", None)
        unit = None if unit is None else str(unit)
        if unit in _TAXIS_CODES:
            return float(scale[3]), unit
    return 1.0, "second"


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _afni_to_transformations(header: AfniRaw) -> tx.List[Transformation]:
    """Convert an AFNI header to transformations, in this order.

    1. voxel -> "physical": a `Scaling` by `|DELTA|` (and the TR of a
       time series);
    2. voxel -> "<view>-cardinal": the cardinal grid, in LPS mm;
    3. voxel -> "<view>": `IJK_TO_DICOM_REAL` (or the cardinal grid), in
       LPS mm.
    """
    voxel_axes = _afni_axes(header)
    voxel_space = CoordinateSystem(name="voxel", axes=voxel_axes, order="F")

    taxis = header.taxis
    scale = [abs(d) for d in header.delta]
    phys_axes = [replace(axis, unit=_MM) for axis in voxel_axes[:3]]
    if len(voxel_axes) > 3:
        axis = voxel_axes[3]
        if axis.type == "time" and taxis and taxis[0] > 0 and taxis[1]:
            phys_axes.append(replace(axis, unit=taxis[1]))
            scale.append(taxis[0])
        else:
            phys_axes.append(replace(axis, unit=None))
            scale.append(1.0)
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=scale)

    def _affine(matrix: np.ndarray, name: str) -> Affine:
        return Affine(
            input=voxel_space,
            output=afni_world(name),
            matrix=np.asarray(matrix, dtype=np.float64)[:3],
        )

    view = header.view
    return [
        vox2phys,
        _affine(header.cardinal_matrix, view + _CARDINAL),
        _affine(header.voxel_to_dicom, view),
    ]
