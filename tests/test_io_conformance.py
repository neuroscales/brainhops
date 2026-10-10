"""Conformance of the file formats to the design of format records (#415).

A file format that follows the design has three layers. Its record class
(`<X>Raw`) holds the header of a file as the format stores it. Its
metadata class (`<X>Metadata`, a subclass of `MetadataFormat`) holds a
record and reads and writes the header alone. Its format object, such as
an image, holds a metadata object and a `raw` array, which is the data as
stored in the file and usually a lazy proxy. The `data` of a format object
is a cached view of `raw`, decoded by a pair of pure functions of the
format. A small format may keep its data in the record instead, as an LTA
file keeps its matrix, and so may a format that is parsed in one pass, as
a FreeSurfer morph keeps its nodes. Such a format has no `raw` field.

Each format that follows the design is listed in `EXEMPLARS`, with what
the checks need to know about it, and every other registered format is
listed in `NOT_MIGRATED`. A test makes sure that the two tables together
name every registered format, so a new format has to choose a table. The
checks are parametrized over the exemplars and over a small format defined
in this module, the toy format, which shows the mechanism end to end in
as little code as possible.
"""

import gzip
import importlib
import os
import pickle
import struct

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import KwOnly, Magic, NoRepr, fields, replace

import brainhops.io as io
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import Metadata
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Scaling,
    Sequence,
    Transformation,
)
from brainhops.io.base._base import Format, register_format
from brainhops.io.base._save import _model
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserContentError,
    WriterError,
    _overrides_from_fileobj,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata import MetadataFormat
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASCoordinatesField
from brainhops.io.transformations.freesurfer.lta import (
    LtaMetadata,
    LtaPhysicalSystem,
    LtaRaw,
    LtaTransformation,
    LtaTransformationPhysToPhys,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
    LtaVoxelSystem,
)
from brainhops.io.transformations.freesurfer.lta._xforms import (
    _matrix_to_disk,
    _matrix_to_model,
)
from brainhops.io.transformations.freesurfer.m3z import (
    M3zMetadata,
    M3zMorph,
)
from brainhops.io.transformations.freesurfer.m3z._xform import (
    _positions_to_disk,
    _positions_to_model,
)

try:
    import nibabel as nb
    from nibabel.arrayproxy import ArrayProxy

    from brainhops.datamodel.transformations import DisplacementField
    from brainhops.io.common.nifti._views import (
        _affine_to_disk,
        _affine_to_model,
        _field_to_disk,
        _field_to_model,
        _image_to_disk,
        _image_to_model,
        _itk_field_to_disk,
        _itk_field_to_model,
    )
    from brainhops.io.images.nifti import NiftiImage, NiftiMetadata
    from brainhops.io.transformations.fsl.fnirt import FnirtWarpField
    from brainhops.io.transformations.itk.nifti import (
        ItkNiftiCoordinatesField,
        ItkNiftiDisplacementField,
    )
    from brainhops.io.transformations.nifti import (
        NiftiRASCoordinatesField,
        NiftiRASDisplacementField,
        NiftiRASToVoxel,
        NiftiVoxelToRAS,
    )
    from brainhops.io.transformations.niftyreg import (
        NiftyRegControlPointGrid,
        NiftyRegDeformationField,
        NiftyRegDisplacementField,
        NiftyRegVelocityField,
        NiftyRegVelocityGrid,
    )
    from brainhops.io.transformations.spm import SpmCoordinatesField
except ImportError:
    nb = None

try:
    import h5py

    from brainhops.io.common.hdf5 import DelayedH5Array
    from brainhops.io.transformations.base.fields import (
        ras_displacement_chain,
    )
    from brainhops.io.transformations.x5 import (
        X5DisplacementField,
        X5Metadata,
        X5Transform,
    )
    from brainhops.io.transformations.x5._blocks import (
        _KINDS as _X5_KINDS,
    )
    from brainhops.io.transformations.x5._blocks import (
        _field_to_disk as _x5_field_to_disk,
    )
    from brainhops.io.transformations.x5._blocks import (
        _field_to_model as _x5_field_to_model,
    )
except ImportError:
    h5py = None

# ----------------------------------------------------------------------
#   TOY FORMAT
# ----------------------------------------------------------------------
# A toy file starts with a header and continues with the data. The header
# holds a magic number, the data type, the shape and the voxel size of the
# data as stored, and a free-form note. The data is stored with its axes
# in reverse order, so that decoding it is not the identity.

_TOY_MAGIC = b"TOY\x01"


def _toy_to_model(raw: ArrayProtocol) -> np.ndarray:
    """Return the data of a toy image from its stored array."""
    return np.asarray(raw).T


def _toy_to_disk(data: ArrayProtocol) -> np.ndarray:
    """Return the stored array of a toy image from its data."""
    return np.ascontiguousarray(np.asarray(data).T)


class ToyRaw(Magic, BinaryFileReader, BinaryFileWriter):
    """Header of a toy file, as the file stores it."""

    shape: tx.Tuple[int, ...] = ()
    """Shape of the stored array, whose axes are in reverse order."""

    dtype: str = "<f8"
    """Data type of the stored array, as a NumPy type string."""

    spacing: tx.Tuple[float, ...] = ()
    """Voxel size along each axis of the stored array."""

    note: str = ""
    """Free-form text that only the record keeps."""

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        with preserve_position(file):
            magic = file.read(len(_TOY_MAGIC))
        return Confidence.CERTAIN if magic == _TOY_MAGIC else Confidence.NO

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "ToyRaw":
        # Only the header is read, and the stream is left after it.
        if file.read(len(_TOY_MAGIC)) != _TOY_MAGIC:
            raise ParserContentError("Not a toy file.")
        dtype = file.read(3).decode("ascii")
        (ndim,) = struct.unpack("<B", file.read(1))
        shape = struct.unpack(f"<{ndim}I", file.read(4 * ndim))
        spacing = struct.unpack(f"<{ndim}d", file.read(8 * ndim))
        (length,) = struct.unpack("<H", file.read(2))
        note = file.read(length).decode("utf-8")
        return cls(shape=shape, dtype=dtype, spacing=spacing, note=note)

    def to_bytes(self, **kwargs) -> bytes:
        ndim = len(self.shape)
        note = self.note.encode("utf-8")
        return b"".join(
            [
                _TOY_MAGIC,
                self.dtype.encode("ascii"),
                struct.pack("<B", ndim),
                struct.pack(f"<{ndim}I", *self.shape),
                struct.pack(f"<{ndim}d", *self.spacing),
                struct.pack("<H", len(note)),
                note,
            ]
        )


class _ToyProxy:
    """Lazy array over the data of a toy file.

    The proxy reads the file each time its values are needed. It keeps a
    path, or the stream that it was given, which the caller keeps open.
    """

    def __init__(
        self,
        source: tx.Any,
        offset: int,
        shape: tx.Tuple[int, ...],
        dtype: str,
    ) -> None:
        self.source = source
        self.offset = offset
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def read_bytes(self) -> bytes:
        """Return the stored data as it is in the file."""
        size = int(np.prod(self.shape)) * self.dtype.itemsize
        if isinstance(self.source, (str, os.PathLike)):
            with open(self.source, "rb") as file:
                file.seek(self.offset)
                return file.read(size)
        with preserve_position(self.source):
            self.source.seek(self.offset)
            return self.source.read(size)

    def __array__(
        self, dtype: tx.Any = None, copy: tx.Optional[bool] = None
    ) -> np.ndarray:
        array = np.frombuffer(self.read_bytes(), self.dtype)
        array = array.reshape(self.shape)
        return array if dtype is None else array.astype(dtype)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        return np.asarray(self)[index]


def _toy_proxy(source: tx.Any, start: int, record: ToyRaw) -> _ToyProxy:
    """Return the proxy of the data that follows a toy header."""
    offset = start + len(record.to_bytes())
    return _ToyProxy(source, offset, record.shape, record.dtype)


def _toy_spacing(transformation: Transformation, ndim: int) -> np.ndarray:
    """Return the voxel size, in the order of the data, of a geometry.

    Raises
    ------
    WriterError
        If the toy format cannot store the transformation.
    """
    if isinstance(transformation, Identity):
        return np.ones(ndim)
    if isinstance(transformation, Scaling):
        return np.asarray(transformation.data, dtype=float)
    raise WriterError("A toy file stores a voxel size, not a general map.")


class ToyMetadata(MetadataFormat, BinaryFileReader, BinaryFileWriter):
    """Metadata of a toy file, which holds its header."""

    EXTENSIONS = (".toy",)

    raw: NoRepr[tx.Optional[ToyRaw]] = None

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return ToyRaw.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "ToyMetadata":
        return cls.from_raw(ToyRaw.from_fileobj(file, **kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        record = self.to_raw()
        if record is None:
            record = ToyRaw()
        return record.to_bytes(**kwargs)


class ToyImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """Image stored in a toy file."""

    EXTENSIONS = (".toy",)

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    metadata: KwOnly[NoRepr[tx.Optional[ToyMetadata]]] = None

    def __post_init__(self, arguments: tx.Any) -> None:
        # The constructor assigns the fields in order, so the default of
        # `raw`, assigned after `data`, erases the array stored by `data=`.
        # The array is therefore stored again. When both are given, `data`
        # takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(cache=True, invalidates=("data",))
    def data(self) -> tx.Optional[np.ndarray]:
        if self.raw is None:
            return None
        return _toy_to_model(self.raw)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.raw = None if value is None else _toy_to_disk(value)

    @smartproperty(unset=(None, "empty"))
    def transformations(self) -> tx.List[Transformation]:
        # The geometry is decoded from the record each time, unless a list
        # of transformations has been assigned.
        if self.metadata is None or self.metadata.raw is None:
            return []
        return [Scaling(list(reversed(self.metadata.raw.spacing)))]

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return ToyMetadata.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def from_filename(cls, filename: tx.Any, **kwargs) -> "ToyImage":
        # The proxy opens the file by name, so it outlives the stream that
        # read the header.
        filename = os.fspath(filename)
        with open(filename, "rb") as file:
            metadata = ToyMetadata.from_fileobj(file, **kwargs)
        proxy = _toy_proxy(filename, 0, metadata.raw)
        return cls(raw=proxy, metadata=metadata)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "ToyImage":
        start = file.tell()
        metadata = ToyMetadata.from_fileobj(file, **kwargs)
        proxy = _toy_proxy(file, start, metadata.raw)
        return cls(raw=proxy, metadata=metadata)

    def to_bytes(self, **kwargs) -> bytes:
        if self.raw is None:
            raise WriterError("A toy image without data cannot be written.")
        # The record of the metadata is the base of the header, and the
        # geometry and the layout of the image are encoded over a copy.
        if self.metadata is None:
            record = ToyRaw()
        else:
            record = self.metadata.to_raw()
        spacing = _toy_spacing(self.transformation, len(self.raw.shape))
        record.shape = tuple(self.raw.shape)
        record.dtype = np.dtype(self.raw.dtype).str
        record.spacing = tuple(float(x) for x in reversed(spacing))
        # A proxy is copied out without decoding, so that an untouched
        # image is written byte for byte.
        if isinstance(self.raw, _ToyProxy):
            data = self.raw.read_bytes()
        else:
            data = np.ascontiguousarray(self.raw).tobytes()
        return record.to_bytes() + data


class _ForeignImage(ImageFormat, SingleScaleImage):
    """Image of another format, which is never registered.

    Its `raw` and `metadata` fields mean nothing to the toy format.
    """

    raw: KwOnly[NoRepr[tx.Any]] = None
    metadata: KwOnly[NoRepr[tx.Any]] = None


class _ForeignMetadata(MetadataFormat):
    """Metadata of another format, which is never registered."""


def _toy_edit_record(record: ToyRaw) -> ToyRaw:
    record.note = "edited"
    return record


def _toy_change_geometry(image: ToyImage) -> None:
    image.transformations = [Scaling([2.0, 3.0, 4.0])]


@pytest.fixture(scope="module", autouse=True)
def _register_the_toy_format() -> tx.Iterator[None]:
    # The toy classes are registered only while this module runs, so that
    # they never reach the readers that other tests use.
    register_format(ToyMetadata)
    register_format(ToyImage)
    yield
    for cls in (ToyMetadata, ToyImage):
        for base in cls.__mro__:
            base.__dict__.get("_REGISTRY", set()).discard(cls)


# ----------------------------------------------------------------------
#   TABLES
# ----------------------------------------------------------------------


class Exemplar(tx.NamedTuple):
    """What the checks need to know about a format that follows the design.

    The format is built from `sample()`, with `build` or else with
    `cls(data=...)`, and written to a file with the given suffix, which is
    the file that the checks read.
    """

    metadata: type
    """Metadata class of the format."""

    suffix: str
    """Suffix of the files that the checks write, such as `".nii.gz"`."""

    proxies: tx.Tuple[type, ...]
    """Types that `raw` may have just after a file is read."""

    to_model: tx.Callable[[tx.Any], tx.Any]
    """Function that decodes `data` from `raw`."""

    to_disk: tx.Callable[[tx.Any], tx.Any]
    """Function that encodes `raw` from `data`."""

    sample: tx.Callable[[], np.ndarray]
    """Function that returns data that the format can store."""

    edit_record: tx.Callable[[tx.Any], tx.Any]
    """Function that edits a copy of a record outside of its geometry."""

    record_edited: tx.Callable[[tx.Any], bool]
    """Function that tells whether a record carries that edit."""

    geometry: tx.Callable[[tx.Any], np.ndarray]
    """Function that returns the geometry of an object as numbers."""

    change_geometry: tx.Callable[[tx.Any], None]
    """Function that changes the geometry of an object in place."""

    foreign: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that builds an object of another format from data.

    It is `None` for a format that refuses every conversion from another
    format, as NiftyReg, ITK and FNIRT do until #312.
    """

    build: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that builds an object of the format from data alone.

    It is needed by a format whose data model is a chain, which has no
    `data` argument, and `cls(data=...)` is used otherwise.
    """

    options: tx.FrozenSet[str] = frozenset()
    """Reader options that the format declares as fields."""

    derived: tx.Tuple[str, ...] = ()
    """Cached views that setting `data` must delete, besides `data`."""

    binary: bool = True
    """Whether the format is read in binary mode.

    A text format decodes its bytes before it parses them, so it reads
    bytes as text rather than through a stream, and the check of a real
    `from_fileobj` applies to binary formats only.
    """

    prefix: str = ""
    """Prefix of the names of the files that the checks write.

    A format that is told from others by the names of its files, such as
    SPM by `y_`, needs it so that the generic writer chooses it.
    """

    stored: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that returns the stored data of a format without `raw`.

    The data of a small format may live in the record of its metadata, as
    the matrix of an LTA file does, and such a format has no `raw` field.
    The checks then read the stored data with this function, which returns
    the data as the writer would store it, and `proxies` are the types
    that it returns just after a file is read. An object built from data
    still has no metadata, because it holds its data in the data model
    until it is written. The function is `None` for a format that stores
    its data in `raw`.
    """

    view: str = "data"
    """Name of the cached view that is decoded from the stored data.

    A format whose model is a chain, such as X5, decodes and caches its
    `transformations` instead of a `data` array.
    """

    data: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that returns the data of an object, or `None` for `data`.

    A format whose model is a chain has no `data`, so the checks read the
    array of one of its elements instead.
    """

    set_data: tx.Optional[tx.Callable[[tx.Any, tx.Any], None]] = None
    """Function that gives an object new data, or `None` to set `data`."""

    cut: tx.Optional[tx.Callable[[tx.Any, tx.Any], None]] = None
    """Function that copies a file without its data, or `None` to cut it.

    The function receives the path of a file and the path of the copy. By
    default the copy holds the bytes that the metadata writes, which are
    the header of the file. An HDF5 file has no header before its data,
    so an X5 copy keeps every group and attribute and stores its fields
    in an external file that does not exist.
    """

    header: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that summarizes metadata without reading the data.

    It is `None` when the bytes that the metadata writes are the header of
    the file, which the checks then compare.
    """


def _nifti_edit_record(record: tx.Any) -> tx.Any:
    record.header["descrip"] = b"edited"
    return record


def _nifti_geometry(image: tx.Any) -> np.ndarray:
    return np.asarray(image.transformation.to(Affine).matrix, dtype=float)


def _nifti_change_geometry(image: tx.Any) -> None:
    image.transformations = [Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])]


def _matrix_geometry(xform: tx.Any) -> np.ndarray:
    return np.asarray(xform.matrix, dtype=float)


def _matrix_change_geometry(xform: tx.Any) -> None:
    xform.data = np.diag([2.0, 3.0, 4.0, 1.0])[:3]


def _data_geometry(xform: tx.Any) -> np.ndarray:
    # A format whose model holds no grid of its own, such as a field of
    # coordinates, has its data as its geometry.
    return np.asarray(xform.data, dtype=float)


def _data_change_geometry(xform: tx.Any) -> None:
    xform.data = np.asarray(xform.data) * 2 + 1


def _chain_geometry(xform: tx.Any) -> np.ndarray:
    # The first link of each chain maps the world to the voxels of the grid.
    return np.asarray(xform.transformations[0].homogeneous_matrix)


def _chain_change_geometry(xform: tx.Any) -> None:
    # The grid of the chain is replaced by a scaled one, around the same
    # field.
    grid = np.diag([2.0, 3.0, 4.0, 1.0])
    chain = list(xform.transformations)
    first, last = chain[0], chain[-1]
    chain[0] = type(first)(
        matrix=np.linalg.inv(grid)[:-1],
        input=first.input,
        output=first.output,
    )
    if len(chain) == 3:
        chain[-1] = type(last)(
            matrix=grid[:-1], input=last.input, output=last.output
        )
    xform.transformations = tuple(chain)


_VOXEL_TO_RAS = np.array(
    [[2.0, 0.0, 0.0, 1.0], [0.0, 4.0, 0.0, 2.0], [0.0, 0.0, 8.0, 3.0]]
)
"""An affine whose inverse is exact in single precision, as NIfTI stores
it."""


def _vectors() -> np.ndarray:
    # Whole numbers, so that adding and removing the grid of a field that
    # stores positions is exact in single precision.
    return np.arange(72, dtype="float32").reshape(2, 3, 4, 3)


def _ras_displacements(data: np.ndarray) -> tx.Any:
    return Sequence(
        [
            RASToVoxel(matrix=np.eye(4)[:-1]),
            DisplacementField(
                data, input=RASToVoxel().output, output=RASToVoxel().output
            ),
            VoxelToRAS(matrix=np.eye(4)[:-1]),
        ]
    )


def _spm_coordinates(data: np.ndarray) -> tx.Any:
    return Sequence(
        [RASToVoxel(matrix=np.eye(4)[:-1]), RASCoordinatesField(field=data)]
    )


EXEMPLARS: tx.Dict[type, Exemplar] = {}
"""The registered formats that follow the design."""

if nb is not None:
    _NIFTI = Exemplar(
        metadata=NiftiMetadata,
        suffix=".nii",
        proxies=(ArrayProxy,),
        to_model=_image_to_model,
        to_disk=_image_to_disk,
        sample=lambda: np.arange(24, dtype="float32").reshape(2, 3, 4),
        edit_record=_nifti_edit_record,
        record_edited=lambda raw: raw.header["descrip"].item() == b"edited",
        geometry=_nifti_geometry,
        change_geometry=_nifti_change_geometry,
    )
    _NIFTI_FIELD = _NIFTI._replace(
        to_model=_field_to_model,
        to_disk=_field_to_disk,
        sample=_vectors,
        geometry=_chain_geometry,
        change_geometry=_chain_change_geometry,
        derived=("transformations",),
    )
    _NIFTYREG = _NIFTI_FIELD
    _ITK = _NIFTI_FIELD._replace(
        to_model=_itk_field_to_model, to_disk=_itk_field_to_disk
    )

    EXEMPLARS[NiftiImage] = _NIFTI._replace(
        foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
    )
    EXEMPLARS[NiftiVoxelToRAS] = _NIFTI._replace(
        # The matrix of an affine lives in the header, so a read affine has
        # no array.
        proxies=(type(None),),
        to_model=_affine_to_model,
        to_disk=_affine_to_disk,
        sample=lambda: _VOXEL_TO_RAS,
        geometry=_matrix_geometry,
        change_geometry=_matrix_change_geometry,
        foreign=lambda data: Affine(data),
        derived=("homogeneous_matrix",),
    )
    EXEMPLARS[NiftiRASCoordinatesField] = _NIFTI_FIELD._replace(
        geometry=_data_geometry,
        change_geometry=_data_change_geometry,
        foreign=lambda data: RASCoordinatesField(field=data),
        derived=("values",),
    )
    EXEMPLARS[NiftiRASDisplacementField] = _NIFTI_FIELD._replace(
        build=lambda data: NiftiRASDisplacementField(raw=_field_to_disk(data)),
        foreign=_ras_displacements,
        options=frozenset({"log", "steps"}),
    )
    EXEMPLARS[SpmCoordinatesField] = _NIFTI_FIELD._replace(
        build=lambda data: SpmCoordinatesField(raw=_field_to_disk(data)),
        foreign=_spm_coordinates,
        prefix="y_",
    )
    for _cls in (
        NiftyRegControlPointGrid,
        NiftyRegDeformationField,
        NiftyRegDisplacementField,
        NiftyRegVelocityField,
        NiftyRegVelocityGrid,
    ):
        EXEMPLARS[_cls] = _NIFTYREG._replace(
            build=lambda data, cls=_cls: cls(raw=_field_to_disk(data))
        )
    for _cls in (ItkNiftiCoordinatesField, ItkNiftiDisplacementField):
        EXEMPLARS[_cls] = _ITK._replace(
            build=lambda data, cls=_cls: cls(raw=_itk_field_to_disk(data))
        )
    EXEMPLARS[FnirtWarpField] = _NIFTI._replace(
        sample=_vectors,
        # FNIRT holds no grid that the model encodes, so its data stands in.
        geometry=_data_geometry,
        change_geometry=_data_change_geometry,
        build=lambda data: FnirtWarpField(raw=data),
        options=frozenset({"moving", "reference", "deformation_type"}),
    )


def _lta_stored(xform: tx.Any) -> tx.Any:
    # The matrix of an LTA file lives in the record, which is parsed in one
    # pass, so it is a tuple of rows just after a read. `to_raw` returns
    # that record unless the transformation holds a matrix of its own.
    return xform.to_raw().affine.matrix


def _lta_edit_record(record: tx.Any) -> tx.Any:
    record.sigma = 2.5
    return record


def _lta_volumes(cls: type) -> tx.Dict[str, tx.Any]:
    """Return the systems of two anonymous volumes, as keyword arguments."""
    return {
        "input": cls.from_raw(LtaRaw.SrcVolumeInfo()),
        "output": cls.from_raw(LtaRaw.DstVolumeInfo()),
    }


_LTA = Exemplar(
    metadata=LtaMetadata,
    suffix=".lta",
    # An LTA file is parsed in one pass and has no lazy data, so the checks
    # read the matrix of the record that would be written. After a read,
    # that record is the one the metadata holds. The metadata, which holds
    # the whole file, never reads data because the file has none beside it.
    proxies=(tuple,),
    to_model=_matrix_to_model,
    to_disk=_matrix_to_disk,
    sample=lambda: _VOXEL_TO_RAS,
    edit_record=_lta_edit_record,
    record_edited=lambda raw: raw.sigma == 2.5,
    geometry=_matrix_geometry,
    change_geometry=_matrix_change_geometry,
    foreign=lambda data: Affine(data, input=RASmm(), output=RASmm()),
    build=lambda data: LtaTransformation(data, input=RASmm(), output=RASmm()),
    derived=("homogeneous_matrix",),
    binary=False,
    stored=_lta_stored,
)
EXEMPLARS[LtaTransformation] = _LTA


# The model of an X5 file is a chain, whose first element here is a field
# of displacements on an identity grid, so that its displacements in voxel
# units are the stored RAS displacements.


def _x5_field(data: tx.Any) -> tx.Any:
    return X5DisplacementField.from_ras(data, np.eye(4))


def _x5_data(xform: tx.Any) -> tx.Any:
    return xform.transformations[0].displacement.data


def _x5_set_data(xform: tx.Any, value: tx.Any) -> None:
    xform.transformations = (_x5_field(value),)


def _x5_stored(xform: tx.Any) -> tx.Any:
    # The field of an X5 file lives in the record, and a chain that was
    # assigned is encoded into a new record.
    return xform.to_raw().nodes[0].transform


def _x5_edit_record(record: tx.Any) -> tx.Any:
    record.header.attrs["Edited"] = "yes"
    return record


def _x5_geometry(xform: tx.Any) -> np.ndarray:
    return _chain_geometry(xform.transformations[0])


def _x5_change_geometry(xform: tx.Any) -> None:
    grid = np.diag([2.0, 3.0, 4.0, 1.0])
    field = X5DisplacementField.from_ras(np.asarray(_x5_data(xform)), grid)
    xform.transformations = (field,)


def _x5_header(metadata: tx.Any) -> tx.Any:
    """Return the record of X5 metadata without the values of its fields.

    The shape of a lazy field is read from the file, but its values are
    not.
    """
    nodes = [
        (
            node.type,
            node.subtype,
            node.representation,
            node.metadata,
            node.dimension_kinds,
            type(node.transform),
            tuple(node.transform.shape),
            np.asarray(node.domain.mapping).tolist(),
            node.domain.size,
        )
        for node in metadata.raw.nodes
    ]
    return metadata.raw.header, nodes


def _x5_cut(source: tx.Any, target: tx.Any) -> None:
    """Copy an X5 file whose fields cannot be read.

    Every group and attribute is copied, and so is every dataset of at
    most two dimensions. A larger dataset is declared in an external file
    that is never written, so that its shape is known but reading its
    values fails.
    """
    missing = str(target) + ".missing"
    with h5py.File(source, "r") as old, h5py.File(target, "w") as new:
        new.attrs.update(old.attrs)

        def copy(name: str, item: tx.Any) -> None:
            if isinstance(item, h5py.Group):
                copied = new.require_group(name)
            elif item.ndim > 2:
                copied = new.create_dataset(
                    name,
                    shape=item.shape,
                    dtype=item.dtype,
                    external=[(missing, 0, h5py.h5f.UNLIMITED)],
                )
            else:
                copied = new.create_dataset(name, data=item[()])
            copied.attrs.update(item.attrs)

        old.visititems(copy)


if h5py is not None:
    EXEMPLARS[X5Transform] = Exemplar(
        metadata=X5Metadata,
        suffix=".x5",
        proxies=(DelayedH5Array,),
        to_model=lambda raw: _x5_field_to_model(raw, _X5_KINDS),
        to_disk=_x5_field_to_disk,
        sample=_vectors,
        edit_record=_x5_edit_record,
        record_edited=lambda raw: raw.header.attrs.get("Edited") == "yes",
        geometry=_x5_geometry,
        change_geometry=_x5_change_geometry,
        foreign=lambda data: Sequence(
            [Sequence(ras_displacement_chain(data, np.eye(4)))]
        ),
        build=lambda data: X5Transform([_x5_field(data)]),
        options=frozenset({"chain", "position"}),
        stored=_x5_stored,
        view="transformations",
        data=_x5_data,
        set_data=_x5_set_data,
        cut=_x5_cut,
        header=_x5_header,
    )

# The model of a morph is a chain, whose field holds the positions of the
# nodes. A morph is a single gzip stream in which the nodes follow the
# header directly, so it is parsed in one pass and its positions live in
# the record. The exemplar is a chain of RAS coordinates, which a morph
# built from a chain can write without the shape of a source image.


def _m3z_chain(data: tx.Any) -> tx.Tuple[tx.Any, ...]:
    return (RASToVoxel(matrix=np.eye(4)[:-1]), RASCoordinatesField(field=data))


def _m3z_data(xform: tx.Any) -> tx.Any:
    return xform.transformations[1].data


def _m3z_set_data(xform: tx.Any, value: tx.Any) -> None:
    chain = list(xform.transformations)
    chain[1] = RASCoordinatesField(field=value)
    xform.transformations = tuple(chain)


def _m3z_stored(xform: tx.Any) -> tx.Any:
    # The positions of a morph live in the record, and a chain that was
    # assigned is encoded into a new record.
    return xform.to_raw().positions


def _m3z_edit_record(record: tx.Any) -> tx.Any:
    # A record is frozen, so the edit is made with `replace`.
    return replace(record, exp_k=7.5)


_M3Z = Exemplar(
    metadata=M3zMetadata,
    suffix=".m3z",
    # A morph is parsed in one pass and has no lazy data, so the positions
    # are an array of the record just after a read.
    proxies=(np.ndarray,),
    to_model=_positions_to_model,
    to_disk=_positions_to_disk,
    sample=_vectors,
    edit_record=_m3z_edit_record,
    record_edited=lambda raw: raw.exp_k == 7.5,
    geometry=_chain_geometry,
    change_geometry=_chain_change_geometry,
    foreign=lambda data: Sequence(list(_m3z_chain(data))),
    build=lambda data: M3zMorph(_m3z_chain(data)),
    stored=_m3z_stored,
    view="transformations",
    data=_m3z_data,
    set_data=_m3z_set_data,
    # The metadata reads the whole file, because the nodes sit inside the
    # gzip stream, right after the header. The bytes that the metadata
    # writes are therefore the whole file, and the check that metadata
    # never reads data compares a full copy of the file: it only shows
    # that `M3zMetadata` and the dispatcher read the file alone.
)
EXEMPLARS[M3zMorph] = _M3Z

NOT_MIGRATED: tx.Tuple[str, ...] = (
    "brainhops.io.images.afni.AfniImage",
    "brainhops.io.images.freesurfer.mgh.MghImage",
    "brainhops.io.images.minc.Minc1Image",
    "brainhops.io.images.minc.Minc2Image",
    "brainhops.io.images.mrtrix.MrtrixImage",
    "brainhops.io.images.nrrd.AttachedNrrdImage",
    "brainhops.io.images.nrrd.DetachedNrrdImage",
    "brainhops.io.images.openslide.AperioImage",
    "brainhops.io.images.openslide.AperioMultiScaleImage",
    "brainhops.io.images.openslide.DicomWsiImage",
    "brainhops.io.images.openslide.DicomWsiMultiScaleImage",
    "brainhops.io.images.openslide.GenericTiffImage",
    "brainhops.io.images.openslide.GenericTiffMultiScaleImage",
    "brainhops.io.images.openslide.HamamatsuImage",
    "brainhops.io.images.openslide.HamamatsuMultiScaleImage",
    "brainhops.io.images.openslide.LeicaImage",
    "brainhops.io.images.openslide.LeicaMultiScaleImage",
    "brainhops.io.images.openslide.MiraxImage",
    "brainhops.io.images.openslide.MiraxMultiScaleImage",
    "brainhops.io.images.openslide.PhilipsImage",
    "brainhops.io.images.openslide.PhilipsMultiScaleImage",
    "brainhops.io.images.openslide.SakuraImage",
    "brainhops.io.images.openslide.SakuraMultiScaleImage",
    "brainhops.io.images.openslide.TrestleImage",
    "brainhops.io.images.openslide.TrestleMultiScaleImage",
    "brainhops.io.images.openslide.VentanaImage",
    "brainhops.io.images.openslide.VentanaMultiScaleImage",
    "brainhops.io.images.openslide.ZeissImage",
    "brainhops.io.images.openslide.ZeissMultiScaleImage",
    "brainhops.io.images.pillow.PillowImage",
    "brainhops.io.images.tiff.TiffImage",
    "brainhops.io.images.tiff.TiffMultiScaleImage",
    "brainhops.io.images.zarr.OmeZarrImage",
    "brainhops.io.images.zarr.ZarrImage",
    "brainhops.io.transformations.elastix.ElastixParameterTransform",
    "brainhops.io.transformations.elastix.ElastixTomlTransform",
    "brainhops.io.transformations.fsl.flirt.FlirtTransform",
    "brainhops.io.transformations.itk.h5.H5Transform",
    "brainhops.io.transformations.itk.mat.MatTransform",
    "brainhops.io.transformations.itk.tfm.TfmTransform",
    "brainhops.io.transformations.matrix.CsvMatrixAffine",
    "brainhops.io.transformations.matrix.Mat73MatrixAffine",
    "brainhops.io.transformations.matrix.MatLegacyMatrixAffine",
    "brainhops.io.transformations.matrix.NpyMatrixAffine",
    "brainhops.io.transformations.matrix.NpzMatrixAffine",
    "brainhops.io.transformations.matrix.TsvMatrixAffine",
    "brainhops.io.transformations.matrix.TxtMatrixAffine",
    "brainhops.io.transformations.niftyreg.NiftyRegAffine",
    "brainhops.io.transformations.zarr.OmeZarrField",
)
"""The registered formats that do not follow the design yet."""

TEST_EXEMPLARS: tx.Dict[type, Exemplar] = {
    ToyImage: Exemplar(
        metadata=ToyMetadata,
        suffix=".toy",
        proxies=(_ToyProxy,),
        to_model=_toy_to_model,
        to_disk=_toy_to_disk,
        sample=lambda: np.arange(24.0).reshape(2, 3, 4),
        edit_record=_toy_edit_record,
        record_edited=lambda record: record.note == "edited",
        geometry=lambda image: _toy_spacing(image.transformation, image.ndim),
        change_geometry=_toy_change_geometry,
        foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
    ),
}
"""Formats defined by the tests, which the checks also run on."""

VARIANTS: tx.List[tx.Tuple[str, type, Exemplar]] = []
"""Other files of the exemplars, which the checks also run on.

Each entry names the case and gives the class and what the checks need to
know about that file, such as another suffix.
"""

if nb is not None:
    VARIANTS.append(
        (
            "NiftiImage-gz",
            NiftiImage,
            EXEMPLARS[NiftiImage]._replace(suffix=".nii.gz"),
        )
    )
    VARIANTS.append(
        (
            "NiftiRASDisplacementField-gz",
            NiftiRASDisplacementField,
            EXEMPLARS[NiftiRASDisplacementField]._replace(suffix=".nii.gz"),
        )
    )

# The views of an LTA file read any LTA file, so they are not registered,
# and each writes its own type once its matrix is set.
VARIANTS.append(
    (
        "LtaTransformationVoxToVox",
        LtaTransformationVoxToVox,
        _LTA._replace(
            foreign=lambda data: Affine(data, **_lta_volumes(LtaVoxelSystem)),
            build=LtaTransformationVoxToVox,
        ),
    )
)
VARIANTS.append(
    (
        "LtaTransformationPhysToPhys",
        LtaTransformationPhysToPhys,
        _LTA._replace(
            foreign=lambda data: Affine(
                data, **_lta_volumes(LtaPhysicalSystem)
            ),
            build=LtaTransformationPhysToPhys,
        ),
    )
)
VARIANTS.append(
    (
        "LtaTransformationRASToRAS",
        LtaTransformationRASToRAS,
        _LTA._replace(build=LtaTransformationRASToRAS),
    )
)

# A morph whose name ends in `.m3d` is written without compression.
VARIANTS.append(("M3zMorph-m3d", M3zMorph, _M3Z._replace(suffix=".m3d")))

CASES = [
    pytest.param(cls, exemplar, id=cls.__name__)
    for cls, exemplar in {**EXEMPLARS, **TEST_EXEMPLARS}.items()
] + [pytest.param(cls, exemplar, id=name) for name, cls, exemplar in VARIANTS]

# Names that belonged to the parsers that the design removes.
_PARSER_STATE = {"image", "header", "_header", "struct", "dataobj", "node"}


def _resolve(name: str) -> tx.Optional[type]:
    """Import a class by its dotted name.

    `None` is returned when the module needs an optional dependency that
    is not installed, since the format is then not registered either.
    """
    module, _, attr = name.rpartition(".")
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as e:
        if (e.name or "").startswith("brainhops"):
            raise
        return None


def _decompressed(path: tx.Any) -> bytes:
    # A gzipped file, such as a `.nii.gz` or a `.m3z` file, is compared
    # after decompression, since only its content must be kept.
    with open(path, "rb") as file:
        content = file.read()
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return content


def _write_header(path: tx.Any, header: bytes) -> None:
    if str(path).endswith(".gz"):
        header = gzip.compress(header)
    with open(path, "wb") as file:
        file.write(header)


def _path(tmp_path: tx.Any, exemplar: Exemplar, name: str) -> tx.Any:
    """Return the path of a file that a check writes."""
    return tmp_path / (exemplar.prefix + name + exemplar.suffix)


def _stored(exemplar: Exemplar, obj: tx.Any) -> tx.Any:
    """Return the data of an object as the file stores it."""
    if exemplar.stored is None:
        return obj.raw
    return exemplar.stored(obj)


def _data(exemplar: Exemplar, obj: tx.Any) -> tx.Any:
    """Return the data of an object."""
    if exemplar.data is None:
        return obj.data
    return exemplar.data(obj)


def _set_data(exemplar: Exemplar, obj: tx.Any, value: tx.Any) -> None:
    """Give an object new data."""
    if exemplar.set_data is None:
        obj.data = value
    else:
        exemplar.set_data(obj, value)


def _header(exemplar: Exemplar, metadata: tx.Any) -> tx.Any:
    """Return what the metadata knows about a file, without its data."""
    if exemplar.header is None:
        return metadata.to_bytes()
    return exemplar.header(metadata)


def _cut(
    exemplar: Exemplar, source: tx.Any, metadata: tx.Any, target: tx.Any
) -> None:
    """Write a copy of a file whose data cannot be read."""
    if exemplar.cut is None:
        _write_header(target, metadata.to_bytes())
    else:
        exemplar.cut(source, target)


def _built(cls: type, exemplar: Exemplar, data: tx.Any) -> tx.Any:
    """Build an object of the format from data alone."""
    if exemplar.build is None:
        return cls(data=data)
    return exemplar.build(data)


def _saved(
    cls: type, exemplar: Exemplar, tmp_path: tx.Any, name: str = "src"
) -> tx.Any:
    """Write a new object of the format and return the path of its file."""
    path = _path(tmp_path, exemplar, name)
    _built(cls, exemplar, exemplar.sample()).save(path)
    return path


# ----------------------------------------------------------------------
#   REGISTRY
# ----------------------------------------------------------------------


def test_every_registered_format_is_in_one_table() -> None:
    assert len(set(NOT_MIGRATED)) == len(NOT_MIGRATED)
    not_migrated = {_resolve(name) for name in NOT_MIGRATED} - {None}
    assert not set(EXEMPLARS) & not_migrated
    registered = {
        cls
        for cls in Format._REGISTRY | MetadataFormat._REGISTRY
        if cls.__module__.startswith("brainhops.")
    }
    # The metadata class of an exemplar is registered with it.
    metadata = {exemplar.metadata for exemplar in EXEMPLARS.values()}
    assert registered == set(EXEMPLARS) | metadata | not_migrated


def test_the_metadata_registry_is_isolated() -> None:
    assert MetadataFormat._is_dispatcher()
    assert ToyMetadata in MetadataFormat._REGISTRY
    assert ToyMetadata not in Format._REGISTRY
    assert MetadataFormat not in Format._REGISTRY


# ----------------------------------------------------------------------
#   CHECKS
# ----------------------------------------------------------------------


def _init_names(klass: type) -> tx.Set[str]:
    """Return the names of the fields that the constructor takes."""
    return {field.public_name for field in fields(klass) if field.init}


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_01_no_parser_state(cls: type, exemplar: Exemplar) -> None:
    names = _init_names
    allowed = names(_model(cls)) | {"raw", "metadata"} | exemplar.options
    assert names(cls) <= allowed
    assert not names(cls) & _PARSER_STATE
    assert names(exemplar.metadata) <= names(Metadata) | {"raw"}
    assert not names(exemplar.metadata) & _PARSER_STATE


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_02_load_is_lazy(cls: type, exemplar: Exemplar, tmp_path) -> None:  # noqa: ANN001
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    assert isinstance(_stored(exemplar, loaded), exemplar.proxies)
    if exemplar.stored is not None:
        assert "raw" not in _init_names(cls)
    repr(loaded)
    assert "_cache_" + exemplar.view not in vars(loaded)
    assert isinstance(loaded.metadata, exemplar.metadata)
    data = _data(exemplar, loaded)
    assert np.array_equal(np.asarray(data), exemplar.sample())
    # The view is decoded once, and reading it leaves the proxy in place.
    assert _data(exemplar, loaded) is data
    assert isinstance(_stored(exemplar, loaded), exemplar.proxies)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_03_setting_data_stores_raw(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    views = (exemplar.view,) + exemplar.derived
    for name in views:
        getattr(loaded, name)
        assert "_cache_" + name in vars(loaded)
    value = exemplar.sample() + 1
    _set_data(exemplar, loaded, value)
    for name in views:
        assert "_cache_" + name not in vars(loaded)
    stored = np.asarray(_stored(exemplar, loaded))
    assert np.array_equal(stored, exemplar.to_disk(value))
    assert np.array_equal(np.asarray(_data(exemplar, loaded)), value)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_04_the_view_round_trips(cls: type, exemplar: Exemplar) -> None:
    value = exemplar.sample()
    back = exemplar.to_model(exemplar.to_disk(value))
    assert back.shape == value.shape
    assert np.array_equal(back, value)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_05_untouched_save_is_byte_identical(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    source = _saved(cls, exemplar, tmp_path)
    target = _path(tmp_path, exemplar, "copy")
    cls.load(source).save(target)
    assert _decompressed(target) == _decompressed(source)
    # The generic writer copies the object with `from_instance` first.
    target = _path(tmp_path, exemplar, "generic")
    io.save(cls.load(source), target)
    assert _decompressed(target) == _decompressed(source)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_06_metadata_never_reads_data(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    source = _saved(cls, exemplar, tmp_path)
    metadata = exemplar.metadata.load(source)
    header = _header(exemplar, metadata)
    # The data of the copy cannot be read, so reading any data would fail.
    # The copy of most formats is the file cut after its header.
    truncated = _path(tmp_path, exemplar, "header")
    _cut(exemplar, source, metadata, truncated)
    assert _header(exemplar, exemplar.metadata.load(truncated)) == header
    dispatched = MetadataFormat.load(truncated)
    assert isinstance(dispatched, exemplar.metadata)
    assert _header(exemplar, dispatched) == header


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_07_metadata_pickles(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    clone = pickle.loads(pickle.dumps(loaded.metadata))
    assert type(clone) is type(loaded.metadata)
    assert clone.to_bytes() == loaded.metadata.to_bytes()


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_08_an_object_built_from_data_saves(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    value = exemplar.sample()
    built = _built(cls, exemplar, value)
    assert built.metadata is None
    stored = np.asarray(_stored(exemplar, built))
    assert np.array_equal(stored, exemplar.to_disk(value))
    path = _path(tmp_path, exemplar, "built")
    built.save(path)
    assert np.array_equal(np.asarray(_data(exemplar, cls.load(path))), value)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_09_copies_keep_raw_and_metadata(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    # Within the format, `from_instance` copies the lazy `raw` and does not
    # read the data.
    copy = cls.from_instance(loaded)
    assert _stored(exemplar, copy) is _stored(exemplar, loaded)
    assert copy.metadata is loaded.metadata
    assert "_cache_" + exemplar.view not in vars(loaded)
    # `replace` passes the data, which takes precedence over `raw`, so the
    # copy holds a decoded array with the same content.
    copy = replace(loaded)
    assert copy.metadata is loaded.metadata
    stored = np.asarray(_stored(exemplar, copy))
    assert np.array_equal(stored, np.asarray(_stored(exemplar, loaded)))
    # The record and the stored array of another format mean nothing to
    # this format, so they are reset. A model that holds data stores it
    # again, and a model that holds a chain keeps the chain, which the
    # writer encodes.
    if exemplar.foreign is None:
        return
    value = exemplar.sample()
    converted = cls.from_instance(exemplar.foreign(value))
    assert converted.metadata is None
    if "data" in _init_names(_model(cls)):
        expected = exemplar.to_disk(value)
        stored = np.asarray(_stored(exemplar, converted))
        assert np.array_equal(stored, expected)
        assert np.array_equal(np.asarray(converted.data), value)
    path = _path(tmp_path, exemplar, "converted")
    converted.save(path)
    assert np.array_equal(np.asarray(_data(exemplar, cls.load(path))), value)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_10_record_edits_survive_and_geometry_wins(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    source = _saved(cls, exemplar, tmp_path)
    loaded = cls.load(source)
    edited = exemplar.edit_record(loaded.metadata.to_raw())
    loaded.metadata = exemplar.metadata.from_raw(edited)
    exemplar.change_geometry(loaded)
    header = loaded.metadata.to_bytes()
    target = _path(tmp_path, exemplar, "edited")
    loaded.save(target)
    # Writing encodes into a copy of the record, never into the record.
    assert loaded.metadata.to_bytes() == header
    back = cls.load(target)
    assert exemplar.record_edited(back.metadata.raw)
    assert np.allclose(exemplar.geometry(back), exemplar.geometry(loaded))
    original = exemplar.geometry(cls.load(source))
    assert not np.allclose(exemplar.geometry(back), original)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_11_public_surface(cls: type, exemplar: Exemplar) -> None:
    if cls in TEST_EXEMPLARS:
        pytest.skip("The test formats are not part of a package.")
    for klass in (cls, exemplar.metadata):
        package, _, module = klass.__module__.rpartition(".")
        assert module.startswith("_"), klass
        exported = importlib.import_module(package).__all__
        assert klass.__name__ in exported, klass
        assert not [name for name in exported if name.startswith("_")]


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_12_adapter_contract(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    for klass in (cls, exemplar.metadata):
        assert not klass._is_dispatcher()
        if exemplar.binary:
            # `from_bytes` reads through a real `from_fileobj` only.
            assert _overrides_from_fileobj(klass)
            assert klass._READ_MODE == "rb"
        else:
            assert klass._READ_MODE == "rt"
    content = _saved(cls, exemplar, tmp_path).read_bytes()
    data = _data(exemplar, cls.from_bytes(content))
    assert np.array_equal(np.asarray(data), exemplar.sample())
    metadata = exemplar.metadata.from_bytes(content)
    assert isinstance(metadata, exemplar.metadata)


# ----------------------------------------------------------------------
#   THE TOY FORMAT THROUGH THE GENERIC ENTRY POINTS
# ----------------------------------------------------------------------


def test_the_generic_loader_reads_a_toy_image(tmp_path) -> None:  # noqa: ANN001
    data = np.arange(6.0).reshape(2, 3)
    path = tmp_path / "image.toy"
    io.save(SingleScaleImage(data, [Scaling([2.0, 3.0])]), path)
    image = io.load(path)
    assert isinstance(image, ToyImage)
    assert isinstance(image.raw, _ToyProxy)
    assert np.array_equal(image.data, data)
    assert np.allclose(image.transformation.data, [2.0, 3.0])
    assert MetadataFormat.sniff(path) is ToyMetadata


def test_data_takes_precedence_over_raw() -> None:
    data = np.arange(6.0).reshape(2, 3)
    image = ToyImage(data, raw=np.zeros((5, 5)))
    assert np.array_equal(image.raw, data.T)
    assert np.array_equal(ToyImage(raw=data.T).data, data)


def test_a_record_is_copied_before_it_is_changed() -> None:
    record = ToyRaw(shape=(3, 2), spacing=(1.0, 1.0), note="kept")
    metadata = ToyMetadata.from_raw(record)
    assert metadata.raw is record
    copy = metadata.to_raw()
    assert copy == record and copy is not record
    image = ToyImage(np.zeros((2, 3)), metadata=metadata)
    image.transformations = [Scaling([4.0, 5.0])]
    image.to_bytes()
    assert record == ToyRaw(shape=(3, 2), spacing=(1.0, 1.0), note="kept")


def test_copied_metadata_keeps_only_a_record_of_its_own_format() -> None:
    metadata = ToyMetadata.from_raw(ToyRaw(note="toy"))
    assert ToyMetadata.from_instance(metadata).raw is metadata.raw
    assert ToyMetadata.from_instance(_ForeignMetadata(raw="other")).raw is None
    assert _ForeignMetadata.from_instance(metadata).raw is None


# ----------------------------------------------------------------------
#   NIFTI
# ----------------------------------------------------------------------

needs_nibabel = pytest.mark.skipif(nb is None, reason="needs nibabel")


@needs_nibabel
@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_an_untouched_scaled_nifti_is_saved_byte_for_byte(
    tmp_path,  # noqa: ANN001
    suffix: str,
) -> None:
    # The values are stored as integers with a slope and an intercept, and
    # the copy must store the same integers with the same scaling.
    values = np.linspace(-3.0, 7.0, 24).reshape(2, 3, 4)
    source = tmp_path / ("scaled" + suffix)
    NiftiImage(data=values).save(source, dtype="int16")
    stored = nb.load(str(source))
    assert stored.get_data_dtype() == np.dtype("int16")
    assert stored.dataobj.slope != 1.0
    target = tmp_path / ("copy" + suffix)
    NiftiImage.load(source).save(target)
    assert _decompressed(target) == _decompressed(source)
    target = tmp_path / ("generic" + suffix)
    io.save(NiftiImage.load(source), target)
    assert _decompressed(target) == _decompressed(source)
    assert np.allclose(NiftiImage.load(target).data, stored.get_fdata())


@needs_nibabel
def test_an_untouched_nifti_keeps_its_extensions_and_description(
    tmp_path,  # noqa: ANN001
) -> None:
    image = nb.Nifti1Image(np.zeros((2, 3, 4), "float32"), np.eye(4))
    image.header["descrip"] = b"kept"
    image.header["cal_max"] = 5.0
    comment = nb.nifti1.Nifti1Extension("comment", b"a comment")
    image.header.extensions.append(comment)
    source = tmp_path / "source.nii"
    nb.save(image, str(source))
    target = tmp_path / "copy.nii"
    loaded = NiftiImage.load(source)
    loaded.data = np.ones((2, 3, 4), "float32")
    loaded.save(target)
    header = nb.load(str(target)).header
    assert header["descrip"].item() == b"kept"
    assert float(header["cal_max"]) == 5.0
    assert [e.get_content() for e in header.extensions] == [b"a comment"]
    assert np.array_equal(nb.load(str(target)).get_fdata(), np.ones((2, 3, 4)))


@needs_nibabel
def test_a_nifti_image_holds_a_proxy_and_a_record(tmp_path) -> None:  # noqa: ANN001
    source = _saved(NiftiImage, EXEMPLARS[NiftiImage], tmp_path)
    loaded = NiftiImage.load(source)
    assert isinstance(loaded.raw, ArrayProxy)
    assert isinstance(loaded.metadata.raw.header, nb.Nifti1Header)
    assert MetadataFormat.load(source).raw.header == loaded.metadata.raw.header
    # Other metadata gives other transformations.
    first = loaded.transformation
    assert loaded.transformation is first
    edited = loaded.metadata.to_raw()
    edited.header.set_sform(np.diag([2.0, 2.0, 2.0, 1.0]), code=2)
    loaded.metadata = NiftiMetadata.from_raw(edited)
    assert loaded.transformation is not first
    expected = np.diag([2.0, 2.0, 2.0, 1.0])[:3]
    assert np.allclose(_nifti_geometry(loaded), expected)


@needs_nibabel
def test_an_untouched_ras_to_voxel_affine_is_saved_byte_for_byte(
    tmp_path,  # noqa: ANN001
) -> None:
    # The inverse of `NiftiVoxelToRAS` is not registered, since a file
    # cannot tell the two apart, so the checks of the registered formats
    # do not reach it.
    source = tmp_path / "source.nii"
    NiftiRASToVoxel(_VOXEL_TO_RAS).save(source)
    loaded = NiftiRASToVoxel.load(source)
    assert loaded.raw is None
    np.testing.assert_array_equal(loaded.data, _VOXEL_TO_RAS)
    target = tmp_path / "copy.nii"
    NiftiRASToVoxel.from_instance(loaded).save(target)
    assert _decompressed(target) == _decompressed(source)
    # The inverse of an affine read from a header reads the same header.
    assert loaded.inverse().metadata is loaded.metadata
