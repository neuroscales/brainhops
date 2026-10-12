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
`raw` and the record describe the file as it was read, or as the object
was last consolidated: assigned data and transformations are pending
until `consolidate` encodes them into a copy of the object, which is
also what a writer writes.

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
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import Inherit, KwOnly, Magic, NoRepr, fields, replace

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
    ParserError,
    WriterError,
    _overrides_from_fileobj,
)
from brainhops.io.common.afni._data import _BrikProxy
from brainhops.io.common.afni._files import afni_dataset_files
from brainhops.io.common.afni._geometry import afni_voxel_to_dicom
from brainhops.io.common.mrtrix._data import _MrtrixProxy
from brainhops.io.common.nrrd._data import _NrrdProxy
from brainhops.io.images.afni import AfniImage, AfniMetadata
from brainhops.io.images.afni._views import (
    _image_to_disk as _afni_to_disk,
)
from brainhops.io.images.afni._views import (
    _image_to_model as _afni_to_model,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.mrtrix import MrtrixImage, MrtrixMetadata
from brainhops.io.images.mrtrix._views import (
    _image_to_disk as _mrtrix_to_disk,
)
from brainhops.io.images.mrtrix._views import (
    _image_to_model as _mrtrix_to_model,
)
from brainhops.io.images.nrrd import (
    AttachedNrrdImage,
    DetachedNrrdImage,
    NrrdMetadata,
)
from brainhops.io.images.nrrd._views import (
    _image_to_disk as _nrrd_to_disk,
)
from brainhops.io.images.nrrd._views import (
    _image_to_model as _nrrd_to_model,
)
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
    from brainhops.io.images.freesurfer.mgh import MghImage, MghMetadata
    from brainhops.io.images.freesurfer.mgh._views import (
        _image_to_disk as _mgh_to_disk,
    )
    from brainhops.io.images.freesurfer.mgh._views import (
        _image_to_model as _mgh_to_model,
    )
    from brainhops.io.images.nifti import NiftiImage, NiftiMetadata
    from brainhops.io.images.nifti._views import (
        _image_to_disk,
        _image_to_model,
    )
    from brainhops.io.transformations.fsl.fnirt import FnirtWarpField
    from brainhops.io.transformations.itk.nifti import (
        ItkNiftiCoordinatesField,
        ItkNiftiDisplacementField,
    )
    from brainhops.io.transformations.itk.nifti._views import (
        _itk_field_to_disk,
        _itk_field_to_model,
    )
    from brainhops.io.transformations.nifti import (
        NiftiRASCoordinatesField,
        NiftiRASDisplacementField,
        NiftiRASToVoxel,
        NiftiVoxelToRAS,
    )
    from brainhops.io.transformations.nifti._views import (
        _affine_to_disk,
        _affine_to_model,
        _field_to_disk,
        _field_to_model,
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
    from scipy.io import netcdf_file

    from brainhops.io.common.minc._data import _MincProxy
    from brainhops.io.common.minc._netcdf import read_netcdf_header
    from brainhops.io.images.minc import Minc1Image, Minc2Image, MincMetadata
    from brainhops.io.images.minc._views import (
        _image_to_disk as _minc_to_disk,
    )
    from brainhops.io.images.minc._views import (
        _image_to_model as _minc_to_model,
    )

    minc_needs = netcdf_file
except ImportError:
    minc_needs = None

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
        record = ToyRaw() if self.raw is None else self.raw
        return record.to_bytes(**kwargs)


class ToyImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """Image stored in a toy file.

    The image is the reference that the formats of the package follow.
    `raw` and the record of the metadata describe the file as it was
    read, or as it was last consolidated. Assigned data and assigned
    transformations are pending: they are stored in the private fields
    `_data` and `_transformations` until `consolidate` encodes them into
    a new `raw` and a new record, or until the image is written.
    """

    EXTENSIONS = (".toy",)

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    metadata: KwOnly[NoRepr[tx.Optional[ToyMetadata]]] = None

    # The image takes the `data` and `transformations` fields over under
    # private names, where `None` means that they are decoded from `raw`
    # and from the record.
    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[np.ndarray]:
        # An assigned array is returned as it is, and the array decoded
        # from `raw` is cached.
        if self.raw is None:
            return None
        return _toy_to_model(self.raw)

    @smartproperty
    def transformations(self) -> tx.List[Transformation]:
        # The geometry is decoded from the record each time, unless a list
        # of transformations, which may be empty, has been assigned.
        if self.metadata is None or self.metadata.raw is None:
            return []
        return [Scaling(list(reversed(self.metadata.raw.spacing)))]

    def consolidate(self, *, inplace: bool = False, **options) -> "ToyImage":
        # Nothing is encoded for an image that has a record and nothing
        # pending, so its copy shares `raw` and the metadata.
        if (
            self._data is None
            and self._transformations is None
            and not options
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            return self if inplace else replace(self)
        raw, record = _toy_encoded(self, **options)
        if self.metadata is None:
            metadata = ToyMetadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                raw=raw,
                metadata=metadata,
            )
        self.raw = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        return self

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

    def to_bytes(self, **options) -> bytes:
        return _toy_written(self.consolidate(**options))

    def to_filename(self, filename: tx.Any, **options) -> None:
        filename = os.fspath(filename)
        if not _toy_reads(self.raw, filename):
            with open(filename, "wb") as file:
                file.write(self.to_bytes(**options))
            return
        # The image is written over the file that its proxy reads, so the
        # image is consolidated in place and then holds what the file
        # holds (#426). The content is read before the file is replaced,
        # and the proxy of the old file, which describes how that file
        # stored the data, is replaced by a proxy of the new file.
        self.consolidate(inplace=True, **options)
        content = _toy_written(self)
        with open(filename, "wb") as file:
            file.write(content)
        self.raw = _toy_proxy(filename, 0, self.metadata.raw)
        self.data = None


def _toy_encoded(
    image: ToyImage, note: tx.Optional[str] = None
) -> tx.Tuple[ArrayProtocol, ToyRaw]:
    """Encode the pending state of a toy image into `raw` and a record.

    The held record is the base of the new one, which is a new object.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image._data is not None:
        raw = _toy_to_disk(image._data)
    else:
        raw = image.raw
    if raw is None:
        raise WriterError("A toy image without data cannot be encoded.")
    if image.metadata is None or image.metadata.raw is None:
        base = ToyRaw()
    else:
        base = image.metadata.raw
    spacing = _toy_spacing(image.transformation, len(raw.shape))
    record = replace(
        base,
        shape=tuple(int(n) for n in raw.shape),
        dtype=np.dtype(raw.dtype).str,
        spacing=tuple(float(x) for x in reversed(spacing)),
        note=base.note if note is None else note,
    )
    return raw, record


def _toy_written(image: ToyImage) -> bytes:
    """Return the content of the file of a consolidated toy image.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image.raw is None:
        raise WriterError("A toy image without data cannot be written.")
    # A proxy is copied out without decoding, so that an untouched image
    # is written byte for byte.
    if isinstance(image.raw, _ToyProxy):
        data = image.raw.read_bytes()
    else:
        data = np.ascontiguousarray(image.raw).tobytes()
    return image.metadata.raw.to_bytes() + data


def _toy_reads(raw: tx.Any, filename: str) -> bool:
    """Tell whether `raw` is a proxy that reads a file by its name."""
    return isinstance(raw, _ToyProxy) and raw.source == filename


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
    """Function that returns the data of an object as its file stores it.

    It is `None` for a format whose `raw` holds the stored data. The data
    of a small format may live in the record of its metadata, as the
    matrix of an LTA file does, and such a format has no `raw` field. The
    matrix of a NIfTI affine lives in the header of its file too, so a
    consolidated affine has no `raw` array. The checks then read the
    stored data with this function, which returns the data as the writer
    would store it, and `proxies` are the types that it returns just after
    a file is read. An object built from data has no metadata, because it
    holds its data in the data model until it is consolidated.
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

    source: tx.Optional[tx.Callable[[tx.Any, tx.Any], None]] = None
    """Function that writes a file of a format that cannot write itself.

    The function receives a path and data. It is `None` for a format that
    writes its own files. MINC cannot be written, so the checks write its
    files with this function, and the checks of writing make sure that
    the format refuses to write instead of checking what it writes.
    """

    renamed: bool = True
    """Whether a copy of a file under another name keeps its bytes.

    A header that names its data file, such as a detached NRRD header,
    changes in that line when it is written under another name, so only
    its copies under the same name, in another directory, are compared.
    """

    header_file: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None
    """Function that returns the header file of an object of several files.

    It is `None` for a format whose objects are stored in one file. An
    AFNI dataset is a `.HEAD` file and a `.BRIK` file, and the checks may
    name either one. The bytes of one file cannot hold such an object, so
    the format must refuse them, and the metadata is read from the bytes
    of the header file. A copy of a file whose data cannot be read is
    then the header file, beside an empty file of data.
    """

    storage: tx.Optional[tx.Dict[str, tx.Any]] = None
    """Options of `consolidate` that change how the data is stored.

    They are `None` for a format whose writer has no such option. A copy
    consolidated with them must hold what a file written with them holds,
    so the checks give it values that the stored type changes.
    """


def _nifti_edit_record(record: tx.Any) -> tx.Any:
    record.header["descrip"] = b"edited"
    return record


def _nifti_geometry(image: tx.Any) -> np.ndarray:
    return np.asarray(image.transformation.to(Affine).matrix, dtype=float)


def _nifti_change_geometry(image: tx.Any) -> None:
    image.transformations = [Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])]


def _header_matrix(xform: tx.Any) -> tx.Optional[np.ndarray]:
    # A NIfTI affine stores its matrix as the best affine of its header.
    if xform.metadata is None:
        return None
    return xform.metadata.raw.header.get_best_affine()


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
        storage={"dtype": "int16"},
    )
    _NIFTYREG = _NIFTI_FIELD
    _ITK = _NIFTI_FIELD._replace(
        to_model=_itk_field_to_model, to_disk=_itk_field_to_disk
    )

    EXEMPLARS[NiftiImage] = _NIFTI._replace(
        foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
        storage={"dtype": "int16"},
    )
    EXEMPLARS[NiftiVoxelToRAS] = _NIFTI._replace(
        # The matrix of an affine lives in the header, so a read affine has
        # no array, and its stored matrix is decoded from the header.
        proxies=(np.ndarray,),
        to_model=_affine_to_model,
        to_disk=_affine_to_disk,
        sample=lambda: _VOXEL_TO_RAS,
        geometry=_matrix_geometry,
        change_geometry=_matrix_change_geometry,
        foreign=lambda data: Affine(data),
        derived=("homogeneous_matrix",),
        stored=_header_matrix,
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
        storage={"dtype": "int16"},
    )


def _lta_stored(xform: tx.Any) -> tx.Any:
    # The matrix of an LTA file lives in the record, which is parsed in one
    # pass, so it is a tuple of rows just after a read. A transformation
    # without a record stores nothing until it is consolidated.
    if xform.metadata is None:
        return None
    return xform.metadata.raw.affine.matrix


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
    # assigned is pending until it is encoded into a new record.
    if xform.metadata is None:
        return None
    return xform.metadata.raw.nodes[0].transform


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


def _hdf5_cut(source: tx.Any, target: tx.Any) -> None:
    """Copy an HDF5 file, such as an X5 file, whose arrays cannot be read.

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
        cut=_hdf5_cut,
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
    # assigned is pending until it is encoded into a new record.
    if xform.metadata is None:
        return None
    return xform.metadata.raw.positions


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

# An MGH file keeps its footer of acquisition parameters and its tags
# after the voxels, and both belong to the record. The edit of the record
# changes one parameter of the footer and the tags. The metadata writes
# the fixed header alone, so the check that metadata never reads data
# compares fixed headers, and the footer of the exemplar is zero.


def _mgh_edit_record(record: tx.Any) -> tx.Any:
    record.header["te"] = 7.5
    record.tags = b"edited"
    return record


def _mgh_record_edited(record: tx.Any) -> bool:
    return float(record.header["te"]) == 7.5 and record.tags == b"edited"


if nb is not None:
    EXEMPLARS[MghImage] = _NIFTI._replace(
        metadata=MghMetadata,
        suffix=".mgh",
        to_model=_mgh_to_model,
        to_disk=_mgh_to_disk,
        edit_record=_mgh_edit_record,
        record_edited=_mgh_record_edited,
        foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
        storage={"dtype": "int16"},
    )

# An AFNI dataset is a `.HEAD` file and a `.BRIK` file. The exemplar
# names the header, so the check of an untouched save compares headers,
# and the variants name the BRIK, so that the same check compares the
# BRIKs, after decompression for a gzipped one. The world of AFNI is LPS,
# so the geometry is compared as the voxel-to-DICOM matrix, whatever the
# world space of an assigned transformation. A record is frozen, so the
# edit is made with `AfniRaw.replace`.


def _afni_header_file(path: tx.Any) -> tx.Any:
    return afni_dataset_files(path)[0]


def _afni_geometry(image: tx.Any) -> np.ndarray:
    return afni_voxel_to_dicom(image.transformation)


_AFNI = Exemplar(
    metadata=AfniMetadata,
    suffix=".HEAD",
    proxies=(_BrikProxy,),
    to_model=_afni_to_model,
    to_disk=_afni_to_disk,
    sample=lambda: np.arange(24, dtype="float32").reshape(2, 3, 4),
    edit_record=lambda record: record.replace(HISTORY_NOTE="edited"),
    record_edited=lambda record: record.get("HISTORY_NOTE") == "edited",
    geometry=_afni_geometry,
    change_geometry=_nifti_change_geometry,
    foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
    header_file=_afni_header_file,
    storage={"datatype": "int16"},
)
EXEMPLARS[AfniImage] = _AFNI

# An MRtrix image is checked as a `.mif` file, which holds the header and
# the voxels, and as a gzipped `.mif.gz` file. A `.mih` header names its
# data file, so a copy under another name differs in that line, and the
# check of an untouched save cannot apply to it. The metadata writes the
# header up to the first voxel, padding included.


def _mrtrix_edit_record(record: tx.Any) -> tx.Any:
    record.keyval["comments"] = "edited"
    return record


_MRTRIX = Exemplar(
    metadata=MrtrixMetadata,
    suffix=".mif",
    proxies=(_MrtrixProxy,),
    to_model=_mrtrix_to_model,
    to_disk=_mrtrix_to_disk,
    sample=lambda: np.arange(24, dtype="float32").reshape(2, 3, 4),
    edit_record=_mrtrix_edit_record,
    record_edited=lambda record: record.keyval.get("comments") == "edited",
    geometry=_nifti_geometry,
    change_geometry=_nifti_change_geometry,
    foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
    storage={"datatype": "int16"},
)
EXEMPLARS[MrtrixImage] = _MRTRIX

# A NRRD image is checked as an attached `.nrrd` file and as a detached
# `.nhdr` header with its `.raw.gz` data file, both gzipped by default.
# The samples of both stay compressed in the file, and an untouched copy
# keeps the compressed bytes. A detached header names its data file, so
# its copies are compared under the same name, in another directory,
# where both files must be identical. The data file named by the copy of
# a detached header that the metadata check writes is the data file of
# the source, which exists, so that check only shows that the metadata
# reads the header alone, as for M3Z.


def _nrrd_edit_record(record: tx.Any) -> tx.Any:
    record.keyvalue["comment"] = "edited"
    return record


_NRRD = Exemplar(
    metadata=NrrdMetadata,
    suffix=".nrrd",
    proxies=(_NrrdProxy,),
    to_model=_nrrd_to_model,
    to_disk=_nrrd_to_disk,
    sample=lambda: np.arange(24, dtype="float32").reshape(2, 3, 4),
    edit_record=_nrrd_edit_record,
    record_edited=lambda record: record.keyvalue.get("comment") == "edited",
    geometry=_nifti_geometry,
    change_geometry=_nifti_change_geometry,
    foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
    storage={"datatype": "int16"},
)
EXEMPLARS[AttachedNrrdImage] = _NRRD
EXEMPLARS[DetachedNrrdImage] = _NRRD._replace(
    suffix=".nhdr", renamed=False, header_file=lambda path: path
)

# MINC files cannot be written, so the checks write them as the MINC
# specification describes, with h5py for MINC2 and with the NetCDF writer
# of scipy for MINC1, which may be gzipped. The voxels are stored as
# floats, which nibabel does not scale. The checks of writing make sure
# that MINC refuses to write. A MINC1 copy that the metadata check reads
# is the file cut after its NetCDF header, and a MINC2 copy keeps every
# group and attribute of the file but not its image. The record is
# compared as it is, since the metadata cannot write bytes.


def _minc_dimensions(data: tx.Any) -> tx.Tuple[str, ...]:
    """Name the dimensions of data in Fortran order, slowest first."""
    return ("zspace", "yspace", "xspace")[3 - np.ndim(data) :]


def _write_minc1(path: tx.Any, data: tx.Any) -> None:
    stored = np.asarray(data, dtype="float32").T
    names = _minc_dimensions(data)
    plain = str(path)[: -len(".gz")] if str(path).endswith(".gz") else path
    f = netcdf_file(str(plain), "w", version=1)
    for name, length in zip(names, stored.shape):
        f.createDimension(name, length)
    for name in names:
        variable = f.createVariable(name, "i", ())
        variable.spacing = b"regular__"
        variable.step = 2.0
    f.createVariable("image", "f", names)[:] = stored
    for name in ("image-min", "image-max"):
        f.createVariable(name, "d", (names[0],))[:] = 0.0
    f.close()
    if plain != path:
        with open(plain, "rb") as file:
            content = file.read()
        os.remove(plain)
        with open(path, "wb") as file:
            file.write(gzip.compress(content))


def _write_minc2(path: tx.Any, data: tx.Any) -> None:
    stored = np.asarray(data, dtype="float32").T
    names = _minc_dimensions(data)
    with h5py.File(path, "w") as f:
        root = f.create_group("minc-2.0")
        group = root.create_group("dimensions")
        for name, length in zip(names, stored.shape):
            dataset = group.create_dataset(name, data=np.int32(0))
            dataset.attrs["length"] = np.int32(length)
            dataset.attrs["step"] = 2.0
        image = root.create_group("image/0")
        dataset = image.create_dataset("image", data=stored)
        dataset.attrs["dimorder"] = np.bytes_(",".join(names).encode())
        for name in ("image-min", "image-max"):
            image.create_dataset(name, data=np.zeros(stored.shape[0]))


def _minc1_cut(source: tx.Any, target: tx.Any) -> None:
    """Copy a MINC1 file cut after its NetCDF header."""
    content = _decompressed(source)
    size = read_netcdf_header(BytesIO(content)).size
    _write_header(target, content[:size])


if minc_needs is not None:
    _MINC = Exemplar(
        metadata=MincMetadata,
        suffix=".mnc",
        proxies=(_MincProxy,),
        to_model=_minc_to_model,
        to_disk=_minc_to_disk,
        sample=lambda: np.arange(24, dtype="float32").reshape(2, 3, 4),
        edit_record=lambda record: replace(record, datatype="<f8"),
        record_edited=lambda record: record.datatype == "<f8",
        geometry=_nifti_geometry,
        change_geometry=_nifti_change_geometry,
        foreign=lambda data: _ForeignImage(data, raw="raw", metadata="meta"),
        cut=_minc1_cut,
        header=lambda metadata: metadata.raw,
        source=_write_minc1,
    )
    EXEMPLARS[Minc1Image] = _MINC
    if h5py is not None:
        EXEMPLARS[Minc2Image] = _MINC._replace(
            cut=_hdf5_cut, source=_write_minc2
        )

NOT_MIGRATED: tx.Tuple[str, ...] = (
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

# An MGZ file is an MGH file compressed as a single gzip stream.
if nb is not None:
    VARIANTS.append(
        (
            "MghImage-mgz",
            MghImage,
            EXEMPLARS[MghImage]._replace(suffix=".mgz"),
        )
    )

# The BRIK of an AFNI dataset may be gzipped.
VARIANTS.append(("AfniImage-BRIK", AfniImage, _AFNI._replace(suffix=".BRIK")))
VARIANTS.append(
    ("AfniImage-BRIK-gz", AfniImage, _AFNI._replace(suffix=".BRIK.gz"))
)

VARIANTS.append(
    ("MrtrixImage-gz", MrtrixImage, _MRTRIX._replace(suffix=".mif.gz"))
)

# A MINC1 file may be gzipped.
if minc_needs is not None:
    VARIANTS.append(
        ("Minc1Image-gz", Minc1Image, _MINC._replace(suffix=".mnc.gz"))
    )

CASES = [
    pytest.param(cls, exemplar, id=cls.__name__)
    for cls, exemplar in {**EXEMPLARS, **TEST_EXEMPLARS}.items()
] + [pytest.param(cls, exemplar, id=name) for name, cls, exemplar in VARIANTS]

IMAGE_CASES = [
    case for case in CASES if issubclass(case.values[0], SingleScaleImage)
]
"""The cases whose format is an image, which decodes its transformations."""

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
    """Return the data of an object as its file stores it."""
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
    if exemplar.cut is not None:
        exemplar.cut(source, target)
    elif exemplar.header_file is not None:
        header = exemplar.header_file(target)
        _write_header(header, metadata.to_bytes())
        if header != target:
            _write_header(target, b"")
    else:
        _write_header(target, metadata.to_bytes())


def _raw(obj: tx.Any) -> tx.Any:
    """Return the `raw` array of an object, or `None` for a format without
    one."""
    return getattr(obj, "raw", None)


def _pending(obj: tx.Any) -> tx.Tuple[tx.Any, tx.Any]:
    """Return the assigned data and transformations of an object.

    They are pending until the object is consolidated, and `None` for a
    field that the format does not have.
    """
    return getattr(obj, "_data", None), getattr(obj, "_transformations", None)


def _same(first: tx.Sequence[tx.Any], second: tx.Sequence[tx.Any]) -> bool:
    """Tell whether two sequences hold the same objects."""
    return all(a is b for a, b in zip(first, second))


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
    if exemplar.source is not None:
        exemplar.source(path, exemplar.sample())
    else:
        _built(cls, exemplar, exemplar.sample()).save(path)
    return path


def _refuses_writing(obj: tx.Any, target: tx.Any) -> None:
    """Check that a format that cannot write refuses to write an object."""
    assert not hasattr(obj, "save")
    with pytest.raises(WriterError):
        io.save(obj, target)


def _placed(cls: type, exemplar: Exemplar, tmp_path: tx.Any) -> tx.Any:
    """Write a new object of the format with a geometry and return its path.

    The geometry is the one that `exemplar.change_geometry` assigns, so
    that it differs from the geometry of an object built from data. A
    format that cannot write keeps the geometry that `exemplar.source`
    writes.
    """
    if exemplar.source is not None:
        return _saved(cls, exemplar, tmp_path, "placed")
    path = _path(tmp_path, exemplar, "placed")
    obj = _built(cls, exemplar, exemplar.sample())
    exemplar.change_geometry(obj)
    obj.save(path)
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
    # A format whose data model is a chain may still take its stored data
    # as `data`, which is pending until the object is consolidated.
    allowed = names(_model(cls)) | {"raw", "metadata", "data"}
    allowed |= exemplar.options
    assert names(cls) <= allowed
    assert not names(cls) & _PARSER_STATE
    assert names(exemplar.metadata) <= names(Metadata) | {"raw"}
    assert not names(exemplar.metadata) & _PARSER_STATE


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_02_load_is_lazy(cls: type, exemplar: Exemplar, tmp_path) -> None:  # noqa: ANN001
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    assert isinstance(_stored(exemplar, loaded), exemplar.proxies)
    repr(loaded)
    assert "_cache_" + exemplar.view not in vars(loaded)
    assert isinstance(loaded.metadata, exemplar.metadata)
    data = _data(exemplar, loaded)
    assert np.array_equal(np.asarray(data), exemplar.sample())
    # The view is decoded once, and reading it leaves the proxy in place.
    assert _data(exemplar, loaded) is data
    assert isinstance(_stored(exemplar, loaded), exemplar.proxies)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_03_assigned_data_is_pending(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    views = (exemplar.view,) + exemplar.derived
    for name in views:
        getattr(loaded, name)
        assert "_cache_" + name in vars(loaded)
    before = np.asarray(_stored(exemplar, loaded))
    metadata = loaded.metadata
    value = exemplar.sample() + 1
    _set_data(exemplar, loaded, value)
    for name in views:
        assert "_cache_" + name not in vars(loaded)
    # The stored data and the record still describe the file, and the
    # data is pending.
    pending = _pending(loaded)
    assert np.array_equal(np.asarray(_stored(exemplar, loaded)), before)
    assert loaded.metadata is metadata
    assert any(field is not None for field in pending)
    assert np.array_equal(np.asarray(_data(exemplar, loaded)), value)
    # Consolidating encodes the data into the stored data of a copy.
    consolidated = loaded.consolidate()
    assert all(field is None for field in _pending(consolidated))
    stored = np.asarray(_stored(exemplar, consolidated))
    assert np.array_equal(stored, exemplar.to_disk(value))
    assert np.array_equal(np.asarray(_data(exemplar, consolidated)), value)
    assert np.array_equal(np.asarray(_stored(exemplar, loaded)), before)
    assert _same(_pending(loaded), pending)


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
    if exemplar.source is not None:
        _refuses_writing(cls.load(source), _path(tmp_path, exemplar, "copy"))
        return
    if exemplar.renamed:
        target = _path(tmp_path, exemplar, "copy")
        cls.load(source).save(target)
        assert _decompressed(target) == _decompressed(source)
        # The generic writer copies the object with `from_instance` first.
        target = _path(tmp_path, exemplar, "generic")
        io.save(cls.load(source), target)
        assert _decompressed(target) == _decompressed(source)
    # Every file of a copy under the same name, in another directory, is
    # the file of the same name beside the source.
    for name, write in (("same", cls.load(source).save), ("io", None)):
        directory = tmp_path / name
        directory.mkdir()
        if write is None:
            io.save(cls.load(source), directory / source.name)
        else:
            write(directory / source.name)
        for copied in directory.iterdir():
            original = tmp_path / copied.name
            assert _decompressed(copied) == _decompressed(original), copied


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
    if exemplar.source is None:
        assert clone.to_bytes() == loaded.metadata.to_bytes()
    else:
        # Metadata that cannot be written is compared through its summary.
        assert _header(exemplar, clone) == _header(exemplar, loaded.metadata)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_08_an_object_built_from_data_saves(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    value = exemplar.sample()
    built = _built(cls, exemplar, value)
    assert built.metadata is None
    assert np.array_equal(np.asarray(_data(exemplar, built)), value)
    # Consolidating encodes the data and a new record, which a format
    # that cannot write cannot encode.
    consolidated = built.consolidate()
    if exemplar.source is None:
        assert isinstance(consolidated.metadata, exemplar.metadata)
    stored = np.asarray(_stored(exemplar, consolidated))
    assert np.array_equal(stored, exemplar.to_disk(value))
    path = _path(tmp_path, exemplar, "built")
    if exemplar.source is not None:
        _refuses_writing(built, path)
        return
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
    assert _raw(copy) is _raw(loaded)
    assert copy.metadata is loaded.metadata
    assert "_cache_" + exemplar.view not in vars(loaded)
    # `replace` copies the stored fields too, so the copy keeps the lazy
    # `raw` and decodes nothing.
    copy = replace(loaded)
    assert copy.metadata is loaded.metadata
    assert _raw(copy) is _raw(loaded)
    assert all(field is None for field in _pending(copy))
    assert "_cache_" + exemplar.view not in vars(loaded)
    assert "_cache_" + exemplar.view not in vars(copy)
    # Assigned data is pending in the copy as well.
    _set_data(exemplar, loaded, exemplar.sample() + 1)
    copy = replace(loaded)
    assert copy.metadata is loaded.metadata
    assert _same(_pending(copy), _pending(loaded))
    # The record and the stored array of another format mean nothing to
    # this format, so they are reset. A model that holds data keeps it
    # pending, and a model that holds a chain keeps the chain, which
    # consolidating encodes.
    if exemplar.foreign is None:
        return
    value = exemplar.sample()
    converted = cls.from_instance(exemplar.foreign(value))
    assert converted.metadata is None
    if "data" in _init_names(_model(cls)):
        assert getattr(converted, "raw", None) is None
        assert np.array_equal(np.asarray(converted.data), value)
        consolidated = converted.consolidate()
        stored = np.asarray(_stored(exemplar, consolidated))
        assert np.array_equal(stored, exemplar.to_disk(value))
    path = _path(tmp_path, exemplar, "converted")
    if exemplar.source is not None:
        _refuses_writing(converted, path)
        return
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
    edited = exemplar.edit_record(deepcopy(loaded.metadata.raw))
    loaded.metadata = exemplar.metadata.from_raw(edited)
    exemplar.change_geometry(loaded)
    target = _path(tmp_path, exemplar, "edited")
    if exemplar.source is not None:
        assert exemplar.record_edited(loaded.metadata.raw)
        _refuses_writing(loaded, target)
        return
    header = loaded.metadata.to_bytes()
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
    path = _saved(cls, exemplar, tmp_path)
    if exemplar.header_file is None:
        data = _data(exemplar, cls.from_bytes(path.read_bytes()))
        assert np.array_equal(np.asarray(data), exemplar.sample())
    else:
        # The bytes of one file cannot hold an object of several files.
        with pytest.raises(ParserError):
            cls.from_bytes(path.read_bytes())
        path = exemplar.header_file(path)
    metadata = exemplar.metadata.from_bytes(path.read_bytes())
    assert isinstance(metadata, exemplar.metadata)


@pytest.mark.parametrize("cls, exemplar", IMAGE_CASES)
def test_13_transformations_are_decoded_unless_assigned(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_placed(cls, exemplar, tmp_path))
    decoded = exemplar.geometry(loaded)
    # After a read, the image stores no transformations, so they are
    # decoded from the record. The copies within the format store none
    # either, so they decode the same record.
    assert loaded._transformations is None
    for copy in (replace(loaded), cls.from_instance(loaded)):
        assert copy._transformations is None
        assert np.allclose(exemplar.geometry(copy), decoded)
    # An empty list is kept, and the copies keep it.
    loaded.transformations = []
    assert loaded.transformations == []
    assert replace(loaded).transformations == []
    assert cls.from_instance(loaded).transformations == []
    # The image is then written as an image built from data alone, by a
    # format that can write.
    target = _path(tmp_path, exemplar, "empty")
    if exemplar.source is not None:
        _refuses_writing(loaded, target)
    else:
        loaded.save(target)
        bare = cls.load(_saved(cls, exemplar, tmp_path, "bare"))
        assert np.allclose(
            exemplar.geometry(cls.load(target)), exemplar.geometry(bare)
        )
    # Assigning `None` decodes the record again.
    loaded.transformations = None
    assert np.allclose(exemplar.geometry(loaded), decoded)


@pytest.mark.parametrize("cls, exemplar", IMAGE_CASES)
def test_14_copies_between_classes_carry_the_decoded_transformations(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_placed(cls, exemplar, tmp_path))
    decoded = exemplar.geometry(loaded)
    # The data model and another format read the decoded transformations,
    # since the record of this format means nothing to them.
    for copy in (
        _model(cls).from_instance(loaded),
        _ForeignImage.from_instance(loaded),
    ):
        assert np.allclose(exemplar.geometry(copy), decoded)
    # This format stores the transformations of another class as given.
    foreign = _ForeignImage(exemplar.sample())
    exemplar.change_geometry(foreign)
    converted = cls.from_instance(foreign)
    assert converted._transformations is not None
    assert np.allclose(
        exemplar.geometry(converted), exemplar.geometry(foreign)
    )


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_15_consolidating_an_untouched_object_changes_nothing(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    raw, metadata = _raw(loaded), loaded.metadata
    # The copy shares `raw` and the metadata, and nothing is decoded.
    consolidated = loaded.consolidate()
    assert consolidated is not loaded
    assert _raw(consolidated) is raw
    assert consolidated.metadata is metadata
    assert "_cache_data" not in vars(loaded)
    assert "_cache_data" not in vars(consolidated)
    assert loaded.consolidate(inplace=True) is loaded
    assert _raw(loaded) is raw
    assert loaded.metadata is metadata


def _changed(cls: type, exemplar: Exemplar, tmp_path: tx.Any) -> tx.Any:
    """Load an object and give it new data and a new geometry."""
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    _set_data(exemplar, loaded, exemplar.sample() + 1)
    exemplar.change_geometry(loaded)
    return loaded


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_16_consolidate_is_idempotent(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = _changed(cls, exemplar, tmp_path)
    raw, metadata, pending = _raw(loaded), loaded.metadata, _pending(loaded)
    if exemplar.source is not None:
        _check_data_only(exemplar, loaded)
        return
    once = loaded.consolidate()
    # Nothing is pending in the copy, whose record is new, and the copy
    # holds what a file written from the object holds.
    assert all(field is None for field in _pending(once))
    assert once.metadata is not metadata
    target = _path(tmp_path, exemplar, "written")
    loaded.save(target)
    back = cls.load(target)
    assert np.array_equal(
        np.asarray(_data(exemplar, once)), np.asarray(_data(exemplar, back))
    )
    assert np.allclose(exemplar.geometry(once), exemplar.geometry(back))
    assert np.allclose(exemplar.geometry(once), exemplar.geometry(loaded))
    # A second consolidation changes nothing.
    twice = once.consolidate()
    assert _raw(twice) is _raw(once)
    assert twice.metadata is once.metadata
    # The object itself keeps its pending state.
    assert _raw(loaded) is raw
    assert loaded.metadata is metadata
    assert _same(_pending(loaded), pending)
    # In place, the object holds the consolidated state.
    assert loaded.consolidate(inplace=True) is loaded
    assert all(field is None for field in _pending(loaded))
    raw, metadata = _raw(loaded), loaded.metadata
    loaded.consolidate(inplace=True)
    assert _raw(loaded) is raw
    assert loaded.metadata is metadata


def _check_data_only(exemplar: Exemplar, loaded: tx.Any) -> None:
    """Check the consolidation of a format that cannot write.

    No record can be encoded, so only the pending data is moved into
    `raw`, and the metadata and the assigned transformations are kept.
    """
    data, chain = _pending(loaded)
    once = loaded.consolidate()
    assert once._data is None
    assert once._transformations is chain
    assert once.metadata is loaded.metadata
    assert np.array_equal(
        np.asarray(_stored(exemplar, once)), exemplar.to_disk(data)
    )
    twice = once.consolidate()
    assert _raw(twice) is _raw(once)
    assert loaded._data is data


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_16_a_storage_override_holds_what_the_file_holds(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    if exemplar.storage is None:
        pytest.skip("The format has no option that changes the stored data.")
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    # The values are not whole numbers, so an integer type that stores
    # them changes them, as the writer of the format does.
    loaded.data = (exemplar.sample() + 1) / 7
    once = loaded.consolidate(**exemplar.storage)
    # The file has the name of the source, in another directory, since a
    # header may name its data file.
    (tmp_path / "stored").mkdir()
    path = _path(tmp_path / "stored", exemplar, "src")
    once.save(path)
    back = cls.load(path)
    # The copy holds the data and the header of the file exactly.
    assert np.asarray(once.data).dtype == np.asarray(back.data).dtype
    assert np.array_equal(np.asarray(once.data), np.asarray(back.data))
    assert _header(exemplar, once.metadata) == _header(exemplar, back.metadata)
    # Nothing is pending in the copy.
    twice = once.consolidate()
    assert twice.raw is once.raw
    assert twice.metadata is once.metadata


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_17_save_writes_the_consolidated_copy(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    if exemplar.source is not None:
        pytest.skip("The format cannot write.")
    loaded = _changed(cls, exemplar, tmp_path)
    raw, metadata, pending = _raw(loaded), loaded.metadata, _pending(loaded)
    saved = _path(tmp_path, exemplar, "saved")
    consolidated = _path(tmp_path, exemplar, "consolidated")
    loaded.save(saved)
    loaded.consolidate().save(consolidated)
    # The files are compared as they are read, since a format may give a
    # new identity to each dataset that it encodes, as AFNI does.
    saved, consolidated = cls.load(saved), cls.load(consolidated)
    assert np.array_equal(
        np.asarray(_data(exemplar, saved)),
        np.asarray(_data(exemplar, consolidated)),
    )
    assert np.allclose(
        exemplar.geometry(saved), exemplar.geometry(consolidated)
    )
    assert np.allclose(exemplar.geometry(saved), exemplar.geometry(loaded))
    # Saving leaves the object as it was.
    assert _raw(loaded) is raw
    assert loaded.metadata is metadata
    assert _same(_pending(loaded), pending)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_18_saving_over_its_own_file_consolidates_the_object(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    path = _saved(cls, exemplar, tmp_path)
    loaded = cls.load(path)
    if exemplar.source is not None or _raw(loaded) is None:
        pytest.skip("Only an object whose `raw` reads its file is affected.")
    # An untouched object is written over the file that it reads.
    loaded.save(path)
    back = cls.load(path)
    assert np.array_equal(np.asarray(loaded.data), exemplar.sample())
    assert np.array_equal(np.asarray(back.data), exemplar.sample())
    assert _header(exemplar, loaded.metadata) == _header(
        exemplar, back.metadata
    )
    # A changed object is consolidated in place, and then holds what the
    # file holds (#426): its `raw` reads the new file.
    loaded = _changed(cls, exemplar, tmp_path)
    path = _path(tmp_path, exemplar, "src")
    expected = np.asarray(_data(exemplar, loaded.consolidate()))
    loaded.save(path)
    assert all(field is None for field in _pending(loaded))
    assert isinstance(_raw(loaded), exemplar.proxies)
    back = cls.load(path)
    assert np.array_equal(np.asarray(loaded.data), np.asarray(back.data))
    assert np.allclose(np.asarray(back.data), expected)
    assert np.allclose(exemplar.geometry(loaded), exemplar.geometry(back))
    # Its record is the header of the new file.
    assert _header(exemplar, loaded.metadata) == _header(
        exemplar, back.metadata
    )


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
    stored = np.zeros((5, 5))
    image = ToyImage(data, raw=stored)
    # Assigned data is read first, and `raw` is kept until the image is
    # consolidated.
    assert image.data is data
    assert image.raw is stored
    assert np.array_equal(image.consolidate().raw, data.T)
    assert np.array_equal(ToyImage(raw=data.T).data, data)
    # Assigning `None` decodes `raw` again.
    image.data = None
    assert np.array_equal(image.data, stored.T)


def test_a_record_is_never_changed() -> None:
    record = ToyRaw(shape=(3, 2), spacing=(1.0, 1.0), note="kept")
    metadata = ToyMetadata.from_raw(record)
    assert metadata.raw is record
    # Metadata consolidates into a copy that holds the same record, since
    # it holds nothing but its record.
    copy = metadata.consolidate()
    assert copy is not metadata and copy.raw is record
    assert metadata.consolidate(inplace=True) is metadata
    image = ToyImage(np.zeros((2, 3)), metadata=metadata)
    image.transformations = [Scaling([4.0, 5.0])]
    image.to_bytes()
    consolidated = image.consolidate(note="changed")
    assert consolidated.metadata.raw.note == "changed"
    assert consolidated.metadata.raw.spacing == (5.0, 4.0)
    assert record == ToyRaw(shape=(3, 2), spacing=(1.0, 1.0), note="kept")
    image.consolidate(inplace=True)
    assert image.metadata is not metadata
    assert record == ToyRaw(shape=(3, 2), spacing=(1.0, 1.0), note="kept")


def test_writer_options_are_encoded_like_a_consolidation(
    tmp_path,  # noqa: ANN001
) -> None:
    path = tmp_path / "image.toy"
    ToyImage(np.arange(6.0).reshape(2, 3)).save(path)
    image = ToyImage.load(path)
    image.save(tmp_path / "noted.toy", note="a longer note")
    assert image.metadata.raw.note == ""
    noted = ToyImage.load(tmp_path / "noted.toy")
    assert noted.metadata.raw.note == "a longer note"
    assert (
        noted.to_bytes() == image.consolidate(note="a longer note").to_bytes()
    )
    # Over its own file, the options are consolidated in place, and the
    # proxy reads the data after the longer header.
    image.save(path, note="a longer note")
    assert image.metadata.raw.note == "a longer note"
    assert np.array_equal(image.data, np.arange(6.0).reshape(2, 3))


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
    edited = loaded.metadata.raw.copy()
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
