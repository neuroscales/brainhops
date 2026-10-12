"""Conformance of the file formats to the design of format records (#415).

A file format that follows the design has three layers. Its record class
(`<X>Raw`) holds the header of a file as the format stores it. Its
metadata class (`<X>Metadata`, a subclass of `MetadataFormat`) holds a
record and reads and writes the header alone. Its format object, such as
an image, holds a metadata object and a `raw` array, which is the data as
stored in the file and usually a lazy proxy. The `data` of a format object
is a cached view of `raw`, decoded by a pair of pure functions of the
format. `raw` and the record describe the file as it was read, or as the
object was last consolidated: assigned data are pending until
`consolidate` encodes them into a copy of the object, which is also what
a writer writes. Transformations that were read or assigned are compared
at `consolidate` with those that the record decodes to, and encoded only
when they differ.

Each format that follows the design is listed in `EXEMPLARS`, with what
the checks need to know about it, and every other registered format is
listed in `NOT_MIGRATED`. A test makes sure that the two tables together
name every registered format, so a new format has to choose a table. The
checks are parametrized over the exemplars and over a small format defined
in this module, the toy format, which shows the mechanism end to end
before any real format is migrated.
"""

import gzip
import importlib
import os
import pickle
import struct
from copy import deepcopy

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
from brainhops.datamodel.transformations import (
    Identity,
    Scaling,
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
from brainhops.io.common._geometry import same_transformations
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata import MetadataFormat

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
    read, or as it was last consolidated. Assigned data are pending: they
    are stored in the private field `_data` until `consolidate` encodes
    them into a new `raw` and a new record, or until the image is written.
    The transformations are decoded from the record on first access and
    stored in the private field `_transformations`, where an assigned list
    is stored too. `consolidate` compares that list with a fresh decode of
    the record, and encodes it only when they differ, so that a list
    changed in place is written and an untouched one keeps the record.
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
        # The geometry is decoded from the record on first access and
        # stored, so that a change made to the list in place is kept. A
        # list, which may be empty, can also be assigned.
        self._transformations = _toy_to_transformations(self.metadata)
        return self._transformations

    def consolidate(self, *, inplace: bool = False, **options) -> "ToyImage":
        # Nothing is encoded for an image that has a record, no pending
        # data, and transformations that are never read or the same as a
        # fresh decode of the record, so its copy shares `raw` and the
        # metadata.
        if (
            self._data is None
            and (
                self._transformations is None
                or same_transformations(
                    self._transformations,
                    _toy_to_transformations(self.metadata),
                )
            )
            and not options
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            if not inplace:
                return replace(self, transformations=None)
            self.transformations = None
            return self
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


def _toy_to_transformations(
    metadata: tx.Optional[ToyMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of a toy image from its record."""
    if metadata is None or metadata.raw is None:
        return []
    return [Scaling(list(reversed(metadata.raw.spacing)))]


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

    The format is built with `cls(data=sample())` and written to a file
    with the given suffix, which is the file that the checks read.
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

    foreign: tx.Callable[[tx.Any], tx.Any]
    """Function that builds an object of another format from data."""

    options: tx.FrozenSet[str] = frozenset()
    """Reader options that the format declares as fields."""

    derived: tx.Tuple[str, ...] = ()
    """Cached views that setting `data` must delete, besides `data`."""

    binary: bool = True
    """Whether the format is read in binary mode."""

    storage: tx.Optional[tx.Dict[str, tx.Any]] = None
    """Options of `consolidate` that change how the data is stored.

    They are `None` for a format whose writer has no such option. A copy
    consolidated with them must hold what a file written with them holds,
    so the checks give it values that the stored type changes.
    """


EXEMPLARS: tx.Dict[type, Exemplar] = {}
"""The registered formats that follow the design."""

NOT_MIGRATED: tx.Tuple[str, ...] = (
    "brainhops.io.images.afni.AfniImage",
    "brainhops.io.images.freesurfer.mgh.MghImage",
    "brainhops.io.images.minc.Minc1Image",
    "brainhops.io.images.minc.Minc2Image",
    "brainhops.io.images.mrtrix.MrtrixImage",
    "brainhops.io.images.nifti.NiftiImage",
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
    "brainhops.io.transformations.freesurfer.lta.LtaTransformation",
    "brainhops.io.transformations.freesurfer.m3z.M3zMorph",
    "brainhops.io.transformations.fsl.flirt.FlirtTransform",
    "brainhops.io.transformations.fsl.fnirt.FnirtWarpField",
    "brainhops.io.transformations.itk.h5.H5Transform",
    "brainhops.io.transformations.itk.mat.MatTransform",
    "brainhops.io.transformations.itk.nifti.ItkNiftiCoordinatesField",
    "brainhops.io.transformations.itk.nifti.ItkNiftiDisplacementField",
    "brainhops.io.transformations.itk.tfm.TfmTransform",
    "brainhops.io.transformations.matrix.CsvMatrixAffine",
    "brainhops.io.transformations.matrix.Mat73MatrixAffine",
    "brainhops.io.transformations.matrix.MatLegacyMatrixAffine",
    "brainhops.io.transformations.matrix.NpyMatrixAffine",
    "brainhops.io.transformations.matrix.NpzMatrixAffine",
    "brainhops.io.transformations.matrix.TsvMatrixAffine",
    "brainhops.io.transformations.matrix.TxtMatrixAffine",
    "brainhops.io.transformations.nifti.NiftiRASCoordinatesField",
    "brainhops.io.transformations.nifti.NiftiRASDisplacementField",
    "brainhops.io.transformations.nifti.NiftiVoxelToRAS",
    "brainhops.io.transformations.niftyreg.NiftyRegAffine",
    "brainhops.io.transformations.niftyreg.NiftyRegControlPointGrid",
    "brainhops.io.transformations.niftyreg.NiftyRegDeformationField",
    "brainhops.io.transformations.niftyreg.NiftyRegDisplacementField",
    "brainhops.io.transformations.niftyreg.NiftyRegVelocityField",
    "brainhops.io.transformations.niftyreg.NiftyRegVelocityGrid",
    "brainhops.io.transformations.spm.y.SpmCoordinatesField",
    "brainhops.io.transformations.x5.X5Transform",
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

CASES = [
    pytest.param(cls, exemplar, id=cls.__name__)
    for cls, exemplar in {**EXEMPLARS, **TEST_EXEMPLARS}.items()
]

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
    with open(path, "rb") as file:
        content = file.read()
    if str(path).endswith(".gz"):
        content = gzip.decompress(content)
    return content


def _write_header(path: tx.Any, header: bytes) -> None:
    if str(path).endswith(".gz"):
        header = gzip.compress(header)
    with open(path, "wb") as file:
        file.write(header)


def _saved(
    cls: type, exemplar: Exemplar, tmp_path: tx.Any, name: str = "src"
) -> tx.Any:
    """Write a new object of the format and return the path of its file."""
    path = tmp_path / (name + exemplar.suffix)
    cls(data=exemplar.sample()).save(path)
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
    assert registered == set(EXEMPLARS) | not_migrated


def test_the_metadata_registry_is_isolated() -> None:
    assert MetadataFormat._is_dispatcher()
    assert ToyMetadata in MetadataFormat._REGISTRY
    assert ToyMetadata not in Format._REGISTRY
    assert MetadataFormat not in Format._REGISTRY


# ----------------------------------------------------------------------
#   CHECKS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_01_no_parser_state(cls: type, exemplar: Exemplar) -> None:
    def names(klass: type) -> tx.Set[str]:
        return {field.public_name for field in fields(klass) if field.init}

    allowed = names(_model(cls)) | {"raw", "metadata"} | exemplar.options
    assert names(cls) <= allowed
    assert not names(cls) & _PARSER_STATE
    assert names(exemplar.metadata) <= names(Metadata) | {"raw"}
    assert not names(exemplar.metadata) & _PARSER_STATE


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_02_load_is_lazy(cls: type, exemplar: Exemplar, tmp_path) -> None:  # noqa: ANN001
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    assert isinstance(loaded.raw, exemplar.proxies)
    repr(loaded)
    assert "_cache_data" not in vars(loaded)
    assert isinstance(loaded.metadata, exemplar.metadata)
    data = loaded.data
    assert np.array_equal(data, exemplar.sample())
    # The view is decoded once, and reading it leaves the proxy in place.
    assert loaded.data is data
    assert isinstance(loaded.raw, exemplar.proxies)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_03_assigned_data_is_pending(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    for name in ("data",) + exemplar.derived:
        getattr(loaded, name)
        assert "_cache_" + name in vars(loaded)
    raw, metadata = loaded.raw, loaded.metadata
    value = exemplar.sample() + 1
    loaded.data = value
    for name in ("data",) + exemplar.derived:
        assert "_cache_" + name not in vars(loaded)
    # `raw` and the record still describe the file, and the data is
    # pending.
    assert loaded.raw is raw
    assert loaded.metadata is metadata
    assert loaded._data is value
    assert np.array_equal(loaded.data, value)
    # Consolidating encodes the data into the `raw` of a copy.
    consolidated = loaded.consolidate()
    assert consolidated._data is None
    assert np.array_equal(
        np.asarray(consolidated.raw), exemplar.to_disk(value)
    )
    assert np.array_equal(consolidated.data, value)
    assert loaded.raw is raw
    assert loaded._data is value


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
    target = tmp_path / ("copy" + exemplar.suffix)
    cls.load(source).save(target)
    assert _decompressed(target) == _decompressed(source)
    # The generic writer copies the object with `from_instance` first.
    target = tmp_path / ("generic" + exemplar.suffix)
    io.save(cls.load(source), target)
    assert _decompressed(target) == _decompressed(source)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_06_metadata_never_reads_data(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    metadata = exemplar.metadata.load(_saved(cls, exemplar, tmp_path))
    header = metadata.to_bytes()
    # The file is cut after its header, so reading any data would fail.
    truncated = tmp_path / ("header" + exemplar.suffix)
    _write_header(truncated, header)
    assert exemplar.metadata.load(truncated).to_bytes() == header
    dispatched = MetadataFormat.load(truncated)
    assert isinstance(dispatched, exemplar.metadata)
    assert dispatched.to_bytes() == header


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
    built = cls(data=value)
    assert built.metadata is None
    assert built.raw is None
    assert np.array_equal(built.data, value)
    # Consolidating encodes the data and a new record.
    consolidated = built.consolidate()
    assert isinstance(consolidated.metadata, exemplar.metadata)
    assert np.array_equal(
        np.asarray(consolidated.raw), exemplar.to_disk(value)
    )
    path = tmp_path / ("built" + exemplar.suffix)
    built.save(path)
    assert np.array_equal(cls.load(path).data, value)


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
    assert copy.raw is loaded.raw
    assert copy.metadata is loaded.metadata
    assert "_cache_data" not in vars(loaded)
    # `replace` copies the stored fields too, so the copy keeps the lazy
    # `raw` and decodes nothing.
    copy = replace(loaded)
    assert copy.metadata is loaded.metadata
    assert copy.raw is loaded.raw
    assert copy._data is None
    assert "_cache_data" not in vars(loaded)
    assert "_cache_data" not in vars(copy)
    # Assigned data is pending in the copy as well.
    value = exemplar.sample() + 1
    loaded.data = value
    copy = replace(loaded)
    assert copy.raw is loaded.raw
    assert copy._data is value
    # The record and the stored array of another format mean nothing to
    # this format, so they are reset, and the data of the other format is
    # pending.
    value = exemplar.sample()
    converted = cls.from_instance(exemplar.foreign(value))
    assert converted.metadata is None
    assert converted.raw is None
    assert np.array_equal(converted.data, value)
    consolidated = converted.consolidate()
    assert np.array_equal(
        np.asarray(consolidated.raw), exemplar.to_disk(value)
    )


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
    header = loaded.metadata.to_bytes()
    target = tmp_path / ("edited" + exemplar.suffix)
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
        # `from_bytes` reads through a real `from_fileobj` only.
        assert _overrides_from_fileobj(klass)
        if exemplar.binary:
            assert klass._READ_MODE == "rb"
    content = _saved(cls, exemplar, tmp_path).read_bytes()
    assert np.array_equal(cls.from_bytes(content).data, exemplar.sample())
    metadata = exemplar.metadata.from_bytes(content)
    assert isinstance(metadata, exemplar.metadata)


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_15_consolidating_an_untouched_object_changes_nothing(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    raw, metadata = loaded.raw, loaded.metadata
    # The copy shares `raw` and the metadata, and nothing is decoded.
    consolidated = loaded.consolidate()
    assert consolidated is not loaded
    assert consolidated.raw is raw
    assert consolidated.metadata is metadata
    assert "_cache_data" not in vars(loaded)
    assert "_cache_data" not in vars(consolidated)
    assert loaded.consolidate(inplace=True) is loaded
    assert loaded.raw is raw
    assert loaded.metadata is metadata


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_16_consolidate_is_idempotent(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    raw, metadata = loaded.raw, loaded.metadata
    value = exemplar.sample() + 1
    loaded.data = value
    exemplar.change_geometry(loaded)
    once = loaded.consolidate()
    # Nothing is pending in the copy, whose record is new.
    assert once._data is None
    assert once._transformations is None
    assert once.metadata is not metadata
    assert np.array_equal(once.data, value)
    assert np.allclose(exemplar.geometry(once), exemplar.geometry(loaded))
    # A second consolidation changes nothing.
    twice = once.consolidate()
    assert twice.raw is once.raw
    assert twice.metadata is once.metadata
    # The object itself keeps its pending state.
    assert loaded.raw is raw
    assert loaded.metadata is metadata
    assert loaded._data is value
    # In place, the object holds the consolidated state.
    assert loaded.consolidate(inplace=True) is loaded
    assert loaded._data is None
    assert loaded._transformations is None
    raw, metadata = loaded.raw, loaded.metadata
    loaded.consolidate(inplace=True)
    assert loaded.raw is raw
    assert loaded.metadata is metadata


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
    path = tmp_path / ("stored" + exemplar.suffix)
    once.save(path)
    back = cls.load(path)
    # The copy holds the data and the header of the file exactly.
    assert np.asarray(once.data).dtype == np.asarray(back.data).dtype
    assert np.array_equal(np.asarray(once.data), np.asarray(back.data))
    assert once.metadata.to_bytes() == back.metadata.to_bytes()
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
    loaded = cls.load(_saved(cls, exemplar, tmp_path))
    raw, metadata = loaded.raw, loaded.metadata
    value = exemplar.sample() + 1
    loaded.data = value
    exemplar.change_geometry(loaded)
    (tmp_path / "saved").mkdir()
    (tmp_path / "consolidated").mkdir()
    name = "copy" + exemplar.suffix
    loaded.save(tmp_path / "saved" / name)
    loaded.consolidate().save(tmp_path / "consolidated" / name)
    assert _decompressed(tmp_path / "saved" / name) == _decompressed(
        tmp_path / "consolidated" / name
    )
    # Saving leaves the object as it was.
    assert loaded.raw is raw
    assert loaded.metadata is metadata
    assert loaded._data is value
    assert loaded._transformations is not None


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_18_saving_over_its_own_file_consolidates_the_object(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    path = _saved(cls, exemplar, tmp_path)
    # An untouched object is written over the file that it reads.
    loaded = cls.load(path)
    loaded.save(path)
    assert np.array_equal(loaded.data, exemplar.sample())
    assert np.array_equal(cls.load(path).data, exemplar.sample())
    # A changed object is consolidated in place, and then holds what the
    # file holds (#426).
    value = exemplar.sample() + 1
    loaded.data = value
    exemplar.change_geometry(loaded)
    loaded.save(path)
    assert loaded._data is None
    assert loaded._transformations is None
    assert isinstance(loaded.raw, exemplar.proxies)
    back = cls.load(path)
    assert np.array_equal(loaded.data, value)
    assert np.array_equal(back.data, value)
    assert np.allclose(exemplar.geometry(loaded), exemplar.geometry(back))


def _compared(cls: type, exemplar: Exemplar, tmp_path: tx.Any) -> tx.Any:
    """Return the path of a file of an image whose list is compared.

    Only images take `transformations` over as a list that is compared
    at `consolidate`, and the check is skipped for any other format.
    """
    path = _saved(cls, exemplar, tmp_path)
    if not isinstance(cls.load(path), ImageFormat):
        pytest.skip("Only an image takes its transformations over as a list.")
    return path


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_19_read_transformations_keep_the_record(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    path = _compared(cls, exemplar, tmp_path)
    loaded = cls.load(path)
    raw, metadata = loaded.raw, loaded.metadata
    # Reading the transformations stores them, and they are the same as
    # those of the record, so nothing is encoded.
    assert loaded.transformations is loaded.transformations
    assert loaded._transformations is not None
    consolidated = loaded.consolidate()
    assert consolidated.raw is raw
    assert consolidated.metadata is metadata
    assert consolidated._transformations is None
    (tmp_path / "copy").mkdir()
    loaded.save(tmp_path / "copy" / path.name)
    assert _decompressed(tmp_path / "copy" / path.name) == _decompressed(path)
    # An equal list that is assigned keeps the record too.
    loaded.transformations = cls.load(path).transformations
    consolidated = loaded.consolidate()
    assert consolidated.raw is raw
    assert consolidated.metadata is metadata


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_19_a_transformation_appended_in_place_is_written(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    path = _compared(cls, exemplar, tmp_path)
    loaded = cls.load(path)
    metadata = loaded.metadata
    # The transformation that the exemplar assigns is appended to the
    # stored list instead, which makes it the preferred one.
    changed = cls.load(path)
    exemplar.change_geometry(changed)
    loaded.transformations.append(changed.transformation)
    assert loaded.consolidate().metadata is not metadata
    target = tmp_path / ("appended" + exemplar.suffix)
    loaded.save(target)
    back = cls.load(target)
    assert np.allclose(exemplar.geometry(back), exemplar.geometry(changed))


@pytest.mark.parametrize("cls, exemplar", CASES)
def test_19_a_copy_of_a_read_image_stays_lazy(
    cls: type,
    exemplar: Exemplar,
    tmp_path,  # noqa: ANN001
) -> None:
    path = _compared(cls, exemplar, tmp_path)
    loaded = cls.load(path)
    assert loaded.transformations is not None
    copied = replace(loaded)
    assert "_cache_data" not in vars(loaded)
    assert "_cache_data" not in vars(copied)
    assert copied.raw is loaded.raw
    assert copied.metadata is loaded.metadata
    assert copied.consolidate().metadata is loaded.metadata


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
