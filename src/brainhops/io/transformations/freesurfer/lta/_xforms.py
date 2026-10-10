"""Affine transformations stored in LTA files."""

__all__ = [
    "LtaFormat",
    "LtaTransformation",
    "LtaTransformationVoxToVox",
    "LtaTransformationPhysToPhys",
    "LtaTransformationRASToRAS",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, fields, replace

# internals
from brainhops._core import path
from brainhops._core.enum import enum_name
from brainhops._core.properties import InvalidatorInAttribute, smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)

# this format
from .._formats import FreesurferTransformationFormat
from ._enums import LtaType, LtaValidity
from ._matrix_utils import (
    _get_phys2phys,
    _get_ras2ras,
    _get_vox2vox,
)
from ._metadata import LtaMetadata
from ._raw import LtaRaw
from ._systems import LtaCoordinateSystem, LtaPhysicalSystem, LtaVoxelSystem

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------


def _record(xform: "LtaTransformation") -> tx.Optional[LtaRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def _record_or_new(
    xform: "LtaTransformation", lta_type: LtaType, volumes: bool
) -> LtaRaw:
    """Return the record of a transformation, or a new record without one.

    The new record has the given type. With `volumes`, it also has an
    empty geometry block for each volume, so that the voxel and physical
    systems of a transformation built from data alone are known, as the
    systems of anonymous volumes named `"src"` and `"dst"`.
    """
    record = _record(xform)
    if record is not None:
        return record
    if volumes:
        return LtaRaw(
            type=lta_type,
            src=LtaRaw.SrcVolumeInfo(),
            dst=LtaRaw.DstVolumeInfo(),
        )
    return LtaRaw(type=lta_type)


def _system(
    cls: tx.Type[LtaCoordinateSystem],
    info: tx.Optional[LtaRaw.VolumeInfo],
) -> tx.Optional[LtaCoordinateSystem]:
    """Return the system of a volume.

    When the record has no geometry for the volume, the system is unknown
    and `None` is returned.
    """
    return None if info is None else cls.from_raw(info)


def _typed_system(
    record: tx.Optional[LtaRaw], info: tx.Optional[LtaRaw.VolumeInfo]
) -> tx.Optional[_systems.CoordinateSystem]:
    """Return the system of a volume as the type of a record defines it.

    The system is `RASmm` for the RAS type and `RSAmm` for the RSA type.
    For the voxel and physical types, it is built from the geometry of the
    volume, `info`. Without a record, the system is unknown and `None` is
    returned.
    """
    if record is None:
        return None
    if record.type == LtaType.LINEAR_RAS_TO_RAS:
        return _systems.RASmm()
    if record.type == LtaType.LINEAR_RSA_TO_RSA:
        return _systems.RSAmm()
    if record.type == LtaType.LINEAR_VOX_TO_VOX:
        return _system(LtaVoxelSystem, info)
    if record.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
        return _system(LtaPhysicalSystem, info)
    raise AssertionError(f"unsupported LTA type: {enum_name(record.type)}")


def _read_only(matrix: np.ndarray) -> np.ndarray:
    """Return a matrix decoded from a record, made read-only.

    The matrix is cached, but the writer reads the record and not the
    cached matrix, so an edit in place would be lost. A new matrix is set
    as data instead.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


# ----------------------------------------------------------------------
#   STORING A MATRIX IN THE RECORD
# ----------------------------------------------------------------------
# The matrix of an LTA file is small and lives in the record, so the data
# of a transformation is a view of the record. Setting the data stores the
# matrix in a new record, which the metadata then holds, and empties the
# private field in which the data model stores its data, so that a copy
# made with `replace` carries the record instead of a decoded matrix.


def _matrix_to_model(matrix: tx.Any) -> np.ndarray:
    """Return the data of a transformation from the matrix of a record.

    The record stores the homogeneous matrix, whose last row is dropped.
    """
    return np.asarray(matrix, dtype=np.float64)[:-1]


def _matrix_to_disk(data: tx.Optional[ArrayProtocol]) -> tx.Tuple:
    """Return the matrix that a record stores for the data of a model.

    The record stores the homogeneous matrix as a tuple of rows of floats.
    Data of `None`, which the data model reads as the identity, is stored
    as the identity.
    """
    if data is None:
        homogeneous = np.eye(4)
    else:
        data = np.asarray(data, dtype=np.float64)
        last = np.zeros((1, data.shape[-1]))
        last[0, -1] = 1.0
        homogeneous = np.concatenate([data, last])
    return tuple(tuple(float(v) for v in row) for row in homogeneous)


def _with_matrix(
    record: LtaRaw, lta_type: LtaType, data: tx.Optional[ArrayProtocol]
) -> LtaRaw:
    """Return a copy of a record that holds other data under a type.

    The other fields of the record, such as the geometries of the volumes,
    are kept.
    """
    affine = LtaRaw.Affine(matrix=_matrix_to_disk(data))
    return replace(record, type=lta_type, affine=affine)


def _store(xform: "LtaTransformation", record: LtaRaw) -> None:
    """Make a transformation hold a record, which carries its new data.

    The other metadata of the transformation is kept, and the private
    field in which the data model stores its data is emptied.
    """
    if xform.metadata is None:
        xform.metadata = LtaMetadata.from_raw(record)
    else:
        xform.metadata = replace(xform.metadata, raw=record)
    xform._data = None


def _set_metadata(
    self: "LtaTransformation", value: tx.Optional[LtaMetadata]
) -> None:
    """Store the metadata and drop the matrix decoded from its record."""
    self._metadata = value
    self.__dict__.pop("_cache_data", None)


def _set_data(
    self: "LtaTransformation", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the matrix in the record, under the type of the record."""
    record = _record_or_new(self, LtaType.LINEAR_VOX_TO_VOX, volumes=False)
    _store(self, _with_matrix(record, record.type, value))


def _set_vox_to_vox_data(
    self: "LtaTransformationVoxToVox", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the matrix in the record as a voxel-to-voxel matrix."""
    lta_type = LtaType.LINEAR_VOX_TO_VOX
    record = _record_or_new(self, lta_type, volumes=True)
    _store(self, _with_matrix(record, lta_type, value))


def _set_phys_to_phys_data(
    self: "LtaTransformationPhysToPhys", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the matrix in the record as a physical-to-physical matrix."""
    lta_type = LtaType.LINEAR_PHYSVOX_TO_PHYSVOX
    record = _record_or_new(self, lta_type, volumes=True)
    _store(self, _with_matrix(record, lta_type, value))


def _set_ras_to_ras_data(
    self: "LtaTransformationRASToRAS", value: tx.Optional[ArrayProtocol]
) -> None:
    """Store the matrix in the record as a RAS-to-RAS matrix."""
    lta_type = LtaType.LINEAR_RAS_TO_RAS
    record = _record_or_new(self, lta_type, volumes=False)
    _store(self, _with_matrix(record, lta_type, value))


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


class LtaFormat(FreesurferTransformationFormat, AffineTransformationFormat):
    """Marker of affine transformations stored in an LTA file."""

    HINTS = ("lta",)


# Text adapters supply byte decoding and encoding before the generic bases.
@register_format
class LtaTransformation(
    LtaFormat,
    TextFileReader,
    TextFileWriter,
    _xforms.Affine,
    TransformationFormat,
):
    """Transformation that can be encoded as a Linear Transform Array.

    LTA, the default linear-transform format of FreeSurfer, can represent
    several kinds of affine. This class is the registered format for `.lta`
    files, which `io.load` reads and `io.save` writes. The views
    [`LtaTransformationVoxToVox`][], [`LtaTransformationPhysToPhys`][] and
    [`LtaTransformationRASToRAS`][] read the same files but are not
    registered, since each would claim every `.lta` file.

    An LTA file is short and is read in one pass, so the whole file is
    held as an [`LtaRaw`][] record by the [`LtaMetadata`][] of the
    transformation. The matrix lives in that record, and the data and the
    coordinate systems of the transformation are read from it. Setting
    the data stores the matrix in a new record, and setting `input` or
    `output` overrides the system that the record defines.

    !!! note "What is written"
        A transformation whose `input` and `output` were not set is written
        as the record that its metadata holds, which also holds its matrix,
        provided that the record defines both systems. Any other
        transformation, including one converted from another affine, is
        written with the LTA type that matches its systems:

        | `input` and `output`  | LTA type                    |
        | --------------------- | --------------------------- |
        | both `RASmm`          | `LINEAR_RAS_TO_RAS`         |
        | both `RSAmm`          | `LINEAR_RSA_TO_RSA`         |
        | both `LtaVoxelSystem` | `LINEAR_VOX_TO_VOX`         |
        | both `LtaPhysicalSystem` | `LINEAR_PHYSVOX_TO_PHYSVOX` |

        Writing any other pair raises [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)

    _metadata: KwOnly[NoRepr[tx.Optional[LtaMetadata]]] = None

    metadata = smartproperty(
        "metadata", fset=_set_metadata, invalidates=_FORGET_VIEWS
    )
    """The metadata of the file, which holds the file as a record.

    A transformation built without data has no metadata. Setting the data
    gives the transformation metadata whose record holds the matrix, and
    assigning other metadata drops the matrix decoded from the old record.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # The constructor stores `data=` in the private field of the data
        # model, which the view does not read. The matrix is therefore
        # stored again, in the record. When both `data` and `metadata` are
        # given, the matrix of `data` replaces the matrix of the record.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The input coordinate system.

        Unless the system was set explicitly, it follows from the LTA type.
        It is `RASmm` for the RAS type and `RSAmm` for the RSA type. For the
        voxel and physical types, it is built from the source geometry, and
        it is `None` when the file records no source geometry. Without a
        record, it is `None`.
        """
        record = _record(self)
        return _typed_system(record, None if record is None else record.src)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The output coordinate system.

        Unless the system was set explicitly, it follows from the LTA type
        and from the destination geometry, in the same way as `input`.
        """
        record = _record(self)
        return _typed_system(record, None if record is None else record.dst)

    @smartproperty(cache=True, fset=_set_data, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` affine matrix, which the `matrix` view reads.

        The matrix is the matrix of the record, without its last row, and
        is read-only. Setting the matrix stores it in a new record under the
        same LTA type. Without a record, the matrix is `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_matrix_to_model(record.affine.matrix))

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a line as the first line of an LTA file.

        The line is scored by [`LtaMetadata.sniff_line`][].
        """
        return LtaMetadata.sniff_line(line, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: LtaRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`LtaRaw`][].

        The record is held by new metadata, without a copy. Keyword
        arguments such as `input`, `output` or `matrix` go to the
        constructor and override the record.
        """
        return cls(metadata=LtaMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from LTA lines.

        The lines are read by [`LtaMetadata.from_lines`][]. Keyword
        arguments such as `input`, `output` or `matrix` go to the
        constructor and override the file.
        """
        return cls(metadata=LtaMetadata.from_lines(lines), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        An [`LtaRaw`][] is read with [`from_raw`][], and any other value as
        the bases read it.
        """
        if isinstance(other, LtaRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy a transformation, sharing the record of an LTA transformation.

        Within the format, the record of the metadata carries the matrix,
        so the matrix is not decoded and passed on, and the copy holds the
        same metadata as `other`. A view copied from another view of the
        format reads the same record in its own coordinate systems.
        """
        if isinstance(other, LtaTransformation):
            kwargs.setdefault("data", None)
        return super().from_instance(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def to_raw(self) -> LtaRaw:
        """Return the [`LtaRaw`][] that encodes the transformation.

        If neither `input` nor `output` was set and the record that the
        metadata holds defines both, that record is returned as it is,
        without a copy, since it also holds the matrix. Otherwise, a new
        record is built. Its type
        follows from the systems, as described in [`LtaTransformation`][],
        and its matrix is `matrix`. Its volume geometries come from the
        systems for the voxel and physical types, and from the current
        record for the RAS and RSA types. Its other fields are copied from
        the current record. A transformation without a record is encoded in
        the same way.

        Raises
        ------
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        record = _record(self)
        overridden = self._input is not None or self._output is not None
        # A record that does not define both systems, such as the record
        # that setting the data of a new transformation creates, does not
        # say what its matrix maps, so the systems decide.
        undefined = self.input is None or self.output is None
        if record is None or overridden or undefined:
            record = _build_raw(self)
        _check_shape(record)
        return record

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the transformation to a path or file object in the LTA format.

        The record is built first, so a transformation that cannot be
        encoded leaves the file untouched.
        """
        return self.to_raw().to_file(file, **kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the LTA file that encodes the transformation."""
        return self.to_raw().to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the LTA file content that encodes the transformation."""
        return self.to_raw().to_text(**kwargs)


class LtaTransformationVoxToVox(LtaTransformation):
    """LTA file interpreted as a voxel-to-voxel affine.

    A transformation built from data alone maps between the voxel systems
    of two anonymous volumes, named `"src"` and `"dst"`.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The voxel system of the source volume, unless set explicitly."""
        lta_type = LtaType.LINEAR_VOX_TO_VOX
        record = _record_or_new(self, lta_type, volumes=True)
        return _system(LtaVoxelSystem, record.src)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The voxel system of the destination volume, unless set."""
        lta_type = LtaType.LINEAR_VOX_TO_VOX
        record = _record_or_new(self, lta_type, volumes=True)
        return _system(LtaVoxelSystem, record.dst)

    @smartproperty(
        cache=True, fset=_set_vox_to_vox_data, invalidates=_FORGET_VIEWS
    )
    def data(self) -> tx.Optional[np.ndarray]:
        """The voxel-to-voxel matrix derived from the record.

        The matrix is read-only. Setting the matrix stores it in a new
        record of the voxel-to-voxel type. Without a record, the matrix is
        `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_get_vox2vox(record)[:-1])


class LtaTransformationPhysToPhys(LtaTransformation):
    """LTA file interpreted as a physical-to-physical affine.

    A transformation built from data alone maps between the physical
    systems of two anonymous volumes, named `"src"` and `"dst"`.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The physical system of the source volume, unless set explicitly."""
        lta_type = LtaType.LINEAR_PHYSVOX_TO_PHYSVOX
        record = _record_or_new(self, lta_type, volumes=True)
        return _system(LtaPhysicalSystem, record.src)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The physical system of the destination volume, unless set."""
        lta_type = LtaType.LINEAR_PHYSVOX_TO_PHYSVOX
        record = _record_or_new(self, lta_type, volumes=True)
        return _system(LtaPhysicalSystem, record.dst)

    @smartproperty(
        cache=True, fset=_set_phys_to_phys_data, invalidates=_FORGET_VIEWS
    )
    def data(self) -> tx.Optional[np.ndarray]:
        """The physical-to-physical matrix derived from the record.

        The matrix is read-only. Setting the matrix stores it in a new
        record of the physical-to-physical type. Without a record, the
        matrix is `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_get_phys2phys(record)[:-1])


class LtaTransformationRASToRAS(LtaTransformation):
    """LTA file interpreted as a RAS-to-RAS affine."""

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The RAS system, unless set explicitly."""
        return _systems.RASmm()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The RAS system, unless set explicitly."""
        return _systems.RASmm()

    @smartproperty(
        cache=True, fset=_set_ras_to_ras_data, invalidates=_FORGET_VIEWS
    )
    def data(self) -> tx.Optional[np.ndarray]:
        """The RAS-to-RAS matrix derived from the record.

        The matrix is read-only. Setting the matrix stores it in a new
        record of the RAS-to-RAS type. Without a record, the matrix is
        `None`.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_get_ras2ras(record)[:-1])


# ----------------------------------------------------------------------
#   ENCODING AN AFFINE AS AN LTA RECORD
# ----------------------------------------------------------------------


def _check_shape(record: LtaRaw) -> None:
    """Refuse a record whose matrix is not the matrix of a 3D affine.

    Raises
    ------
    UnrepresentableTransformationError
        If the homogeneous matrix of the record is not of shape `(4, 4)`.
    """
    shape = record.affine.shape
    if shape != (4, 4):
        raise UnrepresentableTransformationError(
            f"An LTA file encodes a 3D affine, of shape (3, 4), but the "
            f"matrix has shape {(shape[0] - 1, shape[1])}."
        )


def _as_block(
    info: tx.Optional[LtaRaw.VolumeInfo], cls: type
) -> LtaRaw.VolumeInfo:
    """Return a volume geometry as a block of type `cls`.

    When there is no geometry, the block is marked as invalid, which is how
    FreeSurfer writes a missing geometry.
    """
    if info is None:
        return cls(valid=LtaValidity.VOLUME_INFO_INVALID)
    if type(info) is cls:
        return info
    return cls(**{f.name: getattr(info, f.name) for f in fields(cls)})


def _build_raw(xform: LtaTransformation) -> LtaRaw:
    """Encode the matrix and coordinate systems of `xform` as a record.

    See [`LtaTransformation.to_raw`][].
    """
    src_system, dst_system = xform.input, xform.output
    old = _record(xform)
    if old is None:
        old = LtaRaw()
    Src, Dst = LtaRaw.SrcVolumeInfo, LtaRaw.DstVolumeInfo

    def both(cls: type) -> bool:
        return isinstance(src_system, cls) and isinstance(dst_system, cls)

    if both(_systems.RASmm) or both(_systems.RSAmm):
        # A RAS-to-RAS affine does not depend on the volume geometries, but
        # FreeSurfer tools use them, so the geometries of the current record
        # are kept.
        lta_type = (
            LtaType.LINEAR_RAS_TO_RAS
            if both(_systems.RASmm)
            else LtaType.LINEAR_RSA_TO_RSA
        )
        src, dst = _as_block(old.src, Src), _as_block(old.dst, Dst)
    elif both(LtaVoxelSystem) or both(LtaPhysicalSystem):
        lta_type = (
            LtaType.LINEAR_VOX_TO_VOX
            if both(LtaVoxelSystem)
            else LtaType.LINEAR_PHYSVOX_TO_PHYSVOX
        )
        src = _as_block(src_system.raw, Src)
        dst = _as_block(dst_system.raw, Dst)
    else:

        def name(system: tx.Any) -> str:
            if system is None:
                return "an unspecified system"
            return type(system).__name__

        raise UnrepresentableTransformationError(
            f"An LTA file cannot encode an affine from {name(src_system)} "
            f"to {name(dst_system)}: it encodes RASmm to RASmm, RSAmm to "
            f"RSAmm, LtaVoxelSystem to LtaVoxelSystem, and "
            f"LtaPhysicalSystem to LtaPhysicalSystem. Set the input and "
            f"output of the affine to say which it is."
        )

    return LtaRaw(
        type=lta_type,
        nxforms=old.nxforms,
        mean=old.mean,
        sigma=old.sigma,
        affine=LtaRaw.Affine(matrix=_matrix_to_disk(xform.matrix)),
        label=old.label,
        src=src,
        dst=dst,
    )
