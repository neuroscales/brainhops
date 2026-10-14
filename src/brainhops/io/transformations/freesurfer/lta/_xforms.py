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
from bagof.magic import Inherit, NoRepr, fields, replace

# internals
from brainhops._core import path
from brainhops._core.enum import enum_name
from brainhops._core.properties import (
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.common.freesurfer import FreesurferTkrSystem
from brainhops.io.metadata._base import (
    metadata_to_write,
    requested_fields,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)
from brainhops.io.transformations.base._records import (
    matrix_to_disk,
    matrix_to_model,
    read_only,
    record_of,
)

# this format
from .._formats import FreesurferTransformationFormat
from ._enums import LtaType, LtaValidity
from ._matrix_utils import (
    _get_phys2phys,
    _get_ras2ras,
    _get_vox2vox,
    _volume_info,
)
from ._metadata import LtaMetadata
from ._raw import LtaRaw
from ._systems import (
    LtaCoordinateSystem,
    LtaPhysicalSystem,
    LtaVoxelSystem,
    _name,
)

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------


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
    For the `REGISTER_DAT` type, it is the tkregister space of the volume,
    named after the volume, whose geometry stays in the record. For the
    voxel and physical types, it is built from the geometry of the volume,
    `info`. Without a record, the system is unknown and `None` is
    returned.
    """
    if record is None:
        return None
    if record.type == LtaType.LINEAR_RAS_TO_RAS:
        return _systems.RASmm()
    if record.type == LtaType.LINEAR_RSA_TO_RSA:
        return _systems.RSAmm()
    if record.type == LtaType.REGISTER_DAT:
        return FreesurferTkrSystem(**({} if info is None else _name(info)))
    if record.type == LtaType.LINEAR_VOX_TO_VOX:
        return _system(LtaVoxelSystem, info)
    if record.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
        return _system(LtaPhysicalSystem, info)
    raise AssertionError(f"unsupported LTA type: {enum_name(record.type)}")


# ----------------------------------------------------------------------
#   THE MATRIX OF A RECORD
# ----------------------------------------------------------------------
# The matrix of an LTA file is small and lives in the record, so the data
# of a transformation read from a file is a view of the record. A matrix
# that is assigned is held by the data model, in its private field, as
# for any affine, and it is pending: the record is not changed until the
# transformation is consolidated, which encodes a new record. A
# transformation is never given a record that it did not read or encode,
# because the type and the geometries of such a record would be made up,
# and the writer would trust them. The helpers that read the record and
# decode and encode its matrix are shared with the other formats whose
# matrix lives in their record (`io/transformations/base/_records.py`).


def _record_to_model(record: LtaRaw) -> np.ndarray:
    """Return the data of a transformation from a record, in its systems.

    A record of the `REGISTER_DAT` type stores the matrix from the
    tkregister space of `dst` to that of `src`, as FreeSurfer defines it
    (`LTAchangeType` in `utils/transform.cpp`, commit 4f8728e), so the
    data is its inverse, which maps `src` to `dst` as the other types do.
    Any other record stores the matrix from `src` to `dst`.
    """
    matrix = np.asarray(record.affine.matrix, dtype=np.float64)
    if record.type == LtaType.REGISTER_DAT and matrix.shape == (4, 4):
        matrix = np.linalg.inv(matrix)
    return matrix_to_model(matrix)


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
    coordinate systems of the transformation are read from it. The record
    describes the file as it was read, or as the transformation was last
    consolidated. A matrix that is assigned, and an `input` or `output`
    that is set, are pending: they win over the record until
    [`consolidate`][] encodes them into a new record. A transformation
    without a record, such as one built from data alone, holds its matrix
    as any affine does and has no metadata.

    !!! note "What is written"
        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is written as that record, which also
        holds its matrix. The record is written as it was read, even when
        it does not define both systems. Any other transformation,
        including one built from data alone or converted from another
        affine, is consolidated, and written with the LTA type that
        matches its systems:

        | `input` and `output`       | LTA type                    |
        | -------------------------- | --------------------------- |
        | both `RASmm`               | `LINEAR_RAS_TO_RAS`         |
        | both `RSAmm`               | `LINEAR_RSA_TO_RSA`         |
        | both `LtaVoxelSystem`      | `LINEAR_VOX_TO_VOX`         |
        | both `LtaPhysicalSystem`   | `LINEAR_PHYSVOX_TO_PHYSVOX` |
        | both `FreesurferTkrSystem` | `REGISTER_DAT`              |

        Writing any other pair raises [`UnrepresentableTransformationError`][].
        In particular, a tkregister space, a
        [`FreesurferTkrSystem`][brainhops.io.common.freesurfer.FreesurferTkrSystem],
        is never written as scanner RAS.

    !!! note "The `REGISTER_DAT` type"
        An LTA file of type 14, `REGISTER_DAT`, holds the matrix of a
        `register.dat` file, which maps the tkregister RAS space of the
        destination volume to that of the source volume, as FreeSurfer
        defines it (`LTAchangeType` and `ltaReadRegisterDat` in
        `utils/transform.cpp`, commit 4f8728e). The transformation reads
        the inverse of that matrix, so that it maps the source to the
        destination as every other type does, and its systems are the
        tkregister spaces of the two volumes. The views read the file in
        their own systems, as FreeSurfer converts it, which needs both
        volume geometries. A matrix that is assigned between two
        tkregister spaces is written with this type, and its inverse is
        stored.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)

    _metadata: NoRepr[tx.Optional[LtaMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata, and setting
    its data does not create any. Assigning other metadata drops the
    matrix decoded from the old record. A matrix that was assigned is
    pending and is kept, and it wins over the matrix of the new record
    until the transformation is consolidated.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The input coordinate system.

        Unless the system was set explicitly, it follows from the LTA type.
        It is `RASmm` for the RAS type and `RSAmm` for the RSA type. For the
        voxel and physical types, it is built from the source geometry, and
        it is `None` when the file records no source geometry. Without a
        record, it is `None`.
        """
        record = record_of(self)
        return _typed_system(record, None if record is None else record.src)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The output coordinate system.

        Unless the system was set explicitly, it follows from the LTA type
        and from the destination geometry, in the same way as `input`.
        """
        record = record_of(self)
        return _typed_system(record, None if record is None else record.dst)

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record, under the same
        LTA type. Otherwise the matrix is the matrix of the record, without
        its last row, and is read-only, or `None` without a record.
        """
        record = record_of(self)
        if record is None:
            return None
        return read_only(_record_to_model(record))

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

        Parameters
        ----------
        line : str
            The first line of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the line is not the first line
            of an LTA file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return LtaMetadata.sniff_line(line, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: LtaRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`LtaRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : LtaRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `input`, `output` or `matrix`, which go
            to the constructor and override the record.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=LtaMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from LTA lines.

        The lines are read by [`LtaMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Field values, such as `input`, `output` or `matrix`, which go
            to the constructor and override the file.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=LtaMetadata.from_lines(lines), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        An [`LtaRaw`][] is read with [`from_raw`][], and any other value as
        the bases read it.

        Parameters
        ----------
        other : object
            A record, a file, its content, a transformation, a mapping, or
            an instance of a similar class.
        *args : Any
            Constructor arguments. A record and a file take keyword
            arguments only.
        **kwargs : Any
            Options of the reader, and field values otherwise.

        Returns
        -------
        Self
            The new transformation.
        """
        if isinstance(other, LtaRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy a transformation, sharing the record of an LTA transformation.

        Within the format, the record of the metadata carries the matrix,
        so the matrix is not decoded and passed on, and the copy holds the
        same metadata as `other`. The data model would copy the decoded
        matrix, so the rule is kept here. A transformation of the format
        that holds a pending matrix, such as one built from data alone, is
        copied with that matrix, as any affine is. Any other
        transformation, including another view of the format, is converted
        as `other.to(cls)` converts it: a view then reads the same record
        in its own coordinate systems.

        Parameters
        ----------
        other : object
            The instance to copy or convert.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`. A transformation that is converted takes keyword
            options only.
        **kwargs : Any
            Constructor arguments, which take precedence over the fields
            of `other`, or options of the conversion.

        Returns
        -------
        Self
            The copy.

        Raises
        ------
        ConversionError
            If the format cannot hold a transformation that is converted.
        """
        if isinstance(other, cls) and other._data is None:
            kwargs.setdefault("data", None)
        return super().from_instance(other, *args, **kwargs)

    # --- to -----------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. Otherwise, a new record is built. Its type
        follows from the systems, as described in [`LtaTransformation`][],
        and its matrix is `matrix`, or its inverse for the `REGISTER_DAT`
        type. Its volume geometries come from the systems for the voxel
        and physical types, and from the current record for the RAS and
        RSA types. For the `REGISTER_DAT` type, they come from the
        geometry of each tkregister system when it has one, and from the
        current record otherwise. Its other fields are copied from
        the current record. A transformation without a record is encoded in
        the same way. The new record is held by new metadata, and the
        matrix and the systems are then decoded from it.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an LTA file
        holds no field of the vocabulary yet, so a field that is set is
        reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        Self
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        report = ConversionReport(
            source="LtaTransformation", target="LtaTransformation"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, LtaMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        record = record_of(self)
        overridden = (
            self._data is not None
            or self._input is not None
            or self._output is not None
        )
        if record is not None and not overridden:
            if not requested:
                return self if inplace else replace(self)
            if not inplace:
                return replace(self, metadata=metadata)
            self.metadata = metadata
            return self
        record = _build_raw(self)
        _check_shape(record)
        if metadata is None:
            metadata = LtaMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self, metadata=metadata, data=None, input=None, output=None
            )
        self.metadata = metadata
        self.data = None
        self.input = None
        self.output = None
        return self

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the transformation to a path or file object in the LTA format.

        The transformation is consolidated into a copy, whose record is
        written, so a transformation that cannot be encoded leaves the file
        untouched, and the transformation itself is not changed.

        Parameters
        ----------
        file : path or file object
            The path or the open text file.
        **kwargs : Any
            `on_loss` and the fields of the metadata, as for
            [`consolidate`][], and options of the writer of the record.

        Raises
        ------
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        record, options = _lta_written(self, kwargs)
        return record.to_file(file, **options)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the LTA file that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            `on_loss` and the fields of the metadata, as for
            [`consolidate`][], and options of the writer of the record.

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        record, options = _lta_written(self, kwargs)
        return record.to_lines(**options)

    def to_text(self, **kwargs) -> str:
        """Return the LTA file content that encodes the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            `on_loss` and the fields of the metadata, as for
            [`consolidate`][], and options of the writer of the record.

        Returns
        -------
        str
            The content of the file, ending with a newline.

        Raises
        ------
        UnrepresentableTransformationError
            If the LTA format cannot encode the systems or the matrix.
        """
        record, options = _lta_written(self, kwargs)
        return record.to_text(**options)


class LtaTransformationVoxToVox(LtaTransformation):
    """LTA file interpreted as a voxel-to-voxel affine.

    A transformation built from data alone maps between the voxel systems
    of two anonymous volumes, named `"src"` and `"dst"`.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The voxel system of the source volume, unless set explicitly."""
        record = record_of(self)
        info = LtaRaw.SrcVolumeInfo() if record is None else record.src
        return _system(LtaVoxelSystem, info)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The voxel system of the destination volume, unless set."""
        record = record_of(self)
        info = LtaRaw.DstVolumeInfo() if record is None else record.dst
        return _system(LtaVoxelSystem, info)

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The voxel-to-voxel matrix, assigned or derived from the record.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record, under the
        voxel-to-voxel type. Otherwise the matrix is derived from the
        record and is read-only, or `None` without a record.
        """
        record = record_of(self)
        if record is None:
            return None
        return read_only(_get_vox2vox(record)[:-1])


class LtaTransformationPhysToPhys(LtaTransformation):
    """LTA file interpreted as a physical-to-physical affine.

    A transformation built from data alone maps between the physical
    systems of two anonymous volumes, named `"src"` and `"dst"`.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The physical system of the source volume, unless set explicitly."""
        record = record_of(self)
        info = LtaRaw.SrcVolumeInfo() if record is None else record.src
        return _system(LtaPhysicalSystem, info)

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The physical system of the destination volume, unless set."""
        record = record_of(self)
        info = LtaRaw.DstVolumeInfo() if record is None else record.dst
        return _system(LtaPhysicalSystem, info)

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The physical-to-physical matrix, assigned or from the record.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record, under the
        physical-to-physical type. Otherwise the matrix is derived from the
        record and is read-only, or `None` without a record.
        """
        record = record_of(self)
        if record is None:
            return None
        return read_only(_get_phys2phys(record)[:-1])


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

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The RAS-to-RAS matrix, assigned or derived from the record.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record, under the
        RAS-to-RAS type. Otherwise the matrix is derived from the record
        and is read-only, or `None` without a record.
        """
        record = record_of(self)
        if record is None:
            return None
        return read_only(_get_ras2ras(record)[:-1])


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
        # The record stores the homogeneous matrix, whose last row the
        # data model drops. An empty matrix has no row to drop.
        if shape[0] > 0:
            shape = (shape[0] - 1, shape[1])
        raise UnrepresentableTransformationError(
            f"An LTA file encodes a 3D affine, of shape (3, 4), but the "
            f"matrix has shape {shape}."
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

    See [`LtaTransformation.consolidate`][].
    """
    src_system, dst_system = xform.input, xform.output
    old = record_of(xform)
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
    elif both(FreesurferTkrSystem):
        # FreeSurfer stores the matrix of a register.dat file, from the
        # tkregister space of `dst` to that of `src`.
        lta_type = LtaType.REGISTER_DAT
        src = _tkr_block(src_system, old.src, Src)
        dst = _tkr_block(dst_system, old.dst, Dst)
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
            f"RSAmm, LtaVoxelSystem to LtaVoxelSystem, LtaPhysicalSystem "
            f"to LtaPhysicalSystem, and FreesurferTkrSystem to "
            f"FreesurferTkrSystem. Set the input and output of the affine "
            f"to say which it is."
        )

    matrix = matrix_to_disk(xform.matrix)
    if lta_type == LtaType.REGISTER_DAT and np.shape(matrix) == (4, 4):
        try:
            inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        except np.linalg.LinAlgError:
            raise UnrepresentableTransformationError(
                "An LTA file of type REGISTER_DAT stores the inverse of an "
                "affine between tkregister spaces, and this affine is not "
                "invertible."
            ) from None
        matrix = tuple(tuple(float(v) for v in row) for row in inverse)
    return LtaRaw(
        type=lta_type,
        nxforms=old.nxforms,
        mean=old.mean,
        sigma=old.sigma,
        affine=LtaRaw.Affine(matrix=matrix),
        label=old.label,
        src=src,
        dst=dst,
    )


def _tkr_block(
    system: FreesurferTkrSystem,
    old: tx.Optional[LtaRaw.VolumeInfo],
    cls: tx.Type[LtaRaw.VolumeInfo],
) -> LtaRaw.VolumeInfo:
    """Return the geometry block of the volume of a tkregister system.

    The block is encoded from the geometry of the system when it has one,
    and is otherwise that of the current record, or an invalid block.
    """
    if system.geometry is not None:
        return _volume_info(system.geometry, cls)
    return _as_block(old, cls)


def _lta_written(
    xform: LtaTransformation, kwargs: tx.Mapping[str, tx.Any]
) -> tx.Tuple[LtaRaw, tx.Dict[str, tx.Any]]:
    """Return the record that a file written from a transformation holds.

    The record is that of the copy of the transformation consolidated
    with `on_loss` and the fields of the metadata among `kwargs`, and the
    other options, for the writer of the record, are returned with it.

    Raises
    ------
    UnrepresentableTransformationError
        If the LTA format cannot encode the systems or the matrix.
    """
    fields = {
        name: value
        for name, value in kwargs.items()
        if name in FIELDS or name == "on_loss"
    }
    options = {
        name: value for name, value in kwargs.items() if name not in fields
    }
    record = record_of(xform.consolidate(**fields))
    _check_shape(record)
    return record, options
