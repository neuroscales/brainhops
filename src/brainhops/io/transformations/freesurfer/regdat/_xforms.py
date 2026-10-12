"""Affine transformations stored in `register.dat` files."""

__all__ = [
    "RegisterDatFormat",
    "RegisterDatTransformation",
    "RegisterDatTransformationRASToRAS",
]

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import (
    InvalidatorInAttribute,
    smartproperty,
)
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileReader,
    TextFileWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.common.freesurfer._geometry import fs_vox2tkr
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)

# this format
from .._formats import FreesurferTransformationFormat
from ..lta import LtaRaw
from ..lta._matrix_utils import _get_vox2ras
from ..lta._xforms import _matrix_to_disk, _matrix_to_model, _read_only
from ._metadata import RegisterDatMetadata
from ._raw import RegisterDatRaw
from ._systems import RegisterDatTkrSystem

_FORGET_VIEWS = InvalidatorInAttribute("derived_fields")
"""Invalidator that clears the views that the data model derives."""

_VolumeInfo = tx.Optional[LtaRaw.VolumeInfo]


# ----------------------------------------------------------------------
#   READING THE RECORD
# ----------------------------------------------------------------------
# The matrix of a `register.dat` file is small and lives in the record,
# as the matrix of an LTA file does, so the data of a transformation read
# from a file is a view of the record. A matrix that is assigned is held
# by the data model, in its private field, and it is pending: the record
# is not changed until the transformation is consolidated, which encodes
# a new record. The file holds no geometry, so the geometries of the
# reference and movable volumes are given to the transformation, and
# they are never written.


def _record(xform: "RegisterDatTransformation") -> tx.Optional[RegisterDatRaw]:
    """Return the record of the metadata of a transformation, or `None`."""
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


# ----------------------------------------------------------------------
#   FORMATS
# ----------------------------------------------------------------------


class RegisterDatFormat(
    FreesurferTransformationFormat, AffineTransformationFormat
):
    """Marker of affine transformations stored in a `register.dat` file."""

    HINTS = ("regdat",)


# Text adapters supply byte decoding and encoding before the generic bases.
@register_format
class RegisterDatTransformation(
    RegisterDatFormat,
    TextFileReader,
    TextFileWriter,
    _xforms.Affine,
    TransformationFormat,
):
    """Registration stored in a FreeSurfer `register.dat` file.

    `tkregister2` and `bbregister` write their registration to a
    `register.dat` file when they are given `--reg`. The file holds a 4x4
    matrix that maps the tkregister RAS coordinates of the reference
    (target) volume, usually the anatomical volume of the subject, to the
    tkregister RAS coordinates of the movable volume, such as a functional
    volume. FreeSurfer documents this direction in `regio_read_register`
    (`utils/registerio.cpp`, commit 4f8728e) and in `MRItkReg2Native`
    (`utils/mri.cpp`). The tkregister RAS space of a volume is described
    by [`RegisterDatTkrSystem`][]. This class is the registered format for
    `.dat` and `.reg` files, which `io.load` reads and `io.save` writes.

    The file holds no geometry of either volume. The input and the output
    of the transformation are therefore the tkregister spaces of two
    volumes that are not known, unless the geometries are given with the
    options `reference=` and `moving=` when the file is read, or set on
    the transformation later. A geometry is an
    [`LtaRaw.VolumeInfo`][brainhops.io.transformations.freesurfer.lta.LtaRaw],
    the geometry block that LTA files store. With both geometries, the
    view [`RegisterDatTransformationRASToRAS`][] reads the same file as an
    affine between the scanner RAS spaces of the two volumes.

    The whole file is held as a [`RegisterDatRaw`][] record by the
    [`RegisterDatMetadata`][] of the transformation, and the matrix is
    read from it. A matrix that is assigned, and an `input` or `output`
    that is set, are pending until [`consolidate`][] encodes them into a
    new record. A transformation without a record, such as one built from
    data alone, holds its matrix as any affine does and has no metadata.

    !!! note "What is written"
        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is written as that record, byte for
        byte when the record was read from a file. Any other
        transformation is consolidated. When `input` and `output` are both
        [`RegisterDatTkrSystem`][], the matrix is written as it is. When
        they are both `RASmm`, the matrix is taken to map the scanner RAS
        space of the reference volume to that of the movable volume, and
        it is converted with the geometries `reference` and `moving`,
        which are then required. Writing any other pair raises
        [`UnrepresentableTransformationError`][].

        The geometries are not written, so they must be given again to
        read the file as scanner RAS.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dat", ".reg")

    _metadata: NoRepr[tx.Optional[RegisterDatMetadata]] = None

    _reference: KwOnly[_VolumeInfo] = None

    _moving: KwOnly[_VolumeInfo] = None

    metadata = smartproperty("metadata", invalidates=("data", _FORGET_VIEWS))
    """The metadata of the file, which holds the file as a record.

    A transformation built from data alone has no metadata, and setting
    its data does not create any. Assigning other metadata drops the
    matrix decoded from the old record. A matrix that was assigned is
    pending and is kept, and it wins over the matrix of the new record
    until the transformation is consolidated.
    """

    reference = smartproperty("reference", invalidates=("data", _FORGET_VIEWS))
    """The geometry of the reference (target) volume, or `None`.

    It is an option of the reader, since the file does not store it, and
    it is never written. It gives the input system its geometry, and the
    view [`RegisterDatTransformationRASToRAS`][] needs it.
    """

    moving = smartproperty("moving", invalidates=("data", _FORGET_VIEWS))
    """The geometry of the movable volume, or `None`.

    It is an option of the reader, since the file does not store it, and
    it is never written. It gives the output system its geometry, and the
    view [`RegisterDatTransformationRASToRAS`][] needs it.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The input coordinate system.

        Unless it was set explicitly, it is the tkregister space of the
        reference volume, which holds the geometry `reference` when it is
        given.
        """
        return RegisterDatTkrSystem.from_raw(self.reference, "reference")

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The output coordinate system.

        Unless it was set explicitly, it is the tkregister space of the
        movable volume, which holds the geometry `moving` when it is given.
        """
        return RegisterDatTkrSystem.from_raw(self.moving, "moving")

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` affine matrix, which the `matrix` view reads.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] encodes it in a new record. Otherwise the
        matrix is the matrix of the record, without its last row, and is
        read-only, or `None` without a record.
        """
        record = _record(self)
        if record is None:
            return None
        return _read_only(_matrix_to_model(record.matrix))

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file as those of a `register.dat` file.

        The lines are scored by [`RegisterDatMetadata.sniff_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not those of a
            `register.dat` file.
        **kwargs : Any
            Options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return RegisterDatMetadata.sniff_lines(lines, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_raw(cls, raw: RegisterDatRaw, **kwargs) -> tx.Self:
        """Build a transformation from an already parsed [`RegisterDatRaw`][].

        The record is held by new metadata, without a copy.

        Parameters
        ----------
        raw : RegisterDatRaw
            The record of the file.
        **kwargs : Any
            Field values, such as `reference`, `moving` or `matrix`, which
            go to the constructor.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=RegisterDatMetadata.from_raw(raw), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build a transformation from the lines of a `register.dat` file.

        The lines are read by [`RegisterDatMetadata.from_lines`][].

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Field values, such as the geometries `reference` and `moving`,
            which go to the constructor. Any other option is refused.

        Returns
        -------
        Self
            The transformation.
        """
        return cls(metadata=RegisterDatMetadata.from_lines(lines), **kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build a transformation from a record, a file or a data model.

        A [`RegisterDatRaw`][] is read with [`from_raw`][], and any other
        value as the bases read it.

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
        if isinstance(other, RegisterDatRaw) and not args:
            return cls.from_raw(other, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy a transformation, sharing the record of a `register.dat`.

        Within the format, the record of the metadata carries the matrix,
        so the matrix is not decoded and passed on, and the copy holds the
        same metadata and the same geometries as `other`. A transformation
        of the format that holds a pending matrix is copied with that
        matrix, as any affine is. Any other transformation, including
        another view of the format, is converted as `other.to(cls)`
        converts it: a view then reads the same record in its own
        coordinate systems.

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

    def consolidate(self, *, inplace: bool = False) -> tx.Self:
        """Encode the pending matrix and systems into a new record.

        A transformation that holds a record, and whose matrix, `input`
        and `output` were not set, is left as it is, since its record also
        holds its matrix. The geometries `reference` and `moving` are not
        stored in the file, so setting them changes no record. Otherwise,
        a new record is built, whose matrix follows from `matrix` and the
        systems, as described in [`RegisterDatTransformation`][]. Its
        other fields are copied from the current record. Without a
        record, the subject is unknown, the intensity scale is 1, the
        rounding method is `"round"`, and the resolution lines hold the
        first and third voxel sizes of `moving`, as `tkregister2` writes
        them, or 1 when `moving` is not given. The new record is held by
        new metadata, and the matrix and the systems are then decoded
        from it.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this transformation rather than a copy.

        Returns
        -------
        Self
            The consolidated copy, or this transformation with
            `inplace=True`.

        Raises
        ------
        UnrepresentableTransformationError
            If the format cannot encode the systems or the matrix.
        """
        record = _record(self)
        overridden = (
            self._data is not None
            or self._input is not None
            or self._output is not None
        )
        if record is not None and not overridden:
            return self if inplace else replace(self)
        record = _build_raw(self)
        if self.metadata is None:
            metadata = RegisterDatMetadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
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
        """Write the transformation to a path or a file object.

        The transformation is consolidated into a copy, whose record is
        written, so a transformation that cannot be encoded leaves the file
        untouched, and the transformation itself is not changed.

        Parameters
        ----------
        file : path or file object
            The path or the open text file.
        **kwargs : Any
            Options of the writer of the record.

        Raises
        ------
        UnrepresentableTransformationError
            If the format cannot encode the systems or the matrix.
        """
        return _written(self).to_file(file, **kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the `register.dat` file of the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Yields
        ------
        str
            The lines of the file, without line endings.

        Raises
        ------
        UnrepresentableTransformationError
            If the format cannot encode the systems or the matrix.
        """
        return _written(self).to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """Return the content of the `register.dat` file of the transformation.

        The transformation is consolidated into a copy, whose record is
        written.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        str
            The content of the file, ending with a newline.

        Raises
        ------
        UnrepresentableTransformationError
            If the format cannot encode the systems or the matrix.
        """
        return _written(self).to_text(**kwargs)


class RegisterDatTransformationRASToRAS(RegisterDatTransformation):
    """A `register.dat` file read as an affine between scanner RAS spaces.

    The affine maps the scanner RAS coordinates of the reference volume to
    those of the movable volume, in the direction of the file. It is

    ```
    M = tkr2ras(moving) @ R @ inv(tkr2ras(reference))
    ```

    where `R` is the matrix of the file and `tkr2ras(v)` is the matrix
    `vox2ras(v) @ inv(vox2tkr(v))` of a volume `v`. This is the inverse of
    the matrix that `MRItkReg2Native` in `utils/mri.cpp` of FreeSurfer
    (commit 4f8728e) computes, which `lta_convert --inreg` writes as a
    `LINEAR_RAS_TO_RAS` LTA from the movable to the reference volume. The
    geometries `reference` and `moving` are therefore required to read
    the matrix. This class is a view and is not registered.
    """

    @smartproperty
    def input(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The scanner RAS space of the reference, unless set explicitly."""
        return _systems.RASmm()

    @smartproperty
    def output(self) -> tx.Optional[_systems.CoordinateSystem]:
        """The scanner RAS space of the movable volume, unless set."""
        return _systems.RASmm()

    @smartproperty(cache=True, invalidates=_FORGET_VIEWS)
    def data(self) -> tx.Optional[np.ndarray]:
        """The scanner RAS-to-RAS matrix, assigned or derived from the record.

        A matrix that is assigned is returned as it is, and stays pending
        until [`consolidate`][] converts it back to tkregister RAS and
        encodes it in a new record. Otherwise the matrix is derived from
        the record and the two geometries, and is read-only, or `None`
        without a record.

        Raises
        ------
        ValueError
            If the transformation has a record but the geometry of the
            reference or of the movable volume is not given.
        """
        record = _record(self)
        if record is None:
            return None
        reference, moving = _geometries(self)
        tkr = np.asarray(record.matrix, dtype=np.float64)
        return _read_only(_tkr_to_scanner(tkr, reference, moving)[:-1])


# ----------------------------------------------------------------------
#   ENCODING AN AFFINE AS A REGISTER.DAT RECORD
# ----------------------------------------------------------------------


def _geometries(
    xform: RegisterDatTransformation,
) -> tx.Tuple[LtaRaw.VolumeInfo, LtaRaw.VolumeInfo]:
    """Return the geometries of the reference and the movable volume.

    Raises
    ------
    ValueError
        If either geometry is not given.
    """
    if xform.reference is None or xform.moving is None:
        raise ValueError(
            "A register.dat file maps tkregister RAS spaces, and it holds "
            "no volume geometry, so the geometries of the reference and "
            "the movable volume are needed to place it in scanner RAS. "
            "Pass reference=... and moving=... when loading, or set them "
            "on the transformation."
        )
    return xform.reference, xform.moving


def _tkr_to_ras(info: LtaRaw.VolumeInfo) -> np.ndarray:
    """Return the matrix from the tkregister RAS to the scanner RAS of a
    volume.

    It is `vox2ras @ inv(vox2tkr)`, where `vox2tkr` is the matrix of
    `MRIxfmCRS2XYZtkreg` in `utils/mri.cpp` of FreeSurfer.
    """
    vox2ras = _get_vox2ras(info)
    vox2tkr = fs_vox2tkr(info.volume, info.voxelsize)
    return vox2ras @ np.linalg.inv(vox2tkr)


def _tkr_to_scanner(
    matrix: np.ndarray,
    reference: LtaRaw.VolumeInfo,
    moving: LtaRaw.VolumeInfo,
) -> np.ndarray:
    """Return the scanner RAS matrix of a tkregister RAS matrix."""
    return _tkr_to_ras(moving) @ matrix @ np.linalg.inv(_tkr_to_ras(reference))


def _scanner_to_tkr(
    matrix: np.ndarray,
    reference: LtaRaw.VolumeInfo,
    moving: LtaRaw.VolumeInfo,
) -> np.ndarray:
    """Return the tkregister RAS matrix of a scanner RAS matrix.

    This is the matrix that `MRItkRegMtx` in `utils/mri.cpp` of
    FreeSurfer computes from the inverse of `matrix`.
    """
    return np.linalg.inv(_tkr_to_ras(moving)) @ matrix @ _tkr_to_ras(reference)


def _build_raw(xform: RegisterDatTransformation) -> RegisterDatRaw:
    """Encode the matrix and coordinate systems of `xform` as a record.

    See [`RegisterDatTransformation.consolidate`][].

    Raises
    ------
    UnrepresentableTransformationError
        If the format cannot encode the systems or the matrix.
    """
    src_system, dst_system = xform.input, xform.output

    def both(cls: type) -> bool:
        return isinstance(src_system, cls) and isinstance(dst_system, cls)

    data = xform.matrix
    shape = None if data is None else np.shape(data)
    if data is not None and shape != (3, 4):
        raise UnrepresentableTransformationError(
            f"A register.dat file encodes a 3D affine, of shape (3, 4), but "
            f"the matrix has shape {shape}."
        )
    homogeneous = np.asarray(_matrix_to_disk(data), dtype=np.float64)
    if both(RegisterDatTkrSystem):
        matrix = homogeneous
    elif both(_systems.RASmm):
        try:
            reference, moving = _geometries(xform)
        except ValueError as e:
            raise UnrepresentableTransformationError(str(e)) from None
        matrix = _scanner_to_tkr(homogeneous, reference, moving)
    else:

        def name(system: tx.Any) -> str:
            if system is None:
                return "an unspecified system"
            return type(system).__name__

        raise UnrepresentableTransformationError(
            f"A register.dat file cannot encode an affine from "
            f"{name(src_system)} to {name(dst_system)}: it encodes "
            f"RegisterDatTkrSystem to RegisterDatTkrSystem, or RASmm to "
            f"RASmm given the geometries of both volumes. Set the input "
            f"and output of the affine to say which it is."
        )

    old = _record(xform)
    if old is None:
        old = _new_record(xform.moving)
    return RegisterDatRaw(
        subject=old.subject,
        inplane_resolution=old.inplane_resolution,
        slice_thickness=old.slice_thickness,
        intensity=old.intensity,
        matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        float2int=old.float2int,
    )


def _new_record(moving: _VolumeInfo) -> RegisterDatRaw:
    """Return the fields of a record for a transformation without one.

    The resolution lines hold the first and third voxel sizes of the
    movable volume, as `tkregister2` writes them, when its geometry is
    given.
    """
    if moving is None:
        return RegisterDatRaw()
    return RegisterDatRaw(
        inplane_resolution=float(moving.voxelsize[0]),
        slice_thickness=float(moving.voxelsize[2]),
    )


def _written(xform: RegisterDatTransformation) -> RegisterDatRaw:
    """Return the record that a file written from a transformation holds.

    The record is that of the consolidated copy of the transformation.

    Raises
    ------
    UnrepresentableTransformationError
        If the format cannot encode the systems or the matrix.
    """
    return _record(xform.consolidate())
