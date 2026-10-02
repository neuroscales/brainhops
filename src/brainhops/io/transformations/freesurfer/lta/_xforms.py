__all__ = [
    "LtaFormat",
    "LtaTransformation",
    "LtaTransformationVoxToVox",
    "LtaTransformationPhysToPhys",
    "LtaTransformationRasToRas",
]

# stdlib
from functools import partial
from warnings import catch_warnings, simplefilter, warn

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory, fields

# internals
from brainhops._core import path
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    TextFileParserWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    WritableFileBasedTransformation,
)

# local
from .._formats import FreesurferTransformationFormat
from ._enums import LtaType, LtaValidity
from ._matrix_utils import _get_phys2phys, _get_ras2ras, _get_vox2vox
from ._struct import LtaStruct
from ._systems import LtaCoordinateSystem, LtaPhysicalSystem, LtaVoxelSystem


def _system(
    cls: tx.Type[LtaCoordinateSystem],
    info: tx.Optional[LtaStruct.VolumeInfo],
) -> tx.Optional[LtaCoordinateSystem]:
    """The system of a volume, or `None` if the struct records no
    geometry for it: without one, the system is unknown."""
    return None if info is None else cls.from_struct(info)


class LtaFormat(FreesurferTransformationFormat, AffineTransformationFormat):
    """An affine transformation stored in a FreeSurfer LTA file."""

    HINTS = ("lta",)


# `WritableFileBasedTransformation` writes through `FileParserWriter`, which
# knows no encoding: its `sniff_bytes`, `from_bytes` and `to_bytes` raise
# `NotImplementedError`. `TextFileParserWriter` is what bridges bytes to
# text for a text format (as `TextFileParser` does for FLIRT), so it is
# needed as well, and must come first to take precedence.
@register_format
class LtaTransformation(
    LtaFormat,
    TextFileParserWriter,
    _xforms.Affine,
    WritableFileBasedTransformation,
    reverse=False,  # We want `struct` to be the last field.
):
    """
    A transformation than can be encoded as a Linear Transform Array (LTA).

    LTA files are the default format used for linear transformations in
    Freesurfer, and can represent different types of Affine transformations.

    This is the registered format for `.lta` files: `io.load`,
    `io.transformations.load` and `from_other` read them, and `io.save`
    writes them. The views below it (`LtaTransformationVoxToVox`,
    `LtaTransformationPhysToPhys`, `LtaTransformationRasToRas`) read the
    same files, but are not registered: they would claim every `.lta`
    file exactly as well as this class does.

    !!! note "What is written"
        A transformation read from a file, and left untouched, is written
        back as the struct it was read from. One whose `matrix`, `input`
        or `output` has been set -- including one converted from another
        affine -- is written with the LTA type its coordinate systems
        call for:

        | `input` and `output`  | LTA type                    |
        | --------------------- | --------------------------- |
        | both `RASmm`          | `LINEAR_RAS_TO_RAS`         |
        | both `RSAmm`          | `LINEAR_RSA_TO_RSA`         |
        | both `LtaVoxelSystem` | `LINEAR_VOX_TO_VOX`         |
        | both `LtaPhysicalSystem` | `LINEAR_PHYSVOX_TO_PHYSVOX` |

        Any other pair of systems has no LTA encoding, and writing it
        raises
        [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)

    struct: LtaStruct = Factory(LtaStruct, repr=False)

    @property
    def input(self) -> LtaCoordinateSystem:
        """The transformation's input coordinate system.

        Derived from the struct's type and its source volume geometry,
        unless it has been set explicitly.
        """
        if getattr(self, "_input", None) is not None:
            return self._input
        if self.struct.type == LtaType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LtaType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LtaType.LINEAR_VOX_TO_VOX:
            return _system(LtaVoxelSystem, self.struct.src)
        elif self.struct.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LtaPhysicalSystem, self.struct.src)
        raise AssertionError(f"unsupported LTA type: {self.struct.type}")

    @property
    def output(self) -> LtaCoordinateSystem:
        """The transformation's output coordinate system.

        Derived from the struct's type and its destination volume
        geometry, unless it has been set explicitly.
        """
        if getattr(self, "_output", None) is not None:
            return self._output
        if self.struct.type == LtaType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LtaType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LtaType.LINEAR_VOX_TO_VOX:
            return _system(LtaVoxelSystem, self.struct.dst)
        elif self.struct.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LtaPhysicalSystem, self.struct.dst)
        raise AssertionError(f"unsupported LTA type: {self.struct.type}")

    @property
    def matrix(self) -> np.ndarray:
        """The transformation's affine matrix.

        Read from the struct's affine block, unless it has been set
        explicitly.
        """
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return np.asarray(self.struct.affine.matrix, dtype=np.float64)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how likely a line is to be the first line of an LTA
        file. See `LtaStruct.sniff_line`."""
        return LtaStruct.sniff_line(line, error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_(cls, other: tx.Any) -> tx.Self:
        """Build the transformation from a struct, a file, or file
        content.

        !!! warning "Deprecated"
            Use `from_struct` for a struct, `load` for a file, a file
            object or bytes, and `from_text` or `from_lines` for content
            held in memory.
        """
        if isinstance(other, LtaStruct):
            hint = "use from_struct()"
        else:
            hint = (
                "use load() for a file, a file object or bytes, and "
                "from_text() or from_lines() for content held in memory"
            )
        warn(
            f"{cls.__name__}.from_() is deprecated: {hint}.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(other, LtaStruct):
            return cls.from_struct(other)
        with catch_warnings():
            # Warned above, in terms of this class rather than the struct.
            simplefilter("ignore", DeprecationWarning)
            return cls.from_struct(LtaStruct.from_(other))

    @classmethod
    def from_struct(cls, struct: LtaStruct, **kwargs) -> tx.Self:
        """Build the transformation from an already-parsed [`LtaStruct`][].

        Keyword arguments are passed to the constructor, and override
        what the struct says (`input`, `output`, `matrix`).
        """
        return cls(struct=struct, **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build the transformation from an iterable over lines of an
        LTA file.

        Keyword arguments are passed to the constructor, and override
        what the file says (`input`, `output`, `matrix`).
        """
        return cls.from_struct(LtaStruct.from_lines(lines), **kwargs)

    # --- to -----------------------------------------------------------

    def to_struct(self) -> LtaStruct:
        """
        The [`LtaStruct`][] that encodes this transformation.

        A transformation whose `matrix`, `input` and `output` are all
        derived from its struct is encoded by that struct, unchanged.
        Otherwise a struct is built from them: its type is the one the
        coordinate systems call for (see the class documentation), its
        matrix is `matrix`, and its volume geometries are those the
        systems were built from (voxel and physical types) or those of
        the current struct (RAS and RSA types). The other header fields
        are kept from the current struct.

        Raises
        ------
        UnrepresentableTransformationError
            If LTA cannot encode the coordinate systems or the matrix.
        """
        if all(
            getattr(self, name, None) is None
            for name in ("_matrix", "_input", "_output")
        ):
            return self.struct
        return _build_struct(self)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the transformation to a file (path or file-like object)
        in LTA format.

        The struct is built before the file is opened, so a
        transformation that LTA cannot encode is refused without
        creating or truncating the file.
        """
        return self.to_struct().to_file(file, **kwargs)

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """The lines of the LTA file that encodes this transformation."""
        return self.to_struct().to_lines(**kwargs)

    def to_text(self, **kwargs) -> str:
        """The content of the LTA file that encodes this transformation."""
        return self.to_struct().to_text(**kwargs)


class LtaTransformationVoxToVox(LtaTransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a voxel-to-voxel
    affine transformation.
    """

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_VOX_TO_VOX,
            src=LtaStruct.SrcVolumeInfo(),
            dst=LtaStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The voxel system of the struct's source volume, unless it has
        been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LtaVoxelSystem, self.struct.src)

    @property
    def output(self) -> LtaCoordinateSystem:
        """The voxel system of the struct's destination volume, unless it
        has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LtaVoxelSystem, self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        """The voxel-to-voxel affine matrix derived from the struct,
        unless it has been set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_vox2vox(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


class LtaTransformationPhysToPhys(LtaTransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a physical-to-physical
    affine transformation.
    """

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_PHYSVOX_TO_PHYSVOX,
            src=LtaStruct.SrcVolumeInfo(),
            dst=LtaStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The physical system of the struct's source volume, unless it
        has been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LtaPhysicalSystem, self.struct.src)

    @property
    def output(self) -> LtaCoordinateSystem:
        """The physical system of the struct's destination volume,
        unless it has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LtaPhysicalSystem, self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        """The physical-to-physical affine matrix derived from the
        struct, unless it has been set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


class LtaTransformationRasToRas(LtaTransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a RAS-to-RAS
    affine transformation.
    """

    struct: LtaStruct = Factory(
        partial(
            LtaStruct,
            type=LtaType.LINEAR_RAS_TO_RAS,
        ),
        repr=False,
    )

    @property
    def input(self) -> LtaCoordinateSystem:
        """The RAS coordinate system, unless it has been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _systems.RASmm()

    @property
    def output(self) -> LtaCoordinateSystem:
        """The RAS coordinate system, unless it has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _systems.RASmm()

    @property
    def matrix(self) -> np.ndarray:
        """The RAS-to-RAS affine matrix derived from the struct, unless
        it has been set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_ras2ras(self.struct)[:-1]

    @input.setter
    def input(self, value: LtaCoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LtaCoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


# ----------------------------------------------------------------------
#   Encoding an affine as an LTA struct
# ----------------------------------------------------------------------


def _as_block(
    info: tx.Optional[LtaStruct.VolumeInfo], cls: type
) -> LtaStruct.VolumeInfo:
    """A volume geometry, as a block of type `cls` (source or
    destination).

    No geometry gives an invalid block, which is what FreeSurfer writes
    when the geometry is unknown.
    """
    if info is None:
        return cls(valid=LtaValidity.VOLUME_INFO_INVALID)
    if type(info) is cls:
        return info
    return cls(**{f.name: getattr(info, f.name) for f in fields(cls)})


def _build_struct(xform: LtaTransformation) -> LtaStruct:
    """Encode the matrix and coordinate systems of `xform` as a struct.

    See `LtaTransformation.to_struct`.
    """
    src_system, dst_system = xform.input, xform.output
    old = xform.struct
    Src, Dst = LtaStruct.SrcVolumeInfo, LtaStruct.DstVolumeInfo

    def both(cls: type) -> bool:
        return isinstance(src_system, cls) and isinstance(dst_system, cls)

    if both(_systems.RASmm) or both(_systems.RSAmm):
        # The geometry is not part of what a RAS-to-RAS affine means, but
        # FreeSurfer tools use it, so the one already there is kept.
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
        src = _as_block(getattr(src_system, "struct", None), Src)
        dst = _as_block(getattr(dst_system, "struct", None), Dst)
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

    matrix = xform.matrix
    if matrix is None:
        homogeneous = np.eye(4)
    else:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 4):
            raise UnrepresentableTransformationError(
                f"An LTA file encodes a 3D affine, of shape (3, 4), but "
                f"the matrix has shape {matrix.shape}."
            )
        homogeneous = np.concatenate([matrix, [[0.0, 0.0, 0.0, 1.0]]])

    return LtaStruct(
        type=lta_type,
        nxforms=old.nxforms,
        mean=old.mean,
        sigma=old.sigma,
        affine=LtaStruct.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in homogeneous)
        ),
        label=old.label,
        src=src,
        dst=dst,
    )
