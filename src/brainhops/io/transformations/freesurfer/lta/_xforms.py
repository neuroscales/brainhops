__all__ = [
    "LTAFormat",
    "LTATransformation",
    "LTATransformationVoxToVox",
    "LTATransformationPhysToPhys",
    "LTATransformationRASToRAS",
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
from ._enums import LTAType, LTAValidity
from ._matrix_utils import _get_phys2phys, _get_vox2vox
from ._struct import LTAStruct
from ._systems import LTACoordinateSystem, LTAPhysicalSystem, LTAVoxelSystem


def _system(
    cls: tx.Type[LTACoordinateSystem],
    info: tx.Optional[LTAStruct.VolumeInfo],
) -> tx.Optional[LTACoordinateSystem]:
    """The system of a volume, or `None` if the struct records no
    geometry for it: without one, the system is unknown."""
    return None if info is None else cls.from_struct(info)


class LTAFormat(AffineTransformationFormat):
    """An affine transformation stored in a FreeSurfer LTA file."""

    HINTS = ("lta", "freesurfer")


@register_format
class LTATransformation(
    LTAFormat,
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
    writes them. The views below it (`LTATransformationVoxToVox`,
    `LTATransformationPhysToPhys`, `LTATransformationRASToRAS`) read the
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
        | both `LTAVoxelSystem` | `LINEAR_VOX_TO_VOX`         |
        | both `LTAPhysicalSystem` | `LINEAR_PHYSVOX_TO_PHYSVOX` |

        Any other pair of systems has no LTA encoding, and writing it
        raises
        [`UnrepresentableTransformationError`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".lta",)

    struct: LTAStruct = Factory(LTAStruct, repr=False)

    @property
    def input(self) -> LTACoordinateSystem:
        """The transformation's input coordinate system.

        Derived from the struct's type and its source volume geometry,
        unless it has been set explicitly.
        """
        if getattr(self, "_input", None) is not None:
            return self._input
        if self.struct.type == LTAType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LTAType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LTAType.LINEAR_VOX_TO_VOX:
            return _system(LTAVoxelSystem, self.struct.src)
        elif self.struct.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LTAPhysicalSystem, self.struct.src)
        raise AssertionError(f"unsupported LTA type: {self.struct.type}")

    @property
    def output(self) -> LTACoordinateSystem:
        """The transformation's output coordinate system.

        Derived from the struct's type and its destination volume
        geometry, unless it has been set explicitly.
        """
        if getattr(self, "_output", None) is not None:
            return self._output
        if self.struct.type == LTAType.LINEAR_RAS_TO_RAS:
            return _systems.RASmm()
        elif self.struct.type == LTAType.LINEAR_RSA_TO_RSA:
            return _systems.RSAmm()
        elif self.struct.type == LTAType.LINEAR_VOX_TO_VOX:
            return _system(LTAVoxelSystem, self.struct.dst)
        elif self.struct.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return _system(LTAPhysicalSystem, self.struct.dst)
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
    def input(self, value: LTACoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LTACoordinateSystem) -> None:
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
        file. See `LTAStruct.sniff_line`."""
        return LTAStruct.sniff_line(line, error=error, **kwargs)

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
        if isinstance(other, LTAStruct):
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
        if isinstance(other, LTAStruct):
            return cls.from_struct(other)
        with catch_warnings():
            # Warned above, in terms of this class rather than the struct.
            simplefilter("ignore", DeprecationWarning)
            return cls.from_struct(LTAStruct.from_(other))

    @classmethod
    def from_struct(cls, struct: LTAStruct, **kwargs) -> tx.Self:
        """Build the transformation from an already-parsed [`LTAStruct`][].

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
        return cls.from_struct(LTAStruct.from_lines(lines), **kwargs)

    # --- to -----------------------------------------------------------

    def to_struct(self) -> LTAStruct:
        """
        The [`LTAStruct`][] that encodes this transformation.

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


class LTATransformationVoxToVox(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a voxel-to-voxel
    affine transformation.
    """

    struct: LTAStruct = Factory(
        partial(
            LTAStruct,
            type=LTAType.LINEAR_VOX_TO_VOX,
            src=LTAStruct.SrcVolumeInfo(),
            dst=LTAStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LTACoordinateSystem:
        """The voxel system of the struct's source volume, unless it has
        been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LTAVoxelSystem, self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        """The voxel system of the struct's destination volume, unless it
        has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LTAVoxelSystem, self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        """The voxel-to-voxel affine matrix derived from the struct,
        unless it has been set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_vox2vox(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LTACoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


class LTATransformationPhysToPhys(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a physical-to-physical
    affine transformation.
    """

    struct: LTAStruct = Factory(
        partial(
            LTAStruct,
            type=LTAType.LINEAR_PHYSVOX_TO_PHYSVOX,
            src=LTAStruct.SrcVolumeInfo(),
            dst=LTAStruct.DstVolumeInfo(),
        ),
        repr=False,
    )

    @property
    def input(self) -> LTACoordinateSystem:
        """The physical system of the struct's source volume, unless it
        has been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _system(LTAPhysicalSystem, self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        """The physical system of the struct's destination volume,
        unless it has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _system(LTAPhysicalSystem, self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        """The physical-to-physical affine matrix derived from the
        struct, unless it has been set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LTACoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


class LTATransformationRASToRAS(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a RAS-to-RAS
    affine transformation.
    """

    struct: LTAStruct = Factory(
        partial(
            LTAStruct,
            type=LTAType.LINEAR_RAS_TO_RAS,
        ),
        repr=False,
    )

    @property
    def input(self) -> LTACoordinateSystem:
        """The RAS coordinate system, unless it has been set explicitly."""
        if getattr(self, "_input", None) is not None:
            return self._input
        return _systems.RASmm()

    @property
    def output(self) -> LTACoordinateSystem:
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
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem) -> None:
        self._input = value

    @output.setter
    def output(self, value: LTACoordinateSystem) -> None:
        self._output = value

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value


# ----------------------------------------------------------------------
#   Encoding an affine as an LTA struct
# ----------------------------------------------------------------------


def _as_block(
    info: tx.Optional[LTAStruct.VolumeInfo], cls: type
) -> LTAStruct.VolumeInfo:
    """A volume geometry, as a block of type `cls` (source or
    destination).

    No geometry gives an invalid block, which is what FreeSurfer writes
    when the geometry is unknown.
    """
    if info is None:
        return cls(valid=LTAValidity.VOLUME_INFO_INVALID)
    if type(info) is cls:
        return info
    return cls(**{f.name: getattr(info, f.name) for f in fields(cls)})


def _build_struct(xform: LTATransformation) -> LTAStruct:
    """Encode the matrix and coordinate systems of `xform` as a struct.

    See `LTATransformation.to_struct`.
    """
    src_system, dst_system = xform.input, xform.output
    old = xform.struct
    Src, Dst = LTAStruct.SrcVolumeInfo, LTAStruct.DstVolumeInfo

    def both(cls: type) -> bool:
        return isinstance(src_system, cls) and isinstance(dst_system, cls)

    if both(_systems.RASmm) or both(_systems.RSAmm):
        # The geometry is not part of what a RAS-to-RAS affine means, but
        # FreeSurfer tools use it, so the one already there is kept.
        lta_type = (
            LTAType.LINEAR_RAS_TO_RAS
            if both(_systems.RASmm)
            else LTAType.LINEAR_RSA_TO_RSA
        )
        src, dst = _as_block(old.src, Src), _as_block(old.dst, Dst)
    elif both(LTAVoxelSystem) or both(LTAPhysicalSystem):
        lta_type = (
            LTAType.LINEAR_VOX_TO_VOX
            if both(LTAVoxelSystem)
            else LTAType.LINEAR_PHYSVOX_TO_PHYSVOX
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
            f"RSAmm, LTAVoxelSystem to LTAVoxelSystem, and "
            f"LTAPhysicalSystem to LTAPhysicalSystem. Set the input and "
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

    return LTAStruct(
        type=lta_type,
        nxforms=old.nxforms,
        mean=old.mean,
        sigma=old.sigma,
        affine=LTAStruct.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in homogeneous)
        ),
        label=old.label,
        src=src,
        dst=dst,
    )
