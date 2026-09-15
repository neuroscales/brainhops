__all__ = [
    "LTATransformation",
    "LTATransformationVoxToVox",
    "LTATransformationPhysToPhys",
    "LTATransformationRASToRAS",
]

# stdlib
from functools import partial
from os import PathLike

# externals
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

# local
from ._enums import LTAType
from ._matrix_utils import _get_phys2phys, _get_vox2vox
from ._struct import LTAStruct
from ._systems import LTACoordinateSystem, LTAPhysicalSystem, LTAVoxelSystem

_FileLike = tx.Union[tx.IO, PathLike, str]
_LTALike = tx.Union[LTAStruct, _FileLike, bytes, tx.Iterable[str]]


# @register('.lta')
class LTATransformation(
    _xforms.Affine,
    reverse=False,  # We want `struct` to be the last field.
):
    """
    A transformation than can be encoded as a Linear Transform Array (LTA).

    LTA files are the default format used for linear transformations in
    Freesurfer, and can represent different types of Affine transformations.
    """

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
            return _systems.RASCoordinateSystem()
        elif self.struct.type == LTAType.LINEAR_RSA_TO_RSA:
            return _systems.RSACoordinateSystem()
        elif self.struct.type == LTAType.LINEAR_VOX_TO_VOX:
            return LTAVoxelSystem.from_struct(self.struct.src)
        elif self.struct.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return LTAPhysicalSystem.from_struct(self.struct.src)
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
            return _systems.RASCoordinateSystem()
        elif self.struct.type == LTAType.LINEAR_RSA_TO_RSA:
            return _systems.RSACoordinateSystem()
        elif self.struct.type == LTAType.LINEAR_VOX_TO_VOX:
            return LTAVoxelSystem.from_struct(self.struct.dst)
        elif self.struct.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
            return LTAPhysicalSystem.from_struct(self.struct.dst)
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

    @classmethod
    def from_(cls, other: _LTALike) -> tx.Self:
        """Build the transformation from a struct, a file, or file
        content, in any form [`LTAStruct.from_`][] accepts."""
        if isinstance(other, LTAStruct):
            return cls.from_struct(other)
        return cls.from_struct(LTAStruct.from_(other))

    @classmethod
    def from_struct(cls, struct: LTAStruct) -> tx.Self:
        """Build the transformation from an already-parsed [`LTAStruct`][]."""
        return cls(struct=struct)

    @classmethod
    def from_file(cls, file: _FileLike) -> tx.Self:
        """Build the transformation from an LTA file (path or file-like
        object)."""
        return cls.from_struct(LTAStruct.from_file(file))

    @classmethod
    def from_text(cls, text: str) -> tx.Self:
        """Build the transformation from a string in LTA format."""
        return cls.from_struct(LTAStruct.from_text(text))

    @classmethod
    def from_bytes(cls, data: bytes) -> tx.Self:
        """Build the transformation from bytes in LTA format."""
        return cls.from_struct(LTAStruct.from_bytes(data))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str]) -> tx.Self:
        """Build the transformation from an iterable over lines of an
        LTA file."""
        return cls.from_struct(LTAStruct.from_lines(lines))

    @classmethod
    def sniff(cls, other: _LTALike) -> bool:
        """Return whether the content, in any supported form, looks like
        it is in LTA format."""
        if isinstance(other, LTAStruct):
            return True
        return LTAStruct.sniff(other)

    @classmethod
    def sniff_file(cls, file: _FileLike) -> bool:
        """Return whether a file (path or file-like object) looks like
        it is in LTA format."""
        return LTAStruct.sniff_file(file)

    @classmethod
    def sniff_bytes(cls, data: bytes) -> bool:
        """Return whether bytes look like they are in LTA format."""
        return LTAStruct.sniff_bytes(data)

    @classmethod
    def sniff_text(cls, text: str) -> bool:
        """Return whether a string looks like it is in LTA format."""
        return LTAStruct.sniff_text(text)

    @classmethod
    def sniff_line(cls, line: str) -> bool:
        """Return whether a single line looks like the first line of an
        LTA file."""
        return LTAStruct.sniff_line(line)


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
        return LTAVoxelSystem.from_struct(self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        """The voxel system of the struct's destination volume, unless it
        has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return LTAVoxelSystem.from_struct(self.struct.dst)

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
        return LTAPhysicalSystem.from_struct(self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        """The physical system of the struct's destination volume,
        unless it has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return LTAPhysicalSystem.from_struct(self.struct.dst)

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
        return _systems.RASCoordinateSystem()

    @property
    def output(self) -> LTACoordinateSystem:
        """The RAS coordinate system, unless it has been set explicitly."""
        if getattr(self, "_output", None) is not None:
            return self._output
        return _systems.RASCoordinateSystem()

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
