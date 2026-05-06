__all__ = [
    "LTATransformation",
    "LTATransformationVoxToVox",
    "LTATransformationPhysToPhys",
    "LTATransformationRASToRAS"
]

# stdlib
from functools import partial
from os import PathLike

# externals
import typing_extensions as _tx
import numpy as np

# _ext
from brainhops._ext.struct import Factory

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

# local
from ._enums import LTAType
from ._struct import LTAStruct
from ._systems import LTACoordinateSystem, LTAVoxelSystem, LTAPhysicalSystem
from ._matrix_utils import _get_phys2phys, _get_vox2vox


_FileLike = _tx.Union[_tx.IO, PathLike, str]
_LTALike = _tx.Union[LTAStruct, _FileLike, bytes, _tx.Iterable[str]]


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
        assert False, f"unsupported LTA type: {self.struct.type}"

    @property
    def output(self) -> LTACoordinateSystem:
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
        assert False, f"unsupported LTA type: {self.struct.type}"

    @property
    def matrix(self) -> np.ndarray:
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return np.asarray(self.struct.affine.matrix, dtype=np.float64)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem):
        setattr(self, "_input", value)

    @output.setter
    def output(self, value: LTACoordinateSystem):
        setattr(self, "_output", value)

    @matrix.setter
    def matrix(self, value: np.ndarray):
        setattr(self, "_matrix", value)

    @classmethod
    def from_(cls, other: _LTALike) -> _tx.Self:
        if isinstance(other, LTAStruct):
            return cls.from_struct(other)
        return cls.from_struct(LTAStruct.from_(other))

    @classmethod
    def from_struct(cls, struct: LTAStruct) -> _tx.Self:
        return cls(struct=struct)

    @classmethod
    def from_file(cls, file: _FileLike) -> _tx.Self:
        return cls.from_struct(LTAStruct.from_file(file))

    @classmethod
    def from_text(cls, text: str) -> _tx.Self:
        return cls.from_struct(LTAStruct.from_text(text))

    @classmethod
    def from_bytes(cls, data: bytes) -> _tx.Self:
        return cls.from_struct(LTAStruct.from_bytes(data))

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:
        return cls.from_struct(LTAStruct.from_lines(lines))

    @classmethod
    def sniff(cls, other: _LTALike) -> bool:
        if isinstance(other, LTAStruct):
            return True
        return LTAStruct.sniff(other)

    @classmethod
    def sniff_file(cls, file: _FileLike) -> bool:
        return LTAStruct.sniff_file(file)

    @classmethod
    def sniff_bytes(cls, data: bytes) -> bool:
        return LTAStruct.sniff_bytes(data)

    @classmethod
    def sniff_text(cls, text: str) -> bool:
        return LTAStruct.sniff_text(text)

    @classmethod
    def sniff_line(cls, line: str) -> bool:
        return LTAStruct.sniff_line(line)


class LTATransformationVoxToVox(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a voxel-to-voxel
    affine transformation.
    """

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_VOX_TO_VOX,
        src=LTAStruct.SrcVolumeInfo(),
        dst=LTAStruct.DstVolumeInfo()
    ), repr=False)

    @property
    def input(self) -> LTACoordinateSystem:
        if getattr(self, "_input", None) is not None:
            return self._input
        return LTAVoxelSystem.from_struct(self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        if getattr(self, "_output", None) is not None:
            return self._output
        return LTAVoxelSystem.from_struct(self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_vox2vox(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem):
        setattr(self, "_input", value)

    @output.setter
    def output(self, value: LTACoordinateSystem):
        setattr(self, "_output", value)

    @matrix.setter
    def matrix(self, value: np.ndarray):
        setattr(self, "_matrix", value)


class LTATransformationPhysToPhys(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a physical-to-physical
    affine transformation.
    """

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_PHYSVOX_TO_PHYSVOX,
        src=LTAStruct.SrcVolumeInfo(),
        dst=LTAStruct.DstVolumeInfo()
    ), repr=False)

    @property
    def input(self) -> LTACoordinateSystem:
        if getattr(self, "_input", None) is not None:
            return self._input
        return LTAPhysicalSystem.from_struct(self.struct.src)

    @property
    def output(self) -> LTACoordinateSystem:
        if getattr(self, "_output", None) is not None:
            return self._output
        return LTAPhysicalSystem.from_struct(self.struct.dst)

    @property
    def matrix(self) -> np.ndarray:
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem):
        setattr(self, "_input", value)

    @output.setter
    def output(self, value: LTACoordinateSystem):
        setattr(self, "_output", value)

    @matrix.setter
    def matrix(self, value: np.ndarray):
        setattr(self, "_matrix", value)


class LTATransformationRASToRAS(LTATransformation):
    """
    A Linear Transform Array (LTA) file interpreted as a RAS-to-RAS
    affine transformation.
    """

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_RAS_TO_RAS,
    ), repr=False)

    @property
    def input(self) -> LTACoordinateSystem:
        if getattr(self, "_input", None) is not None:
            return self._input
        return _systems.RASCoordinateSystem()

    @property
    def output(self) -> LTACoordinateSystem:
        if getattr(self, "_output", None) is not None:
            return self._output
        return _systems.RASCoordinateSystem()

    @property
    def matrix(self) -> np.ndarray:
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        return _get_phys2phys(self.struct)[:-1]

    @input.setter
    def input(self, value: LTACoordinateSystem):
        setattr(self, "_input", value)

    @output.setter
    def output(self, value: LTACoordinateSystem):
        setattr(self, "_output", value)

    @matrix.setter
    def matrix(self, value: np.ndarray):
        setattr(self, "_matrix", value)
