
__all__ = [
    "LTATransform",
    "LTATransformVoxToVox",
    "LTATransformPhysToPhys",
    "LTATransformRASToRAS"
]

# stdlib
from functools import partial

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


# ----------------------------------------------------------------------
#   COORDINATE SYSTEMS
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
#   TRANSFORMATIONS
# ----------------------------------------------------------------------


class LTATransform(_xforms.Affine, reversed=False):

    struct: LTAStruct = Factory(LTAStruct)

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
    def from_struct(cls, struct: LTAStruct) -> _tx.Self:
        return cls(struct=struct)


class LTATransformVoxToVox(LTATransform):

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_VOX_TO_VOX,
        src=LTAStruct.SrcVolumeInfo(),
        dst=LTAStruct.DstVolumeInfo()
    ))

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



class LTATransformPhysToPhys(LTATransform):

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_PHYSVOX_TO_PHYSVOX,
        src=LTAStruct.SrcVolumeInfo(),
        dst=LTAStruct.DstVolumeInfo()
    ))

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


class LTATransformRASToRAS(LTATransform):

    struct: LTAStruct = Factory(partial(
        LTAStruct,
        type=LTAType.LINEAR_RAS_TO_RAS,
    ))

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
