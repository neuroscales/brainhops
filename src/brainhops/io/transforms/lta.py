
import typing_extensions as _tx
from enum import Enum

from brainhops._ext.struct import Struct
from brainhops.datamodel.axes import SpatialAxis
from brainhops.datamodel.transformations import Affine
from brainhops.datamodel.systems import Spatial3dCoordinateSystem


class LTAType(int, Enum):
    # Affine transformation types
    LINEAR_VOX_TO_VOX = 0
    LINEAR_VOXEL_TO_VOXEL = LINEAR_VOX_TO_VOX
    LINEAR_RAS_TO_RAS = 1
    LINEAR_PHYSVOX_TO_PHYSVOX = 2
    LINEAR_CORONAL_RAS_TO_CORONAL_RAS = 21
    LINEAR_COR_TO_COR = LINEAR_CORONAL_RAS_TO_CORONAL_RAS
    # Transformation file types
    TRANSFORM_ARRAY_TYPE = 10
    MORPH_3D_TYPE = 11
    MNI_TRANSFORM_TYPE = 12
    MATLAB_ASCII_TYPE = 13


_3Ints = _tx.Tuple[int, int, int]
_3Floats = _tx.Tuple[float, float, float]
_4Floats = _tx.Tuple[float, float, float, float]
_3x4Floats = _tx.Tuple[_4Floats, _4Floats, _4Floats]


class LTAStruct(Struct):

    class VolumeInfo(Struct):
        valid: int = 1                              #
        filename: _tx.Optional[str] = None          # Filename of the volume
        volume: _tx.Optional[_3Ints] = None         # 3D shape
        voxelsize: _tx.Optional[_3Floats] = None    # Voxel size
        xras: _tx.Optional[_3Floats] = None         # Columns of the xform
        yras: _tx.Optional[_3Floats] = None         # "
        zras: _tx.Optional[_3Floats] = None         # "
        cras: _tx.Optional[_3Floats] = None         # "

    type: LTAType
    nxforms: int = 1
    mean: _tx.Optional[_3Floats] = None
    sigma: _tx.Optional[float] = None
    affine: _tx.Optional[_3x4Floats] = None
    src: _tx.Optional[VolumeInfo] = None            # Source volume
    dst: _tx.Optional[VolumeInfo] = None            # Destination volume


class LTACoordinateSystem(Spatial3dCoordinateSystem):
    name: _tx.Optional[str] = None
    axes: _tx.List[SpatialAxis] = tuple()
    struct: LTAStruct.VolumeInfo = LTAStruct.VolumeInfo()


class LTAVoxelSystem(LTACoordinateSystem):
    """Voxel space of a volume (source or destination)."""

    @classmethod
    def from_struct(cls, struct: LTAStruct.VolumeInfo) -> "LTAVoxelSystem":
        return cls(
            name=struct.filename,
            axes=[SpatialAxis(name=name, unit="mm") for name in ("x", "y", "z")],
            struct=struct
        )


class LTAScaledSystem(LTACoordinateSystem):
    ...


class LTAPhysicalSystem(LTACoordinateSystem):
    ...


class LTATransform(Affine):

    struct: LTAStruct = LTAStruct()


class LTATransformVoxToVox(LTATransform):
    ...


class LTATransformPhysToPhys(LTATransform):
    ...


class LTATransformRASToRAS(LTATransform):
    ...