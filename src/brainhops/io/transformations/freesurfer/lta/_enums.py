__all__ = ["LTAType", "LTAMatrixType", "LTAValidity"]

# stdlib
from enum import Enum


class LTAType(int, Enum):
    # Affine transformation types
    LINEAR_VOX_TO_VOX = 0
    LINEAR_VOXEL_TO_VOXEL = LINEAR_VOX_TO_VOX
    LINEAR_RAS_TO_RAS = 1
    LINEAR_PHYSVOX_TO_PHYSVOX = 2
    LINEAR_CORONAL_RAS_TO_CORONAL_RAS = 21
    LINEAR_COR_TO_COR = LINEAR_CORONAL_RAS_TO_CORONAL_RAS
    LINEAR_RSA_TO_RSA = LINEAR_COR_TO_COR
    # Transformation file types (invalid in a LTA file)
    TRANSFORM_ARRAY_TYPE = 10
    MORPH_3D_TYPE = 11
    MNI_TRANSFORM_TYPE = 12
    MATLAB_ASCII_TYPE = 13


class LTAMatrixType(int, Enum):
    UNKNOWN_MATRIX = 0
    REAL_MATRIX = 1
    COMPLEX_MATRIX = 2


class LTAValidity(int, Enum):
    VOLUME_INFO_INVALID = 0
    VOLUME_INFO_VALID = 1
