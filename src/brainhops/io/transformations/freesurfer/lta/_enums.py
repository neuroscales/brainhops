__all__ = ["LtaType", "LtaMatrixType", "LtaValidity"]

from enum import Enum


# ruff: disable[E501]
# fmt: off
class LtaType(int, Enum):
    """Type of the affine transformation in an LTA header.

    The type identifies the coordinate systems that the affine maps between,
    for example voxel to voxel or RAS to RAS. Some values are file types
    rather than affine types, and those values are invalid in an LTA file.
    The types that the enum distinguishes, and their values, are:

    | Name                                | Aliases                                  | Value | Description                        |
    |-------------------------------------|------------------------------------------|-------|------------------------------------|
    | `LINEAR_VOX_TO_VOX`                 | `LINEAR_VOXEL_TO_VOXEL`                  | 0     | Affine type.                       |
    | `LINEAR_RAS_TO_RAS`                 |                                          | 1     | Affine type.                       |
    | `LINEAR_PHYSVOX_TO_PHYSVOX`         |                                          | 2     | Affine type.                       |
    | `LINEAR_CORONAL_RAS_TO_CORONAL_RAS` | `LINEAR_COR_TO_COR`, `LINEAR_RSA_TO_RSA` | 21    | Affine type.                       |
    | `TRANSFORM_ARRAY_TYPE`              |                                          | 10    | File type, invalid in an LTA file. |
    | `MORPH_3D_TYPE`                     |                                          | 11    | File type, invalid in an LTA file. |
    | `MNI_TRANSFORM_TYPE`                |                                          | 12    | File type, invalid in an LTA file. |
    | `MATLAB_ASCII_TYPE`                 |                                          | 13    | File type, invalid in an LTA file. |
    """

    # These values are the types of affine transformations.
    LINEAR_VOX_TO_VOX = 0
    LINEAR_VOXEL_TO_VOXEL = LINEAR_VOX_TO_VOX
    LINEAR_RAS_TO_RAS = 1
    LINEAR_PHYSVOX_TO_PHYSVOX = 2
    LINEAR_CORONAL_RAS_TO_CORONAL_RAS = 21
    LINEAR_COR_TO_COR = LINEAR_CORONAL_RAS_TO_CORONAL_RAS
    LINEAR_RSA_TO_RSA = LINEAR_COR_TO_COR
    # These values are file types, which are invalid in an LTA file.
    TRANSFORM_ARRAY_TYPE = 10
    MORPH_3D_TYPE = 11
    MNI_TRANSFORM_TYPE = 12
    MATLAB_ASCII_TYPE = 13
# fmt: on
# ruff: enable[E501]


class LtaMatrixType(int, Enum):
    """Element type of a matrix parsed from an LTA file.

    The element types that the enum distinguishes, and their values, are:

    | Name             | Value |
    |------------------|-------|
    | `UNKNOWN_MATRIX` | 0     |
    | `REAL_MATRIX`    | 1     |
    | `COMPLEX_MATRIX` | 2     |
    """

    UNKNOWN_MATRIX = 0
    REAL_MATRIX = 1
    COMPLEX_MATRIX = 2


class LtaValidity(int, Enum):
    """Flag that records whether a volume-geometry block is populated.

    The two states of the flag, and their values, are:

    | Name                  | Value | Description                 |
    |-----------------------|-------|-----------------------------|
    | `VOLUME_INFO_INVALID` | 0     | The block is not populated. |
    | `VOLUME_INFO_VALID`   | 1     | The block is populated.     |
    """

    VOLUME_INFO_INVALID = 0
    VOLUME_INFO_VALID = 1
