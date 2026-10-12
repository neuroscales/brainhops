import numpy as np

from brainhops._core.enum import enum_name

# internals
from brainhops.io.common.freesurfer._geometry import (
    code2orient,
    fs_phys2ras,
    fs_vox2phys,
    mat2code,
    mat2orient,
)

from ._enums import LtaType
from ._raw import LtaRaw


def _get_vox2phys(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_vox2phys(vol_info.volume, vol_info.voxelsize)


def _get_phys2ras(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_phys2ras(
        vol_info.xras, vol_info.yras, vol_info.zras, vol_info.cras
    )


def _get_vox2ras(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return _get_phys2ras(vol_info) @ _get_vox2phys(vol_info)


# RSA lists the RAS axes in the order R, S, A (see `RSAmm`), so swapping
# the last two spatial axes converts either way: the permutation is its own
# inverse.
_RAS_RSA = [0, 2, 1, 3]


def _rsa2ras(matrix: np.ndarray) -> np.ndarray:
    """Express an RSA-to-RSA matrix as a RAS-to-RAS matrix."""
    return matrix[_RAS_RSA, :][:, _RAS_RSA]


def _get_ras2ras(lta: LtaRaw) -> np.ndarray:
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LtaType.LINEAR_RAS_TO_RAS:
        return matrix
    if lta.type == LtaType.LINEAR_RSA_TO_RSA:
        return _rsa2ras(matrix)
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute RAS-to-RAS matrix without src and dst volume info"
        )
    if lta.type == LtaType.LINEAR_VOX_TO_VOX:
        src_vox2ras = _get_vox2ras(lta.src)
        dst_vox2ras = _get_vox2ras(lta.dst)
        return dst_vox2ras @ matrix @ np.linalg.inv(src_vox2ras)
    if lta.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
        src_phys2ras = _get_phys2ras(lta.src)
        dst_phys2ras = _get_phys2ras(lta.dst)
        return dst_phys2ras @ matrix @ np.linalg.inv(src_phys2ras)
    raise AssertionError(f"unsupported LTA type: {enum_name(lta.type)}")


def _get_phys2phys(lta: LtaRaw) -> np.ndarray:
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
        return matrix
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute phys-to-phys matrix without "
            "src and dst volume info"
        )
    if lta.type == LtaType.LINEAR_VOX_TO_VOX:
        src_vox2phys = _get_vox2phys(lta.src)
        dst_vox2phys = _get_vox2phys(lta.dst)
        return dst_vox2phys @ matrix @ np.linalg.inv(src_vox2phys)
    elif lta.type in (LtaType.LINEAR_RAS_TO_RAS, LtaType.LINEAR_RSA_TO_RSA):
        src_phys2ras = _get_phys2ras(lta.src)
        dst_phys2ras = _get_phys2ras(lta.dst)
        ras2ras = _get_ras2ras(lta)
        return np.linalg.inv(dst_phys2ras) @ ras2ras @ src_phys2ras
    raise AssertionError(f"unsupported LTA type: {enum_name(lta.type)}")


def _get_vox2vox(lta: LtaRaw) -> np.ndarray:
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LtaType.LINEAR_VOX_TO_VOX:
        return matrix
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute vox-to-vox matrix without src and dst volume info"
        )
    if lta.type == LtaType.LINEAR_PHYSVOX_TO_PHYSVOX:
        src_vox2phys = _get_vox2phys(lta.src)
        dst_vox2phys = _get_vox2phys(lta.dst)
        return np.linalg.inv(dst_vox2phys) @ matrix @ src_vox2phys
    if lta.type in (LtaType.LINEAR_RAS_TO_RAS, LtaType.LINEAR_RSA_TO_RSA):
        src_vox2ras = _get_vox2ras(lta.src)
        dst_vox2ras = _get_vox2ras(lta.dst)
        ras2ras = _get_ras2ras(lta)
        return np.linalg.inv(dst_vox2ras) @ ras2ras @ src_vox2ras
    raise AssertionError(f"unsupported LTA type: {enum_name(lta.type)}")


# Orientation helpers shared with the other FreeSurfer formats.
_mat2code = mat2code
_code2orient = code2orient
_mat2orient = mat2orient


def _get_orient(vol_info: LtaRaw.VolumeInfo) -> str:
    vox2ras = _get_vox2ras(vol_info)
    return _mat2orient(vox2ras)
