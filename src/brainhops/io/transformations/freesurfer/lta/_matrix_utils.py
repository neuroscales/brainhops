# externals
import numpy as np

# internals
from brainhops.io.base.freesurfer import (
    code2orient,
    fs_phys2ras,
    fs_vox2phys,
    mat2code,
    mat2orient,
)

# local
from ._enums import LtaType
from ._struct import LtaStruct


def _get_vox2phys(vol_info: LtaStruct.VolumeInfo) -> np.ndarray:
    """Compute the vox2phys matrix from the volume geometry."""
    return fs_vox2phys(vol_info.volume, vol_info.voxelsize)


def _get_phys2ras(vol_info: LtaStruct.VolumeInfo) -> np.ndarray:
    """Compute the phys2ras matrix from the volume geometry."""
    return fs_phys2ras(
        vol_info.xras, vol_info.yras, vol_info.zras, vol_info.cras
    )


def _get_vox2ras(vol_info: LtaStruct.VolumeInfo) -> np.ndarray:
    """Compute the vox2ras matrix from the volume geometry."""
    return _get_phys2ras(vol_info) @ _get_vox2phys(vol_info)


# RSA lists the axes of RAS in the order R, S, A (see `RSAmm`), so
# swapping the last two spatial axes maps either one to the other. The
# permutation is its own inverse.
_RAS_RSA = [0, 2, 1, 3]


def _rsa2ras(matrix: np.ndarray) -> np.ndarray:
    """Express an RSA-to-RSA matrix as a RAS-to-RAS one."""
    return matrix[_RAS_RSA, :][:, _RAS_RSA]


def _get_ras2ras(lta: LtaStruct) -> np.ndarray:
    """Compute the ras2ras matrix from the LTA struct."""
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


def _get_phys2phys(lta: LtaStruct) -> np.ndarray:
    """Compute the phys2phys matrix from the LTA struct."""
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
    raise AssertionError(f"unsupported LTA type: {lta.type}")


def _get_vox2vox(lta: LtaStruct) -> np.ndarray:
    """Compute the vox2vox matrix from the LTA struct."""
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
    raise AssertionError(f"unsupported LTA type: {lta.type}")


# The orientation helpers are shared with the other FreeSurfer formats.
_mat2code = mat2code
_code2orient = code2orient
_mat2orient = mat2orient


def _get_orient(vol_info: LtaStruct.VolumeInfo) -> str:
    """Get the orientation string from the volume info."""
    vox2ras = _get_vox2ras(vol_info)
    return _mat2orient(vox2ras)
