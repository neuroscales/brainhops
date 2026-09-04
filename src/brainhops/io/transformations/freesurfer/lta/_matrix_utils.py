# externals
import numpy as np
import typing_extensions as tx

# internals
from ._enums import LTAType
from ._struct import LTAStruct

# type hints
_3Ints = tx.Tuple[int, int, int]
_3Flips = tx.Tuple[tx.Literal[-1, 1], tx.Literal[-1, 1], tx.Literal[-1, 1]]


def _get_vox2phys(vol_info: LTAStruct.VolumeInfo) -> np.ndarray:
    """Compute the vox2phys matrix from the volume geometry."""
    shift = -0.5 * np.asarray(vol_info.volume) * np.asarray(vol_info.voxelsize)
    vox2phys = np.eye(4)
    vox2phys[0, 0] = vol_info.voxelsize[0]
    vox2phys[1, 1] = vol_info.voxelsize[1]
    vox2phys[2, 2] = vol_info.voxelsize[2]
    vox2phys[:3, 3] = shift
    return vox2phys


def _get_phys2ras(vol_info: LTAStruct.VolumeInfo) -> np.ndarray:
    """Compute the phys2ras matrix from the volume geometry."""
    phys2ras = np.eye(4)
    phys2ras[:3, 0] = vol_info.xras
    phys2ras[:3, 1] = vol_info.yras
    phys2ras[:3, 2] = vol_info.zras
    phys2ras[:3, 3] = vol_info.cras
    return phys2ras


def _get_vox2ras(vol_info: LTAStruct.VolumeInfo) -> np.ndarray:
    """Compute the vox2ras matrix from the volume geometry."""
    return _get_phys2ras(vol_info) @ _get_vox2phys(vol_info)


def _get_ras2ras(lta: LTAStruct) -> np.ndarray:
    """Compute the ras2ras matrix from the LTA struct."""
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LTAType.LINEAR_RAS_TO_RAS:
        return matrix
    if lta.type == LTAType.LINEAR_RSA_TO_RSA:
        # RSA -> RAS = permute first two axes
        return matrix[[1, 0, 2, 3], :][:, [1, 0, 2, 3]]
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute RAS-to-RAS matrix without src and dst volume info"
        )
    if lta.type == LTAType.LINEAR_VOX_TO_VOX:
        src_vox2ras = _get_vox2ras(lta.src)
        dst_vox2ras = _get_vox2ras(lta.dst)
        return dst_vox2ras @ matrix @ np.linalg.inv(src_vox2ras)
    if lta.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
        src_phys2ras = _get_phys2ras(lta.src)
        dst_phys2ras = _get_phys2ras(lta.dst)
    return dst_phys2ras @ matrix @ np.linalg.inv(src_phys2ras)


def _get_phys2phys(lta: LTAStruct) -> np.ndarray:
    """Compute the phys2phys matrix from the LTA struct."""
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
        return matrix
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute phys-to-phys matrix without "
            "src and dst volume info"
        )
    if lta.type == LTAType.LINEAR_VOX_TO_VOX:
        src_vox2phys = _get_vox2phys(lta.src)
        dst_vox2phys = _get_vox2phys(lta.dst)
        return dst_vox2phys @ matrix @ np.linalg.inv(src_vox2phys)
    elif lta.type == LTAType.LINEAR_RAS_TO_RAS:
        src_phys2ras = _get_phys2ras(lta.src)
        dst_phys2ras = _get_phys2ras(lta.dst)
        return np.linalg.inv(dst_phys2ras) @ matrix @ src_phys2ras
    elif lta.type == LTAType.LINEAR_RSA_TO_RSA:
        # RAS -> RSA = permute first two axes
        src_phys2rsa = _get_phys2ras(lta.src)[[1, 0, 2, 3], :][:, [1, 0, 2, 3]]
        dst_phys2rsa = _get_phys2ras(lta.dst)[[1, 0, 2, 3], :][:, [1, 0, 2, 3]]
        return np.linalg.inv(dst_phys2rsa) @ matrix @ src_phys2rsa
    raise AssertionError(f"unsupported LTA type: {lta.type}")


def _get_vox2vox(lta: LTAStruct) -> np.ndarray:
    """Compute the vox2vox matrix from the LTA struct."""
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LTAType.LINEAR_VOX_TO_VOX:
        return matrix
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute vox-to-vox matrix without src and dst volume info"
        )
    if lta.type == LTAType.LINEAR_PHYSVOX_TO_PHYSVOX:
        src_vox2phys = _get_vox2phys(lta.src)
        dst_vox2phys = _get_vox2phys(lta.dst)
        return np.linalg.inv(dst_vox2phys) @ matrix @ src_vox2phys
    if lta.type == LTAType.LINEAR_RAS_TO_RAS:
        src_vox2ras = _get_vox2ras(lta.src)
        dst_vox2ras = _get_vox2ras(lta.dst)
        return np.linalg.inv(dst_vox2ras) @ matrix @ src_vox2ras
    if lta.type == LTAType.LINEAR_RSA_TO_RSA:
        # RSA -> RAS = permute first two axes
        ras2ras = matrix[[1, 0, 2, 3], :][:, [1, 0, 2, 3]]
        src_vox2ras = _get_vox2ras(lta.src)
        dst_vox2ras = _get_vox2ras(lta.dst)
        return np.linalg.inv(dst_vox2ras) @ ras2ras @ src_vox2ras
    raise AssertionError(f"unsupported LTA type: {lta.type}")


def _mat2code(vox2ras: np.ndarray) -> tx.Tuple[_3Ints, _3Flips]:
    """Convert a vox2ras matrix to an orientation code.

    Parameters
    ----------
    vox2ras : np.ndarray
        A 4x4 vox2ras matrix.

    Returns
    -------
    permut : (int, int, int)
        A tuple of three integers representing the permutation of axes.
    flips : ({-1, 1}, {-1, 1}, {-1, 1})
        A tuple of three integers representing the flips of axes.
    """
    vox2ras = vox2ras[:3, :3]  # keep linear part only
    phys2ras = vox2ras / np.linalg.norm(vox2ras, axis=0)
    u, _, vh = np.linalg.svd(phys2ras)
    ortho = u @ vh
    permut = np.abs(ortho + np.random.rand(3, 3) * 1e-6)
    permut = np.round(permut).astype(np.int8)
    permut = np.argmax(permut, axis=0)
    flips = np.sign(np.diag(ortho[permut, :]))
    return permut, flips


def _code2orient(permut: _3Ints, flips: _3Flips) -> str:
    """Convert a permutation and flip code to an orientation string.

    Parameters
    ----------
    permut : (int, int, int)
        A tuple of three integers representing the permutation of axes.
    flips : ({-1, 1}, {-1, 1}, {-1, 1})
        A tuple of three integers representing the flips of axes.

    Returns
    -------
    orient : str
        Three uppercase letters representing the orientation of the axes.
        Letters correspond to each voxel axis (F-ordered) and can take values:
        - 'L' (right-to-left) or 'R' (left-to-right)
        - 'P' (anterior-to-posterior) or 'A' (posterior-to-anterior)
        - 'I' (superior-to-inferior) or 'S' (inferior-to-superior)

    """
    names = [["L", "R"], ["P", "A"], ["I", "S"]]
    name = "".join([names[p][int(f > 0)] for p, f in zip(permut, flips)])
    return name


def _mat2orient(vox2ras: np.ndarray) -> str:
    """Convert a vox2ras matrix to an orientation string."""
    permut, flips = _mat2code(vox2ras)
    return _code2orient(permut, flips)


def _get_orient(vol_info: LTAStruct.VolumeInfo) -> str:
    """Get the orientation string from the volume info."""
    vox2ras = _get_vox2ras(vol_info)
    return _mat2orient(vox2ras)
