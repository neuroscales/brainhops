import numpy as np
import typing_extensions as tx

from brainhops._core.enum import enum_name
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.geometry import Geometry

# internals
from brainhops.io.common.freesurfer._geometry import (
    code2orient,
    fs_geometry_from_vox2ras,
    fs_phys2ras,
    fs_vox2phys,
    fs_vox2tkr,
    mat2code,
    mat2orient,
)
from brainhops.io.common.freesurfer._tkr import scanner_vox2ras

from ._enums import LtaType, LtaValidity
from ._raw import LtaRaw


def _get_vox2phys(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_vox2phys(vol_info.volume, vol_info.voxelsize)


def _get_phys2ras(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_phys2ras(
        vol_info.xras, vol_info.yras, vol_info.zras, vol_info.cras
    )


def _get_vox2ras(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return _get_phys2ras(vol_info) @ _get_vox2phys(vol_info)


def _get_vox2tkr(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_vox2tkr(vol_info.volume, vol_info.voxelsize)


def _get_tkr2ras(vol_info: LtaRaw.VolumeInfo) -> np.ndarray:
    # The tkregister RAS space of the volume, placed in its scanner RAS.
    return _get_vox2ras(vol_info) @ np.linalg.inv(_get_vox2tkr(vol_info))


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
    if lta.type == LtaType.REGISTER_DAT:
        # `MRItkReg2Native(ref=dst, mov=src, R)`, which `LTAchangeType`
        # computes in `utils/transform.cpp` of FreeSurfer (commit
        # 4f8728e). The stored matrix maps the tkregister space of `dst`
        # to that of `src`, so its inverse maps `src` to `dst`.
        src_tkr2ras = _get_tkr2ras(lta.src)
        dst_tkr2ras = _get_tkr2ras(lta.dst)
        tkr2tkr = np.linalg.inv(matrix)
        return dst_tkr2ras @ tkr2tkr @ np.linalg.inv(src_tkr2ras)
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
    elif lta.type in (
        LtaType.LINEAR_RAS_TO_RAS,
        LtaType.LINEAR_RSA_TO_RSA,
        LtaType.REGISTER_DAT,
    ):
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
    if lta.type == LtaType.REGISTER_DAT:
        # The scanner geometries cancel, as in `TransformRegDat2LTA` of
        # `utils/transform.cpp` in FreeSurfer (commit 4f8728e).
        tkr2tkr = np.linalg.inv(matrix)
        src_vox2tkr = _get_vox2tkr(lta.src)
        dst_vox2tkr = _get_vox2tkr(lta.dst)
        return np.linalg.inv(dst_vox2tkr) @ tkr2tkr @ src_vox2tkr
    raise AssertionError(f"unsupported LTA type: {enum_name(lta.type)}")


def _get_tkr2tkr(lta: LtaRaw) -> np.ndarray:
    # The affine from the tkregister RAS space of `src` to that of `dst`.
    # A REGISTER_DAT record stores its inverse.
    matrix = np.asarray(lta.affine.matrix, dtype=np.float64)
    if lta.type == LtaType.REGISTER_DAT:
        return np.linalg.inv(matrix)
    if lta.src is None or lta.dst is None:
        raise ValueError(
            "cannot compute tkr-to-tkr matrix without src and dst volume info"
        )
    src_vox2tkr = _get_vox2tkr(lta.src)
    dst_vox2tkr = _get_vox2tkr(lta.dst)
    if lta.type == LtaType.LINEAR_VOX_TO_VOX:
        return dst_vox2tkr @ matrix @ np.linalg.inv(src_vox2tkr)
    src_tkr2ras = _get_tkr2ras(lta.src)
    dst_tkr2ras = _get_tkr2ras(lta.dst)
    return np.linalg.inv(dst_tkr2ras) @ _get_ras2ras(lta) @ src_tkr2ras


# Orientation helpers shared with the other FreeSurfer formats.
_mat2code = mat2code
_code2orient = code2orient
_mat2orient = mat2orient


def _get_orient(vol_info: LtaRaw.VolumeInfo) -> str:
    vox2ras = _get_vox2ras(vol_info)
    return _mat2orient(vox2ras)


# ----------------------------------------------------------------------
#   VOLUME GEOMETRIES AS DATA MODEL GEOMETRIES
# ----------------------------------------------------------------------


def _volume_geometry(vol_info: LtaRaw.VolumeInfo) -> Geometry:
    """Return the geometry of the volume that a geometry block describes.

    The geometry maps the F-ordered voxels of the volume to scanner RAS,
    as `_get_vox2ras` does. The name of the file of the volume is not
    kept.

    Raises
    ------
    ValueError
        If the block is marked invalid, which is how FreeSurfer writes a
        geometry that is not known.
    """
    if vol_info.valid != LtaValidity.VOLUME_INFO_VALID:
        raise ValueError(
            "This volume geometry is marked invalid (valid = 0), which is "
            "how FreeSurfer writes a geometry that is not known, so it "
            "cannot place the volume in space. Set valid to "
            "LtaValidity.VOLUME_INFO_VALID if the fields are known."
        )
    vox2ras = _get_vox2ras(vol_info)
    voxel = _systems.FVoxelCoordinateSystem(name="voxel")
    vox2ras = _xforms.Affine(vox2ras[:3], input=voxel, output=_systems.RASmm())
    return Geometry(
        transformation=vox2ras,
        shape=tuple(int(n) for n in vol_info.volume),
    )


def _volume_info(
    geometry: Geometry, cls: tx.Type[LtaRaw.VolumeInfo]
) -> LtaRaw.VolumeInfo:
    """Return the geometry block of type `cls` of a data model geometry.

    The block is marked valid and names no file. The voxel size, the
    direction cosines and the centre are those of the voxel-to-RAS
    matrix, as `fs_geometry_from_vox2ras` recovers them.

    Raises
    ------
    TypeError
        If `geometry` is not a `Geometry`.
    ValueError
        If the geometry does not place a 3-D volume in scanner RAS.
    """
    shape, vox2ras = scanner_vox2ras(geometry)
    voxelsize, xras, yras, zras, cras = fs_geometry_from_vox2ras(
        vox2ras, shape
    )
    return cls(
        valid=LtaValidity.VOLUME_INFO_VALID,
        volume=shape,
        voxelsize=voxelsize,
        xras=xras,
        yras=yras,
        zras=zras,
        cras=cras,
    )
