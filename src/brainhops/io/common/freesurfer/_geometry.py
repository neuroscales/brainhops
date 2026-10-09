"""The volume geometry shared by every FreeSurfer format: the
voxel-to-RAS, voxel-to-tkr and voxel-to-physical matrices,
and orientation codes."""

# externals
import numpy as np
import typing_extensions as tx

_3Ints = tx.Tuple[int, int, int]
_3Floats = tx.Tuple[float, float, float]
_3Flips = tx.Tuple[tx.Literal[-1, 1], tx.Literal[-1, 1], tx.Literal[-1, 1]]
_Vec = tx.Sequence[float]


FS_DEFAULT_XRAS: _3Floats = (-1.0, 0.0, 0.0)
"""The direction of the first voxel axis of the default LIA volume."""

FS_DEFAULT_YRAS: _3Floats = (0.0, 0.0, -1.0)
"""The direction of the second voxel axis of the default LIA volume."""

FS_DEFAULT_ZRAS: _3Floats = (0.0, 1.0, 0.0)
"""
The direction of the third voxel axis of the default LIA volume.

When a volume records no valid geometry (an MGH `goodRASFlag` that is
not positive, or an LTA volume marked invalid), FreeSurfer falls back
on these coronal conformed directions, 1 mm voxels and a zero `c_ras`.
"""


def fs_vox2phys(shape: tx.Sequence[int], voxelsize: _Vec) -> np.ndarray:
    """
    Return the `(4, 4)` voxel-to-physical matrix.

    The matrix scales by the voxel size and moves the origin to the centre
    of the volume, at voxel `shape / 2`.
    """
    shape = np.asarray(shape, dtype=np.float64)[:3]
    voxelsize = np.asarray(voxelsize, dtype=np.float64)[:3]
    vox2phys = np.eye(4)
    vox2phys[[0, 1, 2], [0, 1, 2]] = voxelsize
    vox2phys[:3, 3] = -0.5 * shape * voxelsize
    return vox2phys


def fs_phys2ras(xras: _Vec, yras: _Vec, zras: _Vec, cras: _Vec) -> np.ndarray:
    """
    Return the `(4, 4)` physical-to-scanner-RAS matrix.

    Its columns are the three direction cosines and the centre of the
    volume.
    """
    phys2ras = np.eye(4)
    phys2ras[:3, 0] = xras
    phys2ras[:3, 1] = yras
    phys2ras[:3, 2] = zras
    phys2ras[:3, 3] = cras
    return phys2ras


def fs_vox2ras(
    shape: tx.Sequence[int],
    voxelsize: _Vec,
    xras: _Vec,
    yras: _Vec,
    zras: _Vec,
    cras: _Vec,
) -> np.ndarray:
    """Return the `(4, 4)` voxel-to-scanner-RAS matrix."""
    return fs_phys2ras(xras, yras, zras, cras) @ fs_vox2phys(shape, voxelsize)


def fs_vox2tkr(shape: tx.Sequence[int], voxelsize: _Vec) -> np.ndarray:
    """
    Return the `(4, 4)` voxel-to-tkr-RAS matrix, FreeSurfer's `Torig`.

    The matrix uses the default LIA directions and a zero `c_ras`, so it
    depends only on the shape and the voxel size.
    """
    return fs_vox2ras(
        shape,
        voxelsize,
        FS_DEFAULT_XRAS,
        FS_DEFAULT_YRAS,
        FS_DEFAULT_ZRAS,
        (0.0, 0.0, 0.0),
    )


def fs_geometry_from_vox2ras(
    vox2ras: np.ndarray, shape: tx.Sequence[int]
) -> tx.Tuple[_3Floats, _3Floats, _3Floats, _3Floats, _3Floats]:
    """
    Recover `(voxelsize, xras, yras, zras, cras)` from a voxel-to-RAS
    matrix.

    This is the inverse of [`fs_vox2ras`][]. A shear cannot be represented
    and is lost; the cosines are then not orthogonal, as FreeSurfer writes
    them.
    """
    vox2ras = np.asarray(vox2ras, dtype=np.float64)
    linear = vox2ras[:3, :3]
    voxelsize = np.linalg.norm(linear, axis=0)
    # A zero axis gets a zero cosine rather than a division by zero.
    safe = np.where(voxelsize > 0, voxelsize, 1.0)
    cosines = linear / safe
    centre = np.ones(4)
    centre[:3] = 0.5 * np.asarray(shape, dtype=np.float64)[:3]
    cras = (vox2ras @ centre)[:3]

    def _t(x: np.ndarray) -> _3Floats:
        return tuple(float(v) for v in x)

    return (
        _t(voxelsize),
        _t(cosines[:, 0]),
        _t(cosines[:, 1]),
        _t(cosines[:, 2]),
        _t(cras),
    )


def mat2code(vox2ras: np.ndarray) -> tx.Tuple[_3Ints, _3Flips]:
    """
    Compute the orientation code of a `(4, 4)` voxel-to-RAS matrix.

    Returns
    -------
    permut : array of int
        The RAS axis on which each voxel axis runs.
    flips : array of int
        The direction, -1 or 1, of each voxel axis along its RAS axis.
    """
    vox2ras = vox2ras[:3, :3]
    phys2ras = vox2ras / np.linalg.norm(vox2ras, axis=0)
    u, _, vh = np.linalg.svd(phys2ras)
    ortho = u @ vh
    permut = np.abs(ortho + np.random.rand(3, 3) * 1e-6)
    permut = np.round(permut).astype(np.int8)
    permut = np.argmax(permut, axis=0)
    flips = np.sign(np.diag(ortho[permut, :]))
    return permut, flips


def code2orient(permut: _3Ints, flips: _3Flips) -> str:
    """
    Convert an orientation code, as [`mat2code`][] returns it, to a string.

    The string holds one letter per voxel axis, in voxel order: `L`
    (right-to-left) or `R` (left-to-right), `P` (anterior-to-posterior) or
    `A` (posterior-to-anterior), and `I` (superior-to-inferior) or `S`
    (inferior-to-superior).
    """
    names = [["L", "R"], ["P", "A"], ["I", "S"]]
    name = "".join([names[p][int(f > 0)] for p, f in zip(permut, flips)])
    return name


def mat2orient(vox2ras: np.ndarray) -> str:
    """
    Return the orientation string, such as `"LIA"`, of a voxel-to-RAS
    matrix.
    """
    permut, flips = mat2code(vox2ras)
    return code2orient(permut, flips)
