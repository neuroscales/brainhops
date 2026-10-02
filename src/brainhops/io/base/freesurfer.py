"""
The volume geometry shared by every FreeSurfer format.

FreeSurfer describes the world placement of a volume the same way in
every file that records one -- the header of an MGH/MGZ image, the source
and destination blocks of an LTA transform, ... -- with:

- the volume's shape, in voxels (`width, height, depth`);
- the voxel size (`xsize, ysize, zsize`), in millimetres;
- the direction cosines of each voxel axis in RAS (`x_ras`, `y_ras`,
  `z_ras`), which form the columns of the rotation part of the
  voxel-to-RAS matrix;
- the RAS coordinates of the centre of the volume (`c_ras`, `Pxyz_c`).

Three coordinate systems derive from it:

- **scanner RAS**, the world space the volume was acquired in
  (`mri_info --vox2ras`);
- **tkr RAS** (also "surface RAS" or "tkregister RAS"), the space
  FreeSurfer surfaces live in. It has the same voxel sizes but drops the
  direction cosines and `c_ras`: the centre of the volume is the origin,
  and the axes are those of a conformed (LIA) volume
  (`mri_info --vox2ras-tkr`);
- **physical** (or "physvox"), the scaled voxel space shifted so that
  its origin is the centre of the volume. It is the space between
  voxels and scanner RAS used by LTA files of type `LINEAR_PHYSVOX`.

!!! note "The centre of the volume"
    FreeSurfer places the centre of the volume at voxel coordinate
    `shape / 2`, not at `(shape - 1) / 2`. With 0-based voxel
    coordinates whose integers are voxel centres, the centre therefore
    falls half a voxel past the true centre of an even-sized volume.
    This is FreeSurfer's convention, and every matrix here follows it so
    that it matches FreeSurfer and `nibabel` exactly.

The functions take the geometry as plain values, so that each format
reads it from wherever it stores it.

Every FreeSurfer format -- MGH/MGZ images, LTA transforms, ... --
derives from [`FreesurferFormat`][brainhops.io.base.freesurfer.
FreesurferFormat], so that the `"freesurfer"` hint selects them all.
"""

__all__ = [
    "FreesurferFormat",
    "FS_DEFAULT_XRAS",
    "FS_DEFAULT_YRAS",
    "FS_DEFAULT_ZRAS",
    "fs_vox2phys",
    "fs_phys2ras",
    "fs_vox2ras",
    "fs_vox2tkr",
    "fs_geometry_from_vox2ras",
    "mat2orient",
]

# externals
import numpy as np
import typing_extensions as tx

# type hints
_3Ints = tx.Tuple[int, int, int]
_3Floats = tx.Tuple[float, float, float]
_3Flips = tx.Tuple[tx.Literal[-1, 1], tx.Literal[-1, 1], tx.Literal[-1, 1]]
_Vec = tx.Sequence[float]


class FreesurferFormat:
    """
    A format of the FreeSurfer family, whatever it stores.

    It is the shared base of the FreeSurfer image formats (MGH/MGZ) and
    transformation formats (LTA), and carries the `"freesurfer"` hint
    they all answer to. Each format adds its own hints (`"mgh"`,
    `"lta"`, ...), which are then also reachable as `"freesurfer.mgh"`,
    `"freesurfer.lta"`, ...
    """

    HINTS = ("freesurfer",)


FS_DEFAULT_XRAS: _3Floats = (-1.0, 0.0, 0.0)
"""Direction cosine of the first voxel axis of a default (LIA) volume."""

FS_DEFAULT_YRAS: _3Floats = (0.0, 0.0, -1.0)
"""Direction cosine of the second voxel axis of a default (LIA) volume."""

FS_DEFAULT_ZRAS: _3Floats = (0.0, 1.0, 0.0)
"""Direction cosine of the third voxel axis of a default (LIA) volume.

FreeSurfer falls back on these three -- a coronal, "conformed" LIA
orientation -- with a 1 mm voxel size and a zero `c_ras` when a volume
records no valid geometry (an MGH header whose `goodRASFlag` is not
positive, an LTA volume marked invalid).
"""


def fs_vox2phys(shape: tx.Sequence[int], voxelsize: _Vec) -> np.ndarray:
    """
    The `(4, 4)` matrix from voxel to physical (centred scaled voxel) space.

    It scales by the voxel size and moves the origin to the centre of the
    volume, voxel `shape / 2`.
    """
    shape = np.asarray(shape, dtype=np.float64)[:3]
    voxelsize = np.asarray(voxelsize, dtype=np.float64)[:3]
    vox2phys = np.eye(4)
    vox2phys[[0, 1, 2], [0, 1, 2]] = voxelsize
    vox2phys[:3, 3] = -0.5 * shape * voxelsize
    return vox2phys


def fs_phys2ras(xras: _Vec, yras: _Vec, zras: _Vec, cras: _Vec) -> np.ndarray:
    """
    The `(4, 4)` matrix from physical space to scanner RAS.

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
    """The `(4, 4)` matrix from voxel space to scanner RAS."""
    return fs_phys2ras(xras, yras, zras, cras) @ fs_vox2phys(shape, voxelsize)


def fs_vox2tkr(shape: tx.Sequence[int], voxelsize: _Vec) -> np.ndarray:
    """
    The `(4, 4)` matrix from voxel space to tkr (surface) RAS.

    This is FreeSurfer's `Torig`: the voxel-to-RAS matrix of the same
    volume with default (LIA) direction cosines and a zero `c_ras`. It
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
    Decompose a voxel-to-RAS matrix into FreeSurfer's volume geometry.

    This is the inverse of [`fs_vox2ras`][]: the voxel size is the norm
    of each column, the direction cosines are the normalised columns, and
    the centre is the RAS position of voxel `shape / 2`. A matrix with a
    shear, which FreeSurfer cannot store, loses it: the direction cosines
    are then not orthogonal, exactly as FreeSurfer would write them.

    Returns
    -------
    voxelsize, xras, yras, zras, cras : tuple of three floats each
    """
    vox2ras = np.asarray(vox2ras, dtype=np.float64)
    linear = vox2ras[:3, :3]
    voxelsize = np.linalg.norm(linear, axis=0)
    # A degenerate axis has no direction: keep the default rather than
    # divide by zero.
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


def code2orient(permut: _3Ints, flips: _3Flips) -> str:
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


def mat2orient(vox2ras: np.ndarray) -> str:
    """Convert a vox2ras matrix to an orientation string, such as
    `"LIA"`."""
    permut, flips = mat2code(vox2ras)
    return code2orient(permut, flips)
