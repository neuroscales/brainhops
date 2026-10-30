"""The voxel-to-world matrix of an Analyze 7.5 image.

An Analyze header stores the voxel sizes of an image but no orientation,
so the position of the voxels in the world is a convention. brainhops
follows the convention of SPM (`decode_qform0.m` in the `@nifti` class of
SPM), which nibabel follows too (`Spm99AnalyzeHeader.get_origin_affine`).

- The world is the RAS space of NIfTI, in millimetres.
- The first axis of the voxels runs from right to left. This is the
  radiological convention of Analyze, which SPM applies to Analyze files
  by default and nibabel applies through `default_x_flip`. The other
  axes run along the world axes.
- The origin of the world is the voxel whose position the field
  `originator` of the header gives, when that field sets one. SPM reads
  its first three 16-bit integers as the one-based voxel indices of the
  origin, and a field that holds three zeros sets no origin.
- Without an origin, the grid is centred on the origin of the world.

The matrix of SPM maps one-based voxel indices, while the voxel
coordinates of brainhops start at zero. With `vox` the voxel sizes, `o`
the one-based origin and `F = diag(-1, 1, 1)` the flip, SPM maps the
one-based index `i + 1` of a voxel to `F vox (i + 1 - o)`, so the matrix
of the zero-based index `i` is `F diag(vox)` with the translation
`F vox (1 - o)`. Without an origin, SPM takes `o = (shape + 1) / 2`, and
the translation is then `-F vox (shape - 1) / 2`, which centres the grid.
"""

# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

_FLIP = np.asarray([-1.0, 1.0, 1.0])
"""The signs of the world axes along the first three voxel axes."""


def analyze_origin(header: nb.AnalyzeHeader) -> tx.Optional[np.ndarray]:
    """Return the one-based voxel indices of the origin, or `None`.

    The origin is the first three 16-bit integers of the field
    `originator`, as SPM reads them. Three zeros set no origin. A plain
    Analyze header, which does not name the field `origin`, has no
    origin either.

    Parameters
    ----------
    header : nibabel.AnalyzeHeader
        The header.

    Returns
    -------
    numpy.ndarray or None
        The three one-based indices, as floats, or `None`.
    """
    if "origin" not in header.keys():
        return None
    origin = np.asarray(header["origin"][:3], dtype=float)
    if not np.any(origin):
        return None
    return origin


def analyze_to_ras(header: nb.AnalyzeHeader) -> np.ndarray:
    """Return the voxel-to-RAS matrix of the first three axes of a header.

    The matrix maps the zero-based indices of the first three voxel axes
    to RAS millimetres, as the module explains. An image with fewer than
    three axes is treated as if its missing axes had one voxel of size
    one, so that a missing axis adds no translation unless the header
    sets an origin along it.

    Parameters
    ----------
    header : nibabel.AnalyzeHeader
        The header.

    Returns
    -------
    numpy.ndarray
        The `(4, 4)` matrix.
    """
    shape = header.get_data_shape()[:3]
    zooms = header.get_zooms()[:3]
    dims = np.ones(3)
    dims[: len(shape)] = shape
    vox = np.ones(3)
    vox[: len(zooms)] = zooms
    origin = analyze_origin(header)
    if origin is None:
        origin = (dims + 1) / 2
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(_FLIP * vox)
    matrix[:3, 3] = _FLIP * vox * (1 - origin)
    return matrix
