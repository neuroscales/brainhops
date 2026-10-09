"""MRtrix voxel-to-scanner geometry."""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base.parsers import WriterError
from brainhops.io.common._geometry import Arrangement, arrange_voxel_to_ras

# ----------------------------------------------------------------------
#   GEOMETRY (writing)
# ----------------------------------------------------------------------


MRTRIX_POLICY = dict(fill_space=True)
"""
Where MRtrix stores the axes of an array (see
[`plan_axes`][brainhops.io.common._geometry.plan_axes]).

MRtrix reads its first three axes as spatial, and gives the others no
meaning of their own: they are stored after the spatial ones, time first,
then the channels, then the others. A slice with other axes is given a
`z` axis of size one, so they are not read as spatial.
"""


def voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[tx.Any]] = None
) -> Arrangement:
    """
    The voxel-to-RAS geometry of a voxel-to-world transformation, with
    the axes placed where MRtrix stores them.

    The transformation is reduced to an affine (a `Scaling`, a `Sequence`
    of affines, ...). The axes are placed by the types and names their
    spaces declare (`voxel_axes` are those of the data, see
    [`arrange_voxel_to_ras`][brainhops.io.common._geometry.
    arrange_voxel_to_ras] and `MRTRIX_POLICY`). A map with fewer than
    three dimensions is embedded in three. A map over more axes keeps its
    three spatial ones in the `(4, 4)` matrix, when it does not mix them
    with the others, and the scale and offset of the others apart. The
    world space is turned into RAS from the anatomical orientation of its
    axes; a world space with no orientation is taken to be RAS already.

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine representation, or mixes the
        spatial axes with the others.
    """
    return arrange_voxel_to_ras(
        xform, voxel_axes, "MRtrix", "scanner", **MRTRIX_POLICY
    )


def split_voxel_to_scanner(
    matrix: np.ndarray,
) -> tx.Tuple[np.ndarray, tx.Tuple[float, float, float]]:
    """
    Split a voxel-to-scanner matrix into an MRtrix transform and voxel
    sizes.

    The voxel sizes are the lengths of the three direction columns, and
    the transform holds the unit-length directions and the translation:
    `matrix == [transform; 0 0 0 1] @ diag(vox, 1)`.

    Raises
    ------
    WriterError
        If a direction column has zero length, so no voxel size can be
        derived.
    """
    matrix = np.asarray(matrix, dtype=float)
    vox = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.all(vox > 0) or not np.all(np.isfinite(matrix)):
        raise WriterError(
            "The voxel-to-scanner matrix is degenerate or not finite, so "
            "it cannot be written as an MRtrix transform and voxel sizes."
        )
    transform = np.array(matrix[:3, :4], copy=True)
    transform[:, :3] /= vox
    return transform, tuple(float(v) for v in vox)
