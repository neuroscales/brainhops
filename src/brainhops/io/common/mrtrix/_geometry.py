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
Where MRtrix stores the axes of an array, as a policy of
[`plan_axes`][brainhops.io.common._geometry.plan_axes].

A slice with non-spatial axes gets a `z` axis of size one, so that
those axes are not read as spatial.
"""


def voxel_to_ras(
    xform: Transformation, voxel_axes: tx.Optional[tx.List[tx.Any]] = None
) -> Arrangement:
    """
    Return the voxel-to-RAS geometry of a voxel-to-world transformation,
    with the axes placed where MRtrix stores them.

    The axes are placed by what their spaces declare, where `voxel_axes`
    are the axes of the data (see [`arrange_voxel_to_ras`][]).

    Raises
    ------
    UnrepresentableTransformationError
        If the transformation has no affine form, or if it mixes the spatial
        axes with the others.
    """
    return arrange_voxel_to_ras(
        xform, voxel_axes, "MRtrix", "scanner", **MRTRIX_POLICY
    )


def split_voxel_to_scanner(
    matrix: np.ndarray,
) -> tx.Tuple[np.ndarray, tx.Tuple[float, float, float]]:
    """
    Split a voxel-to-scanner matrix into an MRtrix transform of unit
    directions and the voxel sizes.

    Raises
    ------
    WriterError
        If a direction column has zero length, or if the matrix is not
        finite.
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
