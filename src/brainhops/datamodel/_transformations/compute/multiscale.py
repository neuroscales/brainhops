import numpy as np
import typing_extensions as tx

from brainhops._core.affines import axis_scales
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel._multiscale import level_resolution

from ..base import Transformation
from ..concrete import Affine

if tx.TYPE_CHECKING:
    from brainhops.datamodel.multiscale import Multiscale


def nearest_level(
    multiscale: "Multiscale", voxel2world: Transformation
) -> int:
    # Return the index of the scale whose resolution is closest to the
    # grid of `voxel2world`. The comparison is done by
    # `nearest_resolution_index`, which the multiscale image also uses.
    scales = multiscale.scales
    resolutions = [level_resolution(multiscale, i) for i in range(len(scales))]
    return nearest_resolution_index(
        resolutions, voxel2world.to_affine(include="affine", on_error=None)
    )


def nearest_resolution_index(
    resolutions: tx.Sequence[tx.Optional[ArrayProtocol]],
    voxel2world: tx.Optional[Affine],
) -> int:
    """
    Return the index of the resolution closest to a target grid.

    `resolutions` holds the physical grid size of each level, finest first, as
    a vector with one entry per axis, or `None` where it is unknown.
    `voxel2world` is the affine of the grid being matched, already reduced by
    the caller, or `None`. Resolutions are compared in log scale, by the mean
    logarithm of the absolute axis scales, so that a level above the target and
    a level below it are weighed evenly, and the finer level wins a tie. When a
    level or the target is unknown, or when there is at most one level, the
    finest level (0) is returned. Multiscale fields and multiscale images share
    this function, so they cannot disagree.
    """
    if len(resolutions) <= 1:
        return 0
    if any(resolution is None for resolution in resolutions):
        return 0
    if voxel2world is None or voxel2world.matrix is None:
        return 0
    target = axis_scales(voxel2world.matrix)
    if target is None:
        return 0
    log_target = float(np.log(np.abs(np.asarray(target))).mean())
    best_index, best_distance = 0, None
    for index, resolution in enumerate(resolutions):
        log_resolution = float(np.log(np.abs(np.asarray(resolution))).mean())
        distance = abs(log_resolution - log_target)
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index
