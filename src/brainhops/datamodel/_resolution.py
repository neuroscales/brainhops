"""The levels of multiscale transformations that match a grid."""

from bagof.magic import replace

from ._transformations.compute.multiscale import (
    nearest_level,
)
from .geometry import Geometry
from .multiscale import Multiscale
from .transformations import (
    Sequence,
    Transformation,
)


def match_resolution(
    transformation: Transformation, voxel2world: Transformation
) -> Transformation:
    """Replace each multiscale field by the level that matches a grid.

    When an image is resliced onto a grid, a multiscale field in the
    transformation is sampled at the level whose resolution is nearest to
    the grid, as [`nearest_level`][] chooses it. The function looks into
    sequences recursively, and returns a transformation that holds no
    multiscale field unchanged.

    Parameters
    ----------
    transformation : Transformation
        The transformation, which may be or hold a multiscale field.
    voxel2world : Transformation
        The voxel-to-world transformation of the grid, or the
        [`Geometry`][] of the grid.

    Returns
    -------
    Transformation
        The transformation, with each multiscale field replaced by one of
        its levels.
    """
    if isinstance(voxel2world, Geometry):
        voxel2world = voxel2world.transformation
    if isinstance(transformation, Multiscale):
        return transformation.to_singlescale(
            nearest_level(transformation, voxel2world)
        )
    if isinstance(transformation, Sequence):
        parts = transformation.transformations or []
        resolved = [match_resolution(part, voxel2world) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(transformation, transformations=resolved)
        return transformation
    return transformation
