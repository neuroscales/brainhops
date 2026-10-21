import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.affines import axis_scales
from brainhops._core.typing import ArrayProtocol
from brainhops.errors import ConversionError

from ..base import Transformation
from ..concrete import Affine, Identity, TransformationField
from ..inverse import Inverse
from ..sequence import Sequence

if tx.TYPE_CHECKING:
    from brainhops.datamodel.multiscale import Multiscale


def nearest_level(
    multiscale: "Multiscale", voxel2world: Transformation
) -> int:
    # Return the index of the scale whose resolution is closest to the
    # grid of `voxel2world`. The comparison is done by
    # `_nearest_resolution_index`, which the multiscale image also uses.
    scales = multiscale.scales
    resolutions = [multiscale._level_resolution(i) for i in range(len(scales))]
    return _nearest_resolution_index(
        resolutions, _as_affine_ignoring_fields(voxel2world)
    )


def _nearest_resolution_index(
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


def _as_affine(xform: Transformation) -> tx.Optional[Affine]:
    """
    Return the transformation as an `Affine`.

    If the transformation does not reduce to an affine with a defined
    matrix, `None` is returned rather than an error.

    A caller that must know whether the transformation really is affine
    uses this function, not `_as_affine_ignoring_fields`.
    """
    try:
        affine = xform.compute().to(Affine)
    except ConversionError:
        return None
    if not isinstance(affine, Affine) or affine.matrix is None:
        return None
    return affine


def _as_affine_ignoring_fields(
    xform: Transformation,
) -> tx.Optional[Affine]:
    """
    Return the affine part of the transformation, with every field
    treated as the identity.

    Return `None` if what remains is not affine.

    A warp has no affine form, but it is close to an isometry, so the
    affine part still measures the scale of a chain that contains a warp.
    The result answers questions about scale only.
    """
    return _as_affine(_fields_as_identity(xform))


def _fields_as_identity(xform: Transformation) -> Transformation:
    """
    Replace every field with the identity and keep the affine part.

    Fields are recognised by their class, so the typed inverse of a
    field is also replaced.
    """
    if isinstance(xform, TransformationField):
        return Identity(input=xform.input, output=xform.output)
    if isinstance(xform, Inverse):
        forward = xform.forward
        if forward is None:
            return xform
        resolved = _fields_as_identity(forward)
        return xform if resolved is forward else resolved.inverse()
    if isinstance(xform, Sequence):
        parts = xform.transformations or []
        resolved = [_fields_as_identity(part) for part in parts]
        if any(new is not old for new, old in zip(resolved, parts)):
            return replace(xform, transformations=resolved)
        return xform
    return xform
