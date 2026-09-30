# dependencies
import numpy as np
import typing_extensions as tx

# internals
from .errors import ConversionError

if tx.TYPE_CHECKING:
    # internals
    from brainhops.datamodel.systems import CoordinateSystem

    from .base import Transformation


def get_ndim(
    t: "Transformation", default: tx.Optional[int] = None
) -> tx.Optional[int]:
    if t.input and t.input.axes is not None:
        return len(t.input.axes)
    if t.output and t.output.axes is not None:
        return len(t.output.axes)
    return default


def systems_disagree(
    source: tx.Optional["CoordinateSystem"],
    target: tx.Optional["CoordinateSystem"],
) -> bool:
    """
    Whether two adjacent coordinate systems need reconciling.

    `source` is where one transform leaves its coordinates and `target` is
    where the next one expects to find them. A system that is unspecified,
    or that carries no axes, is treated as compatible with its neighbour,
    so only two fully described and unequal systems disagree: not knowing
    is never a reason to refuse.

    This is the precondition of everything that assumes the two ends of a
    boundary line up -- the composers, and the two-argument simplifiers.
    [`adapt`][] is what removes a disagreement.
    """
    if source is None or target is None:
        return False
    if source.axes is None or target.axes is None:
        return False
    return source != target


def boundary_disagrees(
    first: "Transformation", second: "Transformation"
) -> bool:
    """Whether two consecutive transforms disagree on the system they share.

    `first` is applied before `second`, so the boundary is between the
    output of `first` and the input of `second`.
    """
    return systems_disagree(first.output, second.input)


def axis_list(axes: tx.Optional[tx.Any]) -> tx.List[int]:
    """A plain list of integer axis indices (empty for `None`).

    An axis vector may be a numpy array, whose truth value is ambiguous, so
    it is tested against `None` rather than for truthiness.
    """
    if axes is None:
        return []
    return [int(a) for a in axes]


UNREADABLE = object()
"""
Returned by `affine_matrix` for a transform that has no affine reading
(no converter to `Affine`: a projection, a bijection, a field, ...). It is
distinct from `None`, which marks an unparameterized (identity) affine-ish
transform.
"""


def affine_matrix(t: "Transformation") -> tx.Any:
    """The affine matrix of an affine-ish transform, as a numpy array.

    Returns `None` when the transform is matrix-less (an identity), and
    `UNREADABLE` when it cannot be converted to an `Affine` at all.
    Never call it on a lazy `Inverse`: converting one materializes it.
    """
    from .concrete import Affine

    try:
        affine = t.to(Affine)
    except ConversionError:
        return UNREADABLE
    matrix = affine.matrix
    return None if matrix is None else np.asarray(matrix)
