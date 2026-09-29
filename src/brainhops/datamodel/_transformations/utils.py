# dependencies
import typing_extensions as tx

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
