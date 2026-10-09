import numpy as np
import typing_extensions as tx

from brainhops.datamodel._sugar import get_axes
from brainhops.errors import ConversionError

if tx.TYPE_CHECKING:
    from brainhops.datamodel.systems import CoordinateSystem

    from ..base import Transformation


def get_ndim(
    t: "Transformation", default: tx.Optional[int] = None
) -> tx.Optional[int]:
    """Return the axis count of the input system of `t`, or of its output.

    `default` is returned when neither system is closed, because a missing
    or open system says nothing about the count.
    """
    ndim = get_axes(t.input).ndim
    if ndim is None:
        ndim = get_axes(t.output).ndim
    return default if ndim is None else ndim


def systems_disagree(
    source: tx.Optional["CoordinateSystem"],
    target: tx.Optional["CoordinateSystem"],
) -> bool:
    """Return whether two adjacent coordinate systems need reconciling.

    `source` is where one transformation leaves the coordinates and `target` is
    where the next one expects them. Two closed systems disagree when they are
    unequal. A missing or open system disagrees only when its stated axes are
    not
    [`compatible_with`][brainhops.datamodel.systems.CoordinateSystem.compatible_with]
    those of its neighbour, because an unknown axis is never a reason to
    refuse. A disagreement is removed by [`adapt`][].
    """
    source_axes, target_axes = (
        get_axes(source),
        get_axes(target),
    )
    if source_axes.is_open or target_axes.is_open:
        return not source_axes.compatible_with(target_axes)
    return source != target


def boundary_disagrees(
    first: "Transformation", second: "Transformation"
) -> bool:
    """Return whether `first` and `second` disagree on their shared system."""
    return systems_disagree(first.output, second.input)


def with_endpoints(
    t: "Transformation", like: "Transformation"
) -> "Transformation":
    """Carry the endpoints that `like` declares onto `t`.

    When `like` declares none, `t` is returned as the same object, so that a
    lazy inverse that refers to it still cancels.
    """
    edits = {}
    if like._input is not None:
        edits["input"] = like._input
    if like._output is not None:
        edits["output"] = like._output
    return t.to(**edits) if edits else t


def require_endomorphism(t: "Transformation", operator: str) -> None:
    """Refuse a transformation that does not map a space to itself.

    The square root, exponential and logarithm require an endomorphism. The
    check fails when the systems disagree or when the stated axis counts
    differ. A count that is not stated is never a reason to refuse.

    Raises
    ------
    DomainError
        If `t` is not an endomorphism.
    """
    from ....errors import DomainError

    kind = type(t).__name__
    if systems_disagree(t.input, t.output):
        raise DomainError(
            f"The {operator} of a transformation is defined only when it "
            f"maps a space to itself, but this {kind} maps {t.input} to "
            f"{t.output}."
        )
    ni, no = axis_counts(t)
    if ni is not None and no is not None and ni != no:
        raise DomainError(
            f"The {operator} of a transformation is defined only when it "
            f"maps a space to itself, but this {kind} maps {ni} axes to "
            f"{no} axes."
        )


def axis_list(axes: tx.Optional[tx.Any]) -> tx.List[int]:
    """Return axis indices as a list of ints, empty for None.

    The argument is compared with None instead of being tested for truth,
    because the truth value of an array is ambiguous.
    """
    if axes is None:
        return []
    return [int(a) for a in axes]


def axis_counts(
    t: tx.Any,
) -> tx.Tuple[tx.Optional[int], tx.Optional[int]]:
    """Return the input and output axis counts that a transformation states.

    Each count is None when nothing states it. The counts are read from:

    * a matrix shape (`Affine`, `Linear`, `Rotation`), allowing for the
      homogeneous column of an affine;
    * the length of `scale`, `translation` or `permutation`;
    * a field: the last dimension of a displacement field, the spatial rank and
      last dimension of a coordinates field, the rank of a grid;
    * a lazy inverse: the counts of its forward transformation, swapped;
    * a lazy square root, exponential or logarithm: the counts of its
      forward transformation;
    * a sequence: its ends, reading past any leading (or trailing) member that
      preserves the dimension without stating it;
    * the declared input and output systems.

    The declared systems are the fallback, and only a closed system states a
    count. Reading never materializes a lazy inverse or composes a
    sequence.
    """
    from ..base import Transformation
    from ..concrete import (
        Affine,
        CartesianField,
        CoordinatesField,
        DisplacementField,
        Linear,
        Permutation,
        Scaling,
        Translation,
    )
    from ..inverse import Inverse
    from ..operators import Operation
    from ..sequence import Sequence
    from .checkers import _matrix_dims

    if not isinstance(t, Transformation):
        return None, None
    ni: tx.Optional[int] = None
    no: tx.Optional[int] = None
    if isinstance(t, Inverse):
        # A typed inverse such as InverseScaling is also a Scaling, so inverses
        # are checked before the concrete families.
        if t.forward is not None:
            no, ni = axis_counts(t.forward)
    elif isinstance(t, Operation):
        # A lazy operator maps a space to itself, so its counts are those of
        # its forward transformation.
        if t.forward is not None:
            ni, no = axis_counts(t.forward)
    elif isinstance(t, Sequence):
        ni, no = _sequence_ends(list(t.transformations or []))
    elif isinstance(t, (Affine, Linear)):
        dims = _matrix_dims(t)
        if dims is not None:
            ni, no = dims
    else:
        for name, cls in (
            ("scale", Scaling),
            ("translation", Translation),
            ("permutation", Permutation),
        ):
            value = getattr(t, name, None) if isinstance(t, cls) else None
            if value is not None:
                ni = no = len(value)
        if isinstance(t, CartesianField):
            if t.shape is not None:
                ni = no = len(t.shape)
        elif isinstance(t, (DisplacementField, CoordinatesField)):
            # Only the shape is read, so a spline-coefficient field is never
            # decoded.
            field = t.data
            if field is not None:
                no = int(field.shape[-1])
                if isinstance(t, DisplacementField):
                    ni = no
                else:
                    ni = len(field.shape) - 1
    if ni is None:
        ni = get_axes(getattr(t, "_input", None)).ndim
    if no is None:
        no = get_axes(getattr(t, "_output", None)).ndim
    return ni, no


def _sequence_ends(
    members: tx.List["Transformation"],
) -> tx.Tuple[tx.Optional[int], tx.Optional[int]]:
    # Read the axis counts at the two ends of a sequence. A member that
    # states neither count is skipped only if it preserves the dimension.
    ni = _first_count(members, 0)
    no = _first_count(members[::-1], 1)
    return ni, no


def _first_count(
    members: tx.List["Transformation"], side: int
) -> tx.Optional[int]:
    from ..sequence import Sequence
    from .factor import _element_ndim

    for member in members:
        counts = axis_counts(member)
        if counts[side] is not None:
            return counts[side]
        if counts[1 - side] is not None or isinstance(member, Sequence):
            return None
        if _element_ndim(member, -1) != (-1, -1):
            return None
    return None


UNREADABLE = object()
"""Sentinel for a transformation without an affine reading.

It is distinct from None, which marks an affine-like transformation
without parameters, that is, the identity.
"""


def affine_matrix(t: "Transformation") -> tx.Any:
    """Return the affine matrix of an affine-like transformation as an array.

    The result is None for the identity and [`UNREADABLE`][] when there is no
    conversion to Affine. A lazy Inverse must not be passed, because the
    conversion would materialize it.
    """
    from ..concrete import Affine

    try:
        affine = t.to(Affine)
    except ConversionError:
        return UNREADABLE
    matrix = affine.matrix
    return None if matrix is None else np.asarray(matrix)
