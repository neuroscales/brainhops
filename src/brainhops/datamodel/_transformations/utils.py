# dependencies
import numpy as np
import typing_extensions as tx

# api
from brainhops.datamodel.systems import AxisList

# internals
from .errors import ConversionError

if tx.TYPE_CHECKING:
    # internals
    from brainhops.datamodel.systems import CoordinateSystem

    from .base import Transformation


def get_ndim(
    t: "Transformation", default: tx.Optional[int] = None
) -> tx.Optional[int]:
    """The number of axes of the input system of `t`, or else of its output.

    `default` is returned when neither system is closed: a missing or an
    open system says nothing about the number of axes.
    """
    ndim = AxisList.of(t.input).ndim
    if ndim is None:
        ndim = AxisList.of(t.output).ndim
    return default if ndim is None else ndim


def systems_disagree(
    source: tx.Optional["CoordinateSystem"],
    target: tx.Optional["CoordinateSystem"],
) -> bool:
    """
    Whether two adjacent coordinate systems need reconciling.

    `source` is where one transform leaves its coordinates and `target` is
    where the next one expects to find them. Two closed systems disagree
    when they are not equal. A system that is missing or open (its axes
    hold `...`) disagrees with its neighbour only when the
    axes it does state cannot match the neighbour's, i.e. when the two are
    not
    [`compatible_with`][brainhops.datamodel.systems.CoordinateSystem.compatible_with]:
    not knowing is never a reason to refuse.

    This is the precondition of everything that assumes the two ends of a
    boundary line up -- the composers, and the two-argument simplifiers.
    [`adapt`][] is what removes a disagreement.
    """
    source_axes, target_axes = AxisList.of(source), AxisList.of(target)
    if source_axes.is_open or target_axes.is_open:
        return not source_axes.compatible_with(target_axes)
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


def axis_counts(
    t: tx.Any,
) -> tx.Tuple[tx.Optional[int], tx.Optional[int]]:
    """The `(input, output)` axis counts a transformation states.

    Each count is `None` when nothing on `t` states it. They are read, in
    order, from:

    * a matrix shape (`Affine`, `Linear`, `Rotation`), allowing for the
      homogeneous column of an affine;
    * the length of `scale`, `translation` or `permutation`;
    * a field: the last dimension of a displacement field, the spatial rank
      and last dimension of a coordinates field, the rank of a grid;
    * a lazy inverse: its forward's counts, swapped;
    * a sequence: its ends, reading past any leading (or trailing) member
      that preserves the dimension without stating it;
    * the declared input and output systems.

    Only *declared* systems are read, and only a closed system states a
    count: a missing or open system (one whose axes hold `...`) leaves it
    unknown, and is never guessed from. A subspace that declares no system
    derives an open one from its inner system, so it states its counts
    through its declared systems only.

    Reading never materializes a lazy inverse or composes a sequence.
    """
    from .base import Transformation
    from .checkers import _matrix_dims
    from .concrete import (
        Affine,
        CartesianField,
        CoordinatesField,
        DisplacementField,
        Linear,
        Permutation,
        Scaling,
        Translation,
    )
    from .inverse import Inverse
    from .sequence import Sequence

    if not isinstance(t, Transformation):
        return None, None
    ni: tx.Optional[int] = None
    no: tx.Optional[int] = None
    if isinstance(t, Inverse):
        # Before the concrete families: a typed inverse such as
        # `InverseScaling` is also a `Scaling`.
        if t.forward is not None:
            no, ni = axis_counts(t.forward)
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
            field = t.field
            if field is not None:
                no = int(field.shape[-1])
                if isinstance(t, DisplacementField):
                    ni = no
                else:
                    ni = len(field.shape) - 1
    if ni is None:
        ni = AxisList.of(getattr(t, "_input", None)).ndim
    if no is None:
        no = AxisList.of(getattr(t, "_output", None)).ndim
    return ni, no


def _sequence_ends(
    members: tx.List["Transformation"],
) -> tx.Tuple[tx.Optional[int], tx.Optional[int]]:
    # The input count of the first member that states one, and the output
    # count of the last. A member that states neither count is read past
    # only when it is known to preserve the dimension (an identity, a
    # subspace, a matrix-less affine, a field); anything else stops the
    # read, so a count is never taken across a member that may change it.
    ni = _first_count(members, 0)
    no = _first_count(members[::-1], 1)
    return ni, no


def _first_count(
    members: tx.List["Transformation"], side: int
) -> tx.Optional[int]:
    from .factor import _element_ndim
    from .sequence import Sequence

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
