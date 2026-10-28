"""Restriction of a transformation to some of its axes.

[`Transformation.restrict`][brainhops.datamodel.transformations.Transformation.restrict]
reads a transformation from a block of its input axes to a block of its
output axes. This module turns the references that the caller gives into
positions and builds the lazy composition that describes the result,
with a [`Projection`][] before the transformation that creates the
removed inputs and a projection after it that drops the removed outputs.
Nothing is computed here: the pruning of
[`simplify`][brainhops.datamodel._transformations.compute.simplify.simplify]
and the matrix products of `compute` reduce the composition later.
"""

import math
from collections.abc import Iterable, Mapping
from numbers import Real

import typing_extensions as tx

from brainhops.datamodel._axes.list import AxisQuery, _is_query
from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.errors import RestrictionError

from .base import Transformation
from .compute.utils import axis_counts, axis_list
from .meta import Projection, SubspaceTransformation
from .sequence import Sequence

AxisRefs: tx.TypeAlias = tx.Union[
    AxisQuery, tx.Iterable[tx.Union[int, AxisQuery]]
]
"""The axes of one side of a restriction.

The value is a list whose items are positions, names or any other query
of [`AxisQuery`][brainhops.datamodel.axes.AxisQuery], or a single query
or name instead of a list.
"""


def restrict(
    t: Transformation,
    input_axes: AxisRefs,
    output_axes: AxisRefs,
    at: tx.Optional[tx.Mapping[tx.Any, float]] = None,
) -> Transformation:
    """Return `t` read from some of its inputs to some of its outputs.

    The result is the sequence of the projection that creates the
    inputs outside `input_axes`, of `t`, and of the projection that
    drops the outputs outside `output_axes`. A projection that would
    change nothing is left out, so that `t` itself is returned when both
    blocks hold every axis. A subspace that does not state its numbers
    of axes, restricted to the axes that it acts on, gives the
    transformation that it wraps. The method of the same name documents
    the arguments.

    Raises
    ------
    RestrictionError
        If the number of axes of `t` on a side is unknown.
    KeyError
        If no axis has a name that a reference gives.
    IndexError
        If a position lies outside the axes of `t`.
    ValueError
        If a reference matches nothing, if a name is shared, if an axis
        is designated twice, if the axes are not designated in their
        order, or if `at` gives a value to an input that is kept.
    TypeError
        If a reference is neither a position nor a query, or if `at` is
        not a mapping of real values.
    """
    n_in, n_out = axis_counts(t)
    kept_in = _resolve(t, "input", n_in, input_axes)
    kept_out = _resolve(t, "output", n_out, output_axes)
    values = _resolve_at(t, n_in, kept_in, at)

    inner = getattr(t, "transformation", None)
    if (
        (n_in is None or n_out is None)
        and isinstance(t, SubspaceTransformation)
        and inner is not None
        and kept_in == axis_list(t.input_axes)
        and kept_out == axis_list(t.output_axes)
    ):
        # A subspace often states no number of axes, so the projections
        # cannot be built. The subspace passes its other axes through,
        # however, and the restriction removes them on both sides, so
        # the result is the transformation that the subspace wraps,
        # whatever the number of axes is.
        return inner
    for side, ndim in (("input", n_in), ("output", n_out)):
        if ndim is None:
            raise RestrictionError(
                f"Cannot restrict this {type(t).__name__}, because the "
                f"number of its {side} axes is unknown. Give it a closed "
                f"{side} system, so that the axes that the restriction "
                f"removes are known."
            )
    assert n_in is not None and n_out is not None

    removed_in = [p for p in range(n_in) if p not in kept_in]
    removed_out = [p for p in range(n_out) if p not in kept_out]
    chain: tx.List[Transformation] = []
    if removed_in:
        chain.append(
            Projection(
                created=removed_in,
                at=[values.get(p, math.nan) for p in removed_in],
                input=_restricted_system(t.input, n_in, kept_in),
                output=t.input,
            )
        )
    chain.append(t)
    if removed_out:
        chain.append(
            Projection(
                dropped=removed_out,
                input=t.output,
                output=_restricted_system(t.output, n_out, kept_out),
            )
        )
    if len(chain) == 1:
        return t
    return Sequence(chain)


def _resolve(
    t: Transformation,
    side: str,
    ndim: tx.Optional[int],
    refs: tx.Any,
) -> tx.List[int]:
    """Return the positions that the references designate, in order.

    The references are resolved against the system of `t` on `side`, as
    [`AxisSequence.restrict`][brainhops.datamodel.axes.AxisSequence.restrict]
    resolves them, so that a query matches only the explicit axes of an
    open system. A position after the `...` of an open system counts
    from the end. When `t` states its number of axes `ndim`, every
    position is made non-negative, and otherwise the positions are
    returned as they are designated.
    """
    kind = type(t).__name__
    system = getattr(t, side)
    axes = get_axes(system)
    if _is_query(refs) or isinstance(refs, str):
        refs = [refs]
    if isinstance(refs, bytes) or not isinstance(refs, Iterable):
        raise TypeError(
            f"The {side} axes of a restriction are a list of positions and "
            f"queries, or a single query, not a {type(refs).__name__}."
        )
    positions: tx.List[int] = []
    for ref in refs:
        for position in _designate(axes, ref, kind, side, system is None):
            if ndim is not None:
                position = _normalise(position, ndim, side)
            positions.append(position)
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"The {side} axes of a restriction designate the same axis "
            f"more than once: {positions}."
        )
    # Without a number of axes, a negative position cannot be compared
    # with a non-negative one, and such a restriction raises later.
    comparable = ndim is not None or all(p >= 0 for p in positions)
    if comparable and positions != sorted(positions):
        raise ValueError(
            f"A restriction keeps the {side} axes in the order that they "
            f"have in the transformation, but the references designate "
            f"them in another order, at the positions {positions}. "
            f"Compose the result with a Permutation to reorder the axes."
        )
    return positions


def _designate(
    axes: tx.Any, ref: tx.Any, kind: str, side: str, missing: bool
) -> tx.List[int]:
    """Return the positions of the axes that one reference designates.

    A name must designate exactly one axis, and any other query at least
    one, so that a reference never selects nothing in silence.
    """
    if missing:
        where = (
            f"The {kind} has no {side} system, so its {side} axes can only "
            f"be designated by position."
        )
    else:
        where = f"Its {side} axes are named {list(axes.names)}."
    try:
        positions = axes._positions([ref])
    except KeyError:
        raise KeyError(
            f"No {side} axis of the {kind} is named {ref!r}. {where}"
        ) from None
    if not positions:
        label = ref.__name__ if isinstance(ref, type) else repr(ref)
        raise ValueError(
            f"The query {label} matches no {side} axis of the {kind}. {where}"
        )
    return positions


def _normalise(position: int, ndim: int, side: str) -> int:
    """Return the non-negative form of a position among `ndim` axes."""
    if not -ndim <= position < ndim:
        raise IndexError(
            f"The {side} axis position {position} is out of range for a "
            f"transformation of {ndim} {side} axes."
        )
    return position + ndim if position < 0 else position


def _resolve_at(
    t: Transformation,
    ndim: tx.Optional[int],
    kept: tx.List[int],
    at: tx.Any,
) -> tx.Dict[int, float]:
    """Return the value of each removed input that `at` names.

    Each key is a reference to input axes, resolved as one item of the
    input axes, and the value is given to every axis that the key
    designates.
    """
    if at is None:
        return {}
    if not isinstance(at, Mapping):
        raise TypeError(
            f"`at` maps removed input axes to their values, so it is a "
            f"mapping, not a {type(at).__name__}."
        )
    values: tx.Dict[int, float] = {}
    for key, value in at.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"`at` gives the input axis {key!r} the value {value!r}, "
                f"which is not a real number."
            )
        for position in _resolve(t, "input", ndim, [key]):
            if position in kept:
                raise ValueError(
                    f"`at` gives a value to the input axis {key!r}, which "
                    f"the restriction keeps. Only a removed input takes a "
                    f"value."
                )
            if position in values:
                raise ValueError(
                    f"`at` gives more than one value to the input axis at "
                    f"position {position}."
                )
            values[position] = float(value)
    return values


def _restricted_system(
    system: tx.Optional[CoordinateSystem], ndim: int, kept: tx.List[int]
) -> tx.Optional[CoordinateSystem]:
    """Return the system of the kept axes, or None when there is none.

    An open system is closed to the `ndim` axes of the transformation
    first, so that a position after its `...` designates the right axis.
    """
    if system is None:
        return None
    return system.expand(ndim).restrict(kept)
