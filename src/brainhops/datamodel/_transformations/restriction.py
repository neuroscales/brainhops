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

from brainhops.datamodel._sugar import get_axes
from brainhops.datamodel.axes import Axis, AxisQuery
from brainhops.datamodel.systems import AxisSequence, CoordinateSystem
from brainhops.errors import RestrictionError

from .base import Transformation
from .compute.utils import axis_counts, axis_list
from .concrete import Permutation
from .meta import Projection, SubspaceTransformation
from .sequence import Sequence

AxisReference: tx.TypeAlias = tx.Union[int, AxisQuery]
"""One reference to axes: a position, a name or any other axis query."""

AxisReferences: tx.TypeAlias = tx.Union[AxisQuery, tx.Iterable[AxisReference]]
"""The axes of one side of a restriction.

The value is a list whose items are positions, names or any other query
of [`AxisQuery`][brainhops.datamodel.axes.AxisQuery], or a single query
or name instead of a list.
"""

InputValues: tx.TypeAlias = tx.Mapping[AxisReference, float]
"""The values of some removed inputs, keyed by references to them."""

Side: tx.TypeAlias = tx.Literal["input", "output"]
"""The side of a transformation on which axes are kept or removed."""


class KeptAxes(tx.NamedTuple):
    """The axes that a restriction keeps on one side.

    The positions are non-negative and sorted when the number of axes is
    known, and they are given as the references designate them
    otherwise.
    """

    count: tx.Optional[int]
    """Number of axes on this side of the transformation, or None."""

    positions: tx.List[int]
    """Positions of the kept axes."""

    @property
    def removed_positions(self) -> tx.List[int]:
        """Positions of the removed axes, which needs `count`."""
        count = self.count or 0
        return [p for p in range(count) if p not in self.positions]


class ResolvedRestriction(tx.NamedTuple):
    """The arguments of a restriction, read as positions.

    A restriction keeps some inputs and some outputs of a
    transformation, and gives a value to some of the removed inputs.
    """

    inputs: KeptAxes
    """The inputs that the restriction keeps."""

    outputs: KeptAxes
    """The outputs that the restriction keeps."""

    removed_input_values: tx.Dict[int, float]
    """Value of each removed input that `at` names, by position."""


def restrict(
    t: Transformation,
    input_axes: AxisReferences,
    output_axes: AxisReferences,
    at: tx.Optional[InputValues] = None,
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
        is designated twice, or if `at` gives a value to an input that
        is kept.
    TypeError
        If a reference is neither a position nor a query, or if `at` is
        not a mapping of real values.
    """
    restriction = resolve_restriction(t, input_axes, output_axes, at)
    if restriction.inputs.count is None or restriction.outputs.count is None:
        inner = _subspace_restricted_to_its_axes(t, restriction)
        if inner is not None:
            return inner
    require_axis_counts(t, restriction)
    return build_restriction(t, restriction, t.input, t.output)


def resolve_restriction(
    t: Transformation,
    input_axes: AxisReferences,
    output_axes: AxisReferences,
    at: tx.Optional[InputValues] = None,
) -> ResolvedRestriction:
    """Read the arguments of a restriction of `t` as positions.

    The references are resolved against the systems of `t` and the
    numbers of axes that `t` states. The function raises as
    [`restrict`][] does, except for an unknown number of axes, which
    [`require_axis_counts`][] refuses.
    """
    n_in, n_out = axis_counts(t)
    inputs = KeptAxes(n_in, _resolve_axes(t, "input", n_in, input_axes))
    outputs = KeptAxes(n_out, _resolve_axes(t, "output", n_out, output_axes))
    values = _resolve_input_values(t, inputs, at)
    return ResolvedRestriction(inputs, outputs, values)


def require_axis_counts(
    t: Transformation, restriction: ResolvedRestriction
) -> None:
    """Raise when the number of axes of `t` on a side is unknown.

    Raises
    ------
    RestrictionError
        If `restriction` does not know a number of axes.
    """
    for side, kept in (
        ("input", restriction.inputs),
        ("output", restriction.outputs),
    ):
        if kept.count is None:
            raise RestrictionError(
                f"Cannot restrict this {type(t).__name__}, because the "
                f"number of its {side} axes is unknown. Give it a closed "
                f"{side} system, so that the axes that the restriction "
                f"removes are known."
            )


def build_restriction(
    t: Transformation,
    restriction: ResolvedRestriction,
    input: tx.Optional[CoordinateSystem],
    output: tx.Optional[CoordinateSystem],
) -> Transformation:
    """Return the lazy composition that restricts `t` to some axes.

    `input` and `output` are the systems of the full space on each side
    of `t`, which the projections take on the side that faces `t` and
    restrict on the other side. The numbers of axes of `restriction`
    must be known. `t` itself is returned when the restriction keeps
    every axis.
    """
    inputs, outputs = restriction.inputs, restriction.outputs
    assert inputs.count is not None and outputs.count is not None
    removed_inputs = inputs.removed_positions
    removed_outputs = outputs.removed_positions
    values = restriction.removed_input_values
    chain: tx.List[Transformation] = []
    if removed_inputs:
        chain.append(
            Projection(
                created=removed_inputs,
                at=[values.get(p, math.nan) for p in removed_inputs],
                input=_system_of_kept_axes(input, inputs),
                output=input,
            )
        )
    chain.append(t)
    if removed_outputs:
        chain.append(
            Projection(
                dropped=removed_outputs,
                input=output,
                output=_system_of_kept_axes(output, outputs),
            )
        )
    if len(chain) == 1:
        return t
    return Sequence(chain)


def _subspace_restricted_to_its_axes(
    t: Transformation, restriction: ResolvedRestriction
) -> tx.Optional[Transformation]:
    """Return the restriction of a subspace to the axes that it acts on.

    A subspace often states no number of axes, so the projections cannot
    be built. The subspace passes its other axes through, however, and
    the restriction removes them on both sides, so the result is the
    transformation that the subspace wraps, whatever the number of axes
    is. The kept axes stay in the order of the full space, so when the
    subspace lists its axes in another order, a permutation before the
    wrapped transformation and another after it reorder them. None is
    returned when `t` is not such a subspace or when the restriction
    does not keep exactly the axes that it acts on.
    """
    inner = getattr(t, "transformation", None)
    if not isinstance(t, SubspaceTransformation) or inner is None:
        return None
    acted_in = axis_list(t.input_axes)
    acted_out = axis_list(t.output_axes)
    kept_in = restriction.inputs.positions
    kept_out = restriction.outputs.positions
    if kept_in != sorted(acted_in) or kept_out != sorted(acted_out):
        return None
    # Input `j` of the wrapped transformation reads the full axis
    # `acted_in[j]`, which is kept at the rank of that axis among the
    # kept inputs. Output `r` of the result is the full axis
    # `kept_out[r]`, which output `acted_out.index(kept_out[r])` of the
    # wrapped transformation writes.
    before = [kept_in.index(a) for a in acted_in]
    after = [acted_out.index(a) for a in kept_out]
    chain: tx.List[Transformation] = []
    if before != sorted(before):
        chain.append(Permutation(permutation=before))
    chain.append(inner)
    if after != sorted(after):
        chain.append(Permutation(permutation=after))
    return inner if len(chain) == 1 else Sequence(chain)


def _resolve_axes(
    t: Transformation,
    side: Side,
    count: tx.Optional[int],
    refs: AxisReferences,
) -> tx.List[int]:
    """Return the sorted positions that the references designate.

    The references are resolved against the system of `t` on `side` by
    [`AxisSequence.positions`][brainhops.datamodel.axes.AxisSequence.positions],
    so that a query matches only the explicit axes of an open system. A
    position after the `...` of an open system counts from the end.
    When `t` states its number of axes `count`, every position is made
    non-negative, and otherwise the positions are returned as they are
    designated. A restriction keeps a set of axes, so the order of the
    references does not matter and the positions are sorted, but an
    axis must not be designated twice.
    """
    kind = type(t).__name__
    system = getattr(t, side)
    axes = get_axes(system)
    if isinstance(refs, (str, Mapping, Axis)) or callable(refs):
        # A name, a dictionary of fields, a template axis, an axis class
        # or a function stands for a list of one reference.
        refs = [refs]
    if isinstance(refs, bytes) or not isinstance(refs, Iterable):
        raise TypeError(
            f"The {side} axes of a restriction are a list of positions and "
            f"queries, or a single query, not a {type(refs).__name__}."
        )
    positions: tx.List[int] = []
    for ref in refs:
        for position in _positions_of_reference(
            axes, ref, kind, side, system is None
        ):
            if count is not None:
                _check_in_range(position, count, side)
                position = position % count
            positions.append(position)
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"The {side} axes of a restriction designate the same axis "
            f"more than once: {positions}."
        )
    return sorted(positions)


def _positions_of_reference(
    axes: AxisSequence,
    ref: AxisReference,
    kind: str,
    side: Side,
    missing: bool,
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
        positions = axes.positions([ref])
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


def _check_in_range(position: int, count: int, side: Side) -> None:
    """Raise when a position does not lie among `count` axes.

    Raises
    ------
    IndexError
        If `position` lies outside `[-count, count)`.
    """
    if not -count <= position < count:
        raise IndexError(
            f"The {side} axis position {position} is out of range for a "
            f"transformation of {count} {side} axes."
        )


def _resolve_input_values(
    t: Transformation,
    inputs: KeptAxes,
    at: tx.Optional[InputValues],
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
        for position in _resolve_axes(t, "input", inputs.count, [key]):
            if position in inputs.positions:
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


def _system_of_kept_axes(
    system: tx.Optional[CoordinateSystem], kept: KeptAxes
) -> tx.Optional[CoordinateSystem]:
    """Return the system of the kept axes, or None when there is none.

    An open system is closed to the number of axes of the transformation
    first, so that a position after its `...` designates the right axis.
    """
    if system is None:
        return None
    assert kept.count is not None
    return system.expand(kept.count).restrict(kept.positions)
