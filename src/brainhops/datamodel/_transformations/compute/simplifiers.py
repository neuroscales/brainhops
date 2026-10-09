"""Simplifier rules, registered with `simplify.simplifier`.

Simplification rewrites a transformation, or a chain of transformations,
into a cheaper equivalent without computing anything new. The `simplify`
module provides the machinery, a single function that accepts either one
transformation or two consecutive ones. This module holds the rules, which
are registered when it is imported.

A rule that takes one transformation is a downcast. It rewrites the
transformation as the cheapest type that the transformation is known to
belong to, so that an Affine with a diagonal matrix becomes a Scaling. The
resolved [`SimplifyPolicy`][] decides how closely the parameters may be
inspected: an analytic policy reads only the structure, a numeric policy
also reads the values, and a policy of none leaves the transformation
alone. A rule that takes two transformations is a collapse. It replaces
two consecutive transformations by one when doing so costs nothing, for
example when a transformation is next to its lazy inverse. As in a
[`Sequence`][], `first` is applied before `second`, and a rule returns None
when it does not apply.

A simplifier never reads values that the policy forbids. It never
materializes a lazy inverse, because that is the job of
[`Inverse.compute`][], which the composition mode controls. It also never
makes a sequence longer or reconciles the coordinate systems of two
neighbours, which is the job of composition. The note above
`_collapse_pass` explains why declining such pairs loses nothing.
"""

import typing_extensions as tx
from bagof.magic import replace

from brainhops.datamodel import kinds
from brainhops.errors import ConversionError, DomainError

from ..base import Transformation
from ..concrete import (
    Affine,
    CartesianField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Translation,
)
from ..inverse import Inverse
from ..meta import (
    Bijection,
    Projection,
    SubspaceTransformation,
    _same_axes,
)
from ..multiscale import MultiscaleField
from ..operators import Sqrt
from ..sequence import Sequence, _unnest
from .check import is_kind
from .simplify import SimplifyPolicy, SimplifyTable, simplifier
from .simplify import simplify as _simplify
from .utils import with_endpoints as _with_endpoints

NONE = SimplifyPolicy.none
NUMERIC = SimplifyPolicy.numeric


# ======================================================================
#
#                    O N E   I N ,   O N E   O U T
#
# ======================================================================

# The downcast ladder pairs each kind, that is, each set of transformations
# such as the diagonal maps, with the class that represents it, from the
# cheapest to the most general. A transformation becomes the first rung whose
# kind it belongs to, unless it is already an instance of the class of that
# rung. For this reason, a Rotation is never widened to a Linear.
_LADDER: tx.Tuple[tx.Tuple[kinds.Kind, tx.Type[Transformation]], ...] = (
    (kinds.Identity, Identity),
    (kinds.Translation, Translation),
    (kinds.Diagonal, Scaling),
    (kinds.Permutation, Permutation),
    (kinds.SpecialOrthogonal, Rotation),
    (kinds.Linear, Linear),
    (kinds.Affine, Affine),
)


@simplifier
def _(t: Transformation, policy: SimplifyTable) -> Transformation:
    """Downcast a transformation to the cheapest type it belongs to."""
    resolved = policy.resolve(t)
    if resolved is NONE:
        return t
    compute = resolved is NUMERIC
    for node, cls in _LADDER:
        if not is_kind(t, node, compute):
            continue
        if isinstance(t, cls):
            # Rebuilding the transformation would return a new object, and a
            # lazy inverse that refers to the original object through
            # `forward` would then no longer cancel with it.
            return t
        try:
            return t.to(cls)
        except ConversionError:
            # The transformation belongs to the kind but cannot be expressed
            # in its class, because no converter exists or the conversion
            # would lose information. Nothing cheaper is available.
            return t
    return t


@simplifier
def _(t: CartesianField, policy: SimplifyTable) -> Transformation:
    """Leave a grid alone.

    A [`CartesianField`][] is the identity map over its grid, but it is also
    the sampling domain for resampling, which an Identity cannot carry. A grid
    is only removed when a neighbour on each side overwrites its coordinates
    (see `_droppable_grid`).
    """
    return t


@simplifier
def _(t: Inverse, policy: SimplifyTable) -> Transformation:
    """Simplify the forward transformation of an inverse without resolving it.

    Resolving the inverse is the job of [`Inverse.compute`][]. When the
    forward transformation simplifies to the identity, the result is an
    identity. Otherwise, the result is the `inverse()` of the simplified
    forward transformation, which is the lazy inverse typed for its new
    class.
    """
    if policy.resolve(t) is NONE:
        return t
    forward = t.forward
    if forward is None:
        return Identity(input=t.input, output=t.output)
    simplified = _simplify(forward, policy=policy)
    if simplified is forward:
        return t
    if isinstance(simplified, Identity):
        return Identity(input=t.input, output=t.output)
    return _with_endpoints(simplified.inverse(), t)


@simplifier
def _(t: Sqrt, policy: SimplifyTable) -> Transformation:
    """Simplify the transformation under a square root without resolving it.

    Resolving the square root is the job of [`Sqrt.compute`][]. When the
    forward transformation simplifies to the identity, the result is an
    identity. Otherwise, the result is the `sqrt()` of the simplified forward
    transformation. If the simplified transformation refuses to take a square
    root, the wrapper is kept, because a simplifier never raises.
    """
    if policy.resolve(t) is NONE:
        return t
    forward = t.forward
    simplified = _simplify(forward, policy=policy)
    if simplified is forward:
        return t
    if isinstance(simplified, Identity):
        return Identity(input=t.input, output=t.output)
    try:
        rebuilt = simplified.sqrt()
    except (DomainError, NotImplementedError):
        return t
    return _with_endpoints(rebuilt, t)


@simplifier
def _(t: SubspaceTransformation, policy: SimplifyTable) -> Transformation:
    """Simplify the transformation that a subspace embeds.

    A subspace over the same input and output axes whose inner transformation
    is the identity is the identity on the full space. A subspace whose input
    and output axes differ still permutes the coordinates, so it is kept even
    when its inner transformation is the identity.
    """
    resolved = policy.resolve(t)
    if resolved is NONE:
        return t
    same = _same_axes(t)
    inner = t.transformation
    if inner is None:
        return Identity(input=t.input, output=t.output) if same else t
    simplified = _simplify(inner, policy=policy)
    if same and simplified.is_identity(resolved is NUMERIC):
        return Identity(input=t.input, output=t.output)
    if simplified is inner:
        return t
    return t.to(transformation=simplified)


@simplifier
def _(t: Projection, policy: SimplifyTable) -> Transformation:
    """Reduce a projection that drops and creates no axis to the identity."""
    if policy.resolve(t) is NONE:
        return t
    dropped = 0 if t.dropped is None else len(t.dropped)
    created = 0 if t.created is None else len(t.created)
    if dropped == created == 0:
        return Identity(input=t.input, output=t.output)
    return t


@simplifier
def _(t: Bijection, policy: SimplifyTable) -> Transformation:
    """Simplify both directions of an explicit bijection."""
    if policy.resolve(t) is NONE:
        return t
    forward = t.forward
    backward = t.backward
    if forward is not None:
        forward = _simplify(forward, policy=policy)
    if backward is not None:
        backward = _simplify(backward, policy=policy)
    if forward is t.forward and backward is t.backward:
        # Keeping the same object lets an adjacent Inverse of this bijection
        # cancel.
        return t
    return t.to(forward=forward, backward=backward)


@simplifier
def _(t: MultiscaleField, policy: SimplifyTable) -> Transformation:
    """Simplify a multiscale field through its finest scale.

    The elements of a multiscale field are derived from its scales and cannot
    be rebuilt one by one. The field is therefore simplified in the form of
    the scale that it presents, as in [`MultiscaleField.compute`][].
    """
    if policy.resolve(t) is NONE:
        return t
    return _simplify(t._as_sequence(), policy=policy)


@simplifier
def _(t: Sequence, policy: SimplifyTable) -> Transformation:
    """Simplify a sequence by downcasting and collapsing its elements.

    Nested sequences are flattened first. Each of the two passes can create
    new opportunities for the other, so both run in rounds until a round
    changes nothing. The collapse runs first in each round, so that a
    transformation cancels with its lazy inverse before a downcast can
    rebuild either side and break the link between them. Nothing is composed
    or bridged, so the sequence can only get shorter.
    """
    if policy.is_noop():
        return t
    leaves = _unnest(t.transformations)
    while True:
        collapsed = _collapse_pass(leaves, policy)
        simplified = _downcast_pass(collapsed, policy)
        if len(simplified) == len(leaves) and all(
            a is b for a, b in zip(simplified, leaves)
        ):
            break
        leaves = simplified
    return _rebuild(t, leaves)


# ======================================================================
#
#                    T W O   I N ,   O N E   O U T
#
# ======================================================================

# An identity next to another transformation contributes nothing except an
# endpoint, that is, an input or output coordinate system, so it is dropped.
# The other transformation takes over that endpoint, but it is rebuilt only
# when the endpoint is set and differs from its own. A new object would break
# the `forward is` link by which a lazy inverse cancels. The rebuild uses
# `replace` and never `compute()`, which would resolve a lazy inverse.


@simplifier
def _(
    first: Identity, second: Identity, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    return Identity(
        input=first.input or second.input,
        output=second.output or first.output,
    )


@simplifier
def _(
    first: Identity, second: Transformation, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if first.input is None or first.input == second.input:
        return second
    return replace(second, input=first.input)


@simplifier
def _(
    first: Transformation, second: Identity, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if second.output is None or second.output == first.output:
        return first
    return replace(first, output=second.output)


# A transformation next to its own lazy inverse cancels with it. An Inverse
# refers to the transformation that it undoes through `forward`, so the test
# is an O(1) check of object identity that materializes neither side.


@simplifier
def _(
    first: Transformation, second: Inverse, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not _cancels(first, second):
        return None
    return Identity(input=first.input, output=second.output)


@simplifier
def _(
    first: Inverse, second: Transformation, policy: SimplifyTable
) -> tx.Optional[Transformation]:
    if not _cancels(first, second):
        return None
    return Identity(input=first.input, output=second.output)


@simplifier
def _(
    first: SubspaceTransformation,
    second: SubspaceTransformation,
    policy: SimplifyTable,
) -> tx.Optional[Transformation]:
    """Collapse a subspace-wrapped transformation next to its inverse.

    The pair collapses to the identity when three conditions hold. The axes
    written by `first` must be those read by `second`, the pair as a whole
    must not reindex any axis, and the inner transformations must cancel. The
    inner transformations cancel when one is the lazy inverse of the other,
    when both are identities, or when they are sequences whose elements
    cancel once chained together. As a result, a wrapped field cancels with
    its wrapped inverse instead of being resampled.
    """
    if (
        first.output_axes is None
        or second.input_axes is None
        or list(first.output_axes) != list(second.input_axes)
    ):
        return None
    # A pair whose net effect reindexes the axes is deliberately not
    # collapsed here. Such a pair is not the identity, and the composers fold
    # it into a single reindexing subspace transformation.
    same_axes = (first.input_axes is None) == (
        second.output_axes is None
    ) and (
        first.input_axes is None
        or list(first.input_axes) == list(second.output_axes)
    )
    if not same_axes:
        return None
    inner_first = first.transformation
    inner_second = second.transformation
    if _cancels(inner_first, inner_second) or _chain_cancels(
        inner_first, inner_second, policy
    ):
        return Identity(input=first.input, output=second.output)
    # Only the structure is checked, because this rule runs for every pair
    # of adjacent subspaces in each round, and a numeric check would scan a
    # whole field each time. Under a numeric policy, a field that is
    # numerically the identity has already been downcast to an Identity.
    both_identity = (
        inner_first is None or inner_first.is_identity(False)
    ) and (inner_second is None or inner_second.is_identity(False))
    if not both_identity:
        return None
    return Identity(input=first.input, output=second.output)


# ======================================================================
#
#                             H E L P E R S
#
# ======================================================================


def _cancels(first: Transformation, second: Transformation) -> bool:
    """Return whether `[first, second]` cancels to the identity for free.

    The pair cancels when either transformation is the lazy inverse of the
    other, whether that inverse is a typed one such as InverseScaling or a
    generic Inverse. The check is an O(1) test of object identity.
    """
    if isinstance(second, Inverse) and second.forward is first:
        return True
    if isinstance(first, Inverse) and first.forward is second:
        return True
    return False


def _chain_cancels(
    first: tx.Optional[Transformation],
    second: tx.Optional[Transformation],
    policy: SimplifyTable,
) -> bool:
    """Return whether `[first, second]` cancels when either is a sequence.

    The inverse of a sequence is the sequence of the inverses in reverse
    order, which [`_cancels`][] cannot detect. When the two transformations
    are chained into one sequence, that sequence simplifies to the identity
    if each element meets its own lazy inverse.
    """
    if first is None or second is None:
        return False
    if not (isinstance(first, Sequence) or isinstance(second, Sequence)):
        return False
    chained = _simplify(
        Sequence(transformations=[first, second]), policy=policy
    )
    return isinstance(chained, Identity)


def _droppable_grid(t: Transformation, policy: SimplifyTable) -> bool:
    # Decide whether `t` is a grid that its two neighbours make redundant. A
    # grid is the identity map over its coordinates, so `A @ grid @ B` equals
    # `A @ B` when both neighbours overwrite the coordinates. Other fields
    # carry values that the neighbours do not reproduce.
    if not isinstance(t, CartesianField):
        return False
    if policy.resolve(t) is NONE:
        return False
    return t.is_identity(compute=True)


# Simplification never reconciles the coordinate systems of two neighbours,
# either by bridging or by embedding. A genuine cancelling pair never needs
# it, because an Inverse takes the systems of its forward transformation.
# Inside compute, the bridge step (see `_compute_sequence`) has already run
# when simplification starts. The bridge step must run first, because a
# bridge may read the extent of an adjacent grid that the sweep below drops.
# Since `adapt` also embeds operands into the wider axis space, the operands
# are already comparable here. A standalone t.simplify() bridges nothing, so
# it declines a pair that only an embedding would make comparable.


def _collapse_pass(
    leaves: tx.List[Transformation], policy: SimplifyTable
) -> tx.List[Transformation]:
    # Collapse adjacent pairs in one sweep. The sweep uses a stack, so that
    # each merged result is tried again against the neighbour that the merge
    # exposes.
    stack: tx.List[Transformation] = []
    for nxt in leaves:
        while stack:
            # An interior grid is dropped here rather than by a pair rule,
            # because the condition depends on its position. A leading or
            # trailing grid is the sampling domain and stays. The test
            # len(stack) >= 2 means that something precedes the grid, and
            # nxt follows it.
            if len(stack) >= 2 and _droppable_grid(stack[-1], policy):
                stack.pop()
                continue
            merged = _simplify(stack[-1], nxt, policy=policy)
            if merged is None:
                break
            stack.pop()
            nxt = merged
        stack.append(nxt)
    return stack


def _downcast_pass(
    leaves: tx.List[Transformation], policy: SimplifyTable
) -> tx.List[Transformation]:
    # Downcast each element in one sweep. A cache keyed by `id` skips the
    # elements that were already seen. The cache lives only for this sweep,
    # during which `leaves` keeps every element alive, so an `id` cannot be
    # reused by another object.
    cache: tx.Dict[int, Transformation] = {}
    out = []
    for t in leaves:
        key = id(t)
        if key not in cache:
            cache[key] = _simplify(t, policy=policy)
        out.append(cache[key])
    return out


def _rebuild(seq: Sequence, leaves: tx.List[Transformation]) -> Transformation:
    if not leaves:
        return Identity(input=seq.input, output=seq.output)
    if len(leaves) == 1:
        return _with_endpoints(leaves[0], seq)
    if len(leaves) == len(seq.transformations or ()) and all(
        a is b for a, b in zip(leaves, seq.transformations)
    ):
        return seq
    return seq.to(transformations=leaves)
