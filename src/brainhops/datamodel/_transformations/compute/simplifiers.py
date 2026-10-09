"""Simplifier rules, registered with `simplify.simplifier`.

The machinery lives in `simplify`, as one dispatched function with two
arities, and the rules live here and are registered at import.

A rule with one input is a downcast: it rewrites a transformation as the
cheapest type that it is established to belong to, so that an Affine with a
diagonal matrix becomes a Scaling. The resolved [`SimplifyPolicy`][]
decides how hard a parameter may be inspected: an analytic policy reads
structure only, a numeric policy reads values, and a policy of none leaves
the transformation alone. A rule with two inputs is a collapse: it replaces
two consecutive transformations by one when this costs nothing, such as a
transformation next to its lazy inverse. `first` is applied before
`second`, as in a [`Sequence`][], and a rule returns None to decline.

A simplifier never reads values that the policy forbids, never
materializes a lazy inverse (that is the job of [`Inverse.compute`][],
gated by the composition mode), and never lengthens a sequence or
reconciles a boundary, which is the job of composition. The note above
`_collapse_pass` explains why nothing is lost.
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

# Cheapest first. A transformation becomes the first rung whose set it belongs
# to, unless it is already an instance of that rung's class, so a Rotation is
# never widened to a Linear.
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
            # Rebuilding would return a new object and break the `forward is`
            # link by which a lazy inverse cancels.
            return t
        try:
            return t.to(cls)
        except ConversionError:
            # The set has no representation in its class (no converter, or a
            # lossy one), so nothing cheaper exists.
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
    """Simplify the forward transformation of an inverse, unresolved.

    Resolving the inverse is the job of [`Inverse.compute`][]. The inverse of
    an identity is an identity; otherwise the result is the `inverse()` of the
    simplified forward transformation, a typed wrapper of its new family.
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
    """Simplify the forward transformation of a square root, unresolved.

    Resolving the square root is the job of [`Sqrt.compute`][]. The square root
    of an identity is an identity; otherwise the result is the `sqrt()` of the
    simplified forward transformation. If that square root is refused, the
    wrapper is kept, because a simplifier never raises.
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
    is the identity is the identity on the full space. A reindexing subspace
    still permutes the coordinates, so it is kept.
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
    be rebuilt one by one, so the field is simplified as the scale it
    presents, as in [`MultiscaleField.compute`][].
    """
    if policy.resolve(t) is NONE:
        return t
    return _simplify(t._as_sequence(), policy=policy)


@simplifier
def _(t: Sequence, policy: SimplifyTable) -> Transformation:
    """Simplify a sequence by downcasting its leaves and collapsing free pairs.

    Each pass can expose work for the other, so both run until a fixpoint. The
    collapse runs first in each round, so that a transformation cancels with
    its lazy inverse before a downcast can rebuild either side and break the
    link. Nothing is composed or bridged, so the sequence only gets shorter.
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

# An identity contributes only an endpoint. The survivor is rebuilt, with
# `replace` and never compute(), only when that endpoint is set and different,
# because a new object breaks the `forward is` link by which a lazy inverse
# cancels.


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


# An Inverse names its forward transformation, so an O(1) identity check
# suffices.


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

    The pair annihilates when the axes written by `first` are those read by
    `second`, the net map introduces no reindex, and the inner transformations
    cancel: one is the lazy inverse of the other, both are identities, or they
    are sequences whose chained elements cancel. A wrapped field then cancels
    with its wrapped inverse instead of being resampled.
    """
    if (
        first.output_axes is None
        or second.input_axes is None
        or list(first.output_axes) != list(second.input_axes)
    ):
        return None
    # A reindexing inverse pair is deliberately not annihilated: it is not a
    # bare identity, and the composers fold it into one reindexing subspace
    # transformation.
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
    # Only structure is checked: this rule runs for every adjacent subspace
    # pair in each round, and a numeric check would scan a whole field each
    # time. A numerically zero field has already been downcast to an Identity.
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
    """Whether `[first, second]` cancels to the identity for free.

    The pair cancels when either transformation is the lazy inverse of the
    other, typed or generic. The check is an O(1) identity test.
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
    """Whether `[first, second]` cancels when either is a sequence.

    The inverse of a sequence is the sequence of the inverses in reverse
    order, which [`_cancels`][] cannot detect. Chained together, the two
    simplify to the identity when each element meets its own lazy inverse.
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
    # A grid is the identity map over its coordinates, so A @ grid @ B equals A
    # @ B when both neighbours overwrite the coordinates. Other fields carry
    # values that the neighbours do not reproduce.
    if not isinstance(t, CartesianField):
        return False
    if policy.resolve(t) is NONE:
        return False
    return t.is_identity(compute=True)


# Simplification reconciles no boundaries. A genuine cancelling pair never
# needs it, since an Inverse takes the systems of its forward transformation.
# Inside compute, the bridge step (see `_compute_sequence`) has already run,
# and must run first because a bridge may read the extent of an adjacent grid
# that the sweep below drops; since `adapt` also embeds, operands are already
# comparable here. A standalone t.simplify() bridges nothing and declines a
# pair that would need an embedding.


def _collapse_pass(
    leaves: tx.List[Transformation], policy: SimplifyTable
) -> tx.List[Transformation]:
    # Adjacent pairs collapse over a stack, so each result is retried against
    # the newly exposed neighbour.
    stack: tx.List[Transformation] = []
    for nxt in leaves:
        while stack:
            # An interior grid is dropped here because the condition is
            # positional: a leading or trailing grid is the sampling domain and
            # stays. len(stack) >= 2 means that something precedes the grid,
            # and nxt follows it.
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
    # The id-keyed cache skips leaves already seen and lives for this sweep
    # only, so ids cannot be reused.
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
