"""Simplifier implementations, registered through `simplify.simplifier`.

This is to `simplify` what `composers` is to `compose` and `checkers` is to
`check`: the machinery lives in `simplify` (a single `bagof.dispatchers`
function holding both arities as overloads, dispatched by argument count),
every rule lives here and registers at import time.

Two arities
-----------
**One in, one out** -- the *downcast*. A transform is rewritten as the
cheapest type it is established to belong to: an `Affine` whose matrix is
diagonal becomes a `Scaling`, a `Linear` with no matrix becomes an
`Identity`. How hard the parameter may be looked at is the resolved
[`SimplifyPolicy`][]: `analytic` reads structure only, `numeric` reads
values, `none` leaves the transform alone. A container (a sequence, a
wrapper) recurses into what it holds.

**Two in, one out** -- the *collapse*. Two consecutive transforms are
replaced by one, for free: a transform next to its own lazy inverse, an
identity next to anything. `first` is applied before `second`, the order
the two read in a [`Sequence`][]. A rule returns `None` to decline, which
is the ordinary answer.

What a simplifier may not do
----------------------------
* **Read values the policy forbids.** Every rule that looks at a parameter
  first resolves its policy.
* **Materialize a lazy inverse.** Inverting a field is computation, not
  simplification: it is what [`Inverse.compute`][] does, gated by the
  compose mode. A simplifier that inverted a field would defeat the whole
  point of the lazy wrapper, which is to cancel before anyone pays for it.
* **Make a sequence longer, or reconcile a boundary.** Inserting a bridge,
  or embedding an operand in a fuller axis space, is composition's
  business; a pair whose systems disagree is declined instead. See the
  note above `_collapse` for why that loses nothing.
"""

# dependencies
import typing_extensions as tx
from bagof.magic import replace

# datamodel
from brainhops.datamodel import kinds
from brainhops.errors import ConversionError, DomainError

# internals
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

# The downcast ladder, cheapest representation first. A transform is
# rewritten as the first rung whose *set* it is established in -- unless it
# is already an instance of that rung's class, in which case it is already
# at least that specific and is left alone. Pairing each set node with the
# concrete class that represents it is what makes the rewrite a downcast
# and never an upcast: a `Rotation` stops at the rotation rung rather than
# being widened to `Linear` by the rung below.
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
    """Downcast a transform to the cheapest type it belongs to."""
    resolved = policy.resolve(t)
    if resolved is NONE:
        return t
    compute = resolved is NUMERIC
    for node, cls in _LADDER:
        if not is_kind(t, node, compute):
            continue
        if isinstance(t, cls):
            # Already at least this specific: rebuilding it would hand back
            # a new object and break the `forward is` link that a lazy
            # inverse cancels by.
            return t
        try:
            return t.to(cls)
        except ConversionError:
            # Established in the set but not representable in its class
            # (no converter, or the conversion would lose information).
            # Nothing cheaper is available, so stop here.
            return t
    return t


@simplifier
def _(t: CartesianField, policy: SimplifyTable) -> Transformation:
    """Leave a grid alone.

    A [`CartesianField`][] is the identity map over its own grid, so the
    ladder above would collapse it to `Identity`. But a grid is also the
    *sampling domain* onto which data is resampled, and that is not
    recoverable from an `Identity`. A grid is therefore never downcast; it
    is only ever removed as a pair, when a neighbour on each side
    overwrites its coordinates (see `_droppable_grid`).
    """
    return t


@simplifier
def _(t: Inverse, policy: SimplifyTable) -> Transformation:
    """Simplify what an inverse wraps, without ever resolving it.

    Materializing an inverse is computation, not simplification: that is
    [`Inverse.compute`][]'s job, and it is gated by the compose mode.
    Simplifying the forward can still collapse the whole wrapper, because
    the inverse of an identity is an identity.

    The rewrapping goes through the simplified forward's own `inverse()`,
    so the result is the typed wrapper of whatever family the forward
    turned into.
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
    """Simplify what a square root wraps, without ever resolving it.

    As for an inverse, resolving a square root is computation, which is
    [`Operation.compute`][]'s job. Simplifying the forward can still
    collapse the whole wrapper, because the square root of the identity is
    the identity, or make it cheaper: the wrapper is rebuilt through the
    simplified forward's own `sqrt()`, so it becomes the typed wrapper of
    whatever family the forward turned into. A forward whose square root
    is refused is left wrapped as it was, since a simplifier never raises.
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
    """Simplify the transform a subspace embeds.

    A subspace over the same input and output axes whose inner transform
    is (or becomes) the identity is the identity on the full space. A
    subspace that *reindexes* its axes is not: it still permutes
    coordinates, so it is kept.
    """
    resolved = policy.resolve(t)
    if resolved is NONE:
        return t
    same = _same_axes(t)
    inner = t.transformation
    if inner is None:
        # A subspace of nothing: the identity on the same axes, a pure
        # reindex otherwise (kept as is).
        return Identity(input=t.input, output=t.output) if same else t
    simplified = _simplify(inner, policy=policy)
    if same and simplified.is_identity(resolved is NUMERIC):
        return Identity(input=t.input, output=t.output)
    if simplified is inner:
        return t
    return t.to(transformation=simplified)


@simplifier
def _(t: Projection, policy: SimplifyTable) -> Transformation:
    """A projection that neither drops nor creates an axis is the identity."""
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
        # Unchanged: keep object identity so an adjacent `Inverse` of this
        # bijection still cancels.
        return t
    return t.to(forward=forward, backward=backward)


@simplifier
def _(t: MultiscaleField, policy: SimplifyTable) -> Transformation:
    """Simplify a multiscale field through the scale it presents.

    Its elements are derived from its scales, so it cannot be rebuilt
    element-wise the way a plain sequence can. It behaves as its finest
    scale, and simplifying it is simplifying that scale -- the same
    delegation [`MultiscaleField.compute`][] makes.
    """
    if policy.resolve(t) is NONE:
        return t
    return _simplify(t._as_sequence(), policy=policy)


@simplifier
def _(t: Sequence, policy: SimplifyTable) -> Transformation:
    """Simplify a sequence: downcast its leaves, collapse its free pairs.

    The two passes feed each other -- a downcast can expose a collapsible
    pair, and a collapse can expose a downcastable leaf -- so they run to a
    fixpoint. The collapse pass runs first on each round, so that a
    transform meets its own lazy inverse and cancels by object identity
    before any downcast could rebuild either side and break the link.

    Nothing here composes and nothing is bridged: a sequence only ever
    gets shorter.
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

# --- Identity ---------------------------------------------------------
# An identity contributes nothing but an endpoint. It is dropped, and the
# survivor takes over the endpoint the identity named -- but only when that
# endpoint is set and differs, because rebuilding the survivor would hand
# back a new object and break the `forward is` link a lazy inverse cancels
# by (pinned by `test_lazy_inverse.py`). The rebuild goes through
# `replace`, never `compute()`, which would resolve a lazy inverse.


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


# --- Cancellation -----------------------------------------------------
# A transform placed next to its own lazy inverse annihilates it. An
# `Inverse` names the transform it undoes as its `forward`, so the test is
# a plain identity check that materializes neither side: it is O(1) and
# decides from object identity alone.


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


# --- Subspace pairs ---------------------------------------------------


@simplifier
def _(
    first: SubspaceTransformation,
    second: SubspaceTransformation,
    policy: SimplifyTable,
) -> tx.Optional[Transformation]:
    """Collapse a subspace-wrapped transform next to its own inverse.

    Two subspace transforms annihilate when their axes chain (the axes the
    first writes are the axes the second reads), the net map introduces no
    reindex, and their inner transforms compose to the identity -- either
    because one inner is the lazy inverse of the other, or because both
    are already the identity. This is what lets a subspace-wrapped field
    meet its own subspace-wrapped inverse and cancel, rather than the field
    being resampled through a neighbour first. An inner sequence is undone
    by a sequence of lazy inverses, not by one, so a pair whose inner is a
    sequence cancels when the two inners, chained, simplify to the
    identity -- each element meeting its own lazy inverse.
    """
    if (
        first.output_axes is None
        or second.input_axes is None
        or list(first.output_axes) != list(second.input_axes)
    ):
        return None
    # A *reindexing* inverse pair -- whose axes chain (checked above) but
    # whose net map still permutes axes -- is deliberately NOT annihilated
    # here. It does not reduce to the bare identity (it is a pure axis
    # reindex), so it is left to the composers to fold into a single
    # reindexing subspace transform; do not "fix" it into this rule.
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
    # Structure only, never values: this rule is asked about every adjacent
    # subspace pair on every fixpoint iteration, and a numeric check would
    # scan a whole displacement field each time. Nothing is lost -- the
    # `transformation=None` case is `inner is None`, and the lazy-inverse
    # case is `_cancels` above; a numerically-zero field is caught by the
    # downcast pass, which turns it into an `Identity` first.
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
    """Whether ``[first, second]`` cancels to the identity for free.

    `first` is applied before `second`. The two cancel when `second` is the
    lazy inverse of `first`, or `first` is the lazy inverse of `second`. An
    `Inverse` names the transform it undoes as its `forward`, so the test is
    a plain identity check that materializes neither field: it is O(1) and
    decides from object identity alone. This covers both a typed inverse
    and a generic `Inverse(forward=X)`.
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
    """Whether ``[first, second]`` cancels to the identity, either a sequence.

    The inverse of a sequence is the sequence of the inverses of its
    elements, in reverse order, rather than one lazy inverse, so
    [`_cancels`][] cannot see that it undoes the sequence. Chained, the two
    simplify to the identity when each element meets its own lazy inverse,
    which is still decided from object identity alone. Only a pair in
    which a sequence takes part is chained: any other pair is decided by
    `_cancels`.
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
    # Whether `t` is a grid that a neighbour on each side makes redundant.
    # A `CartesianField` is the identity map over its coordinates, so
    # `A @ grid @ B` computes the same field as `A @ B`: the neighbours
    # overwrite the grid's coordinates. Only a grid qualifies -- a general
    # field of coordinates, or of displacements, carries values its
    # neighbours do not reproduce.
    if not isinstance(t, CartesianField):
        return False
    if policy.resolve(t) is NONE:
        return False
    return t.is_identity(compute=True)


# NOTE
#   Simplification does not reconcile boundaries at all -- it neither
#   bridges nor embeds. Two reasons, and the second is the one that settles
#   it:
#
#   * A genuine cancelling pair never needs reconciling. An `Inverse`'s
#     input system *is* its forward's output system, by construction, so
#     the boundary between a transform and its own inverse already agrees.
#
#   * Inside `compute`, the bridge step has already run (see
#     `_compute_sequence`). It has to run first, because a bridge reads its
#     neighbours -- a reversed array-index axis needs the extent an
#     adjacent grid carries, and that grid is one of the things the sweep
#     below may drop. `adapt` embeds as well as bridges, so by the time this
#     pass sees a pair, a lower-dimensional operand has already been embedded
#     into the fuller axis space and the two are comparable.
#
#   What remains is the standalone `t.simplify()` path, which bridges
#   nothing: there, a pair that only an embedding would make comparable is
#   left
#   alone. That is the intended reading of "simplification never lengthens
#   a sequence" -- it declines rather than reaching for machinery that
#   belongs to composition.


def _collapse_pass(
    leaves: tx.List[Transformation], policy: SimplifyTable
) -> tx.List[Transformation]:
    # One sweep collapsing adjacent pairs, over a stack so that each
    # collapse is immediately retried against the new neighbour it exposes.
    stack: tx.List[Transformation] = []
    for nxt in leaves:
        while stack:
            # An interior grid is dropped here rather than by a pair rule,
            # because the rule is positional: a grid is redundant only when
            # a transform on *each* side overwrites its coordinates. A grid
            # that leads or trails the sequence is its sampling domain and
            # stays. `len(stack) >= 2` is exactly "something precedes this
            # grid", and `nxt` is what follows it.
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
    # One sweep downcasting each leaf. A leaf seen before (the same object
    # across iterations) is not re-scanned; the cache is keyed by identity
    # and lives only for this sweep, so no `id` can be reused under it.
    cache: tx.Dict[int, Transformation] = {}
    out = []
    for t in leaves:
        key = id(t)
        if key not in cache:
            cache[key] = _simplify(t, policy=policy)
        out.append(cache[key])
    return out


def _rebuild(seq: Sequence, leaves: tx.List[Transformation]) -> Transformation:
    # The transform a simplified sequence presents. A sequence that
    # collapsed to nothing is the identity between its own endpoints; one
    # that collapsed to a single transform *is* that transform, carrying
    # any endpoint the sequence itself declared.
    if not leaves:
        return Identity(input=seq.input, output=seq.output)
    if len(leaves) == 1:
        return _with_endpoints(leaves[0], seq)
    if len(leaves) == len(seq.transformations or ()) and all(
        a is b for a, b in zip(leaves, seq.transformations)
    ):
        return seq
    return seq.to(transformations=leaves)
