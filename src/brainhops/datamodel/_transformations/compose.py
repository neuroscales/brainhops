"""Dispatch the composition of two transformations to a composer.

`compose(x1, x2)` returns the transform that maps ``x -> x1(x2(x))``: `x2`
is applied first, then `x1`. Note the matrix order -- the operand applied
first is written on the right, as it is in `x1 @ x2`. (A [`Sequence`][]
reads the other way, in application order; the two orders meet at the one
place this module delegates to a pair simplifier.)

Composition is *obligatory*: a caller wants a single transform back, and
"these two cannot be combined" is a [`CompositionError`][], not a return
value. That contract is what separates a composer from a two-argument
simplifier, which may decline with `None` (see the `simplify` module).

Tiers
-----
1. **The cost-free rewrite.** `compose` first asks the pair simplifiers
   whether the two collapse for free -- `X @ X^-1 -> Identity`, `Id @ T ->
   T`. This is not a policy knob: a cost-free rewrite is always the right
   answer when one applies, and for a pair such as `~field @ field` it is
   the *only* answer that exists, since the inverse of a coordinate field
   cannot be materialized at all. Asking first is therefore what guarantees
   a lazy inverse is never materialized when it could have cancelled.

2. **The registered composers.** These read parameters -- they multiply
   matrices and resample fields.

Dispatch within tier 2
----------------------
The registered composers are a [`bagof.dispatchers`][] function keyed by the
pair of operand types (which may be [`Union`][typing.Union]s -- the library
reads a union hint natively). For a concrete pair, the single *most specific*
composer is chosen -- position by position, so a composer on `(Sequence,
Sequence)` beats one on `(Sequence, Transformation)` because `Sequence` is a
subtype of `Transformation`. A genuine specificity tie -- two composers
neither of which is more specific for the concrete pair in hand -- is *not*
resolved by registration order: the library raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], which
propagates out of `compose`. We would rather surface a real ambiguity than
silently pick a bad composer by registration accident.

The one structural near-tie the composers contain is the pair `(Sequence,
Transformation)` and `(Transformation, Sequence)`, which are mutually
incomparable as *rules*. But their only concrete overlap is a `(Sequence,
Sequence)` operand pair, and the registered `(Sequence, Sequence)` composer
is strictly more specific there and wins it outright, so no concrete pair
actually raises. (`bagof` v0.2's pairwise registration check does not see
that dominating third rule, so it emits a benign `RuntimeWarning` about the
incomparable pair at import; the enhancement note in
`docs/design/bagof-dispatchers-migration.md` records it.)

A composer raises [`CompositionError`][] to refuse -- "the types are right
but these two cannot be combined" (for example, two subspace transforms whose
axes do not line up). That propagates straight out, stopping composition. If
no composer applies at all, `compose` raises [`CompositionError`][] too,
after checking whether the two ends of the shared boundary merely disagree.

Unlike the bespoke registry this replaced, the library selects one winner
rather than a nearest-first chain: there is no `NotImplemented`-decline
hand-off to a less specific composer, because no registered composer needs
one (each concrete pair has a single most specific composer that always
produces a result or refuses with `CompositionError`). A general
chain-of-responsibility mode is the enhancement proposed for
`bagof.dispatchers` in `docs/design/bagof-dispatchers-migration.md`; a
composer returning `NotImplemented` is treated here as "no composer applies".

`compose` itself knows nothing about modes. Gating which adjacent
transforms are handed to `compose` is the sequence engine's job, not the
composers'.
"""

# dependencies
import typing_extensions as tx
from bagof.dispatchers import Function, NoMethodError

# internals
from .errors import CompositionError
from .simplify import ANALYTIC_FLOOR, simplify
from .utils import boundary_disagrees

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation


_compose: Function = Function("compose")
"""The dispatched function every registered composer joins."""


def composer(func: tx.Callable) -> tx.Callable:
    """Register a function as a composer of two transformations.

    The composer's declared parameter types (read off its two parameters'
    hints, which may be unions) key it in the dispatcher. A composer *reads
    parameters*; a rewrite that decides from types and object identity alone
    belongs in `simplifiers`, which `compose` consults first.

    Composers carry no `priority`: a genuine specificity tie is left to raise
    [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError] rather
    than be resolved by registration order, so a real ambiguity surfaces
    instead of silently picking a bad composer (see the module docstring).
    """
    _compose.register(func)
    return func


def compose(
    x1: "Transformation",
    x2: "Transformation",
) -> "Transformation":
    """Compose two transformations.

    `x2` is applied first, then `x1`, so the result maps ``x -> x1(x2(x))``.
    A cost-free rewrite is tried first (see the module docstring); failing
    that, the single most specific registered composer is chosen and run. A
    composer that raises [`CompositionError`][] stops composition; if none
    applies, `compose` raises [`CompositionError`][].
    """
    # Tier 1. The pair simplifiers read in application order, so the
    # operands are flipped: `x2` is the one applied first.
    result = simplify(x2, x1, policy=ANALYTIC_FLOOR)
    if result is not None:
        return result

    # Tier 2. The registered, parameter-reading composers. The library picks
    # the single most specific one; a `CompositionError` it raises propagates
    # out and stops composition. Only `NoMethodError` -- "no composer applies"
    # -- is caught here; an `AmbiguousMethodError` from a genuine specificity
    # tie is a real registry fault the maintainer wants surfaced, so it is
    # deliberately NOT caught and propagates out unchanged.
    t1, t2 = type(x1), type(x2)
    try:
        result = _compose(x1, x2)
    except NoMethodError:
        result = NotImplemented
    if result is not NotImplemented:
        return result

    if boundary_disagrees(x2, x1):
        # The composers assume the two ends of the boundary line up, and
        # these two do not. Say so, rather than reporting it as a missing
        # composer: the fix is to reconcile the boundary, not to write one.
        raise CompositionError(
            f"Cannot compose a {t1.__name__} with a {t2.__name__}: they "
            f"disagree on the system they share ({x2.output} vs "
            f"{x1.input}). Reconcile it first -- `adapt` inserts the "
            f"bridge, and `Sequence.compute` does it for you."
        )
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
