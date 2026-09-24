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

Dispatch order within tier 2
----------------------------
For a concrete pair of operand types, every registered composer whose
declared types are ancestors of that pair (with a finite summed hierarchy
`distance`) is a candidate. The candidates are tried nearest-first, with
registration order breaking a tie. The first candidate whose result
``is not NotImplemented`` wins. A composer returns ``NotImplemented`` to
decline and hand off to the next candidate; it raises
[`CompositionError`][] to stop dispatch entirely -- that means "the types
are right but these two cannot be combined" (for example, two subspace
transforms whose axes do not line up). If no candidate applies, or every
candidate declines, `compose` raises [`CompositionError`][].

`compose` itself knows nothing about modes. Gating which adjacent
transforms are handed to `compose` is the sequence engine's job, not the
composers'.
"""

# stdlib
import itertools
import types as _types

# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import safe_get_origin

# internals
from .errors import CompositionError
from .registries import COMPOSERS, COMPOSERS_FASTMAP, distance
from .simplify import ANALYTIC_FLOOR, simplify
from .utils import boundary_disagrees

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation

# `types.UnionType` (the runtime type of a PEP 604 `X | Y` union) only
# exists on Python 3.10+. It is `None` on older interpreters.
_UnionTypes = (tx.Union,)
if hasattr(_types, "UnionType"):
    _UnionTypes += (_types.UnionType,)


def composer(func: tx.Callable) -> tx.Callable:
    """Register a function as a composer of two transformations.

    The composer's declared parameter types key it in the registry. A
    composer *reads parameters*; a rewrite that decides from types and
    object identity alone belongs in `simplifiers`, which `compose`
    consults first.
    """
    types = tuple(tx.get_type_hints(func).values())[:2]
    COMPOSERS[types] = func
    COMPOSERS_FASTMAP.clear()
    return func


def _expand(hint: tx.Any) -> tx.Tuple[tx.Any, ...]:
    # Expand a `X | Y` union hint into its members; a plain type is a
    # one-tuple of itself.
    if safe_get_origin(hint) in _UnionTypes:
        return tx.get_args(hint)
    return (hint,)


def _candidates(t1: type, t2: type) -> tx.Tuple[tx.Callable, ...]:
    # Every registered composer whose declared types are ancestors of
    # `(t1, t2)` with a finite summed hierarchy distance, stably ordered by
    # distance with registration order breaking ties (so the order
    # reproduces the historical first-registered-wins on a tie). The result
    # is cached per concrete pair in `COMPOSERS_FASTMAP`.
    cached = COMPOSERS_FASTMAP.get((t1, t2))
    if cached is not None:
        return cached
    scored = []
    for order, ((T1, T2), func) in enumerate(COMPOSERS.items()):
        best = float("inf")
        for A, B in itertools.product(_expand(T1), _expand(T2)):
            dist = distance(t1, A) + distance(t2, B)
            if dist < best:
                best = dist
        if best < float("inf"):
            scored.append((func, best, order))
    scored.sort(key=lambda s: (s[1], s[2]))
    funcs = tuple(s[0] for s in scored)
    COMPOSERS_FASTMAP[(t1, t2)] = funcs
    return funcs


def compose(
    x1: "Transformation",
    x2: "Transformation",
) -> "Transformation":
    """Compose two transformations.

    `x2` is applied first, then `x1`, so the result maps ``x -> x1(x2(x))``.
    A cost-free rewrite is tried first (see the module docstring); failing
    that, the registered composers are tried in dispatch order and the
    first result that ``is not NotImplemented`` is returned. A composer
    that raises [`CompositionError`][] stops dispatch; if none applies or
    all decline, `compose` raises [`CompositionError`][].
    """
    # Tier 1. The pair simplifiers read in application order, so the
    # operands are flipped: `x2` is the one applied first.
    result = simplify(x2, x1, policy=ANALYTIC_FLOOR)
    if result is not None:
        return result

    # Tier 2. The registered, parameter-reading composers.
    t1, t2 = type(x1), type(x2)
    for func in _candidates(t1, t2):
        result = func(x1, x2)
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
