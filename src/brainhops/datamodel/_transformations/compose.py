"""Dispatch the composition of two transformations to a composer.

`compose(x1, x2)` returns the transform that maps ``x -> x1(x2(x))``: `x2`
is applied first, then `x1`. It picks the composer to run by walking the
registry of composers and ordering the ones that apply.

Dispatch order
--------------
For a concrete pair of operand types, every registered composer whose
declared types are ancestors of that pair (with a finite summed hierarchy
`distance`) is a candidate. The candidates are tried in order of:

1. **priority** -- higher first;
2. **hierarchy distance** -- nearer declared types first;
3. **registration order** -- the first-registered composer breaks a tie.

The first candidate whose result ``is not NotImplemented`` wins. A composer
returns ``NotImplemented`` to decline and hand off to the next candidate; it
raises [`CompositionError`][] to stop dispatch entirely -- that means "the
types are right but these two cannot be combined" (for example, two subspace
transforms whose axes do not line up). If no candidate applies, or every
candidate declines, `compose` raises [`CompositionError`][].

Priority tiers
--------------
* `ANALYTIC` (see [`registries`][]) is reserved for a composer that decides
  purely from the operand types and object identity, reading no parameter.
  Today that is the inverse-cancel pair, ``X @ X^-1 -> Identity``.
* priority ``0`` is the default, used by the numeric composers that read and
  combine parameters (matrices, fields, ...).

Ordering the analytic tier ahead of the numeric one enforces the invariant
that a **cost-free rewrite is tried before any parameter-reading one**: a
transform placed next to its own lazy inverse cancels to the identity before
the numeric composer that would materialize the inverse ever runs, so a lazy
inverse is never materialized when it could have cancelled.

`compose` itself knows nothing about modes. Gating which adjacent transforms
are handed to `compose` is the sequence engine's job, not the composers'.
"""

# stdlib
import itertools
import types as _types
from functools import partial

# dependencies
import typing_extensions as tx

# core
from brainhops._core.typing import safe_get_origin

# internals
from .errors import CompositionError
from .registries import COMPOSERS, COMPOSERS_FASTMAP, distance

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation

# `types.UnionType` (the runtime type of a PEP 604 `X | Y` union) only
# exists on Python 3.10+. It is `None` on older interpreters.
_UnionTypes = (tx.Union,)
if hasattr(_types, "UnionType"):
    _UnionTypes += (_types.UnionType,)


def composer(
    func: tx.Optional[tx.Callable] = None,
    *,
    priority: int = 0,
) -> tx.Callable:
    """Register a function as a composer of two transformations.

    Usable bare (``@composer``) or with a priority
    (``@composer(priority=ANALYTIC)``). The composer's declared parameter
    types key it in the registry, alongside its dispatch `priority`.
    """
    if func is None:
        return partial(composer, priority=priority)
    types = tuple(tx.get_type_hints(func).values())[:2]
    COMPOSERS[types] = (func, priority)
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
    # `(-priority, distance)` with registration order breaking ties (so the
    # order reproduces the historical first-registered-wins on a tie). The
    # result is cached per concrete pair in `COMPOSERS_FASTMAP`.
    cached = COMPOSERS_FASTMAP.get((t1, t2))
    if cached is not None:
        return cached
    scored = []
    for order, ((T1, T2), (func, priority)) in enumerate(COMPOSERS.items()):
        best = float("inf")
        for A, B in itertools.product(_expand(T1), _expand(T2)):
            dist = distance(t1, A) + distance(t2, B)
            if dist < best:
                best = dist
        if best < float("inf"):
            scored.append((func, priority, best, order))
    scored.sort(key=lambda s: (-s[1], s[2], s[3]))
    funcs = tuple(s[0] for s in scored)
    COMPOSERS_FASTMAP[(t1, t2)] = funcs
    return funcs


def compose(
    x1: "Transformation",
    x2: "Transformation",
) -> "Transformation":
    """Compose two transformations.

    `x2` is applied first, then `x1`, so the result maps ``x -> x1(x2(x))``.
    The composer to run is chosen by dispatch order (see the module
    docstring): the candidates are tried in turn and the first result that
    ``is not NotImplemented`` is returned. A composer that raises
    [`CompositionError`][] stops dispatch; if none applies or all decline,
    `compose` raises [`CompositionError`][].
    """
    t1, t2 = type(x1), type(x2)
    for func in _candidates(t1, t2):
        result = func(x1, x2)
        if result is not NotImplemented:
            return result
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
