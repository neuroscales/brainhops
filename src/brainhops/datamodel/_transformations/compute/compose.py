"""Dispatcher for composing two transformations.

`compose(x1, x2)` is the map `x -> x1(x2(x))`, so `x2` is applied
first, in the same order as the matrix product `x1 @ x2`. A
[`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
elements in the opposite order, the order of application, and the two
conventions meet where the pair simplifier is called. Composition
cannot decline: a failure raises [`CompositionError`][] and never
returns None, unlike a two-argument simplifier.

Composition proceeds in two tiers. The first tier is a cost-free
rewrite by the pair simplifiers of [`simplify`][], such as
`X @ X^-1 -> Identity` or `Id @ T -> T`. This tier is not a policy
option. It is the only answer for the composition of a field with the
inverse of a field, which cannot be materialized, and it guarantees
that a lazy inverse is never materialized when it could cancel. The
second tier consists of the registered composers, which read the
parameters, for example to multiply matrices or resample fields.

The composers are dispatched by a
[`Function`][bagof.dispatchers.Function] keyed by the pair of operand
types, which may be unions. The most specific method wins position by
position, so `(Sequence, Sequence)` beats `(Sequence, Transformation)`.
A genuine tie raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError], which
propagates out of `compose` instead of being resolved by registration
order. The only structural near-tie, between
`(Sequence, Transformation)` and `(Transformation, Sequence)`, overlaps
at `(Sequence, Sequence)`, where a dedicated composer wins.

A composer that receives operands of the right types but cannot
combine them, such as subspace transformations with misaligned axes,
raises [`CompositionError`][], which stops the composition. The
function knows nothing of modes: which adjacent transformations reach
`compose` is decided by the sequence engine.
"""

import typing_extensions as tx
from bagof.dispatchers import Function, NoMethodError

from brainhops.errors import CompositionError

from .simplify import ANALYTIC_FLOOR, simplify
from .utils import boundary_disagrees

if tx.TYPE_CHECKING:
    from ..base import Transformation


_compose: Function = Function("compose")
"""Dispatched function on which every composer is registered."""


def composer(func: tx.Callable) -> tx.Callable:
    """Register a composer, keyed by the hints of its two parameters.

    A composer reads the parameters of its operands. Decisions that follow
    from types or identity alone belong in `simplifiers`, which `compose`
    consults first. Genuine ties between composers raise
    [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]
    rather than being resolved by priority.
    """
    _compose.register(func)
    return func


def compose(
    x1: "Transformation",
    x2: "Transformation",
) -> "Transformation":
    """Compose two transformations into `x -> x1(x2(x))`.

    A cost-free rewrite is tried first, then the most specific composer.

    Parameters
    ----------
    x1 : Transformation
        Transformation applied second.
    x2 : Transformation
        Transformation applied first.

    Returns
    -------
    Transformation
        The composed transformation.

    Raises
    ------
    CompositionError
        If the composer refuses the pair, or if no composer applies.
    """
    # Pair simplifiers use the order of application.
    result = simplify(x2, x1, policy=ANALYTIC_FLOOR)
    if result is not None:
        return result

    # Only NoMethodError means that no composer applies. AmbiguousMethodError
    # is deliberately left uncaught, since it reveals a fault in the registry.
    t1, t2 = type(x1), type(x2)
    try:
        result = _compose(x1, x2)
    except NoMethodError:
        result = NotImplemented
    if result is not NotImplemented:
        return result

    if boundary_disagrees(x2, x1):
        # The fix is to reconcile the boundary, not to write a composer, so the
        # error says so.
        raise CompositionError(
            f"Cannot compose a {t1.__name__} with a {t2.__name__}: they "
            f"disagree on the system they share ({x2.output} vs "
            f"{x1.input}). Reconcile it first -- `adapt` inserts the "
            f"bridge, and `Sequence.compute` does it for you."
        )
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
