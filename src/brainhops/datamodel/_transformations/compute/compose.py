"""Dispatcher for composing two transformations.

`compose(x1, x2)` is the map `x -> x1(x2(x))`, so `x2` is applied
first, in the same order as the matrix product `x1 @ x2`. A
[`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
elements in the opposite order, which is the order of application. The
two conventions meet inside `compose`, which passes its operands to the
pair simplifiers in the order of application.

Unlike a two-argument simplifier, which may decline by returning None,
composition never declines. When two transformations cannot be
combined, `compose` raises [`CompositionError`][].

Composition proceeds in two tiers. The first tier is a cost-free
rewrite by the pair simplifiers of [`simplify`][], such as
`X @ X^-1 -> Identity` or `Id @ T -> T`. This tier always runs and
cannot be turned off by a policy. For the composition of a field with
the inverse of a field, it gives the only possible answer, because the
inverse of a field cannot be materialized. Running this tier first also
guarantees that a lazy inverse is never materialized when it could have
cancelled instead. The second tier consists of the registered
composers, which read the parameters of the operands, for example to
multiply matrices or to resample fields.

The composer to run is chosen by a
[`Function`][bagof.dispatchers.Function] from the types of the two
operands. Each composer declares the operand types it accepts, and a
declared type may be a union. The most specific composer wins, with
the types compared position by position, so a composer for
`(Sequence, Sequence)` is preferred to one for
`(Sequence, Transformation)`. When two composers are equally specific,
the dispatcher raises
[`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]. The
error propagates out of `compose` instead of being resolved by the
order of registration. The registered composers contain only one
near-tie, between `(Sequence, Transformation)` and
`(Transformation, Sequence)`. The only operands that both of these
composers accept are two sequences, and in that case a dedicated
composer for `(Sequence, Sequence)` wins.

A composer may receive operands of the right types and still be unable
to combine them, for example two subspace transformations whose axes do
not line up. Such a composer raises [`CompositionError`][], which stops
the composition. The `compose` function knows nothing of modes. The
sequence engine decides which adjacent transformations are passed to
`compose`.
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
    """Register a composer for the operand types given by its two hints.

    A composer reads the parameters of its operands. Decisions that follow
    from the types or the identity of the operands alone belong in
    `simplifiers`, which `compose` consults first. When two composers are
    equally specific for a pair of operands, a call raises
    [`AmbiguousMethodError`][bagof.dispatchers.AmbiguousMethodError]
    instead of choosing one of them by priority.
    """
    _compose.register(func)
    return func


def compose(
    x1: "Transformation",
    x2: "Transformation",
) -> "Transformation":
    """Compose two transformations into `x -> x1(x2(x))`.

    A cost-free rewrite by a pair simplifier is tried first. If no such
    rewrite applies, the most specific registered composer is called.

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
        # When the two systems at the boundary disagree, the fix is to
        # reconcile them rather than to write a composer, and the error
        # message says so.
        raise CompositionError(
            f"Cannot compose a {t1.__name__} with a {t2.__name__}: they "
            f"disagree on the system they share ({x2.output} vs "
            f"{x1.input}). Reconcile it first -- `adapt` inserts the "
            f"bridge, and `Sequence.compute` does it for you."
        )
    raise CompositionError(f"No composer found for types: {t1}, {t2}")
