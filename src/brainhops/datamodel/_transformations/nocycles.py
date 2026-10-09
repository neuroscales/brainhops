"""Late-bound registries shared by the transformation modules.

A low-level module sometimes needs a class or a routine defined in a module
that imports it. Each registry below starts empty and is filled at import time
by the module that defines its entry, so that every module can import this one
without creating an import cycle.
"""

import typing_extensions as tx

if tx.TYPE_CHECKING:
    from .base import Transformation
    from .operators import Operation
    from .sequence import Sequence


OPERATORS: tx.Dict[str, tx.Type["Operation"]] = {}
"""Front doors of the lazy operators, keyed by method name.

A key is the name of the [`Transformation`][] method that applies the operator,
such as `"inverse"` or `"sqrt"`. The value is the front door of the operator: a
polymorphic class that, given a transformation, builds the typed wrapper suited
to the family of that transformation. The front doors are used by the
`inverse()` and `sqrt()` methods, and by a lazy wrapper that rebuilds itself
after its forward is edited, since a new encoding can change the class of the
forward.
"""


def register_operator(name: str) -> tx.Callable[[type], type]:
    """Return a class decorator that registers a front door in [`OPERATORS`][].

    Parameters
    ----------
    name
        Name of the operator, which is also the name of the transformation
        method that applies it.

    Returns
    -------
    callable
        Decorator that registers a class under `name` and returns the class
        unchanged.
    """

    def decorate(cls: tx.Type["Operation"]) -> tx.Type["Operation"]:
        OPERATORS[name] = cls
        return cls

    return decorate


ADAPT: tx.Optional[tx.Callable[..., "Sequence"]] = None
"""Routine that reconciles two consecutive transformations.

The routine bridges a boundary where the coordinate systems of the two
transformations differ only by a reordering, rescaling or flipping of shared
axes. It also embeds a transformation that acts on a subset of the axes into
the fuller axis space of its neighbour, where it acts as the identity on the
extra axes, so that a lower-dimensional transformation can meet a
higher-dimensional one without changing dimensionality.

The routine is defined in the `adaptors` module, which imports this one, and
registers itself at import time, so that the sequence code can reach it without
an import cycle. The value is `None` until then.
"""


def register_adapt(func: tx.Callable[..., "Sequence"]) -> None:
    """Register the routine stored in [`ADAPT`][] and return it.

    Because the routine is returned, the function can be used as a decorator.
    """
    global ADAPT
    ADAPT = func
    return func


SEQUENCE: tx.Optional[tx.Type["Sequence"]] = None
"""The registered [`Sequence`][] class, or `None` until it is registered."""


def register_sequence(cls: tx.Type["Sequence"]) -> None:
    global SEQUENCE
    SEQUENCE = cls
    return cls


# The root of the concrete hierarchy is registered so that code below `base`,
# such as `simplify`, can recognise a transformation without importing `base`
# at load time.
TRANSFORMATION: tx.Optional[tx.Type["Transformation"]] = None


def register_transformation(cls: tx.Type["Transformation"]) -> None:
    """Register the root class of the concrete hierarchy and return it."""
    global TRANSFORMATION
    TRANSFORMATION = cls
    return cls
