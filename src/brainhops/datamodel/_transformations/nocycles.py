"""The late-bound singletons every cyclic import goes through.

The singletons (`OPERATORS`, `ADAPT`, `SEQUENCE`, `TRANSFORMATION`) live here
because they must be importable by every module of the package without
forming a cycle: a module registers into them at import time, so that a
lower layer can reach a higher one without importing it.

Dispatch itself no longer lives here. `compose`, `convert` and `simplify`
dispatch through `bagof.dispatchers` functions in their own modules, and
`is_kind` -- whose membership question combines every applicable checker
rather than picking one winner -- wraps a `bagof.dispatchers` function with
its own reducer in `check`. The bespoke `Dispatcher` base class and the
`type_distance` metric they all shared are gone; see the migration memo
under `docs/design/` for how each maps onto the library's model.
"""

# dependencies
import typing_extensions as tx

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation
    from .operators import Operation
    from .sequence import Sequence


# --- operators --------------------------------------------------------

OPERATORS: tx.Dict[str, tx.Type["Operation"]] = {}
"""
The front door of each unary operator, keyed by the name of the
`Transformation` method that applies it: `inverse`, `sqrt`.

A front door is polymorphic: calling it picks the typed wrapper of the
transformation handed to it. That is what a transformation's own
`inverse()`/`sqrt()` builds, and what a lazy wrapper rebuilds itself
through when its forward is edited -- a new encoding can change the
forward's class, and the typed wrapper is then chosen again for it.

The wrappers live in modules that import this one, so they register here
at import time.
"""


def register_operator(name: str) -> tx.Callable[[type], type]:
    """Register the front door of the operator that `name` applies."""

    def decorate(cls: tx.Type["Operation"]) -> tx.Type["Operation"]:
        OPERATORS[name] = cls
        return cls

    return decorate


# --- adapt ------------------------------------------------------------

ADAPT: tx.Optional[tx.Callable[..., "Sequence"]] = None
"""
The adaptor lives in `adaptors`, which imports this module. It
registers itself here at import time, so the sequence machinery can call
it without importing that module at load time and forming a cycle. The
adaptor reconciles two consecutive transforms: it bridges a boundary
where the two systems merely reorder, rescale or flip their shared axes,
and it embeds a transform that acts on a subset of a boundary's axes in
the fuller axis space, leaving the extra axes as the identity, so a
lower-dimensional transform meets a higher-dimensional neighbour without
a dimensionality change.
"""


def register_adapt(func: tx.Callable[..., "Sequence"]) -> None:
    """Register the routine that reconciles two consecutive transforms."""
    global ADAPT
    ADAPT = func
    return func


# --- sequence ---------------------------------------------------------

SEQUENCE: tx.Optional[tx.Type["Sequence"]] = None
"""Registered `Sequence` class, to avoid cyclic imports."""


def register_sequence(cls: tx.Type["Sequence"]) -> None:
    global SEQUENCE
    SEQUENCE = cls
    return cls


# --- base -------------------------------------------------------------

# The concrete `base.Transformation` root, registered at its import time so
# that `modes._lower_key` can recognize a concrete transformation class as a
# key without importing `base` at the top level (which would cycle).
TRANSFORMATION: tx.Optional[tx.Type["Transformation"]] = None


def register_transformation(cls: tx.Type["Transformation"]) -> None:
    """Register the concrete `Transformation` root class."""
    global TRANSFORMATION
    TRANSFORMATION = cls
    return cls
