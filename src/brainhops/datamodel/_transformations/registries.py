"""The late-bound singletons every cyclic import goes through.

The singletons (`INVERSE`, `ADAPT`, `SEQUENCE`, `TRANSFORMATION`) live here
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
    from .inverse import Inverse
    from .operators import Exp, Log, Sqrt
    from .sequence import Sequence


# --- inverse ----------------------------------------------------------

INVERSE_CACHE = "_inverse_param_cache"
"""
The materialized inverse parameter is cached on the *forward* transform,
under this attribute name, rather than on the wrapper. A wrapper rebuilt
by `replace` or `.to(...)` keeps the same forward transform, so the cache
survives the rebuild and the inversion is not run again. The cache
assumes the forward transform is not mutated in place after it is
wrapped: a forward transform whose parameter is replaced by editing the
same object would keep serving the stale inverse.
"""

INVERSE: tx.Optional[tx.Type["Inverse"]] = None
"""Registered `Inverse` class, to avoid cyclic imports."""


def register_inverse(cls: tx.Type["Inverse"]) -> None:
    global INVERSE
    INVERSE = cls
    return cls


# --- operators --------------------------------------------------------

SQRT: tx.Optional[tx.Type["Sqrt"]] = None
"""Registered `Sqrt` class, to avoid cyclic imports."""

EXP: tx.Optional[tx.Type["Exp"]] = None
"""Registered `Exp` class, to avoid cyclic imports."""

LOG: tx.Optional[tx.Type["Log"]] = None
"""Registered `Log` class, to avoid cyclic imports."""


def register_sqrt(cls: tx.Type["Sqrt"]) -> tx.Type["Sqrt"]:
    global SQRT
    SQRT = cls
    return cls


def register_exp(cls: tx.Type["Exp"]) -> tx.Type["Exp"]:
    global EXP
    EXP = cls
    return cls


def register_log(cls: tx.Type["Log"]) -> tx.Type["Log"]:
    global LOG
    LOG = cls
    return cls


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


# --- compose / convert / simplify -------------------------------------
#
# These three no longer keep a table here. Their rules are registered into
# `bagof.dispatchers` functions in the `compose`, `convert` and `simplify`
# modules: `compose` and `convert` are single-winner most-specific dispatch
# (`convert` on `type(x)` and, via a `type[...]` parameter, on the target
# class value), and `simplify` uses one function per arity. See the migration
# memo under `docs/design/` for why they map onto the library's model and why
# `is_kind` (above) stays bespoke.
