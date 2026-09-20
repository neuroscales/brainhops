# stdlib
from functools import partial

# dependencies
import typing_extensions as tx

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation
    from .inverse import Inverse
    from .sequence import Sequence

# The materialized inverse parameter is cached on the *forward* transform,
# under this attribute name, rather than on the wrapper. A wrapper rebuilt
# by `replace` or `.to(...)` keeps the same forward transform, so the cache
# survives the rebuild and the inversion is not run again. The cache
# assumes the forward transform is not mutated in place after it is
# wrapped: a forward transform whose parameter is replaced by editing the
# same object would keep serving the stale inverse.
INVERSE_CACHE = "_inverse_param_cache"

# Each forward transformation type is paired with the `Inverse` subclass
# that represents its inverse. `_lazy_inverse` looks the wrapper up here,
# and the table is filled in once the wrapper classes are defined below.
INVERSE_WRAPPERS: tx.Dict[tx.Type["Transformation"], tx.Type["Inverse"]] = {}

# The adaptor lives in `adaptors`, which imports this module. It
# registers itself here at import time, so the sequence machinery can call
# it without importing that module at load time and forming a cycle. The
# adaptor reconciles two consecutive transforms: it bridges a boundary
# where the two systems merely reorder, rescale or flip their shared axes,
# and it lifts a transform that acts on a subset of a boundary's axes into
# the fuller axis space, leaving the extra axes as the identity, so a
# lower-dimensional transform meets a higher-dimensional neighbour without
# a dimensionality change.
ADAPT: tx.Optional[tx.Callable[..., "Sequence"]] = None


def register_adapt(func: tx.Callable[..., "Sequence"]) -> None:
    """Register the routine that reconciles two consecutive transforms."""
    global ADAPT
    ADAPT = func


SEQUENCE: tx.Optional[tx.Type["Sequence"]] = None


def register_sequence(cls: tx.Type["Sequence"]) -> None:
    """Register the routine that reconciles two consecutive transforms."""
    global SEQUENCE
    SEQUENCE = cls


INVERSE: tx.Optional[tx.Type["Inverse"]] = None


def register_inverse(cls: tx.Type["Inverse"]) -> None:
    """Register the routine that reconciles two consecutive transforms."""
    global INVERSE
    INVERSE = cls


# The concrete `base.Transformation` root, registered at its import time so
# that `modes._lower_key` can recognize a concrete transformation class as a
# key without importing `base` at the top level (which would cycle).
TRANSFORMATION: tx.Optional[tx.Type["Transformation"]] = None


def register_transformation(cls: tx.Type["Transformation"]) -> None:
    """Register the concrete `Transformation` root class."""
    global TRANSFORMATION
    TRANSFORMATION = cls


XFORM_PAIR = tx.Tuple[tx.Type["Transformation"], tx.Type["Transformation"]]

CONVERTER = tx.Callable[["Transformation"], "Transformation"]
CONVERTER_REGISTRY = tx.Dict[XFORM_PAIR, CONVERTER]
CONVERTERS: CONVERTER_REGISTRY = {}
CONVERTERS_FASTMAP: CONVERTER_REGISTRY = {}

COMPOSER = tx.Callable[["Transformation", "Transformation"], "Transformation"]
# Each declared type-pair maps to a LIST of `(composer, priority)` entries,
# so several composers may share a signature at different priorities -- for
# example the numeric `(Subspace, Subspace)` composer and the ANALYTIC
# `(Subspace, Subspace)` cancel composer. A higher-priority tier (the
# analytic cancel composers) is tried ahead of the numeric composers
# regardless of hierarchy distance.
COMPOSER_REGISTRY = tx.Dict[XFORM_PAIR, tx.List[tx.Tuple[COMPOSER, int]]]
COMPOSERS: COMPOSER_REGISTRY = {}
# The fastmap caches, per concrete pair, the ordered tuple of candidate
# `(composer, priority)` entries `compose` should try, already sorted by
# dispatch order.
COMPOSERS_FASTMAP: tx.Dict[
    XFORM_PAIR, tx.Tuple[tx.Tuple[COMPOSER, int], ...]
] = {}

# Dispatch priority for a composer that decides purely from the types and
# object identity of its operands, without reading any parameter (today,
# the inverse-cancel pair `X @ X^-1 -> Identity`). It is tried ahead of the
# numeric composers, which register at the default priority 0.
ANALYTIC = 1


def distance(t1: type, t2: type, oriented: bool = True) -> int:
    """Compute the distance between two types in the class hierarchy."""
    # TODO: handle type hints (Union, Any)
    if t1 == t2:
        return 0
    if issubclass(t1, t2):
        return 1 + min(map(partial(distance, t2=t2), t1.__bases__))
    if issubclass(t2, t1) and not oriented:
        return 1 + min(map(partial(distance, t2=t1), t2.__bases__))
    return float("inf")
