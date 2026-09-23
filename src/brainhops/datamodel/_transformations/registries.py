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


def inverse_wrapper(cls: tx.Type["Transformation"]) -> tx.Type["Inverse"]:
    """The `Inverse` subclass that represents the inverse of `cls`.

    Only the base transformation types are registered in
    `INVERSE_WRAPPERS`, so a subclass -- a format-specific affine such as
    an `LPSToVoxel`, or any other refinement a reader declares -- has no
    entry of its own and is served by the entry of its nearest registered
    ancestor. The MRO is walked rather than the table scanned, so the
    answer is the *most derived* registered base and does not depend on
    the order the table happens to be filled in.

    Parameters
    ----------
    cls : type
        A forward transformation type.

    Returns
    -------
    wrapper : type
        The `Inverse` subclass that wraps `cls`.

    Raises
    ------
    KeyError
        If neither `cls` nor any of its bases is registered.
    """
    for base in cls.__mro__:
        wrapper = INVERSE_WRAPPERS.get(base)
        if wrapper is not None:
            return wrapper
    raise KeyError(cls)


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


XFORM_PAIR = tx.Tuple[tx.Type["Transformation"], tx.Type["Transformation"]]

CONVERTER = tx.Callable[["Transformation"], "Transformation"]
CONVERTER_REGISTRY = tx.Dict[XFORM_PAIR, CONVERTER]
CONVERTERS: CONVERTER_REGISTRY = {}
CONVERTERS_FASTMAP: CONVERTER_REGISTRY = {}

COMPOSER = tx.Callable[["Transformation", "Transformation"], "Transformation"]
# Each registered composer is stored with its dispatch priority, so a
# higher-priority tier (such as the analytic cancel composers) is tried
# ahead of the numeric composers regardless of hierarchy distance.
COMPOSER_REGISTRY = tx.Dict[XFORM_PAIR, tx.Tuple[COMPOSER, int]]
COMPOSERS: COMPOSER_REGISTRY = {}
# The fastmap caches, per concrete pair, the ordered tuple of candidate
# composers `compose` should try, already sorted by dispatch order.
COMPOSERS_FASTMAP: tx.Dict[XFORM_PAIR, tx.Tuple[COMPOSER, ...]] = {}

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
