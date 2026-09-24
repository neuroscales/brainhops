# stdlib
from functools import partial

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel.hierarchy import Transformation as TransformationSet

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation
    from .inverse import Inverse
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


# --- adapt ------------------------------------------------------------

ADAPT: tx.Optional[tx.Callable[..., "Sequence"]] = None
"""
The adaptor lives in `adaptors`, which imports this module. It
registers itself here at import time, so the sequence machinery can call
it without importing that module at load time and forming a cycle. The
adaptor reconciles two consecutive transforms: it bridges a boundary
where the two systems merely reorder, rescale or flip their shared axes,
and it lifts a transform that acts on a subset of a boundary's axes into
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


# --- convert ----------------------------------------------------------

PairOfTypes = tx.Tuple[tx.Type["Transformation"], tx.Type["Transformation"]]

Converter = tx.Callable[["Transformation"], "Transformation"]
ConverterRegistry = tx.Dict[PairOfTypes, Converter]

CONVERTERS: ConverterRegistry = {}
CONVERTERS_FASTMAP: ConverterRegistry = {}

# --- compose ----------------------------------------------------------

Composer = tx.Callable[["Transformation", "Transformation"], "Transformation"]
ComposerRegistry = tx.Dict[PairOfTypes, Composer]
"""
A composer *fuses* two transforms into one by reading their parameters
(multiplying matrices, resampling fields). Cost-free rewrites that decide
from types and object identity alone are not composers: they are two-argument
simplifiers (see `SIMPLIFIERS`), which `compose` consults first.
"""

COMPOSERS: ComposerRegistry = {}
COMPOSERS_FASTMAP: tx.Dict[PairOfTypes, tx.Tuple[Composer, ...]] = {}
"""
The fastmap caches, per concrete pair, the ordered tuple of candidate
composers `compose` should try, already sorted by dispatch order.
"""

# --- check ------------------------------------------------------------

Kind = tx.Union[
    tx.Type[TransformationSet],  # a hierarchy node
    tx.Type["Transformation"]    # a concrete/field/wrapper class
]

Checker = tx.Callable[["Transformation", bool], bool]
"""
A checker decides whether a transform is *established* in a hierarchy set:
`checker(t, compute) -> bool` (`compute=False` analytic, `True` numeric).
It is keyed by `(source transform type, hierarchy kind node)`, dispatched
like the composers/converters (nearest source in the class hierarchy wins).
"""

CheckerKey = tx.Tuple["Transformation", Kind]
CheckerRegistry = tx.Dict[CheckerKey, Checker]


CHECKERS: CheckerRegistry = {}
CHECKERS_FASTMAP: tx.Dict[
    tx.Type["Transformation"],
    tx.Tuple[tx.Tuple[Kind, Checker], ...]
] = {}
"""
The fastmap caches, per concrete source type, the (kind node, checker)
pairs that apply to it, already reduced to the nearest source per kind.
"""


KIND_ALIASES: tx.Dict[str, Kind] = {}
"""
Registry of known transformation kinds, keyed by name (lower-case).

A kind is either a hierarchy set node, or a concrete/field/wrapper class
(or tuple of classes) matched by `isinstance`. See `check` for what
separates the two.

The registry is populated by `checkers` at import time.
"""

# --- simplify ---------------------------------------------------------

LeafSimplifier = tx.Callable[..., "Transformation"]
"""
A one-argument simplifier: `f(t, policy) -> Transformation`. It is *total*
(it always returns a transform, possibly `t` itself) and it rewrites one
transform into an equivalent, cheaper one. `policy` is a `SimplifyTable`,
which the simplifier resolves against `t`.
"""

PairSimplifier = tx.Callable[..., tx.Optional["Transformation"]]
"""
A two-argument simplifier:
`f(first, second, policy) -> Transformation | None`.
It is *partial*: `None` means "these two do not collapse", which is the
common answer and so must not be an exception. The arguments are in
**application order** -- `first` is applied before `second`, exactly as the
two read in a [`Sequence`][]. Note that this is the reverse of `compose`,
which takes its operands in matrix order; `compose` flips them at the single
point where it delegates here.
"""

SimplifierKey = tx.Union[
    tx.Type["Transformation"],                # a leaf simplifier
    PairOfTypes,                              # a pair simplifier
]
SimplifierRegistry = tx.Dict[
    SimplifierKey, tx.Union[LeafSimplifier, PairSimplifier]
]

SIMPLIFIERS: SimplifierRegistry = {}
SIMPLIFIERS_FASTMAP: SimplifierRegistry = {}
"""
One registry holds both arities, keyed by a single type (leaf) or by a pair
of types (pair). `simplify()` dispatches on the number of transforms it is
given, so the two never collide.
"""

# --- utils ----------------------------------------------------------


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
