"""Simplification: the optional, never-failing rewrites.

A *simplifier* rewrites transforms into an equivalent but cheaper
representation. It comes in two arities, dispatched by [`simplify`][]
through two `bagof.dispatchers` functions, one per arity:

* **one in, one out** -- `f(t, policy) -> Transformation`. Total: it always
  returns a transform, possibly `t` itself. This is the leaf downcast (an
  `Affine` whose matrix is a permutation becomes a `Permutation`).

* **two in, one out** -- `f(first, second, policy) -> Transformation | None`.
  Partial: `None` means "these two do not collapse", which is the common
  answer for an arbitrary adjacent pair and so must not be an exception.
  `first` is applied before `second`, in the order the two read in a
  [`Sequence`][]. This is the cost-free pair rewrite (`X` next to `X^-1`
  collapses to the identity; an identity next to anything disappears).

Contract
--------
Simplification is *optional* and *total*: it never raises, and declining is
always a legitimate answer. This is what separates it from
[`compose`][brainhops.datamodel._transformations.compose.compose], which is
*obligatory* and *partial* -- a caller of `compose` wants one object back,
and "no" there is a `CompositionError`. A sweep over a sequence asks about
every adjacent pair and is told "no" most of the time, so it needs the
simplify contract, not the compose one.

A simplifier may never make a sequence longer, and never reconciles a
boundary: inserting a bridge, or embedding an operand in a fuller axis
space, is composition's business. A pair whose systems disagree is simply
declined. (Inside `compute`, that pair has already been reconciled before
the simplify pass sees it -- see `sequence._compute_sequence`.)

`compose` consults the pair simplifiers first, as its cheapest tier: that is
not a policy knob on `compose` but the observation that a cost-free rewrite
is always the right answer when one applies -- and, for a pair such as
`~field @ field`, the only answer that exists at all.

Policy
------
Every simplifier takes its policy as a [`SimplifyTable`][], under the name
`policy`: a map from a [`TransformationFamily`][] to a
[`SimplifyPolicy`][], with a `None` fallback, so a caller can say "read the
values of affines but do not touch the fields". Each simplifier resolves it
against the transform in hand -- `none` (do not touch it), `analytic`
(structure only) or `numeric` (read values) -- rather than being handed a
single resolved level, so that a wrapper resolves its *inner* against the
caller's own keys.
"""

__all__ = [
    "SimplifyLike",
    "SimplifyPolicy",
    "SimplifyTable",
    "ANALYTIC_FLOOR",
    "simplifier",
    "simplify",
    "get_simplifier",
    "get_pair_simplifiers",
]

# stdlib
import inspect
import itertools
from collections.abc import Mapping

# dependencies
import typing_extensions as tx
from bagof.converters import get_converter
from bagof.dispatchers import Function, NoMethodError

# datamodel
from brainhops.datamodel.enums import SimplifyPolicy

# internals
from . import registries
from .modes import (
    Family,
    FamilyLike,
    _is_family_like,
    is_family,
    normalize_family,
)
from .utils import boundary_disagrees

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation

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
**application order** -- `first` is applied before `second`.
"""


# ======================================================================
#
#                              T Y P I N G
#
# ======================================================================


PolicyLike = tx.Union[SimplifyPolicy, bool, str, None]
"""Something that can be normalized to a [`SimplifyPolicy`][]."""

SimplifyLike = tx.Union[
    PolicyLike,
    FamilyLike,
    tx.Iterable[FamilyLike],
    tx.Mapping[FamilyLike, PolicyLike],
]
"""Possible input to the `simplify` argument of [`compute()`][]."""


# ======================================================================
#
#                            D I S P A T C H
#
# ======================================================================


# Two `bagof.dispatchers` functions, one per arity, so a call never has to
# tell a leaf simplifier from a pair one: `simplify` picks the function by
# how many transforms it was given.
#
# A leaf simplifier is *total* -- exactly one applies, the most specific, the
# way a method override wins -- which is `bagof.dispatchers`' native
# single-winner model. A pair simplifier is *partial*: it returns `None` to
# decline, and several may apply to one concrete pair with none more specific
# than another (`(Identity, Transformation)` against `(Transformation,
# Inverse)`, say). The bespoke registry broke that tie by registration order;
# here each pair carries a `priority` that decreases with registration order,
# so the earliest-registered wins a specificity tie, exactly as before. The
# `None`-decline contract is preserved -- the chosen simplifier returning
# `None` is the decline -- and, for the rules registered here, the winner's
# decision never differs from a lower candidate's, so single-winner dispatch
# reproduces the old chain of responsibility. (Where it would not, in general,
# is the gap written up in `docs/design/bagof-dispatchers-migration.md`.)
_simplify_leaf: Function = Function("simplify_leaf")
_simplify_pair: Function = Function("simplify_pair")
_pair_order = itertools.count()


def _arity_from_hints(func: tx.Callable) -> int:
    # The number of transform operands a bare `@simplifier` declares: every
    # parameter but `policy` and the catch-alls. One is a leaf, two a pair.
    n = sum(
        1
        for name, param in inspect.signature(func).parameters.items()
        if name != "policy"
        and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )
    if n not in (1, 2):
        raise TypeError(
            f"a simplifier takes one or two transformations (plus `policy`), "
            f"but {func.__qualname__} declares {n}"
        )
    return n


def _register_leaf(func: tx.Callable) -> tx.Callable:
    _simplify_leaf.register(func)
    return func


def _register_pair(func: tx.Callable) -> tx.Callable:
    # Earlier registrations win a specificity tie: a strictly decreasing
    # priority reproduces the bespoke registry's registration-order tie-break.
    _simplify_pair.register(func, priority=-next(_pair_order))
    return func


@tx.overload
def simplifier(T: tx.Type["Transformation"]) -> tx.Callable:
    """Return a decorator registering a leaf simplifier for `T`."""
    ...


@tx.overload
def simplifier(
    T1: tx.Type["Transformation"], T2: tx.Type["Transformation"]
) -> tx.Callable:
    """Return a decorator registering a pair simplifier for `(T1, T2)`."""
    ...


@tx.overload
def simplifier(func: tx.Callable) -> tx.Callable:
    """Register a simplifier, reading its types from its hints."""
    ...


def simplifier(*args) -> tx.Callable:
    """Register a simplifier.

    Used bare (`@simplifier`), the operand types are read from the type
    hints of every parameter but `policy`: one such parameter registers a
    leaf simplifier, two register a pair simplifier. Used with explicit
    types (`@simplifier(Identity, Transformation)`), those types key it
    instead -- the form a generated simplifier that carries no hints needs.
    """
    if len(args) == 1 and not isinstance(args[0], type):
        func = args[0]
        arity = _arity_from_hints(func)
        return _register_leaf(func) if arity == 1 else _register_pair(func)
    if not args or len(args) > 2 or not all(isinstance(a, type) for a in args):
        raise TypeError(
            "simplifier() takes a function, or one or two transformation "
            f"types, not {args!r}"
        )
    types = args

    def decorator(func: tx.Callable) -> tx.Callable:
        # Overlay the operand hints, leaving `policy` (and any catch-all) as
        # written, so the function is registered under the given types.
        overlay = (*types, SimplifyTable)
        if len(types) == 1:
            _simplify_leaf.register(overlay)(func)
        else:
            _simplify_pair.register(overlay, priority=-next(_pair_order))(func)
        return func

    return decorator


def get_simplifier(T: type) -> tx.Optional[LeafSimplifier]:
    """The leaf simplifier registered nearest to `T` in the class hierarchy.

    A leaf simplifier is *total*, so exactly one answers: the most specific
    one wins, the way a method override does.
    """
    try:
        return _simplify_leaf.resolve(T, SimplifyTable)
    except NoMethodError:
        return None


def get_pair_simplifiers(T1: type, T2: type) -> tx.Tuple[PairSimplifier, ...]:
    """The pair simplifier that applies to `(T1, T2)`, or an empty tuple.

    A pair simplifier is *partial* and may decline. The bespoke registry
    returned every applicable candidate, nearest first, for `simplify` to
    try in turn; `bagof.dispatchers` instead selects the single most
    specific one (ties broken by registration order, see [`simplifier`][]),
    so this returns that one -- as a one-tuple, to keep the shape callers
    expect -- or an empty tuple when none applies.
    """
    try:
        return (_simplify_pair.resolve(T1, T2, SimplifyTable),)
    except NoMethodError:
        return ()


def simplify(
    *transformations: "Transformation",
    policy: SimplifyLike = SimplifyPolicy.analytic,
) -> tx.Optional["Transformation"]:
    """Simplify one transform, or a pair of consecutive transforms.

    Parameters
    ----------
    *transformations : Transformation
        One transform to downcast, or two consecutive transforms --
        `first` applied before `second` -- to collapse into one.
    policy : simplify policy
        How hard each transform may be looked at. Anything
        [`SimplifyTable.from_like`][] accepts; it is normalized to a
        [`SimplifyTable`][] before dispatch.

    Returns
    -------
    Transformation or None
        For one transform, the simplified transform -- `t` itself when
        nothing applies. For a pair, the single transform the two collapse
        to, or `None` when they do not collapse.
    """
    if not transformations or len(transformations) > 2:
        raise TypeError(
            "simplify() takes one or two transformations, "
            f"not {len(transformations)}"
        )

    Transformation = registries.TRANSFORMATION
    if Transformation is not None and not all(
        isinstance(t, Transformation) for t in transformations
    ):
        raise TypeError(
            "simplify() takes transformations positionally and its policy "
            "as the `policy` keyword"
        )

    policy = SimplifyTable.from_like(policy)
    if policy.resolve(*transformations) is SimplifyPolicy.none:
        return transformations[0] if len(transformations) == 1 else None

    if len(transformations) == 1:
        (t,) = transformations
        try:
            return _simplify_leaf(t, policy=policy)
        except NoMethodError:
            # The root type is always registered, so this can only mean the
            # registry was not imported.
            raise ValueError(
                f"no simplifier registered for {type(t)}"
            ) from None

    first, second = transformations

    if boundary_disagrees(first, second):
        # Every pair rule assumes the two ends of the boundary line up: a
        # rule that drops one of the operands, or replaces both by an
        # identity, would swallow the reordering, rescaling or flip that
        # sits between them. Enforcing it here rather than in each rule is
        # what keeps a rule added later from forgetting it.
        #
        # Declining is the whole response. Reconciling the boundary is
        # `adapt`'s job, and it inserts an element -- which simplification
        # may not do. `compute` bridges before it simplifies, so within a
        # sequence the pair is reconciled by the time it gets here.
        return None

    try:
        # The single most specific pair simplifier (ties broken by
        # registration order). It returns `None` to decline, which -- like
        # no simplifier applying at all -- is the ordinary "these two do not
        # collapse" answer, not an error.
        return _simplify_pair(first, second, policy=policy)
    except NoMethodError:
        return None


# ======================================================================
#
#                              P O L I C Y
#
# ======================================================================

_POLICY_RANK = {
    SimplifyPolicy.none: 0,
    SimplifyPolicy.analytic: 1,
    SimplifyPolicy.numeric: 2,
}

# `bagof` converter: resolves a `SimplifyPolicy` by value and (after
# lower-casing) by name.
_to_policy = get_converter(SimplifyPolicy)


def normalize_policy(value: tx.Any) -> SimplifyPolicy:
    """Normalize a single policy value to a [`SimplifyPolicy`][]."""
    if value is True:
        return SimplifyPolicy.numeric
    if value is False or value is None:
        return SimplifyPolicy.none
    if isinstance(value, SimplifyPolicy):
        return value
    if isinstance(value, str):
        value = value.lower()
    try:
        return _to_policy(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"invalid simplify policy {value!r}; expected one of "
            "False/'none', 'analytic', True/'numeric'"
        ) from e


def _safest_policy(*policies: SimplifyPolicy) -> SimplifyPolicy:
    # The least aggressive (safest) policy. Never compare policies with `<`:
    # `SimplifyPolicy` is a `StrEnum`, so string order would rank
    # ``"analytic" < "none"``. Always go through `_POLICY_RANK`.
    return min(policies, key=_POLICY_RANK.__getitem__)


def _is_policy_like(value: tx.Any) -> bool:
    if value is None or value is True or value is False:
        return True
    if isinstance(value, SimplifyPolicy):
        return True
    if isinstance(value, str) and value.lower() in (
        "none",
        "analytic",
        "numeric",
    ):
        return True
    return False


# ======================================================================
#
#                               T A B L E
#
# ======================================================================


class SimplifyTable(dict):
    """
    A `Dict[TransformationFamily, SimplifyPolicy]` mapping.

    The `None` key holds the fallback policy, used for a transform that
    matches no other entry.
    """

    def __setitem__(self, family: Family, policy: SimplifyPolicy) -> None:
        # Colliding lowered keys keep the SAFEST policy
        #   >>> the same tie-break the resolution rule uses for several
        #   >>> matching entries.
        if family in self:
            policy = _safest_policy(self[family], policy)
        super().__setitem__(family, policy)

    @classmethod
    def from_like(cls, value: SimplifyLike) -> tx.Self:
        """
        Normalize any accepted `simplify=` input into a simplify table.

        | Input        | Output             |
        |--------------|--------------------|
        | `None`       | `{None: none}`     |
        | `False`      | `{None: none}`     |
        | `True`       | `{None: numeric}`  |
        | `"none"`     | `{None: none}`     |
        | `"analytic"` | `{None: analytic}` |
        | `"numeric"`  | `{None: numeric}`  |
        | a `policy: SimplifyPolicy`                 | `{None: policy}`                                |
        | a `key: str &verbar; type`                 | `{None: none, lowered(key): analytic}`          |
        | a `Iterable[key: str]`                     | `{None: none, lowered(key): analytic, ...}`     |
        | a `Dict[key: str, policy: SimplifyPolicy}` | `{None: <fallback>, lowered(key): policy, ...}` |
        """  # noqa: E501
        # A mapping is either already a `SimplifyTable` or can be easily
        # normalised to one. We'll let `from_mapping` handle it.
        if isinstance(value, Mapping):
            return cls.from_mapping(value)

        # A single policy value (bool, None, str, SimplifyPolicy) sets the
        # fallback and names no family.
        if _is_policy_like(value):
            return cls({None: normalize_policy(value)})

        # Otherwise the value names one or more families, which are given
        # the `analytic` policy while the `None` fallback stays `none`:
        # "look at these kinds, leave everything else alone".
        return cls.from_families(value)

    @classmethod
    def from_families(cls, value: SimplifyLike) -> tx.Self:
        """Build a table from one family key, or an iterable of them."""
        # A `(kind, ndim)` pair is family-like *and* iterable, so the
        # single-key test comes first.
        keys = [value]
        if not _is_family_like(value):
            if isinstance(value, str) or not isinstance(value, tx.Iterable):
                raise ValueError(f"invalid simplify policy: {value!r}")
            keys = list(value)
        table = cls()
        table[None] = SimplifyPolicy.none
        for key in keys:
            table[normalize_family(key)] = SimplifyPolicy.analytic
        return table

    @classmethod
    def from_mapping(cls, value: Mapping) -> tx.Self:
        """Build a table from a `{family: policy}` mapping."""
        if isinstance(value, cls):
            return value

        fallback = SimplifyPolicy.analytic  # the `None`-absent default
        table = cls()
        for family, policy in value.items():
            policy = normalize_policy(policy)
            if family is None:
                fallback = policy
            else:
                table[normalize_family(family)] = policy
        table[None] = fallback
        return table

    def is_noop(self) -> bool:
        """Whether this table leaves every transform untouched."""
        return all(p is SimplifyPolicy.none for p in self.values())

    def resolve(self, *transformations: "Transformation") -> SimplifyPolicy:
        """
        Resolve the simplify policy for one transform, or the policy shared
        by several.

        Every entry whose family admits the transform applies, and the
        safest of them wins; when none does, the `None` fallback applies.
        Given several transforms, the safest of their policies wins, so a
        pair is only rewritten as hard as its most protected member allows.

        Parameters
        ----------
        *transformations : Transformation
            The transforms to resolve.

        Returns
        -------
        SimplifyPolicy
            The resolved policy.
        """
        return _safest_policy(*map(self._resolve_one, transformations))

    def _resolve_one(self, t: "Transformation") -> SimplifyPolicy:
        matched = [
            policy
            for family, policy in self.items()
            if family is not None and is_family(t, family)
        ]
        return _safest_policy(*matched) if matched else self[None]


ANALYTIC_FLOOR = SimplifyTable({None: SimplifyPolicy.analytic})
"""
The table that allows a structural rewrite of anything and a numeric
rewrite of nothing. It is the floor every caller gets for free: `compose`
uses it for the cost-free tier it tries before fusing anything.
"""
