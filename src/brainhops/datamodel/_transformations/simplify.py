"""Simplification: the optional, never-failing rewrites.

A *simplifier* rewrites transforms into an equivalent but cheaper
representation. It comes in two arities, both dispatched by [`simplify`][]
and both registered in the same `registries.SIMPLIFIERS` table:

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
boundary: inserting a bridge, or lifting an operand into a fuller axis
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
from collections.abc import Mapping
from functools import partial

# dependencies
import typing_extensions as tx

# bagof
from bagof.converters import get_converter

# datamodel
from brainhops.datamodel.enums import SimplifyPolicy

# internals
from . import registries
from .modes import (
    Family,
    FamilyLike,
    _is_family_like,
    matches_mode,
    normalize_family,
)
from .registries import (
    SIMPLIFIERS,
    SIMPLIFIERS_FASTMAP,
    LeafSimplifier,
    PairSimplifier,
    distance,
)
from .utils import boundary_disagrees

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation


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
    instead.
    """
    if len(args) == 1 and not isinstance(args[0], type):
        func = args[0]
        SIMPLIFIERS[_key_from_hints(func)] = func
        SIMPLIFIERS_FASTMAP.clear()
        return func
    if not args or not all(isinstance(a, type) for a in args):
        raise TypeError(
            "simplifier() takes a function, or one or two transformation "
            f"types, not {args!r}"
        )
    return partial(_register, key=args[0] if len(args) == 1 else args)


def _register(func: tx.Callable, key: tx.Any) -> tx.Callable:
    SIMPLIFIERS[key] = func
    SIMPLIFIERS_FASTMAP.clear()
    return func


def _key_from_hints(func: tx.Callable) -> tx.Any:
    # The operand types, read from the hints of every parameter that is not
    # the policy. One operand keys a leaf simplifier, two key a pair.
    hints = tx.get_type_hints(func)
    types = tuple(
        hints[name]
        for name, param in inspect.signature(func).parameters.items()
        if name != "policy"
        and param.kind
        not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    )
    if len(types) == 1:
        return types[0]
    if len(types) == 2:
        return types
    raise TypeError(
        f"a simplifier takes one or two transformations (plus `policy`), "
        f"but {func.__qualname__} declares {len(types)}"
    )


def get_simplifier(T: type) -> tx.Optional[LeafSimplifier]:
    """The leaf simplifier registered nearest to `T` in the class hierarchy.

    A leaf simplifier is *total*, so exactly one answers: the nearest one
    wins, the way a method override does.
    """
    if T in SIMPLIFIERS_FASTMAP:
        return SIMPLIFIERS_FASTMAP[T]
    best_distance, best_func = float("inf"), None
    for key, func in SIMPLIFIERS.items():
        if isinstance(key, tuple):
            continue  # a pair simplifier
        dist = distance(T, key)
        if dist < best_distance:
            best_distance, best_func = dist, func
    if best_distance == float("inf"):
        best_func = None
    SIMPLIFIERS_FASTMAP[T] = best_func
    return best_func


def get_pair_simplifiers(
    T1: type, T2: type
) -> tx.Tuple[PairSimplifier, ...]:
    """The pair simplifiers that apply to `(T1, T2)`, nearest first.

    A pair simplifier is *partial*, so several may apply and each may
    decline: the candidates form a chain of responsibility, exactly as the
    composers do. They are ordered by summed hierarchy distance, with
    registration order breaking a tie.
    """
    key = (T1, T2)
    cached = SIMPLIFIERS_FASTMAP.get(key)
    if cached is not None:
        return cached
    scored = []
    for order, (registered, func) in enumerate(SIMPLIFIERS.items()):
        if not isinstance(registered, tuple):
            continue  # a leaf simplifier
        A, B = registered
        dist = distance(T1, A) + distance(T2, B)
        if dist < float("inf"):
            scored.append((func, dist, order))
    scored.sort(key=lambda s: (s[1], s[2]))
    funcs = tuple(s[0] for s in scored)
    SIMPLIFIERS_FASTMAP[key] = funcs
    return funcs


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
    root = registries.TRANSFORMATION
    if not transformations or len(transformations) > 2:
        raise TypeError(
            "simplify() takes one or two transformations, "
            f"not {len(transformations)}"
        )
    if root is not None and not all(
        isinstance(t, root) for t in transformations
    ):
        raise TypeError(
            "simplify() takes transformations positionally and its policy "
            "as the `policy` keyword"
        )
    policy = SimplifyTable.from_like(policy)

    if len(transformations) == 1:
        t, = transformations
        func = get_simplifier(type(t))
        if func is None:
            # The root type is always registered, so this can only mean the
            # registry was not imported.
            raise ValueError(f"no simplifier registered for {type(t)}")
        return func(t, policy=policy)

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
    for func in get_pair_simplifiers(type(first), type(second)):
        result = func(first, second, policy=policy)
        if result is not None:
            return result
    # No pair simplifier applies, or every one declined: the two do not
    # collapse. That is the ordinary answer, not an error.
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
    if (
        isinstance(value, str) and
        value.lower() in ("none", "analytic", "numeric")
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
            if family is not None and matches_mode(t, family)
        ]
        return _safest_policy(*matched) if matched else self[None]


ANALYTIC_FLOOR = SimplifyTable({None: SimplifyPolicy.analytic})
"""
The table that allows a structural rewrite of anything and a numeric
rewrite of nothing. It is the floor every caller gets for free: `compose`
uses it for the cost-free tier it tries before fusing anything.
"""
