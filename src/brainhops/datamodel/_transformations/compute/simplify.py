"""Simplification: optional rewrites into an equivalent, cheaper form.

A simplifier has one of two arities, registered as overloads of a single
dispatched function that [`simplify`][] selects by the number of arguments.
A simplifier with one input, `f(t, policy)`, always returns a
transformation, possibly `t` itself; it downcasts a leaf, for example an
Affine whose matrix is a permutation into a Permutation. A simplifier with
two inputs, `f(first, second, policy)`, rewrites a pair at no cost (a
transformation next to its inverse becomes the identity) or returns None to
decline, which is a common answer and not an exception. `first` is applied
before `second`, as in a [`Sequence`][].

Simplification is optional and never raises, unlike composition with
[`compose`][brainhops.datamodel._transformations.compose.compose], whose
refusal raises a CompositionError. The sweep of a sequence asks every
adjacent pair and mostly hears no, so it needs this contract. A simplifier
never lengthens a sequence and never reconciles a boundary, which is the
business of composition, so a pair whose systems disagree is declined.
`compose` tries the pair simplifiers first, because a cost-free rewrite is
always right when it applies, and for a pair such as `~field @ field` it is
the only answer.

Each simplifier receives, as `policy`, a [`SimplifyTable`][] that maps a
[`TransformationFamily`][] to a [`SimplifyPolicy`][], with a fallback under
the key None. The simplifier resolves the table against the transformation
in hand to none (leave it), analytic (read structure only) or numeric (read
values). A table is passed instead of a resolved level so that a wrapper
can resolve its inner transformation against the caller's own keys.
"""

__all__ = [
    "SimplifyLike",
    "SimplifyPolicy",
    "SimplifyTable",
    "ANALYTIC_FLOOR",
    "simplifier",
    "simplify",
    "get_simplifiers",
]

import inspect
from collections.abc import Mapping

import typing_extensions as tx
from bagof.converters import get_converter
from bagof.dispatchers import Function, NoMethodError

from brainhops.datamodel.enums import SimplifyPolicy

from .. import nocycles
from ..modes import (
    Family,
    FamilyLike,
    _is_family_like,
    is_family,
    normalize_family,
)
from .utils import boundary_disagrees

if tx.TYPE_CHECKING:
    from ..base import Transformation

LeafSimplifier = tx.Callable[..., "Transformation"]
"""Simplifier `f(t, policy)`, which always returns a transformation."""

PairSimplifier = tx.Callable[..., tx.Optional["Transformation"]]
"""Simplifier `f(first, second, policy)`, which may return None."""


# ======================================================================
#
#                              T Y P I N G
#
# ======================================================================


PolicyLike: tx.TypeAlias = tx.Union[SimplifyPolicy, bool, str, None]
"""Value that normalizes to a [`SimplifyPolicy`][]."""

SimplifyLike: tx.TypeAlias = tx.Union[
    PolicyLike,
    FamilyLike,
    tx.Iterable[FamilyLike],
    tx.Mapping[FamilyLike, PolicyLike],
]
"""Accepted value of the `simplify` argument of [`compute()`][]."""


# ======================================================================
#
#                            D I S P A T C H
#
# ======================================================================


# Both arities are overloads of one function, with call shapes that never
# compete. A leaf rule is total and the most specific one wins. A pair rule is
# partial and several may apply with none more specific, so `simplify` walks
# _simplify.candidates() most specific first, ties in registration order, until
# one does not decline.
_simplify: Function = Function("simplify")


def _arity_from_hints(func: tx.Callable) -> int:
    # Every parameter except `policy` and the catch-alls is an operand.
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
    # A genuine tie between two leaves raises AmbiguousMethodError when a call
    # reaches it.
    _simplify.register(func)
    return func


def _register_pair(func: tx.Callable) -> tx.Callable:
    # Specificity ties between pairs (about 255) are not defects: candidates()
    # returns them in registration order, and no single-winner call is made on
    # a pair.
    _simplify.register(func)
    return func


@tx.overload
def simplifier(T: tx.Type["Transformation"]) -> tx.Callable:
    """Return a decorator that registers a leaf simplifier for `T`."""
    ...


@tx.overload
def simplifier(
    T1: tx.Type["Transformation"], T2: tx.Type["Transformation"]
) -> tx.Callable:
    """Return a decorator that registers a pair simplifier for `(T1, T2)`."""
    ...


@tx.overload
def simplifier(func: tx.Callable) -> tx.Callable:
    """Register a simplifier, reading its operand types from its hints."""
    ...


def simplifier(*args) -> tx.Callable:
    """Register a simplifier.

    Used bare, as `@simplifier`, the decorator reads the operand types from the
    hints of every parameter except `policy`: one operand makes a leaf
    simplifier, and two make a pair simplifier. Called with explicit types, as
    in `@simplifier(Identity, Transformation)`, it keys the simplifier on those
    types instead, which suits generated simplifiers that carry no hints.
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
        # Overlay the operand hints and leave `policy` and the catch-alls as
        # written.
        overlay = (*types, SimplifyTable)
        _simplify.register(overlay)(func)
        return func

    return decorator


@tx.overload
def get_simplifiers(T: type) -> tx.Optional[LeafSimplifier]:
    """Return the leaf simplifier for `T`, or None."""
    ...


@tx.overload
def get_simplifiers(T1: type, T2: type) -> tx.Tuple[PairSimplifier, ...]:
    """Return the pair simplifiers for `(T1, T2)`, most specific first."""
    ...


def get_simplifiers(*types: type) -> tx.Any:
    """Return the simplifiers registered for one or two types.

    For one type, the leaf simplifier is returned, which is the single most
    specific rule, or None if no rule is registered. For two types, every
    applicable pair simplifier is returned, most specific first with ties in
    registration order. This is the order in which [`simplify`][] tries them,
    and the tuple is empty when no rule applies.

    Raises
    ------
    TypeError
        If the number of types is neither one nor two.
    """
    if len(types) == 1:
        try:
            return _simplify.resolve(types[0], SimplifyTable).function
        except NoMethodError:
            return None
    if len(types) == 2:
        return tuple(
            method.function
            for method in _simplify.resolve_candidates(
                types[0], types[1], SimplifyTable
            )
        )
    raise TypeError(
        "get_simplifiers() takes one transformation type (a leaf) or two "
        f"(a pair), not {len(types)}"
    )


def simplify(
    *transformations: "Transformation",
    policy: SimplifyLike = SimplifyPolicy.analytic,
) -> tx.Optional["Transformation"]:
    """Simplify one transformation, or a pair of consecutive transformations.

    Parameters
    ----------
    *transformations
        One transformation to downcast, or two consecutive ones (`first` before
        `second`) to collapse.
    policy
        How hard each transformation may be inspected. Anything that
        [`SimplifyTable.from_like`][] accepts is normalized to a
        [`SimplifyTable`][] before dispatch.

    Returns
    -------
    Transformation or None
        For one transformation, the simplified transformation, which is the
        transformation itself if nothing applies. For a pair, the single
        collapsed transformation, or None if the pair does not collapse.

    Raises
    ------
    TypeError
        If zero or more than two arguments are given, or if an argument is not
        a transformation.
    ValueError
        If no leaf simplifier is registered for the type, which only happens
        when the rules have not been imported.
    """
    if not transformations or len(transformations) > 2:
        raise TypeError(
            "simplify() takes one or two transformations, "
            f"not {len(transformations)}"
        )

    Transformation = nocycles.TRANSFORMATION
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
            return _simplify(t, policy=policy)
        except NoMethodError:
            # The root type always has a rule, so this error means that the
            # rules were not imported.
            raise ValueError(
                f"no simplifier registered for {type(t)}"
            ) from None

    first, second = transformations

    if boundary_disagrees(first, second):
        # Pair rules assume that the two sides of the boundary line up; a rule
        # that drops an operand would otherwise swallow the reordering or flip
        # between them. Reconciling is the job of `adapt`, which inserts an
        # element, and compute bridges before simplifying.
        return None

    # The first rule that does not decline gives the collapse. When every rule
    # declines, or none applies, the pair does not collapse, which is not an
    # error.
    for candidate in _simplify.candidates(first, second, policy=policy):
        result = candidate(first, second, policy=policy)
        if result is not None:
            return result
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

# Resolves a SimplifyPolicy by value and by lower-cased name.
_to_policy = get_converter(SimplifyPolicy)


def normalize_policy(value: tx.Any) -> SimplifyPolicy:
    """Normalize a single policy value to a [`SimplifyPolicy`][].

    True stands for numeric, False and None stand for none, and a string is
    matched case-insensitively.

    Raises
    ------
    ValueError
        If the value names no policy.
    """
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
    # SimplifyPolicy is a StrEnum, so comparing with < would follow string
    # order and rank 'analytic' below 'none'. Always rank through _POLICY_RANK.
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
    """Table that maps transformation families to simplification policies.

    The key None holds the fallback policy, which applies to transformations
    that match no other entry.
    """

    def __setitem__(self, family: Family, policy: SimplifyPolicy) -> None:
        # Colliding keys keep the safest policy, the same tie-break as in
        # resolution.
        if family in self:
            policy = _safest_policy(self[family], policy)
        super().__setitem__(family, policy)

    @classmethod
    def from_like(cls, value: SimplifyLike) -> tx.Self:
        """Normalize any accepted `simplify=` value into a table.

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
        if isinstance(value, Mapping):
            return cls.from_mapping(value)

        # A single policy value sets the fallback and names no family.
        if _is_policy_like(value):
            return cls({None: normalize_policy(value)})

        # Otherwise the value names families, which become analytic while the
        # fallback stays none: look at these kinds and leave everything else
        # alone.
        return cls.from_families(value)

    @classmethod
    def from_families(cls, value: SimplifyLike) -> tx.Self:
        """Build a table from one family key or an iterable of them."""
        # A (kind, ndim) pair is both family-like and iterable, so the single-
        # key test comes first.
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
        """Build a table from a mapping of families to policies."""
        if isinstance(value, cls):
            return value

        fallback = SimplifyPolicy.analytic  # no None key
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
        """Whether the table leaves every transformation untouched."""
        return all(p is SimplifyPolicy.none for p in self.values())

    def resolve(self, *transformations: "Transformation") -> SimplifyPolicy:
        """Resolve the policy of one or several transformations.

        Every entry whose family admits a transformation applies, and the
        safest of those policies wins; when no entry matches, the fallback
        applies. For several transformations, the safest of their policies
        wins, so a pair is rewritten only as hard as its most protected member
        allows.
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
"""Table that allows structural rewrites of anything, numeric ones of nothing.

This is the floor that every caller gets for free. `compose` uses it for
the cost-free tier that it tries before fusing anything.
"""
