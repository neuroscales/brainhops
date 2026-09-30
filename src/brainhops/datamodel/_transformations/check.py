"""Dispatch a kind-membership question to a checker.

`is_kind(t, kind, compute=False)` is the single membership predicate used by
both `compute(mode)` (which kinds do we compose) and `simplify(policy)` (how
hard each leaf is looked at). It answers whether `t`'s membership in `kind`
is *established*: either by declaration (the `.register` promise, answered
by `isinstance`), or from structure (`compute=False`, analytic) or from
values (`compute=True`, numeric). Not-established is never refuted.

Two kinds of kind
-----------------
A `kind` is either

* a **kind node** -- a class defined in [`kinds`][], such as `kinds.Affine`.
  It denotes a *set* of maps, so membership can be established beyond the
  declared type: a `Linear` whose matrix happens to be diagonal is a member
  of the diagonal set. Kind nodes are the only kinds a checker may be
  registered against.

* a **class kind** -- a concrete transform (`Affine`), a field
  (`CartesianField`), a wrapper (`InverseAffine`), a container (`Sequence`).
  It denotes a *representation*, not a set, so membership is exactly
  `isinstance` and nothing establishes it beyond that. `"affine"` is the
  set; `Affine` is the class.

Either way a kind is *one* class, never several: one kind is one node of the
hierarchy, and a concrete transform is a node of it too -- it is registered
into it (`@kinds.Affine.register`), so `issubclass(Affine,
kinds.TransformationKind)` holds. What separates the two is
[`kinds.is_kind_node`][]: a kind node is a *real* (non-virtual) subclass of
[`kinds.TransformationKind`][], while a concrete transform is only ever a
virtual subclass of the node it registered to. What the separation decides is
whether membership can be established beyond `isinstance`.

Checkers
--------
A concrete or wrapper transform establishes membership in a kind node beyond
its declared type through a *checker*, registered against its own class in
[`is_kind`][], which is a [`Dispatcher`][] -- the same pattern as `convert`.
`@checker` on a function

    def _(query: Affine, kind: type[kinds.Translation], compute: bool) -> bool

declares "an `Affine` may be established as a member of the translation
set", and decides it for a given instance. The pair of types is read from
the first two hints, so a checker is written the way it is called; the
explicit form `@checker(Source, Node)` keys a function that carries no such
hints (a generated one). The implementations live in `checkers`, next to
nothing but each other, and register at import time; this module only holds
the machinery.

Dispatch
--------
Declared membership (`isinstance`) settles it affirmatively first, and a
class kind stops there -- it *is* `isinstance`. Otherwise every checker that
*applies* is asked, and their answers are OR-ed. A checker applies when

* its **source** is a supertype of `type(t)`, and it is the nearest such
  source registered against its node -- ordinary method resolution, so a
  wrapper (`Inverse`) shadows the concrete base it also inherits from, and a
  typed `InverseAffine` uses the recursing `Inverse` checker rather than the
  matrix checker that would read (and materialize) it;

* its **node** is a **subset** of the queried kind. That direction is what
  makes the answer sound: establishing `t in C` for a `C subset of kind`
  proves `t in kind`, while the converse proves nothing. It is also why a
  checker is passed the node that was *queried* rather than the one it was
  registered against -- a wrapper reasons about the question it was asked.

The disjunction is not belt and braces: the lattice is a DAG, and two
routes into a kind are routinely *incomparable*, so neither can stand for
the other. An `Affine` holding a permutation matrix is established in the
orthogonal set through the permutation node, and one holding a rotation
through the special-orthogonal node; a `Scaling` is established in the
bijections through the invertible-diagonal node, and in the translations
through the identity node. Ask only the nearest of those and the answer is
still sound, but it turns on which sibling happens to sit nearer -- so all
of them are asked, nearest first, and the first `True` wins.
"""

__all__ = [
    "Checker",
    "Family",
    "FamilyLike",
    "Kind",
    "KindLike",
    "checker",
    "get_checker",
    "is_family",
    "is_kind",
    "normalize_family",
    "normalize_kind",
    "register_kind_alias",
]

# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import kinds

# internals
from .registries import Dispatcher, type_distance

# typing
if tx.TYPE_CHECKING:
    from .base import Transformation as _Transformation

Transformation: tx.TypeAlias = "_Transformation"
Kind: tx.TypeAlias = tx.Type[kinds.TransformationKind]
"""
One node of the [`kinds`][] hierarchy -- a set node or, since every
concrete transform is registered into the hierarchy, a class kind.
"""
Family: tx.TypeAlias = kinds.TransformationFamily
Key: tx.TypeAlias = tx.Tuple[tx.Type[Transformation], Kind]
Checker: tx.TypeAlias = tx.Callable[[Transformation, Kind, bool], bool]
"""
A checker decides whether a transform is *established* in a kind node:
`checker(t, kind, compute) -> bool` (`compute=False` analytic, `True`
numeric). It is keyed by `(source transform type, kind node)` and is given
the node that was queried, which may be a superset of the one it was
registered against.
"""

KindLike: tx.TypeAlias = tx.Union[
    Kind,
    str,  # a NAME/SYMBOL or a registered alias
]
FamilyLike: tx.TypeAlias = tx.Union[
    Family,
    tx.Tuple[KindLike, tx.Optional[int]],
    KindLike,
    int,  # a dimension (the whole set, at that dimension)
]

KIND_ALIASES: tx.Dict[str, Kind] = {}
"""
The kinds a [`kinds`][] NAME or SYMBOL does not already name, keyed by name
(lower-case). A kind is either a kind node or a class matched by
`isinstance`; see the module docstring for what separates the two. The table
is populated by `checkers` at import time.
"""


# --- Dispatcher -------------------------------------------------------


class IsKind(Dispatcher[Key, Checker]):
    """
    A registry of the functions that check whether an instance is of a
    certain kind.

    The registry is a [`Dispatcher`][] that maps a `(transformation type,
    kind node)` pair to a checker function. When called with a transform and
    a kind, it asks every checker that applies, nearest first, and stops at
    the first one to establish membership -- see the module docstring for
    which checkers apply, and why more than one may.
    """

    @classmethod
    def key_from_func(cls, func: Checker) -> Key:
        hints = tx.get_type_hints(func)
        inp, out, *_ = hints.values()
        if tx.get_origin(out) is type:
            out = tx.get_args(out)[0]
        return inp, out

    @classmethod
    def key_from_args(
        cls, x: Transformation, kind: Kind, *args, **kwargs
    ) -> Key:
        return type(x), kind

    @classmethod
    def key_distance(
        cls, key: Key, registered: Key
    ) -> tx.Optional[tx.Tuple[float, float]]:
        # The source comes first: among the checkers registered against one
        # node, only the one whose source is nearest is asked (see
        # `candidates`), and that is what lets a wrapper shadow the concrete
        # base it also inherits from. The kind then orders the nodes, so the
        # largest subset of the question is asked first.
        source, kind = key
        registered_source, registered_kind = registered
        if not issubclass(source, registered_source):
            return None  # not an ancestor of the queried transform
        # Nearest *on the MRO*, not by `type_distance`: a `Generic`
        # subscription such as `Inverse[Affine]` inserts a real intermediate
        # class, which puts a whole extra step between `InverseAffine` and
        # `Inverse` and would make `type_distance` rank the concrete `Affine`
        # base nearer. The MRO has no such trouble: the wrapper and its
        # subscription both precede the base there, which is precisely the
        # precedence a subclass declared them with.
        mro = source.__mro__
        try:
            source_rank = mro.index(registered_source)
        except ValueError:
            source_rank = len(mro)  # an ancestor that is not on the MRO
        # The kind is *contravariant*: the registered node must be a subset
        # of the queried one, and the largest such subset is the nearest.
        kind_distance = type_distance(registered_kind, kind)
        if kind_distance == float("inf"):
            return None  # not a subset of the queried kind
        return source_rank, kind_distance

    def __setitem__(self, key: Key, func: Checker) -> None:
        # Every registration funnels through here, whichever form it was
        # written in, so this is where a key is checked.
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                f"a checker is keyed by a (transformation type, kind node) "
                f"pair, not by {key!r}"
            )
        _, kind = key
        if not kinds.is_transformation_set(kind):
            raise TypeError(
                f"a checker may only be registered against a kind node, not "
                f"{kind!r}. A class kind is matched by isinstance and needs "
                f"no checker."
            )
        super().__setitem__(key, func)

    @classmethod
    def candidate_group(cls, registered: Key) -> Kind:
        # The checkers registered against one node compete: whether a
        # transform is in that node has one answer, given by the checker
        # whose source is nearest, exactly as a method override would. Two
        # *different* nodes do not compete -- each proves its own set, and
        # incomparable sets prove different things -- so both are asked.
        _, node = registered
        return node

    def __call__(
        self, x: Transformation, kind: KindLike, compute: bool = False
    ) -> bool:
        """
        Whether `x`'s membership in `kind` is established at level `compute`.

        For example, for a `Linear` transformation:

        * `compute=False` (analytic) inspects structure only and assumes
          invertibility from shape (a square matrix is presumed invertible,
          a wide one surjective, a tall one injective).

        * `compute=True` (numeric) reads values and refines with rank.

        Parameters
        ----------
        x : Transformation
            The transformation whose membership is questioned.
        kind : kind-like
            A kind node, a class kind, or a string that names one -- a
            [`kinds`][] NAME or SYMBOL (`"affine"`, `"SO(3)"`) or a
            registered alias (`"inverse"`, `"displacements"`).
        compute : bool, default=False
            Whether values may be read to establish membership.

        Returns
        -------
        bool
            Whether membership is established.
        """
        kind = normalize_kind(kind)
        if isinstance(x, kind):
            return True
        if not kinds.is_transformation_set(kind):
            # A class kind denotes a representation, not a set: `isinstance`
            # is the whole answer, and nothing establishes it beyond that.
            return False
        compute = bool(compute)
        # Nothing applying is an answer -- "not established" -- so an empty
        # disjunction is exactly right, and never an error.
        return any(
            check(x, kind, compute)
            for check in self.candidates((type(x), kind))
        )


# --- Public API -------------------------------------------------------

is_kind: IsKind = IsKind()
"""Public membership predicate, that dispatches on its input types."""

checker = is_kind.register
"""Decorator to register a checker function."""

get_checker = is_kind.get
"""Get the checker function for a `(transformation type, kind node)` pair."""


def is_family(x: Transformation, family: FamilyLike) -> bool:
    """Whether a transform belongs to a [`kinds.TransformationFamily`][].

    A family is a kind and, optionally, a dimensionality; a transform
    belongs to it when its kind membership is established (at analytic:
    resolution never reads a value to decide which family admits a leaf)
    and its endpoints do not contradict the dimension.

    Parameters
    ----------
    x : Transformation
        The transformation whose membership is questioned.
    family : family-like
        A [`kinds.TransformationFamily`][], a `(kind, ndim)` pair, a bare
        dimension, a kind, or a string that names a kind.

    Returns
    -------
    bool
        Whether the family admits `x`.
    """
    family = normalize_family(family)
    if family.kind is not None and not is_kind(x, family.kind, compute=False):
        return False
    if family.ndim is None:
        return True
    # An endpoint that does not say how many axes it has cannot contradict
    # the dimension, so it does not reject: the family asks for a dimension
    # it can read, not for one it must be told.
    #
    # FIXME
    #   In many transforms, the ndim can be guessed from the content of the
    #   xform, even if the input/output spaces are not set (e.g. the shape
    #   of the matrix or the field).
    for endpoint in (x.input, x.output):
        axes = getattr(endpoint, "axes", None)
        if axes is not None and len(axes) != family.ndim:
            return False
    return True


# --- Public helpers ---------------------------------------------------


def register_kind_alias(
    name: str, cls: tx.Union[Kind, tx.Tuple[Kind, ...]]
) -> None:
    """Name a class, or a kind node, as a kind key.

    Wrapper and field kinds have no kind node -- their membership depends on
    their contents -- so a name that resolves to one is matched by
    `isinstance` (see [`is_kind`][]). A name that resolves to a kind node
    (e.g. `"scaling"` -> the diagonal set) keeps set semantics.
    """
    if not isinstance(cls, tuple):
        cls = (cls,)
    for a_cls in cls:
        KIND_ALIASES[name.lower()] = a_cls


def normalize_kind(kind_like: KindLike) -> Kind:
    """
    Resolve a kind-like object to a kind.

    Parameters
    ----------
    kind_like : kind-like
        A kind node, a class kind, or a string that names one.

    Returns
    -------
    kind : Kind
        A kind node, or a class kind.

    Raises
    ------
    ValueError
        If `kind_like` is a string that does not name a known kind.
    TypeError
        If `kind_like` is not a kind at all -- a tuple of kinds, say, which
        is several kinds and therefore none.
    """
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return alias
        try:
            return kinds.TransformationKind.parse(kind_like)
        except ValueError:
            raise ValueError(
                f"unknown transformation kind {kind_like!r}: not a kind "
                f"name/symbol and not one of {sorted(KIND_ALIASES)}"
            ) from None
    if not isinstance(kind_like, type):
        # `isinstance` would accept a tuple of classes, so the refusal has
        # to be explicit: one kind is one node of the hierarchy. Ask about
        # the base the classes share, or ask twice.
        raise TypeError(
            f"not a transformation kind: {kind_like!r}. A kind is one class "
            f"-- a node of `kinds`, or a concrete transformation class -- or "
            f"a string naming one."
        )
    return kind_like


def normalize_family(family_like: FamilyLike) -> Family:
    """
    Resolve a family-like object to a [`kinds.TransformationFamily`][].

    This is the single entry point that turns a user-written mode or
    simplify key into the `(kind, ndim)` pair the tables are keyed by.

    Parameters
    ----------
    family_like : family-like
        A [`kinds.TransformationFamily`][], a `(kind, ndim)` pair, a bare
        dimension, a kind, or a string that names a kind.

    Returns
    -------
    family : kinds.TransformationFamily
        The normalized family.

    Raises
    ------
    ValueError
        If the kind is a string that names nothing known, or if the input
        is not family-like at all.
    """
    # A `(kind, ndim)` pair names its dimension explicitly. It is the only
    # tuple a family reads: a tuple of *kinds* is not a kind.
    kind_like, ndims = family_like, ()
    if isinstance(family_like, tuple):
        if not kinds.is_family_tuple(family_like):
            raise ValueError(
                f"Invalid (kind, ndim) pair: {family_like!r}. A family pairs "
                f"one kind with one dimension (or None)."
            )
        kind_like, *ndims = family_like
    # A string goes through the alias table first, so that a name the kind
    # hierarchy does not know (a wrapper, a field, a friendlier spelling)
    # normalizes like any other key.
    if isinstance(kind_like, str):
        alias = KIND_ALIASES.get(kind_like.lower())
        if alias is not None:
            return kinds.TransformationFamily(alias, *ndims)
    return kinds.TransformationFamily.parse(kind_like, *ndims)
