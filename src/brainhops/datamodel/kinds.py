r"""
This module defines a transformation

The class A is a subclass of B if A can be converted to B without loss.
For example, Linear is a subclass of Affine. This is akin to set theory
in mathematics.

This differs from the usual "type hierarchy", where inheriting classes
can be more specialized then their parents.

We implement both types of hierarchy, so that users can easily check
whether a transformation can be conceptually thought as a given class
of transformation (e.g. "is this transformation a linear transform?"),
without it being conflated with type inheritance
(e.g. "is this transformation an instance of the Linear type?").

In our hierarchy, transformation kinds correspond to sets, and
inheritance describes set inclusion: `issubclass(A, B)` implies
`A ⊆ B`. Transformations map from ℝⁿ to ℝᵐ and can be composed when
their domains and codomains match. Homeomorphisms require n = m.

Some of these transformation sets (but not all!) are groups under the
composition operator. This is indicated by the [`group`][] decorator.
The group structure refers to transformations of a fixed space ℝⁿ under
composition; these groups act on ℝⁿ. If a set A is a subset of set B,
and both A and B are groups, then A is a subgroup of B.

Many linear transformations, when constrained to be invertible, can
be thought of as members of a Lie group, i.e. of a smooth manifold,c
and their group operations (composition and inversion) are smooth maps.
This is indicated by the [`liegroup`][] decorator. Note that the set A
may be a subgroup of a Lie group B without being a Lie group itself!

Lie groups
----------

```
G | > 0 | = 1 | ⋉ T
======================
GL ------------- Aff             General Linear     | Affine
 | \             |
 |   GL+ ------- Aff+            Positive Linear    | Positive Affine
 |    | \        |
 |    |  SL ---- SAff            Special Linear     | Special Affine
 |    |   |      |
CO ------------- Sim             Conformal          | Affine Conformal
 | \  |   |      |
 |   CO+ ------- Sim+            Special Conformal  | Special Affine Conformal
 |    |   |      |
 O ------------- E               Orthogonal         | Euclidean
 | \  | /        |
 S   SO -------- SE              Special Orthogonal | Special Euclidean
 | /             |
 I ------------- T               Identity           | Translation
```

General Lie groups (GL/CO/O) are in general not "connected", and
instead composed of two disconnected components. One that contains
the identity, and one that does not (the "flipped" version).

Positive Lie groups (SO/CO+/GL+) are restricted to transformations with
positive determinant, and therefore exclude flips. They can be defined
as the component from the corresponding general group that contains the
identity transform.

All classical linear group can be extended with the translation group
(⋉ T) to define affine groups.

!!! info
    * The Special Euclidean group (SE) contains transformations that are
      classicaly referred to as "rigid-body" transformations. They
      preserve angles and volumes.

    * The Special Conformal Euclidean group (Sim+) contains
      transformations that are classicaly referred to as "similitude"
      (or conformal) transformations. They only preserve (oriented) angles.

!!! warning

    Python preserves the declared order of bases in the method resolution
    order (MRO), whereas set inclusion imposes no order between unrelated
    supersets. These ordering constraints can conflict, even when the
    inclusion hierarchy is acyclic. Resolving a conflict may require
    reordering bases throughout the hierarchy or explicitly listing
    ancestors that are already inherited indirectly. Neither change
    alters the intended set inclusions.
"""

# ======================================================================
#   LIST OF ALL SETS
# ======================================================================
#
# The Symbol column gives the SYMBOL attribute of each kind, which is
# the exact string that `TransformationKind.parse` and
# `TransformationFamily.parse` accept. An empty Symbol cell means that
# the kind has no SYMBOL and can only be parsed from its name or from
# one of its aliases. A test in `tests/test_hierarchy_membership.py`
# checks this table against the code.
#
# +-------------------------------------+--------+---------+-----------+
# | Name                                | Symbol | Def.    | Property  |
# +-------------------------------------+--------+---------+-----------+
# |                              GENERAL                               |
# +-------------------------------------+--------+---------+-----------+
# | TransformationKind                  |        |         |           |
# | Transformation                      | Map    |         |           |
# | Injection                           |        |         |           |
# | Surjection                          |        |         |           |
# | Bijection                           |        |         |           |
# | Diffeomorphism                      | Diff   |         |det Df ≠ 0 |
# | VolumePreservingDiffeomorphism      |        |         |det Df =±1 |
# | OrientationPreservingDiffeomorphism | Diff+  |         |det Df > 0 |
# | ConformalDiffeomorphism             |        |         |           |
# | SpecialDiffeomorphism               | SDiff  |         |det Df = 1 |
# +-------------------------------------+--------+---------+-----------+
# |                              MATRIX                                |
# +-------------------------------------+--------+---------+-----------+
# | Matrix                              |        |         |           |
# | InvertibleMatrix                    |        |         | det ≠ 0   |
# | OrientationPreservingMatrix         |        |         | det > 0   |
# | VolumePreservingMatrix              |        |         | det =±1   |
# +-------------------------------------+--------+---------+-----------+
# |                              AFFINE                                |
# +-------------------------------------+--------+---------+-----------+
# | Affine                              |        |         | ⊄ Gp      |
# | InvertibleAffine                    | Aff    |         | det ≠ 0   |
# | PositiveAffine                      | Aff+   |         | det > 0   |
# | VolumePreservingAffine              |        |         | det =±1   |
# | SpecialAffine                       | SAff   | SL  ⋉ T | det = 1   |
# | ConformalEuclidean                  | Sim    | CO  ⋉ T | det ≠ 0   |
# | SpecialConformalEuclidean           | Sim+   | CO+ ⋉ T | det > 0   |
# | Euclidean                           | E      | O   ⋉ T | det =±1   |
# | SpecialEuclidean                    | SE     | SO  ⋉ T | det = 1   |
# | Dilation                            | ℝ* ⋉ T | ℝ*  ⋉ T | det ≠ 0   |
# | PositiveDilation                    | ℝ+ ⋉ T | ℝ+  ⋉ T | det > 0   |
# | Translation                         | T      |         | det = 1   |
# +-------------------------------------+--------+---------+-----------+
# |                              LINEAR                                |
# +-------------------------------------+--------+---------+-----------+
# | Linear                              |        |         | ⊄ Gp      |
# | InvertibleLinear                    | GL     |         | det ≠ 0   |
# | PositiveLinear                      | GL+    |         | det > 0   |
# | SpecialLinear                       | SL     |         | det = 1   |
# | ConformalOrthogonal                 | CO     | O  x ℝ+ | det ≠ 0   |
# | SpecialConformalOrthogonal          | CO+    | SO x ℝ+ | det > 0   |
# | Orthogonal                          | O      |         | det =±1   |
# | SpecialOrthogonal                   | SO     |         | det = 1   |
# | GeneralizedPermutation              | S ⋉ Δ  | S ⋉ Δ   | det ≠ 0   |
# | SignedPermutation                   | B      | C₂ⁿ ⋊ S | det ≠ 0   |
# | Permutation                         | S      |         | det ≠ 0   |
# | EvenPermutation                     | A      |         | det = 1   |
# | OddPermutation                      | S \ A  |         | det = -1  |
# | Diagonal                            |        |         | ⊄ Gp      |
# | InvertibleDiagonal                  | Δ      |         | det ≠ 0   |
# | PositiveDiagonal                    | Δ+     |         | det > 0   |
# | SpecialDiagonal                     | Δ_S    | Δ ∩ SL  | det = 1   |
# | OrthogonalDiagonal                  | Δ_O    | Δ ∩ O   | det =±1   |
# | SpecialOrthogonalDiagonal           | Δ_SO   | Δ ∩ SO  | det = 1   |
# | Multiplicative                      | ℝ      |         | ⊄ Gp      |
# | InvertibleMultiplicative            | ℝ*     |         | det ≠ 0   |
# | PositiveMultiplicative              | ℝ+     |         | det > 0   |
# +-------------------------------------+--------+---------+-----------+
# |                             IDENTITY                               |
# +-------------------------------------+--------+---------+-----------+
# | Identity                            | I      |         | det = 1   |
# +-------------------------------------+--------+---------+-----------+
#
# ======================================================================

__all__ = []

# stdlib
import re
from abc import ABC
from collections.abc import Sequence as AbcSequence
from functools import lru_cache

import typing_extensions as tx
from bagof.magic import Magic, replace

from brainhops._core.compat import PLACEHOLDER, partial

# --- API helpers ------------------------------------------------------


def public(obj: tx.Any) -> tx.Any:
    # Mark an object as public, which adds its name to __all__
    __all__.append(obj.__name__)
    return obj


# ======================================================================
#
#                           R E G I S T R I E S
#
# ======================================================================

_Type: tx.TypeAlias = tx.Type["TransformationKind"]
_Decorator: tx.TypeAlias = tx.Callable[["_Type"], "_Type"]
_Set: tx.TypeAlias = tx.Set[_Type]
_StrMap: tx.TypeAlias = tx.Dict[str, _Type]
_TypeMap: tx.TypeAlias = tx.Dict[_Type, _Type]

GROUPS: _Set = set()
LIE_GROUPS: _Set = set()
CONNECTED: _Set = set()
SIMPLYCONNECTED: _Set = set()
CLOSEDUNDER: tx.Dict[_Type, _Set] = {}
NONEMBEDDABLE: _Set = set()

NAMETOCLASS: _StrMap = {}
FSYMBOLTOCLASS: _StrMap = {}
SYMBOLTOCLASS: _StrMap = {}
INVERTIBLE_OF: _TypeMap = {}
NONINVERTIBLE_OF: _TypeMap = {}


# --- checks -----------------------------------------------------------


@public
def is_group(cls: _Type) -> bool:
    """Return whether a set of transformations forms a group.

    A class is recognized as a group when it was marked with the
    [`group`][] or [`liegroup`][] decorator.
    """
    return cls in GROUPS


@public
def is_lie_group(cls: _Type) -> bool:
    """Return whether a set of transformations forms a Lie group.

    A class is recognized as a Lie group when it was marked with the
    [`liegroup`][] decorator.
    """
    return cls in LIE_GROUPS


@public
def is_connected(cls: _Type) -> bool:
    """Return whether a set of transformations is connected.

    A class is recognized as connected when it was marked with the
    [`connected`][] or [`simplyconnected`][] decorator.
    """
    return cls in CONNECTED


@public
def is_simplyconnected(cls: _Type) -> bool:
    """Return whether a set of transformations is simply connected.

    A class is recognized as simply connected when it was marked with
    the [`simplyconnected`][] decorator.
    """
    return cls in SIMPLYCONNECTED


@public
def is_invertible(cls: _Type) -> bool:
    """
    Return whether a set of transformations is known to be invertible.
    """
    return issubclass(cls, Bijection)


@public
@lru_cache(maxsize=None)  # noqa: UP033
def is_closedunder(cls: _Type, subcls: _Type) -> bool:
    """
    Return whether a set of transformations is closed under composition
    with another set.

    This is always the case when `subcls` is a subgroup of `cls`, or
    when `subcls` is a subset of `cls` and `cls` is closed.

    There can be more special cases, which are registered with the
    [`closedunder`][] decorator.
    """
    # 1) Subset of group
    if is_group(cls) and issubclass(subcls, cls):
        return True
    # 2) Registered closure
    if subcls in CLOSEDUNDER.get(cls, set()):
        return True
    # 3) cls is closed under a superset of subcls
    for supcls in subcls.__bases__:
        if is_closedunder(cls, supcls):
            return True
    return False


@public
def is_closed(cls: _Type) -> bool:
    """
    Return whether a set of transformations is closed under composition.
    """
    return is_closedunder(cls, cls)


@public
def is_transformation_set(cls: tx.Any) -> bool:
    """
    Return whether an object is a node of this hierarchy, i.e. whether
    it denotes a *set* of maps rather than a representation of one.

    True for a class defined in this module (a real subclass of
    [`TransformationKind`][]), False for a concrete transformation, a
    field, a wrapper or a container. A concrete transformation is
    only ever a *virtual* subclass of the node it registered to, and
    virtual registration does not touch the MRO, so the test separates
    the two cleanly.
    """
    if not isinstance(cls, type):
        return False
    return TransformationKind in cls.__mro__


@public
def is_embeddable(cls: _Type) -> bool:
    """
    Return whether a set of transformations is known to embed into a
    higher-dimensional space by padding with the identity.

    The embedding is `T -> T ⊕ I`, block-diagonal: `T` acts on the axes
    it already acted on, and the new ones pass through. The set is
    embeddable when the result stays in the same set one dimension up,
    `G(n) ⊕ I ⊆ G(n+k)`. For example:

    * Every element of SO(3) is an element of SO(4) that way;
    * Most elements of ℝ*(3) (a scaling by the same factor along every
      axis) are not elements of ℝ*(4), since e.g. `diag(2, 2, 2, 1)`
      is not isotropic in ℝ⁴.

    !!! note

        This is an *embedding*: an injective homomorphism placing one
        group inside a bigger one.
    """
    # I don't try to infer embeddability from the subsets or supersets of
    # cls, because it is not very robust and depends very much on the
    # definition of embeddability (which "properties" do we want conserved).
    # Here, the property is "belong to the same group type" (but not the
    # exact same group since we have changed dimensions). For example,
    # all elements of SO(3) are elements of SO(4) when embedded from ℝ^3
    # to ℝ^4. But SO(3) has subgroups that are embeddable and others that
    # are not, and it has supergroups that are embeddable and other that
    # are not.
    return cls not in NONEMBEDDABLE


# --- generators -------------------------------------------------------


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_sets() -> tx.FrozenSet[type]:
    """Return all known transformation sets."""
    return frozenset(NAMETOCLASS.values())


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_groups() -> tx.FrozenSet[type]:
    """Return all known transformation groups."""
    return frozenset(GROUPS)


@public
@lru_cache(maxsize=1)  # noqa: UP033
def all_lie_groups() -> tx.FrozenSet[type]:
    """Return all known Lie groups."""
    return frozenset(LIE_GROUPS)


@public
def as_invertible(cls: _Type) -> _Type:
    """
    Return the invertible subset of a set of transformations, if any.

    Raises
    ------
    TypeError
        If the set is not invertible and has no known invertible subset.
    """
    node = INVERTIBLE_OF.get(cls, cls)
    if not is_invertible(node):
        raise TypeError(
            f"{cls} is not a known invertible set of transformations"
        )
    return node


@public
def as_unrestricted(cls: _Type) -> _Type:
    """
    Return the superset of a set of transformations that does not
    eclude non-invertible transformations (if any).
    """
    return NONINVERTIBLE_OF.get(cls, cls)


# --- decorators -------------------------------------------------------


def group(cls: _Type) -> _Type:
    """
    Mark the set of transformations as forming a group under composition.

    wiki: https://en.wikipedia.org/wiki/Group_(mathematics)
    """
    GROUPS.add(cls)
    return cls


def liegroup(cls: _Type) -> _Type:
    """
    Mark the set of transformations as forming a Lie group under composition.

    wiki: https://en.wikipedia.org/wiki/Lie_group
    """
    LIE_GROUPS.add(cls)
    return group(cls)


def connected(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being connected.

    wiki: https://en.wikipedia.org/wiki/Connected_space
    """
    CONNECTED.add(cls)
    return cls


def simplyconnected(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being simply connected.

    wiki: https://en.wikipedia.org/wiki/Simply_connected_space
    """
    SIMPLYCONNECTED.add(cls)
    return connected(cls)


def invertible_subset_of(cls: _Type) -> _Decorator:
    """
    Mark the set of transformations as the invertible subset of another set.

    wiki: https://en.wikipedia.org/wiki/Inverse_element
    """

    def decorator(subcls: _Type) -> _Type:
        INVERTIBLE_OF[cls] = subcls
        NONINVERTIBLE_OF[subcls] = cls
        return subcls

    return decorator


def closedunder(cls: _Type) -> _Decorator:
    """
    Mark the set of transformations as being closed under compatible
    composition with another set.

    The argument is a set of transformations that is a superset of `cls`.

    wiki: https://en.wikipedia.org/wiki/Closure_(mathematics)
    """

    def decorator(subcls: _Type) -> _Type:
        CLOSEDUNDER.setdefault(cls, set()).add(subcls)
        return subcls

    return decorator


def closed(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being closed under
    compatible composition.

    wiki: https://en.wikipedia.org/wiki/Closure_(mathematics)
    """
    return closedunder(cls)(cls)


def nonembeddable(cls: _Type) -> _Type:
    """
    Mark the set of transformations as *not* embedding into a
    higher-dimensional space by padding with the identity.

    See [`is_embeddable`][] for what the embedding is.
    """
    NONEMBEDDABLE.add(cls)
    return cls


def alias(cls: _Type, *names: str) -> _Type:
    """
    Register a transformation set under an additional name.

    This is useful for sets that have multiple common names.
    """
    if isinstance(cls, str):
        return partial(alias, PLACEHOLDER, cls, *names)

    if cls.__name__ not in __all__:
        __all__.append(cls.__name__)

    if "ALIAS" not in cls.__dict__:
        cls.ALIAS = []  # ensure not inherited
    if isinstance(cls.ALIAS, str):
        cls.ALIAS = [cls.ALIAS]
    if isinstance(cls.ALIAS, tuple):
        cls.ALIAS = list(cls.ALIAS)
    cls.ALIAS.extend(names)
    for name in names:
        NAMETOCLASS[name.lower()] = cls
        globals()[name] = cls
        __all__.append(name)
    return cls


# ======================================================================
#
#                           F A M I L I E S
#
# ======================================================================

Kind: tx.TypeAlias = tx.Type["TransformationKind"]
Dim: tx.TypeAlias = tx.Optional[int]
KindLike: tx.TypeAlias = tx.Union[Kind, str]
FamilyTuple: tx.TypeAlias = tx.Tuple[KindLike, Dim, Dim]
FamilyTupleLike: tx.TypeAlias = tx.Union[
    tx.Tuple[KindLike], tx.Tuple[KindLike, Dim], tx.Tuple[KindLike, Dim, Dim]
]
FamilyLike: tx.TypeAlias = tx.Union[tx.Self, FamilyTupleLike, KindLike, int]
MISSING = object()


@public
class TransformationFamily(
    AbcSequence, Magic, eq=True, hash=True, frozen=True
):
    """
    A parametric family of transformations.

    A family is parameterized by a base type and, optionally,
    input and/or output dimensions.
    """

    # --- attributes ---------------------------------------------------

    kind: Kind
    """All transformations in this family are instances of this type."""

    ndim: Dim = MISSING
    """The dimensionality of the transformations in this family, if known."""

    odim: Dim = MISSING
    """
    The output dimensionality of the transformations in this family, if known.
    """

    def __post_init__(self) -> None:
        if self.ndim is MISSING:
            object.__setattr__(self, "ndim", None)
        if self.odim is MISSING:
            object.__setattr__(self, "odim", self.ndim)

    @property
    def name(self) -> str:
        """The preferred name of the transformation family."""
        return self.kind.__name__

    @property
    def names(self) -> tx.Tuple[str, ...]:
        """The names of the transformation family."""
        name = self.kind.__name__
        alias = self.kind.__dict__.get("ALIAS", ())
        if isinstance(alias, str):
            alias = (alias,)
        return (name, *alias)

    @property
    def symbol(self) -> tx.Optional[str]:
        """The symbol of the transformation family."""
        return self.kind.__dict__.get("SYMBOL")

    @property
    def fsymbol(self) -> tx.Optional[str]:
        """
        The symbol of the transformation family, with its dimension.

        If the family dimensionality is unknown, the dimension is the
        placeholder `{n}`.
        """
        if (fsymbol := self.kind.__dict__.get("FSYMBOL")) is None:
            return None
        ndim = "{n}" if self.ndim is None else self.ndim
        odim = "{m}" if self.odim is None else self.odim
        fsymbol = fsymbol.format(n=ndim, m=odim)
        return fsymbol

    # --- magic --------------------------------------------------------
    # A family has three items, so it unpacks into (kind, ndim, odim).

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> tx.Union[Kind, Dim]:
        if index == 0:
            return self.kind
        if index == 1:
            return self.ndim
        if index == 2:
            return self.odim
        raise IndexError(
            f"index {index} out of range for TransformationFamily"
        )

    def __iter__(self) -> tx.Iterator[tx.Union[Kind, Dim]]:
        yield self.kind
        yield self.ndim
        yield self.odim

    # --- to -----------------------------------------------------------

    def to_tuple(self) -> tx.Tuple[Kind, Dim, Dim]:
        """Convert to a (kind, ndim, odim) tuple."""
        return (self.kind, self.ndim, self.odim)

    # --- from ---------------------------------------------------------

    @classmethod
    def parse(
        cls, repr: FamilyLike, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Return a transformation family from its name, symbol or type.

        The input `repr` can be a family, a tuple
        `(kind[, ndim[, odim]])`, a bare input dimension, a kind, or a
        string that holds the name, an alias or the symbol of a kind.

        When `ndim` or `odim` is left at its default value, the family
        keeps the dimension that `repr` carries, so parsing a family
        returns the same family. A value that is passed explicitly
        replaces the dimension carried by `repr`, and passing `None`
        explicitly means that the dimension is unknown. When only
        `ndim` is passed, `odim` follows the new `ndim` if `repr` does
        not give an output dimension of its own. A family always holds
        an output dimension, so for a family an output dimension equal
        to the input dimension is treated as following it, and
        `parse(parse("SO(3)"), ndim=None)` and
        `parse("SO(3)", ndim=None)` both give a family whose two
        dimensions are unknown.

        The cases are tested in an order that never passes a value to
        a check that cannot handle it, because a bare `issubclass`
        would raise on an `int` or a `str`.
        """
        # --- already a family ---
        if isinstance(repr, TransformationFamily):
            if ndim is MISSING:
                ndim = repr.ndim
                if odim is MISSING:
                    odim = repr.odim
            elif odim is MISSING and repr.odim != repr.ndim:
                odim = repr.odim
            if (repr.ndim, repr.odim) == (ndim, odim):
                return repr
            return replace(repr, ndim=ndim, odim=odim)

        # --- a (kind[, ndim[, odim]]) tuple ---
        if isinstance(repr, tuple):
            if not is_family_tuple(repr):
                raise ValueError(
                    f"Invalid tuple: {repr!r}. A family pairs one kind "
                    f"with zero, one or two dimensions."
                )
            return cls.from_tuple(repr, ndim, odim)

        # --- a bare dimension ---
        # `bool` is an `int`, so it is excluded explicitly.
        if isinstance(repr, int) and not isinstance(repr, bool):
            if ndim is MISSING:
                ndim = repr
            return cls(Transformation, ndim, odim)

        # --- a kind ---
        if isinstance(repr, type):
            return cls(repr, ndim, odim)

        # --- a name or a symbol ---
        if isinstance(repr, str):
            try:
                return cls.from_name(repr, ndim, odim)
            except ValueError:
                return cls.from_symbol(repr, ndim, odim)

        # --- error ---
        raise ValueError(f"Invalid transformation kind or family: {repr!r}")

    @classmethod
    def from_tuple(
        cls, values: FamilyTupleLike, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Return a transformation family from a tuple.

        The tuple has one to three elements, `(kind[, ndim[, odim]])`,
        which is also the form that `to_tuple` returns. An explicit
        `ndim` or `odim` argument replaces the matching element.
        """
        values = list(values)
        if len(values) < 3:
            values += [MISSING] * (3 - len(values))
        if ndim is not MISSING:
            values[1] = ndim
        if odim is not MISSING:
            values[2] = odim
        return cls.parse(*values)

    @classmethod
    def from_name(
        cls, name: str, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Return a transformation family from its name."""
        kind = NAMETOCLASS.get(name.lower())
        if kind is None:
            raise ValueError(f"Invalid name for transformation kind: {name}")
        return cls.parse(kind, ndim, odim)

    @classmethod
    def from_symbol(
        cls, symbol: str, ndim: Dim = MISSING, odim: Dim = MISSING
    ) -> tx.Self:
        """Return a transformation family from its symbol."""
        kind = SYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim, odim)

        kind = FSYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim, odim)

        for fsymbol, kind in FSYMBOLTOCLASS.items():
            if "{n}" not in fsymbol:
                continue
            pattern = _fsymbol_to_pattern(fsymbol)
            if not (match := pattern.match(symbol)):
                continue
            if ndim is MISSING:
                ndim = match.groupdict().get("n", ndim)
                if isinstance(ndim, str):
                    ndim = int(ndim)
            if odim is MISSING:
                odim = match.groupdict().get("m", odim)
                if isinstance(odim, str):
                    odim = int(odim)
            return cls.parse(kind, ndim, odim)

        raise ValueError(
            f"Invalid symbol for transformation kind or family: {symbol}"
        )


def is_family_tuple(value: tuple) -> bool:
    """Return whether a tuple describes a transformation family.

    A family tuple has one to three elements, `(kind[, ndim[, odim]])`.
    The first element is a kind or a string, and each of the others is
    a dimension, which is an `int` or `None`. A `bool` is an `int` in
    Python but not a dimension, so it is rejected.
    """
    if len(value) not in (1, 2, 3):
        return False
    kind = value[0]
    if isinstance(kind, type) and not issubclass(kind, TransformationKind):
        return False
    if not isinstance(kind, (str, type)):
        return False
    if len(value) > 1:
        ndim = value[1]
        if ndim is not None and (
            isinstance(ndim, bool) or not isinstance(ndim, int)
        ):
            return False
    if len(value) > 2:
        ndim = value[2]
        if ndim is not None and (
            isinstance(ndim, bool) or not isinstance(ndim, int)
        ):
            return False
    return True


def _fsymbol_to_pattern(fsymbol: str) -> re.Pattern:
    """
    Convert an FSYMBOL template (which may contain '{n}' one or more
    times) into a regex pattern with a single named group 'n', reused
    via backreference for repeated occurrences.
    """
    pattern = re.escape(fsymbol)
    for name in ("n", "m"):
        slot = re.escape(f"{{{name}}}")
        pattern = pattern.replace(slot, rf"(?P<{name}>\d+)", 1)
        pattern = pattern.replace(slot, rf"(?P={name})")
    return re.compile("^" + pattern + "$")


# ======================================================================
#
#                           H I E R A R C H Y
#
# ======================================================================


@public
class TransformationKind(ABC):
    """The root of the transformation

    A subclass declares its `SYMBOL` (a short mathematical symbol, such
    as `"SO"`) and its `FSYMBOL` (the same symbol with a dimension
    placeholder, such as `"SO({n})"`), and its `ALIAS` (one or more
    plain-text names). These are recorded automatically on subclassing,
    and are what [`TransformationFamily.parse`][] resolves a string against.

    The class name is automatically registered as a name, and is the
    preferred name of the transformation kind.

    Aliases can also be registered with the [`alias`][] decorator.
    """

    SYMBOL: str
    FSYMBOL: str
    ALIAS: tx.Union[str, tx.List[str], tx.Tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        NAMETOCLASS[cls.__name__.lower()] = cls
        if alias := cls.__dict__.get("ALIAS"):
            if isinstance(alias, str):
                alias = (alias,)
            for name in alias:
                NAMETOCLASS[name.lower()] = cls
        if fsymb := cls.__dict__.get("FSYMBOL"):
            FSYMBOLTOCLASS[fsymb] = cls
        if symb := cls.__dict__.get("SYMBOL"):
            SYMBOLTOCLASS[symb] = cls

    def __new__(cls, subcls: type) -> type:
        """Register a concrete transformation type to this set."""
        return cls.register(subcls)

    @classmethod
    def parse(cls, repr: KindLike) -> tx.Type[tx.Self]:
        """Return a transformation kind from its name, symbol or type."""
        return TransformationFamily.parse(repr).kind


@closed  # assuming domains are matching
@alias("Morphism", "Map")
class Transformation(TransformationKind):
    """Any coordinate transformation.

    alias: Morphism, Map
    """

    SYMBOL = "Map"
    FSYMBOL = "Map(ℝ^{n},ℝ^{m})"


@closed
@alias("VolumePreservingTransformation", "VolumePreservingMap")
class VolumePreserving(Transformation):
    """
    A transformation that preserves volumes **locally** .

    If the map is smooth, this means |det Df| = 1.
    """


@closed
@alias("OrientationPreservingTransformation", "OrientationPreservingMap")
class OrientationPreserving(Transformation):
    """
    A transformation that preserves orientation.

    If the map is smooth, this means det Df > 0.
    """


@closed
@nonembeddable
@alias("Conformal", "ConformalMap", "ConformalTransformation")
@alias("AnglePreservingTransformation", "AnglePreservingMap")
class AnglePreserving(Transformation):
    """
    A transformation that preserves angles.

    If the map is smooth, this means `(Df).T @ (Df) = λ(x) I`.

    alias: Conformal

    wiki: https://en.wikipedia.org/wiki/Conformal_map
    """


@closed  # assuming domains are matching
@alias("Injective", "InjectiveMap", "InjectiveTransformation")
class Injection(Transformation):
    """A one-to-one transformation (distinct inputs map to distinct outputs).

    alias: Injective

    wiki: https://en.wikipedia.org/wiki/Injective_function
    """


@closed  # assuming domains are matching
@alias("Surjective", "SurjectiveMap", "SurjectiveTransformation")
class Surjection(Transformation):
    """An onto transformation (every output is reached).

    alias: Surjective

    wiki: https://en.wikipedia.org/wiki/Surjective_function
    """


@group
@invertible_subset_of(Transformation)
@alias("Bijective", "BijectiveTransformation", "BijectiveMap")
@alias("Invertible", "InvertibleTransformation", "InvertibleMap")
class Bijection(Injection, Surjection):
    """An invertible transformation.

    A bijection is both injective and surjective.

    alias: Bijective, Invertible

    wiki: https://en.wikipedia.org/wiki/Bijection
    """


@group
@public
class Homeomorphism(Bijection):
    """
    A continuous bijection with a continuous inverse.

    wiki: https://en.wikipedia.org/wiki/Homeomorphism
    """


@group
@public
class Diffeomorphism(Homeomorphism):
    """A smooth homeomorphism, with a smooth inverse.

    wiki: https://en.wikipedia.org/wiki/Diffeomorphism
    """

    SYMBOL = "Diff"
    FSYMBOL = "Diff(ℝ^{n})"


@group
@public
class VolumePreservingDiffeomorphism(VolumePreserving, Diffeomorphism):
    """A diffeomorphism that preserves volumes."""


@group
@public
class OrientationPreservingDiffeomorphism(
    OrientationPreserving, Diffeomorphism
):
    """A diffeomorphism that preserves orientation."""

    SYMBOL = "Diff+"
    FSYMBOL = "Diff+(ℝ^{n})"


@group
@nonembeddable
@alias("AnglePreservingDiffeomorphism")
class ConformalDiffeomorphism(AnglePreserving, Diffeomorphism):
    """A diffeomorphism that preserves angles."""


@group
@alias("VolumeAndOrientationPreservingDiffeomorphism")
class SpecialDiffeomorphism(
    VolumePreservingDiffeomorphism,
    OrientationPreservingDiffeomorphism,
):
    """A diffeomorphism that preserves both volume and orientation.

    Preserves the standard volume form: det(Df(x)) = 1 everywhere.
    Includes all connected components satisfying this condition.
    """

    SYMBOL = "SDiff"
    FSYMBOL = "SDiff(ℝ^{n})"


# ----------------------------------------------------------------------
#   L I E  /  M A T R I X
# ----------------------------------------------------------------------


@closed
@alias("MatrixTransformation")
class Matrix(Transformation):
    """Transformation that can be represented as a matrix.

    This includes linear operators, as well as more general affine
    operators that can be represented as linear operators acting on
    homogeneous coordinates.

    wiki: https://en.wikipedia.org/wiki/Transformation_matrix
    """


@group
@invertible_subset_of(Matrix)
@alias("InvertibleMatrixTransformation")
class InvertibleMatrix(Matrix, Diffeomorphism):
    """A matrix transformation that is invertible.

    A matrix map is smooth, and an invertible one has a smooth (matrix)
    inverse, so this is where the smooth end of the lattice attaches: every
    invertible affine transformation is a diffeomorphism, and a volume
    preserving one (SL, SAff) is one of those.

    wiki: https://en.wikipedia.org/wiki/Invertible_matrix
    """


@group
@connected
@public
class OrientationPreservingMatrix(
    OrientationPreservingDiffeomorphism, InvertibleMatrix
):
    """An invertible matrix transformation with positive determinant."""


@group
@public
class VolumePreservingMatrix(VolumePreservingDiffeomorphism, InvertibleMatrix):
    """An invertible matrix transformation with determinant ± 1."""


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------


@closed
@alias("AffineTransformation", "AffineMap", "AffineMatrix")
class Affine(Matrix):
    """An affine transformation. May not be invertible.

    wiki: https://en.wikipedia.org/wiki/Affine_transformation
    """


@liegroup
@invertible_subset_of(Affine)
@alias("InvertibleAffineMatrix", "AffineGroup")
@alias("InvertibleAffineTransformation", "InvertibleAffineMap")
class InvertibleAffine(Affine, InvertibleMatrix):
    """
    An invertible affine transformation

    Invertible affine transformations form a Lie group, named Aff.

    This group includes reflections, and has therefore two connected
    components.

    symbol: Aff = GL ⋉ T

    wiki: https://en.wikipedia.org/wiki/Affine_group
    """

    SYMBOL = "Aff"
    FSYMBOL = "Aff(ℝ^{n})"


@liegroup
@connected
@alias("PositiveAffineMatrix", "PositiveAffineGroup")
@alias("PositiveAffineTransformation", "PositiveAffineMap")
class PositiveAffine(InvertibleAffine, OrientationPreservingMatrix):
    """
    Affine transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Aff+ = GL+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Affine_group
    """

    SYMBOL = "Aff+"
    FSYMBOL = "Aff+(ℝ^{n})"


@liegroup
@alias(
    "VolumePreservingAffineMap",
    "VolumePreservingAffineTransformation",
)
class VolumePreservingAffine(InvertibleAffine, VolumePreservingMatrix):
    """
    An affine transformation with determinant ±1 -- preserves volumes.
    """


@liegroup
@connected
@alias(
    "SpecialAffineMap",
    "SpecialAffineGroup",
    "SpecialAffineTransformation",
)
class SpecialAffine(
    PositiveAffine, VolumePreservingAffine, SpecialDiffeomorphism
):
    """
    An affine transformation with determinant +1 -- preserves volumes

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: SAff = SL ⋉ T

    wiki: https://en.wikipedia.org/wiki/Affine_group#Special_affine_group
    """

    SYMBOL = "SAff"
    FSYMBOL = "SAff({n})"


@liegroup
@nonembeddable
@alias(
    "ConformalEuclideanGroup",
    "ConformalEuclideanMap",
    "ConformalEuclideanTransformation",
    "Similarity",
    "Similitude",
    "AffineSimilarity",
    "AffineSimilitude",
)
class ConformalEuclidean(InvertibleAffine, ConformalDiffeomorphism):
    """
    An affine transformation that preserves angles.

    This group includes reflections, and has therefore two connected
    components.

    symbol: Sim = CO ⋉ T

    alias: Similarity, Similitude

    wiki: https://en.wikipedia.org/wiki/Similarity_(geometry)#In_Euclidean_space
    """

    SYMBOL = "Sim"
    FSYMBOL = "Sim({n})"


@liegroup
@connected
@nonembeddable
@alias(
    "SpecialConformalEuclideanMap",
    "SpecialConformalEuclideanGroup",
    "SpecialConformalEuclideanTransformation",
    "ConformalSpecialEuclidean",
    "ConformalSpecialEuclideanMap",
    "ConformalSpecialEuclideanGroup",
    "ConformalSpecialEuclideanTransformation",
    "DirectSimilarity",
    "DirectSimilitude",
    "DirectAffineSimilarity",
    "DirectAffineSimilitude",
)
class SpecialConformalEuclidean(ConformalEuclidean, PositiveAffine):
    """
    An affine transformation that preserves angles and orientation.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Sim+ = CO+ ⋉ T

    alias: DirectSimilarity, DirectSimilitude

    wiki: https://en.wikipedia.org/wiki/Similarity_(geometry)#In_Euclidean_space
    """

    SYMBOL = "Sim+"
    FSYMBOL = "Sim+({n})"


@liegroup
@alias("EuclideanGroup", "EuclideanMap", "EuclideanTransformation")
class Euclidean(ConformalEuclidean, VolumePreservingAffine):
    """
    A euclidean transformation with determinant ± 1

    symbol: E = O ⋉ T

    wiki: https://en.wikipedia.org/wiki/Euclidean_group
    """

    SYMBOL = "E"
    FSYMBOL = "E({n})"


@liegroup
@connected
@alias(
    "SpecialEuclideanGroup",
    "SpecialEuclideanMap",
    "SpecialEuclideanTransformation",
    "Rigid",
    "RigidTransformation",
)
class SpecialEuclidean(SpecialConformalEuclidean, Euclidean, SpecialAffine):
    """
    A euclidean transformation with determinant +1

    symbol: SE = SO ⋉ T

    alias: RigidTransformation

    wiki: https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries
    """  # noqa: E501

    SYMBOL = "SE"
    FSYMBOL = "SE({n})"


@liegroup
@nonembeddable
@alias(
    "DilationGroup",
    "HomothetyTranslation",
    "HomothecyTranslation",
)
class Dilation(ConformalEuclidean):
    """
    A dilation is a scaling and a translation.

    symbol: ℝ* ⋉ T

    alias: HomothetyTranslation

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ* ⋉ T"
    FSYMBOL = "ℝ* ⋉ T({n})"


@liegroup
@simplyconnected
@nonembeddable
@public
class PositiveDilation(Dilation, SpecialConformalEuclidean):
    """
    A dilation with a positive scaling factor.

    symbol: ℝ+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ+ ⋉ T"
    FSYMBOL = "ℝ+ ⋉ T({n})"


@liegroup
@simplyconnected
@public
class Translation(PositiveDilation, SpecialEuclidean):
    """A translation.

    symbol: T

    wiki: https://en.wikipedia.org/wiki/Translation_(geometry)#As_a_group
    """

    SYMBOL = "T"
    FSYMBOL = "T({n})"


# ----------------------------------------------------------------------
#   L I N E A R
# ----------------------------------------------------------------------


@closed
@alias("LinearTransformation", "LinearMap")
class Linear(Affine):
    """A linear transformation. May not be invertible."""


@liegroup
@invertible_subset_of(Linear)
@alias(
    "InvertibleGroup",
    "InvertibleLinearMap",
    "InvertibleLinearTransformation",
)
class InvertibleLinear(Linear, InvertibleAffine):
    """
    An invertible linear transformation.

    Invertible linear transformation form a Lie group, named the
    General Linear group (GL).

    This group includes reflections, and has therefore two connected
    components.

    symbol: GL

    wiki: https://en.wikipedia.org/wiki/General_linear_group
    """

    SYMBOL = "GL"
    FSYMBOL = "GL({n})"


@liegroup
@connected
@alias("PositiveLinearTransformation", "PositiveLinearMap")
class PositiveLinear(InvertibleLinear, PositiveAffine):
    """
    Linear transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: GL+
    """

    SYMBOL = "GL+"
    FSYMBOL = "GL+({n})"


@liegroup
@connected
@alias("SpecialLinearTransformation", "SpecialLinearMap")
class SpecialLinear(PositiveLinear, SpecialAffine):
    """
    A linear transformation with determinant 1 -- preserves volumes .

    A linear map is an affine map whose translation vanishes, so SL sits
    inside SAff = SL ⋉ T -- the horizontal link of the diagram above -- and
    reaches `VolumePreservingDiffeomorphism` through it.

    symbol: SL

    wiki: https://en.wikipedia.org/wiki/Special_linear_group
    """

    SYMBOL = "SL"
    FSYMBOL = "SL({n})"


@liegroup
@nonembeddable
@alias(
    "ConformalOrthogonalGroup",
    "ConformalOrthogonalMap",
    "ConformalOrthogonalTransformation",
)
class ConformalOrthogonal(InvertibleLinear, ConformalEuclidean):
    """
    A linear transformation that preserves angles (CO)

    symbol: CO = O x ℝ+

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """

    SYMBOL = "CO"
    FSYMBOL = "CO({n})"


@liegroup
@connected
@nonembeddable
@alias(
    "SpecialConformalOrthogonalGroup",
    "SpecialConformalOrthogonalMap",
    "SpecialConformalOrthogonalTransformation",
    "ConformalSpecialOrthogonal",
    "ConformalSpecialOrthogonalMap",
    "ConformalSpecialOrthogonalTransformation",
)
class SpecialConformalOrthogonal(
    ConformalOrthogonal, PositiveLinear, SpecialConformalEuclidean
):
    """
    A linear transformation that preserves angles and orientation (CO+)

    symbol: CO+ = SO x ℝ+

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group

    !!! note

        A scaled rotation `c R` has determinant `c**n`, which is positive
        but only equal to one when `c` is. CO+ therefore sits under GL+
        (`PositiveLinear`), in the positive-determinant column of the
        diagram above, and *not* under SL: it does not preserve volumes.
        The rotations do -- see [`SpecialOrthogonal`][], which declares SL
        for itself.
    """

    SYMBOL = "CO+"
    FSYMBOL = "CO+({n})"


@liegroup
@alias("OrthogonalTransformation", "OrthogonalMap", "OrthogonalGroup")
class Orthogonal(ConformalOrthogonal, Euclidean):
    """
    A orthogonal matrix (AA' = I) with determinant ±1

    symbol: O

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group
    """

    SYMBOL = "O"
    FSYMBOL = "O({n})"


@liegroup
@connected
@alias(
    "SpecialOrthogonalGroup",
    "SpecialOrthogonalMap",
    "SpecialOrthogonalTransformation",
    "Rotation",
    "RotationTransformation",
)
class SpecialOrthogonal(
    Orthogonal,
    SpecialConformalOrthogonal,
    SpecialEuclidean,
    SpecialLinear,
):
    """
    A orthogonal matrix (AA' = I) with determinant +1

    symbol: SO

    alias: Rotation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group

    !!! note

        SL is declared here rather than inherited: a rotation preserves
        volumes, but the conformal group it also belongs to does not (see
        [`SpecialConformalOrthogonal`][]), so no ancestor can carry the
        fact.
    """  # noqa: E501

    SYMBOL = "SO"
    FSYMBOL = "SO({n})"


@liegroup
@alias(
    "GeneralizedPermutationMatrix",
    "Monomial",
    "MonomialGroup",
    "MonomialMatrix",
    "MonomialTransformation",
)
class GeneralizedPermutation(InvertibleLinear):
    """A generalized permutation.

    Permutations are invertible by definition.

    The generalized permutation group is the semidirect product of the
    symmetric group (permutations) and the group of invertible diagonal
    matrices.

    symbol: S ⋉ Δ

    wiki: https://en.wikipedia.org/wiki/Generalized_permutation_matrix
    """

    SYMBOL = "S ⋉ Δ"
    FSYMBOL = "S_{n} ⋉ Δ({n})"


@liegroup
@alias(
    "SignedPermutationMatrix", "SignedPermutationGroup", "HyperoctahedralGroup"
)
class SignedPermutation(GeneralizedPermutation, Orthogonal):
    """A signed permutation.

    A generalized permutation with non-zero entries ±1.

    symbol: B = C₂ⁿ ⋊ S

    alias: HyperoctahedralGroup

    wiki: https://en.wikipedia.org/wiki/Generalized_permutation_matrix#Signed_permutation_group
    """  # noqa: E501

    SYMBOL = "B"
    FSYMBOL = "B_{n}"


@liegroup
@alias("PermutationMatrix", "PermutationGroup", "SymmetricGroup")
class Permutation(SignedPermutation, Orthogonal):
    """A permutation.

    A generalized permutation with non-zero entries +1.

    Permutations form the symmetric group, named S.

    symbol: S

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    """

    SYMBOL = "S"
    FSYMBOL = "S_{n}"


@liegroup
@alias("AlternatingGroup", "EvenPermutationGroup")
class EvenPermutation(Permutation, SpecialOrthogonal):
    """An even permutation.

    A permutation with determinant +1.

    A permutation matrix is orthogonal, and an even one has determinant
    +1, so the even permutations are exactly the permutations that lie in
    SO(n). That edge is what makes SO(n) closed under composition with an
    even permutation -- and hence what lets a subspace transform that
    reindexes its axes by an even permutation stay a rotation.

    Even permutations form the (finite discrete) Alternating Group A.

    symbol: A

    wiki: https://en.wikipedia.org/wiki/Alternating_group
    """

    SYMBOL = "A"
    FSYMBOL = "A_{n}"


@public
class OddPermutation(Permutation):
    """An odd permutation.

    A permutation with determinant -1.

    !!! note

        This is the *coset* `S \\ A`, not a group: composing two odd
        permutations gives an even one, so the set is not closed under
        composition and does not contain the identity.

    symbol: S \\ A

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    wiki: https://en.wikipedia.org/wiki/Alternating_group
    """

    SYMBOL = "S \\ A"
    FSYMBOL = "S_{n} \\ A_{n}"


@closed
@alias(
    "DiagonalMatrix",
    "DiagonalTransformation",
    "Scaling",
    "ScalingTransformation",
)
class Diagonal(Linear):
    """A diagonal matrix, may not be invertible.

    wiki: https://en.wikipedia.org/wiki/Diagonal_matrix
    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """


@liegroup
@invertible_subset_of(Diagonal)
@alias(
    "InvertibleDiagonalMatrix",
    "InvertibleDiagonalTransformation",
)
class InvertibleDiagonal(Diagonal, GeneralizedPermutation):
    """An invertible diagonal matrix.

    Invertible diagonal matrices form a Lie group.

    This group includes reflections, and has therefore 2**n connected
    components.

    symbol: Δ
    """

    SYMBOL = "Δ"
    FSYMBOL = "Δ({n})"


@liegroup
@simplyconnected
@alias(
    "PositiveDiagonalMatrix",
    "PositiveDiagonalTransformation",
    "PositiveScaling",
    "PositiveScalingTransformation",
)
class PositiveDiagonal(InvertibleDiagonal, PositiveLinear):
    """A diagonal matrix with positive entries.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Δ+
    """

    SYMBOL = "Δ+"
    FSYMBOL = "Δ+({n})"


@liegroup
@public
class SpecialDiagonal(InvertibleDiagonal, SpecialLinear):
    """A diagonal matrix with determinant +1.

    symbol: Δ_S
    """

    SYMBOL = "Δ_S"
    FSYMBOL = "Δ_S({n})"


@liegroup
@alias(
    "OrthogonalDiagonalMatrix",
    "OrthogonalDiagonalTransformation",
    "CardinalReflection",
    "CardinalReflectionGroup",
)
class OrthogonalDiagonal(InvertibleDiagonal, SignedPermutation):
    """
    A diagonal matrix with entries ±1.

    symbol: Δ_O
    """

    SYMBOL = "Δ_O"
    FSYMBOL = "Δ_O({n})"


@liegroup
@alias(
    "SpecialOrthogonalDiagonalMatrix",
    "SpecialOrthogonalDiagonalTransformation",
)
class SpecialOrthogonalDiagonal(
    OrthogonalDiagonal,
    SpecialDiagonal,
    SpecialOrthogonal,
):
    """An orthogonal diagonal matrix with determinant +1.

    symbol: Δ_SO
    """

    SYMBOL = "Δ_SO"
    FSYMBOL = "Δ_SO({n})"


@closed
@nonembeddable
@alias(
    "MultiplicativeTransformation",
    "Multiplication",
    "ScaledIdentityMatrix",
)
class Multiplicative(Diagonal):
    """A scaling with the same factor in all dimensions.

    symbol: ℝ = {c I : c ∈ ℝ}
    """

    SYMBOL = "ℝ"
    FSYMBOL = "ℝ({n})"


@liegroup
@nonembeddable
@invertible_subset_of(Multiplicative)
@alias(
    "MultiplicativeGroup",
    "InvertibleMultiplicativeTransformation",
    "InvertibleMultiplication",
    "Homothety",
    "HomothetyTransformation",
    "Homothecy",
    "HomothecyTransformation",
    "HomogeneousDilation",
)
class InvertibleMultiplicative(
    Multiplicative, InvertibleDiagonal, ConformalOrthogonal
):
    """A scaling with the same nonzero factor in all dimensions.

    An isotropic scaling preserves angles: `c I` is `|c|` times the
    orthogonal `sign(c) I`, so ℝ* lies in CO.

    symbol: ℝ* = {c I : c ∈ ℝ*}

    alias: Homothety, HomogeneousDilation

    wiki: https://en.wikipedia.org/wiki/Multiplicative_group
    wiki: https://en.wikipedia.org/wiki/Homothety
    """

    SYMBOL = "ℝ*"
    FSYMBOL = "ℝ*({n})"


@liegroup
@simplyconnected
@nonembeddable
@alias(
    "PositiveMultiplicativeGroup",
    "PositiveMultiplicativeTransformation",
    "PositiveMultiplication",
    "PositiveHomothety",
    "PositiveHomothetyTransformation",
)
class PositiveMultiplicative(
    InvertibleMultiplicative, PositiveDiagonal, SpecialConformalOrthogonal
):
    """A scaling with the same positive factor in all dimensions.

    A positive factor is a non-zero one, so this is a subgroup of ℝ*
    (`InvertibleMultiplicative`) and not merely a subset of ℝ
    (`Multiplicative`), which it reaches through it.

    symbol: ℝ+ = {c I : c ∈ ℝ+*}

    alias: PositiveHomothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """

    SYMBOL = "ℝ+"
    FSYMBOL = "ℝ+({n})"


# ----------------------------------------------------------------------
#   I D E N T I T Y
# ----------------------------------------------------------------------


@liegroup
@simplyconnected
@alias(
    "IdentityMatrix",
    "IdentityMap",
    "IdentityTransformation",
)
class Identity(
    Translation,
    EvenPermutation,
    SpecialOrthogonalDiagonal,
    PositiveMultiplicative,
    SpecialOrthogonal,
):
    """The identity transformation.

    symbol: I

    wiki: https://en.wikipedia.org/wiki/Identity_function
    """

    SYMBOL = "I"
    FSYMBOL = "I({n})"
