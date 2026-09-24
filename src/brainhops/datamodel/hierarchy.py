r"""
This module defines a transformation

The class A is a subclass of B if A can be converted to B without loss.
For example, Linear is a subclass of Affine. This is akin to set theory
in mathematics.

This differs from the usual "type hierarchy", where inheriting classes
are typically more specialized then their parents.

We implement both types of hierarchy, so that users can easily check
whether a transformation can be conceptually thought as a given class
of transformation (e.g. "is this transformation a linear transform?"),
without it being conflated with type inheritance
(e.g. "is this transformation an instance of the Linear class?").

In our hierarchy, transformations types correspond to sets, and
inheritance describes set inclusion (i.e., `isubclass(A, B)` implies
`A ⊆ B`). All transformations act on the n-dimensional euclidean space
ℝⁿ, and are assigned a "composition" operator.

Some of these transformation sets (but not all!) are groups under the
composition operator. This is indicated by the [`group`][] decorator.
If a set A is a subset of set B, and both A and B are groups, then A is
a subgroup of B.

Many linear transformations, when constrained to be invertible, can
be thought of as members of a Lie group, i.e. they are smooth manifolds,
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
CO ------------- CE              Conformal          | Affine Conformal
 | \  |   |      |
 |   CSO ------- CSE             Special Conformal  | Special Affine Conformal
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

Positive Lie groups (SO/CSO/GL+) are restricted to transformations with
positive determinant, and therefore exclude flips. They can be defined
as the component from the corresponding general group that contains the
identity transform.

Note that the Special Linear group (SL) can have negative determinants,
but is restricted to unit-norm determinants (±1, i.e. volume
preserving).

All classical linear group can be extended with the translation group
(⋉ T) to define affine groups.

!!! info
    * The Special Euclidean group (SE) contains transformations that are
      classicaly referred to as "rigid-body" transformations. They
      preserve angles and volumes.

    * The Special Conformal Euclidean group (CSE) contains
      transformations that are classicaly referred to as "similitude"
      (or conformal) transformations. They only preserve angles.
"""

# fmt: off
# ruff: disable[E501]
__all__ = [
    "is_group",
    "is_lie_group",
    "is_connected",
    "is_simplyconnected",
    "group",
    "liegroup",
    "connected",
    "simplyconnected",
    "TransformationFamily",
    # --- GENERAL ------------------------------------------------------
    "TransformationBaseClass",
    "Transformation",
    "Morphism",                                     # ^ alias
    "InjectiveTransformation",
    "SurjectiveTransformation",
    "BijectiveTransformation",
    "Bijection",                                    # ^ alias
    "Isomorphism",
    "Diffeomorphism",                               # Diff
    "VolumePreservingDiffeomorphism",               # SDiff
    # --- MATRIX -------------------------------------------------------
    "MatrixTransformation",
    "InvertibleMatrixTransformation",
    "PositiveDefiniteMatrixTransformation",
    # --- AFFINE -------------------------------------------------------
    "AffineTransformation",                         # <not a group>
    "InvertibleAffineTransformation",               # Aff              (det ≠ 0)
    "PositiveAffineTransformation",                 # Aff+             (det > 0)
    "SpecialAffineTransformation",                  # SAff = SL  ⋉ T   (det = 1)
    "VolumePreservingAffineTransformation",         # ^ alias
    "ConformalEuclideanTransformation",             # CE   = CO  ⋉ T   (det ≠ 0)
    "SpecialConformalEuclideanTransformation",      # CSE  = CSO ⋉ T   (det > 0)
    "Similitude",                                   # ^ alias
    "EuclideanTransformation",                      # E    = O   ⋉ T   (det =±1)
    "SpecialEuclideanTransformation",               # SE   = SO  ⋉ T   (det = 1)
    "RigidTransformation",                          # ^ alias
    "Dilation",                                     # ℝ* ⋉ T           (det ≠ 0)
    "PositiveDilation",                             # ℝ+ ⋉ T           (det > 0)
    "Translation",                                  # T                (det = 1)
    # --- LINEAR -------------------------------------------------------
    "LinearTransformation",                         # M = matrix set
    "InvertibleLinearTransformation",               # GL               (det ≠ 0)
    "PositiveLinearTransformation",                 # GL+              (det > 0)
    "SpecialLinearTransformation",                  # SL               (det = 1)
    "ConformalOrthogonalTransformation",            # CO  = O x ℝ*     (det ≠ 0)
    "SpecialConformalOrthogonalTransformation",     # CSO = O x ℝ+     (det > 0)
    "OrthogonalTransformation",                     # O                (det =±1)
    "SpecialOrthogonalTransformation",              # SO               (det = 1)
    "Rotation",                                     # ^ alias
    "GeneralizedPermutation",                       # S ⋉ Δ            (det ≠ 0)
    "SignedPermutation",                            # S ⋉ Δ+           (det ≠ 0)
    "Permutation",                                  # S                (det ≠ 0)
    "DiagonalTransformation",                       # <not a group>
    "InvertibleDiagonalTransformation",             # Δ                (det ≠ 0)
    "PositiveDiagonalTransformation",               # Δ+               (det > 0)
    "Reflection",                                   #
    "FiniteReflection",                             #
    "SpecialDiagonalTransformation",                # SΔ               (det =±1)
    "CardinalReflection",                           # ^ alias
    "MultiplicativeTransformation",                 # ℝ
    "InvertibleMultiplicativeTransformation",       # ℝ*               (det ≠ 0)
    "Homothety",                                    # ^ alias
    "PositiveMultiplicativeTransformation",         # ℝ+               (det > 0)
    "PositiveHomothety",                            # ^ alias
    # --- IDENTITY -----------------------------------------------------
    "IdentityTransformation",                       # I                (det = 1)
]
# ruff: enable[E501]
# fmt: on

# stdlib
import re
from abc import ABC
from collections.abc import Sequence as AbcSequence
from functools import lru_cache

import typing_extensions as tx
from bagof.magic import Magic, replace

# ======================================================================
#
#                           R E G I S T R I E S
#
# ======================================================================

_Type = tx.Type["TransformationBaseClass"]
_Decorator = tx.Callable[["_Type"], "_Type"]
_Set = tx.Set[_Type]
_StrMap = tx.Dict[str, _Type]
_TypeMap = tx.Dict[_Type, _Type]

GROUPS: _Set = set()
LIE_GROUPS: _Set = set()
CONNECTED: _Set = set()
SIMPLYCONNECTED: _Set = set()
CLOSEDUNDER: tx.Dict[_Type, _Set] = {}
NONLIFTABLE: _Set = set()

NAMETOCLASS: _StrMap = {}
FSYMBOLTOCLASS: _StrMap = {}
SYMBOLTOCLASS: _StrMap = {}
INVERTIBLE_OF: _TypeMap = {}
NONINVERTIBLE_OF: _TypeMap = {}


# --- checks -----------------------------------------------------------


def is_group(cls: _Type) -> bool:
    """Return whether a set of transformations forms a group.

    A class is recognized as a group when it, or one of its ancestors,
    was marked with the [`group`][] or [`liegroup`][] decorator.
    """
    return cls in GROUPS


def is_lie_group(cls: _Type) -> bool:
    """Return whether a set of transformations forms a Lie group.

    A class is recognized as a Lie group when it, or one of its
    ancestors, was marked with the [`liegroup`][] decorator.
    """
    return cls in LIE_GROUPS


def is_connected(cls: _Type) -> bool:
    """Return whether a set of transformations is connected.

    A class is recognized as connected when it, or one of its
    ancestors, was marked with the [`connected`][] or
    [`simplyconnected`][] decorator.
    """
    return cls in CONNECTED


def is_simplyconnected(cls: _Type) -> bool:
    """Return whether a set of transformations is simply connected.

    A class is recognized as simply connected when it, or one of its
    ancestors, was marked with the [`simplyconnected`][] decorator.
    """
    return cls in SIMPLYCONNECTED


def is_invertible(cls: _Type) -> bool:
    """
    Return whether a set of transformations is known to be invertible.
    """
    return issubclass(cls, BijectiveTransformation)


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


def is_closed(cls: _Type) -> bool:
    """
    Return whether a set of transformations is closed under composition.
    """
    return is_closedunder(cls, cls)


def is_liftable(cls: _Type) -> bool:
    """
    Return whether a set of transformations is known to be liftable to a
    higher-dimensional space.
    """
    # I don't try to infer liftability from the subsets or supersets of
    # cls, because it is not very robust and depends very much on the
    # definition of liftability (which "properties" do we want converved).
    # Here, the property is "belong to the same group type" (but not the
    # exact same group since we have changed dimensions). For example,
    # all elements of SO(3) are elements of SO(4) when lifted from ℝ^3
    # to ℝ^4. But SO(3) has subgroups that are liftable and others that
    # are not, and it has supergroups that are liftable and other that
    # are not.
    return cls not in NONLIFTABLE


# --- generators -------------------------------------------------------


@lru_cache(maxsize=1)  # noqa: UP033
def all_sets() -> tx.FrozenSet[type]:
    """Return all known transformation sets."""
    return frozenset(NAMETOCLASS.values())


@lru_cache(maxsize=1)  # noqa: UP033
def all_groups() -> tx.FrozenSet[type]:
    """Return all known transformation groups."""
    return frozenset(GROUPS)


@lru_cache(maxsize=1)  # noqa: UP033
def all_lie_groups() -> tx.FrozenSet[type]:
    """Return all known Lie groups."""
    return frozenset(LIE_GROUPS)


def as_invertible(cls: _Type) -> _Type:
    """
    Return the invertible subset of a set of transformations, if any.

    If the set is not known to have an invertible subset, return `None`.
    """
    node = INVERTIBLE_OF.get(cls, cls)
    if not is_invertible(node):
        raise TypeError(
            f"{cls} is not a known invertible set of transformations"
        )
    return node


def as_noninvertible(cls: _Type) -> _Type:
    """
    Return the non-invertible superset of a set of transformations, if any.

    If the set is not known to be a subset of a non-invertible set, return
    `None`.
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
    """
    def decorator(subcls: _Type) -> _Type:
        INVERTIBLE_OF[cls] = subcls
        NONINVERTIBLE_OF[subcls] = cls
        return subcls

    return decorator


def closedunder(cls: _Type) -> _Decorator:
    """
    Mark the set of transformations as being closed under composition
    with another set.

    The argument is a set of transformations that is a superset of `cls`.
    """
    def decorator(subcls: _Type) -> _Type:
        CLOSEDUNDER.setdefault(cls, set()).add(subcls)
        return subcls

    return decorator


def closed(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being closed under composition.
    """
    return closedunder(cls)(cls)


def nonliftable(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being non-liftable to a
    higher-dimensional space.
    """
    NONLIFTABLE.add(cls)
    return cls


# ======================================================================
#
#                           F A M I L I E S
#
# ======================================================================

Kind = tx.Type["TransformationBaseClass"]
Dim = tx.Optional[int]
KindLike = tx.Union[Kind, str]
FamilyLike = tx.Union[tx.Self, tx.Tuple[KindLike, Dim], KindLike]


class TransformationFamily(
    # `eq`/`hash` are written by hand below: a family compares and hashes
    # as its `(kind, ndim)` pair, so a plain tuple is an equal key.
    AbcSequence, Magic, eq=False, hash=False, frozen=True
):
    """
    A parametric family of transformations.

    A family is parameterized by a base type and, optionally, a dimensionality.
    """

    # --- attributes ---------------------------------------------------

    kind: Kind
    """All transformations in this family are instances of this type."""

    ndim: Dim = None
    """The dimensionality of the transformations in this family, if known."""

    @property
    def name(self) -> str:
        """The preferred name of the transformation family."""
        return self.names[0]

    @property
    def names(self) -> tx.Tuple[str, ...]:
        """The names of the transformation family."""
        names = self.kind.NAME
        if isinstance(names, str):
            names = (names,)
        return names

    @property
    def symbol(self) -> tx.Optional[str]:
        """The symbol of the transformation family."""
        return getattr(self.kind, "SYMBOL", None)

    @property
    def fsymbol(self) -> tx.Optional[str]:
        """
        The symbol of the transformation family, with its dimension.

        If the family dimensionality is unknown, the dimension is the
        placeholder `{n}`.
        """
        fsymbol = getattr(self.kind, "FSYMBOL", None)
        ndim = self.ndim
        if fsymbol and ndim is not None:
            fsymbol = fsymbol.format(n=ndim)
        return fsymbol

    # --- magic --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tx.Union[Kind, Dim]:
        if index == 0:
            return self.kind
        if index == 1:
            return self.ndim
        raise IndexError(
            f"index {index} out of range for TransformationFamily"
        )

    def __iter__(self) -> tx.Iterator[tx.Union[Kind, Dim]]:
        yield self.kind
        yield self.ndim

    # A family *is* its `(kind, ndim)` pair, so it compares and hashes like
    # one. Tables keyed by families (the simplify table, the mode list) can
    # then be read with a plain tuple, and two spellings of the same family
    # collide on one key.
    def __eq__(self, other: tx.Any) -> bool:
        if isinstance(other, TransformationFamily):
            return self.to_tuple() == other.to_tuple()
        if isinstance(other, tuple):
            return self.to_tuple() == other
        return NotImplemented

    def __ne__(self, other: tx.Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    # --- to -----------------------------------------------------------

    def to_tuple(self) -> tx.Tuple[Kind, Dim]:
        """Convert to a (kind, ndim) tuple."""
        return (self.kind, self.ndim)

    # --- from ---------------------------------------------------------

    @classmethod
    def parse(cls, repr: FamilyLike, ndim: Dim = None) -> tx.Self:
        """Return a transformation family from its name, symbol or type.

        An explicit `ndim` wins over one carried by `repr`; `ndim=None`
        means "whatever `repr` says", so parsing a family is idempotent.

        The cases are ordered so that no branch is reached with a value it
        cannot type-check: a bare `issubclass` would raise on an `int` or a
        `str`.
        """
        # --- already a family ---
        if isinstance(repr, TransformationFamily):
            if ndim is not None and repr.ndim != ndim:
                repr = replace(repr, ndim=ndim)
            return repr
        # --- a (kind, ndim) pair ---
        # A two-tuple whose second element is a dimension is a pair; any
        # other tuple is a tuple *of kinds* (an `isinstance` kind such as
        # the "field" alias), which is kept whole.
        if isinstance(repr, tuple) and _is_kind_ndim_pair(repr):
            return cls.from_tuple(repr)
        # --- a bare dimension ---
        # `bool` is an `int`, so it is excluded explicitly.
        if isinstance(repr, int) and not isinstance(repr, bool):
            return cls(Transformation, repr if ndim is None else ndim)
        # --- a type, or a tuple of types ---
        if isinstance(repr, type) or isinstance(repr, tuple):
            return cls(repr, ndim)
        # --- a name or a symbol ---
        if isinstance(repr, str):
            try:
                return cls.from_name(repr, ndim)
            except ValueError:
                return cls.from_symbol(repr, ndim)
        raise ValueError(f"invalid transformation family: {repr!r}")

    @classmethod
    def from_tuple(cls, pair: tx.Tuple[KindLike, Dim]) -> tx.Self:
        """Return a transformation family from its `(kind, ndim)` pair."""
        kind, ndim = pair
        return cls.parse(kind, ndim)

    @classmethod
    def from_name(cls, name: str, ndim: Dim = None) -> tx.Self:
        """Return a transformation family from its name."""
        kind = NAMETOCLASS.get(name.lower())
        if kind is None:
            raise ValueError(f"invalid name for transformation family: {name}")
        return cls.parse(kind, ndim)

    @classmethod
    def from_symbol(cls, symbol: str, ndim: Dim = None) -> tx.Self:
        """Return a transformation family from its symbol."""
        kind = SYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim)

        kind = FSYMBOLTOCLASS.get(symbol)
        if kind is not None:
            return cls.parse(kind, ndim)

        for fsymbol, kind in FSYMBOLTOCLASS.items():
            if "{n}" not in fsymbol:
                continue
            pattern = _fsymbol_to_pattern(fsymbol)
            match = re.match(pattern, symbol)
            if match:
                return cls.parse(kind, int(match.group("n")))

        raise ValueError(f"invalid symbol for transformation family: {symbol}")


def _is_kind_ndim_pair(value: tx.Tuple) -> bool:
    """Whether a tuple reads as a `(kind, ndim)` pair rather than as a
    tuple of kinds.

    A `(kind, ndim)` pair has exactly two elements whose second one is a
    dimension: an `int` or `None`. Every other tuple -- including a
    two-tuple of two classes, such as the "field" kind alias -- is a tuple
    of kinds, matched by `isinstance` as a whole.
    """
    if len(value) != 2:
        return False
    ndim = value[1]
    if isinstance(ndim, bool):
        return False
    return ndim is None or isinstance(ndim, int)


def _fsymbol_to_pattern(fsymbol: str) -> str:
    """
    Convert an FSYMBOL template (which may contain '{n}' one or more
    times) into a regex pattern with a single named group 'n', reused
    via backreference for repeated occurrences.
    """
    parts = fsymbol.split("{n}")
    pattern = re.escape(parts[0])
    for i, part in enumerate(parts[1:]):
        if i == 0:
            pattern += r"(?P<n>\d+)"
        else:
            pattern += r"(?P=n)"
        pattern += re.escape(part)
    return "^" + pattern + "$"


# ======================================================================
#
#                           H I E R A R C H Y
#
# ======================================================================


class TransformationBaseClass(ABC):
    """The root of the transformation

    A subclass declares its `SYMBOL` (a short mathematical symbol, such
    as `"SO"`), its `FSYMBOL` (the same symbol with a dimension
    placeholder, such as `"SO({n})"`), and its `NAME` (one or more
    plain-text names). These are recorded automatically on subclassing,
    and are what [`TransformationFamily.parse`][] resolves a string against.
    """

    SYMBOL: str
    FSYMBOL: str
    NAME: tuple

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "NAME", None):
            for name in cls.NAME:
                NAMETOCLASS[name.lower()] = cls
        if getattr(cls, "FSYMBOL", None):
            FSYMBOLTOCLASS[cls.FSYMBOL] = cls
        if getattr(cls, "SYMBOL", None):
            SYMBOLTOCLASS[cls.SYMBOL] = cls


@closed  # assuming domains are matching
class Transformation(TransformationBaseClass):
    """Any coordinate transformation.

    alias: Morphism
    """

    SYMBOL = "Trans"
    FSYMBOL = "Trans({n})"
    NAME = ("Transformation", "Morphism")


Morphism = Transformation


@closed  # assuming domains are matching
class InjectiveTransformation(Transformation):
    """A one-to-one transformation (distinct inputs map to distinct outputs).

    wiki: https://en.wikipedia.org/wiki/Injective_function
    """

    NAME = (
        "InjectiveTransformation",
        "Injective",
        "Injection",
    )


@closed  # assuming domains are matching
class SurjectiveTransformation(Transformation):
    """An onto transformation (every output is reached).

    wiki: https://en.wikipedia.org/wiki/Surjective_function
    """

    NAME = (
        "SurjectiveTransformation",
        "Surjective",
        "Surjection",
    )


@group
@invertible_subset_of(Transformation)
class BijectiveTransformation(
    InjectiveTransformation, SurjectiveTransformation
):
    """An invertible transformation.

    A bijection is both injective and surjective.

    alias: Bijection

    wiki: https://en.wikipedia.org/wiki/Bijection
    """

    NAME = (
        "BijectiveTransformation",
        "Bijective",
        "Bijection",
        "InvertibleTransformation",
        "Invertible",
    )


Bijection = Bijective = BijectiveTransformation
Invertible = InvertibleTransformation = BijectiveTransformation


@group
class Isomorphism(BijectiveTransformation):
    """
    In all useful senses, bijective morphisms are isomorphisms,
    but there are mathematical spaces on which this is not true.
    We introduce a separate class to make mathematicians happy.

    wiki: https://en.wikipedia.org/wiki/Isomorphism
    """

    NAME = ("Isomorphism",)


@group
class Diffeomorphism(Isomorphism):
    """Smooth invertible transformation, with a smooth inverse.

    wiki: https://en.wikipedia.org/wiki/Diffeomorphism
    """

    SYMBOL = "Diff"
    FSYMBOL = "Diff(ℝ^{n})"
    NAME = ("Diffeomorphism",)


@group
class VolumePreservingDiffeomorphism(Diffeomorphism):
    """A diffeomorphism that preserves volumes."""

    SYMBOL = "SDiff"
    FSYMBOL = "SDiff(ℝ^{n})"
    NAME = ("VolumePreservingDiffeomorphism",)


# ----------------------------------------------------------------------
#   L I E  /  M A T R I X
# ----------------------------------------------------------------------


@closed
class MatrixTransformation(Transformation):
    """Transformation that can be represented as a matrix.

    wiki: https://en.wikipedia.org/wiki/Transformation_matrix
    """

    NAME = (
        "MatrixTransformation",
        "Matrix",
    )


@group
@invertible_subset_of(MatrixTransformation)
class InvertibleMatrixTransformation(
    MatrixTransformation, BijectiveTransformation
):
    """A matrix transformation that is invertible.

    wiki: https://en.wikipedia.org/wiki/Invertible_matrix
    """

    NAME = (
        "InvertibleMatrixTransformation",
        "InvertibleMatrix",
    )


@group
@simplyconnected
class PositiveDefiniteMatrixTransformation(InvertibleMatrixTransformation):
    """An invertible matrix transformation with positive determinant.

    wiki: https://en.wikipedia.org/wiki/Positive-definite_matrix
    """

    NAME = (
        "PositiveDefiniteMatrixTransformation",
        "PositiveDefiniteMatrix",
    )


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------


@closed
class AffineTransformation(MatrixTransformation):
    """An affine transformation. May not be invertible.

    wiki: https://en.wikipedia.org/wiki/Affine_transformation
    """

    NAME = (
        "AffineTransformation",
        "Affine",
    )


@liegroup
@invertible_subset_of(AffineTransformation)
class InvertibleAffineTransformation(
    AffineTransformation, InvertibleMatrixTransformation
):
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
    NAME = (
        "InvertibleAffineTransformation",
        "InvertibleAffine",
    )


@liegroup
@simplyconnected
class PositiveAffineTransformation(
    InvertibleAffineTransformation, PositiveDefiniteMatrixTransformation
):
    """
    Affine transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Aff+ = GL+ ⋉ T
    """

    SYMBOL = "Aff+"
    FSYMBOL = "Aff+(ℝ^{n})"
    NAME = (
        "PositiveAffineTransformation",
        "PositiveAffine",
    )


@liegroup
@simplyconnected
class SpecialAffineTransformation(
    PositiveAffineTransformation, VolumePreservingDiffeomorphism
):
    """
    An affine transformation with determinant +1 -- preserves volumes

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: SAff = SL ⋉ T

    alias: VolumePreservingAffineTransformation

    wiki: https://en.wikipedia.org/wiki/Affine_group#Special_affine_group
    """

    SYMBOL = "SAff"
    FSYMBOL = "SAff({n})"
    NAME = (
        "SpecialAffineTransformation",
        "SpecialAffine",
        "VolumePreservingAffineTransformation",
        "VolumePreservingAffine",
    )


VolumePreservingAffineTransformation = SpecialAffineTransformation


@liegroup
@nonliftable
class ConformalEuclideanTransformation(InvertibleAffineTransformation):
    """
    An affine transformation that preserves angles (up to their sign).

    This group includes reflections, and has therefore two connected
    components.

    symbol: CE = CO ⋉ T
    """

    SYMBOL = "CE"
    FSYMBOL = "CE({n})"
    NAME = (
        "ConformalEuclideanTransformation",
        "ConformalEuclidean",
    )


@liegroup
@connected
@nonliftable
class SpecialConformalEuclideanTransformation(
    ConformalEuclideanTransformation
):
    """
    An affine transformation that preserves angles

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: CSE = CSO ⋉ T

    alias: Similitude
    """

    SYMBOL = "CSE"
    FSYMBOL = "CSE({n})"
    NAME = (
        "SpecialConformalEuclideanTransformation",
        "SpecialConformalEuclidean",
        "Similitude",
        "SimilitudeTransformation",
    )


Similitude = SpecialConformalEuclideanTransformation


@liegroup
class EuclideanTransformation(ConformalEuclideanTransformation):
    """
    A euclidean transformation with determinant ± 1

    symbol: E = O ⋉ T

    wiki: https://en.wikipedia.org/wiki/Euclidean_group
    """

    SYMBOL = "E"
    FSYMBOL = "E({n})"
    NAME = (
        "EuclideanTransformation",
        "Euclidean",
    )


@liegroup
@connected
class SpecialEuclideanTransformation(
    SpecialConformalEuclideanTransformation, EuclideanTransformation
):
    """
    A euclidean transformation with determinant +1

    symbol: SE = SO ⋉ T

    alias: RigidTransformation

    wiki: https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries
    """  # noqa: E501

    SYMBOL = "SE"
    FSYMBOL = "SE({n})"
    NAME = (
        "SpecialEuclideanTransformation",
        "SpecialEuclidean",
        "RigidTransformation",
        "Rigid",
    )


RigidTransformation = SpecialEuclideanTransformation


@liegroup
@nonliftable
class Dilation(ConformalEuclideanTransformation):
    """
    A dilation is a similitude with no rotation, i.e. a scaling with
    translation.

    symbol: ℝ* ⋉ T

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ* ⋉ T"
    FSYMBOL = "ℝ* ⋉ T({n})"
    NAME = ("Dilation",)


@liegroup
@simplyconnected
@nonliftable
class PositiveDilation(Dilation, SpecialConformalEuclideanTransformation):
    """
    A dilation with a positive scaling factor.

    symbol: ℝ+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ+ ⋉ T"
    FSYMBOL = "ℝ+ ⋉ T({n})"
    NAME = ("PositiveDilation",)


@liegroup
@simplyconnected
class Translation(PositiveDilation, SpecialEuclideanTransformation):
    """A translation.

    symbol: T

    wiki: https://en.wikipedia.org/wiki/Translation_(geometry)#As_a_group
    """

    SYMBOL = "T"
    FSYMBOL = "T({n})"
    NAME = ("Translation",)


# ----------------------------------------------------------------------
#   L I N E A R
# ----------------------------------------------------------------------


@closed
class LinearTransformation(AffineTransformation):
    """A linear transformation. May not be invertible."""

    NAME = (
        "LinearTransformation",
        "Linear",
    )


@liegroup
@invertible_subset_of(LinearTransformation)
class InvertibleLinearTransformation(
    LinearTransformation, InvertibleAffineTransformation
):
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
    NAME = (
        "InvertibleLinearTransformation",
        "InvertibleLinear",
    )


@liegroup
@connected
class PositiveLinearTransformation(
    InvertibleLinearTransformation, PositiveAffineTransformation
):
    """
    Linear transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: GL+
    """

    SYMBOL = "GL+"
    FSYMBOL = "GL+({n})"
    NAME = (
        "PositiveLinearTransformation",
        "PositiveLinear",
    )


@liegroup
@connected
class SpecialLinearTransformation(
    PositiveLinearTransformation, VolumePreservingDiffeomorphism
):
    """
    A linear transformation with determinant 1 -- preserves volumes .

    symbol: SL

    wiki: https://en.wikipedia.org/wiki/Special_linear_group
    """

    SYMBOL = "SL"
    FSYMBOL = "SL({n})"
    NAME = (
        "SpecialLinearTransformation",
        "SpecialLinear",
    )


@liegroup
@nonliftable
class ConformalOrthogonalTransformation(InvertibleLinearTransformation):
    """
    A linear transformation that preserves (absolute) angles (CO)

    symbol: CO = O x ℝ* (if n is even)
                 O x ℝ+ (if n is odd)

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """

    SYMBOL = "CO"
    FSYMBOL = "CO({n})"
    NAME = (
        "ConformalOrthogonalTransformation",
        "ConformalOrthogonal",
    )


@liegroup
@connected
@nonliftable
class SpecialConformalOrthogonalTransformation(
    ConformalOrthogonalTransformation, SpecialLinearTransformation
):
    """
    A linear transformation that preserves angles (CSO)

    symbol: CSO = SO x ℝ+

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """

    SYMBOL = "CSO"
    FSYMBOL = "CSO({n})"
    NAME = (
        "SpecialConformalOrthogonalTransformation",
        "SpecialConformalOrthogonal",
        "ConformalSpecialOrthogonalTransformation",
        "ConformalSpecialOrthogonal",
    )


@liegroup
class OrthogonalTransformation(
    ConformalOrthogonalTransformation, EuclideanTransformation
):
    """
    A orthogonal matrix (AA' = I) with determinant ±1

    symbol: O

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group
    """

    SYMBOL = "O"
    FSYMBOL = "O({n})"
    NAME = (
        "OrthogonalTransformation",
        "Orthogonal",
    )


@liegroup
@connected
class SpecialOrthogonalTransformation(
    OrthogonalTransformation,
    SpecialConformalOrthogonalTransformation,
    SpecialEuclideanTransformation,
):
    """
    A orthogonal matrix (AA' = I) with determinant +1

    symbol: SO

    alias: Rotation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group
    """  # noqa: E501

    SYMBOL = "SO"
    FSYMBOL = "SO({n})"
    NAME = (
        "SpecialOrthogonalTransformation",
        "SpecialOrthogonal",
        "Rotation",
        "RotationTransformation",
    )


Rotation = SpecialOrthogonalTransformation


@group
class GeneralizedPermutation(InvertibleLinearTransformation):
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
    NAME = ("GeneralizedPermutation",)


@group
class SignedPermutation(GeneralizedPermutation, OrthogonalTransformation):
    """A signed permutation.

    A generalized permutation with non-zero entries ±1.

    symbol: S ⋉ SΔ

    wiki: https://en.wikipedia.org/wiki/Generalized_permutation_matrix#Signed_permutation_group
    """  # noqa: E501

    SYMBOL = "S ⋉ SΔ"
    FSYMBOL = "S_{n} ⋉ SΔ({n})"
    NAME = ("SignedPermutation",)


@group
class Permutation(SignedPermutation, OrthogonalTransformation):
    """A permutation.

    A generalized permutation with non-zero entries +1.

    Permutations form the symmetric group, named S.

    symbol: S

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    """

    SYMBOL = "S"
    FSYMBOL = "S_{n}"
    NAME = ("Permutation",)


@group
class EvenPermutation(Permutation, SpecialOrthogonalTransformation):
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
    NAME = ("EvenPermutation",)


@group
class OddPermutation(Permutation):
    """An odd permutation.

    A permutation with determinant -1.

    symbol: S \\ A

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    wiki: https://en.wikipedia.org/wiki/Alternating_group
    """

    SYMBOL = "S \\ A"
    FSYMBOL = "S_{n} \\ A_{n}"
    NAME = ("OddPermutation",)



@closed
class DiagonalTransformation(LinearTransformation):
    """A diagonal matrix, may not be invertible.

    wiki: https://en.wikipedia.org/wiki/Diagonal_matrix
    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """

    NAME = (
        "DiagonalTransformation",
        "Diagonal",
        "Scaling",
    )


@liegroup
@invertible_subset_of(DiagonalTransformation)
class InvertibleDiagonalTransformation(
    DiagonalTransformation, GeneralizedPermutation
):
    """An invertible diagonal matrix.

    Invertible diagonal matrices form a Lie group.

    This group includes reflections, and has therefore two connected
    components.

    symbol: Δ
    """

    SYMBOL = "Δ"
    FSYMBOL = "Δ({n})"
    NAME = (
        "InvertibleDiagonalTransformation",
        "InvertibleDiagonal",
    )


@liegroup
@simplyconnected
class PositiveDiagonalTransformation(
    InvertibleDiagonalTransformation, PositiveLinearTransformation
):
    """A diagonal matrix with positive entries.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Δ+
    """

    SYMBOL = "Δ+"
    FSYMBOL = "Δ+({n})"
    NAME = (
        "PositiveDiagonalTransformation",
        "PositiveDiagonal",
        "PositiveScaling",
    )


@group
class Reflection(OrthogonalTransformation):
    """A reflection.

    This group includes all reflections along any hyperplane that goes
    through the origin. This hyperplane does not have to be axis-aligned.

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    NAME = ("Reflection",)


@group
class FiniteReflection(Reflection):
    """Base class for finite reflection groups.

    A finite reflection group is a reflection group limited to a finite
    number of reflection planes.

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    NAME = ("FiniteReflection",)


@group
class SpecialDiagonalTransformation(
    InvertibleDiagonalTransformation, SignedPermutation, FiniteReflection
):
    """A diagonal matrix with entries ±1.

    symbol: SΔ

    alias: CardinalReflection

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    SYMBOL = "SΔ"
    FSYMBOL = "SΔ({n})"
    NAME = (
        "SpecialDiagonalTransformation",
        "SpecialDiagonal",
    )


CardinalReflection = SpecialDiagonalTransformation


@closed
@nonliftable
class MultiplicativeTransformation(
    DiagonalTransformation,
):
    """A scaling with the same factor in all dimensions.

    symbol: ℝ
    """

    SYMBOL = "ℝ"
    FSYMBOL = "ℝ"
    NAME = (
        "MultiplicativeTransformation",
        "Multiplicative",
    )


@liegroup
@nonliftable
@invertible_subset_of(MultiplicativeTransformation)
class InvertibleMultiplicativeTransformation(
    MultiplicativeTransformation, InvertibleDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.

    symbol: ℝ*

    alias: Homothety

    wiki: https://en.wikipedia.org/wiki/Multiplicative_group
    wiki: https://en.wikipedia.org/wiki/Homothety
    """

    SYMBOL = "ℝ*"
    FSYMBOL = "ℝ*"
    NAME = (
        "InvertibleMultiplicativeTransformation",
        "InvertibleMultiplicative",
        "Homothety",
        "HomothetyTransformation",
    )


Homothety = InvertibleMultiplicativeTransformation


@liegroup
@simplyconnected
@nonliftable
class PositiveMultiplicativeTransformation(
    MultiplicativeTransformation, PositiveDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.

    symbol: ℝ+

    alias: PositiveHomothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """

    SYMBOL = "ℝ+"
    FSYMBOL = "ℝ+"
    NAME = (
        "PositiveMultiplicativeTransformation",
        "PositiveMultiplicative",
        "PositiveHomothety",
        "PositiveHomothetyTransformation",
    )


PositiveHomothety = PositiveMultiplicativeTransformation


# ----------------------------------------------------------------------
#   I D E N T I T Y
# ----------------------------------------------------------------------


@liegroup
@simplyconnected
class IdentityTransformation(
    Translation,
    Permutation,
    PositiveMultiplicativeTransformation,
    SpecialOrthogonalTransformation,
):
    """The identity transformation.

    symbol: I

    wiki: https://en.wikipedia.org/wiki/Identity_function
    """

    SYMBOL = "I"
    FSYMBOL = "I({n})"
    NAME = (
        "IdentityTransformation",
        "Identity",
    )
