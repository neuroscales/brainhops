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

The implication only goes that way: a declared edge is a true inclusion,
but a true inclusion need not be declared. Inclusion is a DAG and an MRO
is a linear order, so some true edges cannot be added at all -- ℝ* ⊆ ℝ* ⋉ T
(`InvertibleMultiplicative` inside `Dilation`) is one: the identity lies at
the foot of both the linear and the affine chain, and declaring that edge
makes C3 order the two chains in contradictory ways. Membership questions
are therefore asked of [`is_kind`][], which reads the values as well as
the edges, and never of `issubclass` alone.

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
    "is_transformation_set",
    "is_kind_ndim_pair",
    "is_lie_group",
    "is_connected",
    "is_simplyconnected",
    "group",
    "liegroup",
    "connected",
    "simplyconnected",
    "TransformationFamily",
    # --- GENERAL ------------------------------------------------------
    "TransformationKind",
    "Transformation",
    "Morphism",                                     # ^ alias
    "Injection",
    "Surjection",
    "Bijection",
    "Bijection",                                    # ^ alias
    "Isomorphism",
    "Diffeomorphism",                               # Diff
    "VolumePreservingDiffeomorphism",               # SDiff
    # --- MATRIX -------------------------------------------------------
    "Matrix",
    "InvertibleMatrix",
    "PositiveDefiniteMatrix",
    # --- AFFINE -------------------------------------------------------
    "Affine",                         # <not a group>
    "InvertibleAffine",               # Aff              (det ≠ 0)
    "PositiveAffine",                 # Aff+             (det > 0)
    "SpecialAffine",                  # SAff = SL  ⋉ T   (det = 1)
    "VolumePreservingAffineTransformation",         # ^ alias
    "ConformalEuclidean",             # CE   = CO  ⋉ T   (det ≠ 0)
    "SpecialConformal",      # CSE  = CSO ⋉ T   (det > 0)
    "Similitude",                                   # ^ alias
    "Euclidean",                      # E    = O   ⋉ T   (det =±1)
    "SpecialEuclidean",               # SE   = SO  ⋉ T   (det = 1)
    "RigidTransformation",                          # ^ alias
    "Dilation",                                     # ℝ* ⋉ T           (det ≠ 0)
    "PositiveDilation",                             # ℝ+ ⋉ T           (det > 0)
    "Translation",                                  # T                (det = 1)
    # --- LINEAR -------------------------------------------------------
    "Linear",                         # M = matrix set
    "InvertibleLinear",               # GL               (det ≠ 0)
    "PositiveLinear",                 # GL+              (det > 0)
    "SpecialLinear",                  # SL               (det = 1)
    "ConformalOrthogonal",            # CO  = O x ℝ*     (det ≠ 0)
    "SpecialConformalOrthogonal",     # CSO = O x ℝ+     (det > 0)
    "Orthogonal",                     # O                (det =±1)
    "SpecialOrthogonal",              # SO               (det = 1)
    "Rotation",                                     # ^ alias
    "GeneralizedPermutation",                       # S ⋉ Δ            (det ≠ 0)
    "SignedPermutation",                            # S ⋉ Δ+           (det ≠ 0)
    "Permutation",                                  # S                (det ≠ 0)
    "Diagonal",                       # <not a group>
    "InvertibleDiagonal",             # Δ                (det ≠ 0)
    "PositiveDiagonal",               # Δ+               (det > 0)
    "Reflection",                                   #
    "FiniteReflection",                             #
    "SpecialDiagonal",                # SΔ               (det =±1)
    "CardinalReflection",                           # ^ alias
    "Multiplicative",                 # ℝ
    "InvertibleMultiplicative",       # ℝ*               (det ≠ 0)
    "Homothety",                                    # ^ alias
    "PositiveMultiplicative",         # ℝ+               (det > 0)
    "PositiveHomothety",                            # ^ alias
    # --- IDENTITY -----------------------------------------------------
    "Identity",                       # I                (det = 1)
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

_Type = tx.Type["TransformationKind"]
_Decorator = tx.Callable[["_Type"], "_Type"]
_Set = tx.Set[_Type]
_StrMap = tx.Dict[str, _Type]
_TypeMap = tx.Dict[_Type, _Type]

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
    return issubclass(cls, Bijection)


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


def is_embeddable(cls: _Type) -> bool:
    """
    Return whether a set of transformations is known to embed into a
    higher-dimensional space by padding with the identity.

    The embedding is `T -> T ⊕ I`, block-diagonal: `T` acts on the axes
    it already acted on, and the new ones pass through. The set is
    embeddable when the result stays in the same set one dimension up,
    `G(n) ⊕ I ⊆ G(n+k)`. Every element of SO(3) is an element of SO(4)
    that way; no element of ℝ* (a scaling by the same factor along every
    axis) is, since `diag(2, 2, 2, 1)` is not isotropic in ℝ⁴.

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

    wiki: https://en.wikipedia.org/wiki/Inverse_element
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

    wiki: https://en.wikipedia.org/wiki/Closure_(mathematics)
    """

    def decorator(subcls: _Type) -> _Type:
        CLOSEDUNDER.setdefault(cls, set()).add(subcls)
        return subcls

    return decorator


def closed(cls: _Type) -> _Type:
    """
    Mark the set of transformations as being closed under composition.

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


# ======================================================================
#
#                           F A M I L I E S
#
# ======================================================================

Kind = tx.Type["TransformationKind"]
Dim = tx.Optional[int]
KindLike = tx.Union[Kind, str]
FamilyLike = tx.Union[tx.Self, tx.Tuple[KindLike, Dim], KindLike]


class TransformationFamily(
    AbcSequence, Magic, eq=True, hash=True, frozen=True
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
    # > allows unpacking into a 2-tuple

    def __len__(self) -> int:
        return 2

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
        if isinstance(repr, tuple):
            if not is_kind_ndim_pair(repr):
                raise ValueError(
                    f"Invalid (kind, ndim) pair: {repr!r}. A family pairs one "
                    f"kind with one dimension (or None)."
                )
            return cls.from_tuple(repr)
        # --- a bare dimension ---
        # `bool` is an `int`, so it is excluded explicitly.
        if isinstance(repr, int) and not isinstance(repr, bool):
            return cls(Transformation, repr if ndim is None else ndim)
        # --- a kind ---
        if isinstance(repr, type):
            return cls(repr, ndim)
        # --- a name or a symbol ---
        if isinstance(repr, str):
            try:
                return cls.from_name(repr, ndim)
            except ValueError:
                return cls.from_symbol(repr, ndim)
        raise ValueError(f"Invalid transformation kind or family: {repr!r}")

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
            raise ValueError(f"Invalid name for transformation kind: {name}")
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

        raise ValueError(
            f"Invalid symbol for transformation kind or family: {symbol}"
        )


def is_kind_ndim_pair(value: tx.Tuple) -> bool:
    """Whether a tuple is a well-formed `(kind, ndim)` pair.

    Exactly two elements, whose second one is a dimension: an `int` or
    `None`. A `bool` is an `int` in Python but not a dimension, so it is
    excluded.
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


class TransformationKind(ABC):
    """The root of the transformation

    A subclass declares its `SYMBOL` (a short mathematical symbol, such
    as `"SO"`), its `FSYMBOL` (the same symbol with a dimension
    placeholder, such as `"SO({n})"`), and its `NAME` (one or more
    plain-text names). These are recorded automatically on subclassing,
    and are what [`TransformationFamily.parse`][] resolves a string against.
    """

    SYMBOL: str
    FSYMBOL: str
    NAME: tx.Tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "NAME", None):
            for name in cls.NAME:
                NAMETOCLASS[name.lower()] = cls
        if getattr(cls, "FSYMBOL", None):
            FSYMBOLTOCLASS[cls.FSYMBOL] = cls
        if getattr(cls, "SYMBOL", None):
            SYMBOLTOCLASS[cls.SYMBOL] = cls

    @classmethod
    def parse(cls, repr: KindLike) -> tx.Type[tx.Self]:
        """Return a transformation kind from its name, symbol or type."""
        return TransformationFamily.parse(repr).kind


@closed  # assuming domains are matching
class Transformation(TransformationKind):
    """Any coordinate transformation.

    alias: Morphism
    """

    SYMBOL = "Trans"
    FSYMBOL = "Trans({n})"
    NAME = ("Transformation", "Morphism")


Morphism = Transformation


@closed  # assuming domains are matching
class Injection(Transformation):
    """A one-to-one transformation (distinct inputs map to distinct outputs).

    wiki: https://en.wikipedia.org/wiki/Injective_function
    """

    NAME = (
        "Injection",
        "Injective",
        "InjectiveTransformation",
    )


InjectiveTransformation = Injective = Injection


@closed  # assuming domains are matching
class Surjection(Transformation):
    """An onto transformation (every output is reached).

    wiki: https://en.wikipedia.org/wiki/Surjective_function
    """

    NAME = (
        "Surjection",
        "Surjective",
        "SurjectiveTransformation",
    )


SurjectiveTransformation = Surjective = Surjection


@group
@invertible_subset_of(Transformation)
class Bijection(Injection, Surjection):
    """An invertible transformation.

    A bijection is both injective and surjective.

    alias: Bijection

    wiki: https://en.wikipedia.org/wiki/Bijection
    """

    NAME = (
        "Bijection",
        "Bijective",
        "Invertible",
        "BijectiveTransformation",
        "InvertibleTransformation",
    )


BijectiveTransformation = Bijective = Bijection
InvertibleTransformation = Invertible = Bijection


@group
class Isomorphism(Bijection):
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
class Matrix(Transformation):
    """Transformation that can be represented as a matrix.

    wiki: https://en.wikipedia.org/wiki/Transformation_matrix
    """

    NAME = (
        "Matrix",
        "MatrixTransformation",
    )


MatrixTransformation = Matrix


@group
@invertible_subset_of(Matrix)
class InvertibleMatrix(Matrix, Diffeomorphism):
    """A matrix transformation that is invertible.

    A matrix map is smooth, and an invertible one has a smooth (matrix)
    inverse, so this is where the smooth end of the lattice attaches: every
    invertible affine transformation is a diffeomorphism, and a volume
    preserving one (SL, SAff) is one of those.

    wiki: https://en.wikipedia.org/wiki/Invertible_matrix
    """

    NAME = (
        "InvertibleMatrix",
        "InvertibleMatrixTransformation",
    )


InvertibleMatrixTransformation = InvertibleMatrix


@group
@simplyconnected
class PositiveDefiniteMatrix(InvertibleMatrix):
    """An invertible matrix transformation with positive determinant.

    wiki: https://en.wikipedia.org/wiki/Positive-definite_matrix
    """

    NAME = (
        "PositiveDefiniteMatrix",
        "PositiveDefiniteMatrixTransformation",
    )


PositiveDefiniteMatrixTransformation = PositiveDefiniteMatrix


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------


@closed
class Affine(Matrix):
    """An affine transformation. May not be invertible.

    wiki: https://en.wikipedia.org/wiki/Affine_transformation
    """

    NAME = (
        "Affine",
        "AffineMatrix",
        "AffineTransformation",
    )


AffineTransformation = AffineMatrix = Affine


@liegroup
@invertible_subset_of(Affine)
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
    NAME = (
        "InvertibleAffine",
        "InvertibleAffineMatrix",
        "InvertibleAffineTransformation",
        "AffineGroup",
    )


InvertibleAffineTransformation = InvertibleAffineMatrix = InvertibleAffine
AffineGroup = InvertibleAffine


@liegroup
@simplyconnected
class PositiveAffine(InvertibleAffine, PositiveDefiniteMatrix):
    """
    Affine transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Aff+ = GL+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Affine_group
    """

    SYMBOL = "Aff+"
    FSYMBOL = "Aff+(ℝ^{n})"
    NAME = (
        "PositiveAffine",
        "PositiveAffineGroup",
        "PositiveDefiniteAffine",
        "PositiveAffineMatrix",
        "PositiveDefiniteAffineMatrix",
        "PositiveAffineTransformation",
        "PositiveDefiniteAffineTransformation",
    )


PositiveAffineTransformation = PositiveAffine
PositiveDefiniteAffineTransformation = PositiveDefiniteAffine = PositiveAffine
PositiveDefiniteAffineMatrix = PositiveAffineGroup = PositiveAffine


@liegroup
@simplyconnected
class SpecialAffine(PositiveAffine, VolumePreservingDiffeomorphism):
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
        "SpecialAffine",
        "SpecialAffineTransformation",
        "SpecialAffineGroup",
        "VolumePreservingAffine",
        "VolumePreservingAffineTransformation",
    )


SpecialAffineTransformation = SpecialAffineGroup = SpecialAffine
VolumePreservingAffineTransformation = VolumePreservingAffine = SpecialAffine


@liegroup
@nonembeddable
class ConformalEuclidean(InvertibleAffine):
    """
    An affine transformation that preserves angles (up to their sign).

    This group includes reflections, and has therefore two connected
    components.

    symbol: CE = CO ⋉ T
    """

    SYMBOL = "CE"
    FSYMBOL = "CE({n})"
    NAME = (
        "ConformalEuclidean",
        "ConformalEuclideanGroup",
        "ConformalEuclideanTransformation",
    )


ConformalEuclideanTransformation = ConformalEuclideanGroup = ConformalEuclidean


@liegroup
@connected
@nonembeddable
class SpecialConformal(ConformalEuclidean, PositiveAffine):
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
        "SpecialConformalEuclidean",
        "SpecialConformalEuclideanGroup",
        "SpecialConformalEuclideanTransformation",
        "ConformalSpecialEuclidean",
        "ConformalSpecialEuclideanGroup",
        "ConformalSpecialEuclideanTransformation",
        "Similitude",
        "SimilitudeTransformation",
    )


SpecialConformalEuclideanGroup = SpecialConformal
SpecialConformalEuclideanTransformation = SpecialConformal
ConformalSpecialEuclidean = SpecialConformal
ConformalSpecialEuclideanGroup = SpecialConformal
ConformalSpecialEuclideanTransformation = SpecialConformal
SimilitudeTransformation = Similitude = SpecialConformal


@liegroup
class Euclidean(ConformalEuclidean):
    """
    A euclidean transformation with determinant ± 1

    symbol: E = O ⋉ T

    wiki: https://en.wikipedia.org/wiki/Euclidean_group
    """

    SYMBOL = "E"
    FSYMBOL = "E({n})"
    NAME = (
        "Euclidean",
        "EuclideanGroup",
        "EuclideanTransformation",
    )


EuclideanTransformation = EuclideanGroup = Euclidean


@liegroup
@connected
class SpecialEuclidean(SpecialConformal, Euclidean, SpecialAffine):
    """
    A euclidean transformation with determinant +1

    symbol: SE = SO ⋉ T

    alias: RigidTransformation

    wiki: https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries
    """  # noqa: E501

    SYMBOL = "SE"
    FSYMBOL = "SE({n})"
    NAME = (
        "SpecialEuclidean",
        "SpecialEuclideanTransformation",
        "SpecialEuclideanGroup",
        "Rigid",
        "RigidTransformation",
    )


SpecialEuclideanTransformation = SpecialEuclideanGroup = SpecialEuclidean
RigidTransformation = Rigid = SpecialEuclidean


@liegroup
@nonembeddable
class Dilation(ConformalEuclidean):
    """
    A dilation is a similitude with no rotation, i.e. a scaling with
    translation.

    symbol: ℝ* ⋉ T

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ* ⋉ T"
    FSYMBOL = "ℝ* ⋉ T({n})"
    NAME = ("Dilation", "HomothetyTranslations", "HomothecyTranslations")


HomothetyTranslations = HomothecyTranslations = Dilation


@liegroup
@simplyconnected
@nonembeddable
class PositiveDilation(Dilation, SpecialConformal):
    """
    A dilation with a positive scaling factor.

    symbol: ℝ+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """

    SYMBOL = "ℝ+ ⋉ T"
    FSYMBOL = "ℝ+ ⋉ T({n})"
    NAME = ("PositiveDilation",)


@liegroup
@simplyconnected
class Translation(PositiveDilation, SpecialEuclidean):
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
class Linear(Affine):
    """A linear transformation. May not be invertible."""

    NAME = (
        "Linear",
        "LinearTransformation",
    )


LinearTransformation = Linear


@liegroup
@invertible_subset_of(Linear)
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
    NAME = (
        "InvertibleLinear",
        "InvertibleLinearTransformation",
    )


InvertibleLinearTransformation = InvertibleLinear


@liegroup
@connected
class PositiveLinear(InvertibleLinear, PositiveAffine):
    """
    Linear transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: GL+
    """

    SYMBOL = "GL+"
    FSYMBOL = "GL+({n})"
    NAME = (
        "PositiveLinear",
        "PositiveLinearTransformation",
        "PositiveDefiniteLinear",
        "PositiveDefiniteLinearTransformation",
    )


PositiveLinearTransformation = PositiveLinear
PositiveDefiniteLinearTransformation = PositiveDefiniteLinear = PositiveLinear


@liegroup
@connected
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
    NAME = (
        "SpecialLinear",
        "SpecialLinearTransformation",
    )


SpecialLinearTransformation = SpecialLinear


@liegroup
@nonembeddable
class ConformalOrthogonal(InvertibleLinear, ConformalEuclidean):
    """
    A linear transformation that preserves (absolute) angles (CO)

    symbol: CO = O x ℝ* (if n is even)
                 O x ℝ+ (if n is odd)

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """

    SYMBOL = "CO"
    FSYMBOL = "CO({n})"
    NAME = (
        "ConformalOrthogonal",
        "ConformalOrthogonalTransformation",
    )


ConformalOrthogonalTransformation = ConformalOrthogonal


@liegroup
@connected
@nonembeddable
class SpecialConformalOrthogonal(
    ConformalOrthogonal, PositiveLinear, SpecialConformal
):
    """
    A linear transformation that preserves angles (CSO)

    symbol: CSO = SO x ℝ+

    !!! note

        A scaled rotation `c R` has determinant `c**n`, which is positive
        but only equal to one when `c` is. CSO therefore sits under GL+
        (`PositiveLinear`), in the positive-determinant column of the
        diagram above, and *not* under SL: it does not preserve volumes.
        The rotations do -- see [`SpecialOrthogonal`][], which declares SL
        for itself.

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """

    SYMBOL = "CSO"
    FSYMBOL = "CSO({n})"
    NAME = (
        "SpecialConformalOrthogonal",
        "SpecialConformalOrthogonalTransformation",
        "ConformalSpecialOrthogonal",
        "ConformalSpecialOrthogonalTransformation",
    )


SpecialConformalOrthogonalTransformation = SpecialConformalOrthogonal
ConformalSpecialOrthogonal = SpecialConformalOrthogonal
ConformalSpecialOrthogonalTransformation = SpecialConformalOrthogonal


@liegroup
class Orthogonal(ConformalOrthogonal, Euclidean):
    """
    A orthogonal matrix (AA' = I) with determinant ±1

    symbol: O

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group
    """

    SYMBOL = "O"
    FSYMBOL = "O({n})"
    NAME = (
        "Orthogonal",
        "OrthogonalTransformation",
    )


OrthogonalTransformation = Orthogonal


@liegroup
@connected
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

    !!! note

        SL is declared here rather than inherited: a rotation preserves
        volumes, but the conformal group it also belongs to does not (see
        [`SpecialConformalOrthogonal`][]), so no ancestor can carry the
        fact.

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group
    """  # noqa: E501

    SYMBOL = "SO"
    FSYMBOL = "SO({n})"
    NAME = (
        "SpecialOrthogonal",
        "SpecialOrthogonalTransformation",
        "Rotation",
        "RotationTransformation",
    )


SpecialOrthogonalTransformation = SpecialOrthogonal
RotationTransformation = Rotation = SpecialOrthogonal


@group
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
    NAME = (
        "GeneralizedPermutation",
        "GeneralizedPermutationMatrix",
        "Monomial",
        "MonomialMatrix",
        "MonomialTransformation",
    )


GeneralizedPermutationMatrix = GeneralizedPermutation
Monomial = MonomialMatrix = MonomialTransformation = GeneralizedPermutation


@group
class SignedPermutation(GeneralizedPermutation, Orthogonal):
    """A signed permutation.

    A generalized permutation with non-zero entries ±1.

    symbol: S ⋉ SΔ

    wiki: https://en.wikipedia.org/wiki/Generalized_permutation_matrix#Signed_permutation_group
    """  # noqa: E501

    SYMBOL = "S ⋉ SΔ"
    FSYMBOL = "S_{n} ⋉ SΔ({n})"
    NAME = ("SignedPermutation", "SignedPermutationMatrix")


SignedPermutationMatrix = SignedPermutation


@group
class Permutation(SignedPermutation, Orthogonal):
    """A permutation.

    A generalized permutation with non-zero entries +1.

    Permutations form the symmetric group, named S.

    symbol: S

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    """

    SYMBOL = "S"
    FSYMBOL = "S_{n}"
    NAME = ("Permutation", "PermutationMatrix", "SymmetricGroup")


SymmetricGroup = PermutationMatrix = Permutation


@group
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
    NAME = ("EvenPermutation", "AlternatingGroup")


AlternatingGroup = EvenPermutation


class OddPermutation(Permutation):
    """An odd permutation.

    A permutation with determinant -1.

    !!! note

        This is the *coset* `S \\ A`, not a group: composing two odd
        permutations gives an even one, so the set is not closed under
        composition and does not contain the identity. It is the one node
        here that is neither.

    symbol: S \\ A

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    wiki: https://en.wikipedia.org/wiki/Alternating_group
    """

    SYMBOL = "S \\ A"
    FSYMBOL = "S_{n} \\ A_{n}"
    NAME = ("OddPermutation",)


@closed
class Diagonal(Linear):
    """A diagonal matrix, may not be invertible.

    wiki: https://en.wikipedia.org/wiki/Diagonal_matrix
    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """

    NAME = (
        "Diagonal",
        "DiagonalMatrix",
        "DiagonalTransformation",
        "Scaling",
        "ScalingTransformation",
    )


DiagonalMatrix = DiagonalTransformation = Diagonal
ScalingTransformation = Scaling = Diagonal


@liegroup
@invertible_subset_of(Diagonal)
class InvertibleDiagonal(Diagonal, GeneralizedPermutation):
    """An invertible diagonal matrix.

    Invertible diagonal matrices form a Lie group.

    This group includes reflections, and has therefore two connected
    components.

    symbol: Δ
    """

    SYMBOL = "Δ"
    FSYMBOL = "Δ({n})"
    NAME = (
        "InvertibleDiagonal",
        "InvertibleDiagonalMatrix",
        "InvertibleDiagonalTransformation",
    )


InvertibleDiagonalMatrix = InvertibleDiagonal
InvertibleDiagonalTransformation = InvertibleDiagonal


@liegroup
@simplyconnected
class PositiveDiagonal(InvertibleDiagonal, PositiveLinear):
    """A diagonal matrix with positive entries.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Δ+
    """

    SYMBOL = "Δ+"
    FSYMBOL = "Δ+({n})"
    NAME = (
        "PositiveDiagonal",
        "PositiveDiagonalMatrix",
        "PositiveDiagonalTransformation",
        "PositiveScaling",
        "PositiveScalingTransformation",
    )


PositiveDiagonalMatrix = PositiveDiagonal
PositiveDiagonalTransformation = PositiveDiagonal
PositiveDiagonalTransformation = PositiveScaling = PositiveDiagonal


@group
class Reflection(Orthogonal):
    """A reflection.

    This group includes all reflections along any hyperplane that goes
    through the origin. This hyperplane does not have to be axis-aligned.

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    NAME = ("Reflection", "ReflectionGroup")


ReflectionGroup = Reflection


@group
class FiniteReflection(Reflection):
    """Base class for finite reflection groups.

    A finite reflection group is a reflection group limited to a finite
    number of reflection planes.

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    NAME = ("FiniteReflection", "FiniteReflectionGroup")


FiniteReflectionGroup = FiniteReflection


@group
class SpecialDiagonal(InvertibleDiagonal, SignedPermutation, FiniteReflection):
    """A diagonal matrix with entries ±1.

    symbol: SΔ

    alias: CardinalReflection

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """

    SYMBOL = "SΔ"
    FSYMBOL = "SΔ({n})"
    NAME = (
        "SpecialDiagonal",
        "SpecialDiagonalMatrix",
        "SpecialDiagonalTransformation",
        "CardinalReflection",
        "CardinalReflectionGroup",
    )


SpecialDiagonalMatrix = SpecialDiagonalTransformation = SpecialDiagonal
CardinalReflectionGroup = CardinalReflection = SpecialDiagonal


@closed
@nonembeddable
class Multiplicative(Diagonal):
    """A scaling with the same factor in all dimensions.

    symbol: ℝ
    """

    SYMBOL = "ℝ"
    FSYMBOL = "ℝ"
    NAME = (
        "Multiplicative",
        "MultiplicativeTransformation",
        "Multiplication",
        "ScaledIdentityMatrix",
    )


MultiplicativeTransformation = Multiplication = Multiplicative
ScaledIdentityMatrix = Multiplicative


@liegroup
@nonembeddable
@invertible_subset_of(Multiplicative)
class InvertibleMultiplicative(
    Multiplicative, InvertibleDiagonal, ConformalOrthogonal
):
    """A scaling with the same positive factor in all dimensions.

    An isotropic scaling preserves angles: `c I` is `|c|` times the
    orthogonal `sign(c) I`, so ℝ* lies in CO.

    symbol: ℝ*

    alias: Homothety

    wiki: https://en.wikipedia.org/wiki/Multiplicative_group
    wiki: https://en.wikipedia.org/wiki/Homothety
    """

    SYMBOL = "ℝ*"
    FSYMBOL = "ℝ*"
    NAME = (
        "InvertibleMultiplicative",
        "InvertibleMultiplicativeTransformation",
        "InvertibleMultiplication",
        "MultiplicativeGroup",
        "Homothety",
        "HomothetyTransformation",
        "Homothecy",
        "HomothecyTransformation",
        "HomogeneousDilation",
    )


InvertibleMultiplication = InvertibleMultiplicative
InvertibleMultiplicativeTransformation = InvertibleMultiplicative
HomothetyTransformation = Homothety = InvertibleMultiplicative
HomothecyTransformation = Homothecy = InvertibleMultiplicative
MultiplicativeGroup = HomogeneousDilation = InvertibleMultiplicative


@liegroup
@simplyconnected
@nonembeddable
class PositiveMultiplicative(
    InvertibleMultiplicative, PositiveDiagonal, SpecialConformalOrthogonal
):
    """A scaling with the same positive factor in all dimensions.

    A positive factor is a non-zero one, so this is a subgroup of ℝ*
    (`InvertibleMultiplicative`) and not merely a subset of ℝ
    (`Multiplicative`), which it reaches through it.

    symbol: ℝ+

    alias: PositiveHomothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """

    SYMBOL = "ℝ+"
    FSYMBOL = "ℝ+"
    NAME = (
        "PositiveMultiplicative",
        "PositiveMultiplicativeTransformation",
        "PositiveMultiplicativeGroup",
        "PositiveMultiplication",
        "PositiveHomothety",
        "PositiveHomothetyTransformation",
    )


PositiveMultiplication = PositiveMultiplicative
PositiveMultiplicativeTransformation = PositiveMultiplicative
PositiveHomothetyTransformation = PositiveHomothety = PositiveMultiplicative
PositiveMultiplicativeGroup = PositiveMultiplicative


# ----------------------------------------------------------------------
#   I D E N T I T Y
# ----------------------------------------------------------------------


@liegroup
@simplyconnected
class Identity(
    Translation,  # T
    EvenPermutation,  # A   -- the identity permutation is even
    SpecialDiagonal,  # SΔ  -- and so a cardinal reflection group
    PositiveMultiplicative,  # ℝ+
    SpecialOrthogonal,  # SO
):
    """The identity transformation.

    Every set marked [`group`][] contains the identity, so the bases above
    are the *maximal* such sets: `Permutation` and `Reflection`, say, are
    reached through the tighter `EvenPermutation` and `SpecialDiagonal`.

    symbol: I

    wiki: https://en.wikipedia.org/wiki/Identity_function
    """

    SYMBOL = "I"
    FSYMBOL = "I({n})"
    NAME = (
        "Identity",
        "IdentityMatrix",
        "IdentityTransformation",
    )


IdentityTransformation = IdentityMatrix = Identity
