r"""
This module defines a transformation hierarchy.

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
composition operator. This is indicated by the `@group` decorator.
If a set A is a subset of set B, and both A and B are groups, then A is
a subgroup of B.

Many linear transformations, when constrained to be invertible, can
be thought of as members of a Lie group, i.e. they are smooth manifolds,
and their group operations (composition and inversion) are smooth maps.
This is indicated by the `@liegroup` decorator. Note that the set A may
be a subgroup of a Lie group B without being a Lie group itself!

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

::note
    * The Special Euclidean group (SE) contains transformations that are
      classicaly referred to as "rigid-body" transformations. They
      preserve angles and volumes.

    * The Special Conformal Euclidean group (CSE) contains
      transformations that are classicaly referred to as "similitude"
      (or conformal) transformations. They only preserve angles.
"""
import re

import typing_extensions as _tx

# fmt: off
__all__ = [
    "Transformation",
    "Morphism",                                     # ^ alias
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
    "MultiplicativeTransformation",                 # ℝ
    "InvertibleMultiplicativeTransformation",       # ℝ*               (det ≠ 0)
    "Homothety",                                    # ^ alias
    "PositiveMultiplicativeTransformation",         # ℝ+               (det > 0)
    "PositiveHomothety",                            # ^ alias
    # --- IDENTITY -----------------------------------------------------
    "IdentityTransformation",                       # I                (det = 1)
]
# fmt: on
from abc import ABC

GROUPS = set()
LIE_GROUPS = set()
CONNECTED = set()
SIMPLYCONNECTED = set()
NAMETOCLASS = {}
FSYMBOLTOCLASS = {}
SYMBOLTOCLASS = {}


def is_group(cls) -> bool:
    return cls in GROUPS


def is_lie_group(cls) -> bool:
    return cls in LIE_GROUPS


def is_connected(cls) -> bool:
    return cls in CONNECTED


def is_simplyconnected(cls) -> bool:
    return cls in SIMPLYCONNECTED


def group(cls):
    """
    Mark the set of transformations as forming a group under composition.

    wiki: https://en.wikipedia.org/wiki/Group_(mathematics)
    """
    GROUPS.add(cls)
    return cls


def liegroup(cls):
    """
    Mark the set of transformations as forming a Lie group under composition.

    wiki: https://en.wikipedia.org/wiki/Lie_group
    """
    LIE_GROUPS.add(cls)
    return group(cls)


def connected(cls):
    """
    Mark the set of transformations as being connected.

    wiki: https://en.wikipedia.org/wiki/Connected_space
    """
    CONNECTED.add(cls)
    return cls


def simplyconnected(cls):
    """
    Mark the set of transformations as being simply connected.

    wiki: https://en.wikipedia.org/wiki/Simply_connected_space
    """
    SIMPLYCONNECTED.add(cls)
    return connected(cls)


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


def parseType(s: _tx.Union[str, type, int]):
    """
    Resolve a string to its corresponding class in the transformation
    hierarchy, and extract a dimension if the string encodes one.

    Checks, in order if s is string:
      1. NAMETOCLASS    -- exact match on `NAME`    (e.g. "Rotation")
      2. SYMBOLTOCLASS  -- exact match on `SYMBOL`  (e.g. "SO")
      3. FSYMBOLTOCLASS -- pattern match on `FSYMBOL`, which may contain
         one or more `{n}` placeholders for the dimension (e.g. "SO(3)"
         matches template "SO({n})", extracting n=3; all occurrences of
         `{n}` in a template must agree on the same value)

    Returns
    -------
    cls : type or None
    dim : int or None
    """
    if isinstance(s, str):
        if s in NAMETOCLASS:
            return NAMETOCLASS[s], None
        if s in SYMBOLTOCLASS:
            return SYMBOLTOCLASS[s], None
        if s in FSYMBOLTOCLASS:
            return FSYMBOLTOCLASS[s], None
        for fsymbol, cls in FSYMBOLTOCLASS.items():
            if "{n}" not in fsymbol:
                continue
            pattern = _fsymbol_to_pattern(fsymbol)
            m = re.match(pattern, s)
            if m:
                return cls, int(m.group("n"))
        ValueError(f"invalid string for type lookup: {s}")
    if isinstance(s, int):
        return Transformation, s
    return s, None


class TransformationBaseClass(ABC):
    SYMBOL: str
    FSYMBOL: str
    NAME: tuple

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "NAME", None):
            for name in cls.NAME:
                NAMETOCLASS[name] = cls
        if getattr(cls, "FSYMBOL", None):
            FSYMBOLTOCLASS[cls.FSYMBOL] = cls
        if getattr(cls, "SYMBOL", None):
            SYMBOLTOCLASS[cls.SYMBOL] = cls


class Transformation(TransformationBaseClass):
    """Any coordinate transformation.

    alias: Morphism
    """

    SYMBOL = "Trans"
    FSYMBOL = "Trans({n})"
    NAME = ("Transformation",)


Morphism = Transformation


class BijectiveTransformation(Transformation):
    """An invertible transformation.

    alias: Bijection

    wiki: https://en.wikipedia.org/wiki/Bijection
    """
    NAME = ("BijectiveTransformation", "Bijective", "Bijection",)


Bijection = BijectiveTransformation


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


class MatrixTransformation(Transformation):
    """Transformation that can be represented as a matrix.

    wiki: https://en.wikipedia.org/wiki/Transformation_matrix
    """
    NAME = ("MatrixTransformation", "Matrix",)


@group
class InvertibleMatrixTransformation(
    MatrixTransformation,
    BijectiveTransformation
):
    """A matrix transformation that is invertible.

    wiki: https://en.wikipedia.org/wiki/Invertible_matrix
    """
    NAME = ("InvertibleMatrixTransformation", "InvertibleMatrix",)


@group
@simplyconnected
class PositiveDefiniteMatrixTransformation(InvertibleMatrixTransformation):
    """An invertible matrix transformation with positive determinant.

    wiki: https://en.wikipedia.org/wiki/Positive-definite_matrix
    """
    NAME = ("PositiveDefiniteMatrixTransformation", "PositiveDefiniteMatrix",)


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------


class AffineTransformation(MatrixTransformation):
    """An affine transformation. May not be invertible.

    wiki: https://en.wikipedia.org/wiki/Affine_transformation
    """
    NAME = ("AffineTransformation", "Affine",)


@liegroup
class InvertibleAffineTransformation(
    AffineTransformation,
    InvertibleMatrixTransformation
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
    NAME = ("InvertibleAffineTransformation", "InvertibleAffine",)


@liegroup
@simplyconnected
class PositiveAffineTransformation(
    InvertibleAffineTransformation,
    PositiveDefiniteMatrixTransformation
):
    """
    Affine transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Aff+ = GL+ ⋉ T
    """
    SYMBOL = "Aff+"
    FSYMBOL = "Aff+(ℝ^{n})"
    NAME = ("PositiveAffineTransformation", "PositiveAffine",)


@liegroup
@simplyconnected
class SpecialAffineTransformation(
    PositiveAffineTransformation,
    VolumePreservingDiffeomorphism
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
    NAME = ("SpecialAffineTransformation", "SpecialAffine",
            "VolumePreservingAffineTransformation", "VolumePreservingAffine",)


VolumePreservingAffineTransformation = SpecialAffineTransformation


@liegroup
class ConformalEuclideanTransformation(InvertibleAffineTransformation):
    """
    An affine transformation that preserves angles (up to their sign).

    This group includes reflections, and has therefore two connected
    components.

    symbol: CE = CO ⋉ T
    """
    SYMBOL = "CE"
    FSYMBOL = "CE({n})"
    NAME = ("ConformalEuclideanTransformation", "ConformalEuclidean",)


@liegroup
@connected
class SpecialConformalEuclideanTransformation(ConformalEuclideanTransformation):
    """
    An affine transformation that preserves angles

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: CSE = CSO ⋉ T

    alias: Similitude
    """
    SYMBOL = "CSE"
    FSYMBOL = "CSE({n})"
    NAME = ("SpecialConformalEuclideanTransformation",
            "SpecialConformalEuclidean",
            "Similitude",
            "SimilitudeTransformation",)


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
    NAME = ("EuclideanTransformation", "Euclidean",)


@liegroup
@connected
class SpecialEuclideanTransformation(SpecialConformalEuclideanTransformation):
    """
    A euclidean transformation with determinant +1

    symbol: SE = SO ⋉ T

    alias: RigidTransformation

    wiki: https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries
    """
    SYMBOL = "SE"
    FSYMBOL = "SE({n})"
    NAME = ("SpecialEuclideanTransformation",
            "SpecialEuclidean", "RigidTransformation", "Rigid",)


RigidTransformation = SpecialEuclideanTransformation


@liegroup
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
class Translation(PositiveDilation):
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


class LinearTransformation(AffineTransformation):
    """A linear transformation. May not be invertible."""
    NAME = ("LinearTransformation", "Linear",)


@liegroup
class InvertibleLinearTransformation(
    LinearTransformation,
    InvertibleAffineTransformation
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
    NAME = ("InvertibleLinearTransformation", "InvertibleLinear",)


@liegroup
@connected
class PositiveLinearTransformation(
    InvertibleLinearTransformation,
    PositiveAffineTransformation
):
    """
    Linear transformation with a positive determinant.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: GL+
    """
    SYMBOL = "GL+"
    FSYMBOL = "GL+({n})"
    NAME = ("PositiveLinearTransformation", "PositiveLinear",)


@liegroup
@connected
class SpecialLinearTransformation(
    PositiveLinearTransformation,
    VolumePreservingDiffeomorphism
):
    """
    A linear transformation with determinant 1 -- preserves volumes .

    symbol: SL

    wiki: https://en.wikipedia.org/wiki/Special_linear_group
    """
    SYMBOL = "SL"
    FSYMBOL = "SL({n})"
    NAME = ("SpecialLinearTransformation", "SpecialLinear",)


@liegroup
class ConformalOrthogonalTransformation(InvertibleLinearTransformation):
    """
    A linear transformation that preserves (absolute) angles (CO)

    symbol: CO = O x ℝ* (if n is even)
                 O x ℝ+ (if n is odd)

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """
    SYMBOL = "CO"
    FSYMBOL = "CO({n})"
    NAME = ("ConformalOrthogonalTransformation", "ConformalOrthogonal",)


@liegroup
@connected
class SpecialConformalOrthogonalTransformation(
    ConformalOrthogonalTransformation,
    SpecialLinearTransformation
):
    """
    An linear transformation that preserves angles (CSO)

    symbol: CSO = SO x ℝ+

    alias: Dilation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """
    SYMBOL = "CSO"
    FSYMBOL = "CSO({n})"
    NAME = ("SpecialConformalOrthogonalTransformation",
            "SpecialConformalOrthogonal",)


@liegroup
class OrthogonalTransformation(
    ConformalOrthogonalTransformation,
    EuclideanTransformation
):
    """
    A orthogonal matrix (AA' = I) with determinant ±1

    symbol: O

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group
    """
    SYMBOL = "O"
    FSYMBOL = "O({n})"
    NAME = ("OrthogonalTransformation", "Orthogonal",)


@liegroup
@connected
class SpecialOrthogonalTransformation(
    OrthogonalTransformation,
    SpecialConformalOrthogonalTransformation,
    SpecialEuclideanTransformation
):
    """
    A orthogonal matrix (AA' = I) with determinant +1

    symbol: SO

    alias: Rotation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group
    """
    SYMBOL = "SO"
    FSYMBOL = "SO({n})"
    NAME = ("SpecialOrthogonalTransformation",
            "SpecialOrthogonal", "Rotation", "RotationTransformation",)


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
    """
    SYMBOL = "S ⋉ SΔ"
    FSYMBOL = "S_{n} ⋉ SΔ({n})"
    NAME = ("SignedPermutation",)


@group
class Permutation(SignedPermutation, OrthogonalTransformation):
    """A permutation.

    A generalized permutation with non-zero entries +1.

    Permutations form the symmetric Lie group, named S.

    symbol: S

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    """
    SYMBOL = "S"
    FSYMBOL = "S_{n}"
    NAME = ("Permutation",)


class DiagonalTransformation(LinearTransformation):
    """A diagonal matrix, may not be invertible.

    wiki: https://en.wikipedia.org/wiki/Diagonal_matrix
    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """
    NAME = ("DiagonalTransformation", "Diagonal",)


@liegroup
class InvertibleDiagonalTransformation(
    DiagonalTransformation,
    GeneralizedPermutation
):
    """An invertible diagonal matrix.

    Invertible diagonal matrices form a Lie group.

    This group includes reflections, and has therefore two connected
    components.

    symbol: Δ
    """
    SYMBOL = "Δ"
    FSYMBOL = "Δ({n})"
    NAME = ("InvertibleDiagonalTransformation", "InvertibleDiagonal",)


@liegroup
@simplyconnected
class PositiveDiagonalTransformation(
    InvertibleDiagonalTransformation,
    PositiveLinearTransformation
):
    """A diagonal matrix with positive entries.

    This group does not include reflections, and has therefore a
    single connected component.

    symbol: Δ+
    """
    SYMBOL = "Δ+"
    FSYMBOL = "Δ+({n})"
    NAME = ("PositiveDiagonalTransformation", "PositiveDiagonal",)


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
    InvertibleDiagonalTransformation,
    SignedPermutation,
    FiniteReflection
):
    """A diagonal matrix with entries ±1.

    symbol: SΔ

    alias: CardinalReflection

    wiki: https://en.wikipedia.org/wiki/Reflection_group
    """
    SYMBOL = "SΔ"
    FSYMBOL = "SΔ({n})"
    NAME = ("SpecialDiagonalTransformation", "SpecialDiagonal",)


CardinalReflection = SpecialDiagonalTransformation


class MultiplicativeTransformation(
    DiagonalTransformation,
):
    """A scaling with the same factor in all dimensions.

    symbol: ℝ
    """
    SYMBOL = "ℝ"
    FSYMBOL = "ℝ"
    NAME = ("MultiplicativeTransformation", "Multiplicative",)


@liegroup
class InvertibleMultiplicativeTransformation(
    MultiplicativeTransformation,
    InvertibleDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.

    symbol: ℝ*

    alias: Homothety

    wiki: https://en.wikipedia.org/wiki/Multiplicative_group
    wiki: https://en.wikipedia.org/wiki/Homothety
    """
    SYMBOL = "ℝ*"
    FSYMBOL = "ℝ*"
    NAME = ("InvertibleMultiplicativeTransformation",
            "InvertibleMultiplicative", "Homothety", "HomothetyTransformation",)


Homothety = InvertibleMultiplicativeTransformation


@liegroup
@simplyconnected
class PositiveMultiplicativeTransformation(
    MultiplicativeTransformation,
    PositiveDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.

    symbol: ℝ+

    alias: PositiveHomothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """
    SYMBOL = "ℝ+"
    FSYMBOL = "ℝ+"
    NAME = ("PositiveMultiplicativeTransformation", "PositiveMultiplicative",
            "PositiveHomothety", "PositiveHomothetyTransformation",)


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
    SpecialOrthogonalTransformation
):
    """The identity transformation.

    symbol: I

    wiki: https://en.wikipedia.org/wiki/Identity_function
    """
    SYMBOL = "I"
    FSYMBOL = "I({n})"
    NAME = ("IdentityTransformation", "Identity",)
