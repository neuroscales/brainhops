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
without it being conflated with type inheritence 
(e.g. "is this transformation an instance of the Linear class?").

Many linear transformations, when constrained to be invertible, can
be thought of as members of a Lie group. Lie groups are therefore 
heavily used in our hierarchy.

Lie groups
----------

```
G | > 0 | = 1 | ⋉ T
======================
GL ------------- Aff             General Linear     | Affine
 | \
 |   GL+ ------- Aff+
 |    | \
 |    |  SL ---- SAff            Special Linear     | Special Affine
 |    |   | 
CO ------------- CE              Conformal          | Affine Conformal
 | \  |   |
 |   CSO ------- CSE             Special Conformal  | Special Affine Conformal
 |    |   |
 O ------------- E               Orthogonal         | Euclidean
 | \  | /
 S   SO -------- SE              Special Orthogonal | Special Euclidean
 | /
 I ------------- T               Identity           | Translations
```

General Lie groups (GL/CO/O) are in general not "connected", and
instead composed of two disconnected components. One that contains
the identity, and one that does not (the "flipped" version).

Positive Lie groups (SO/CSO/GL+) are restricted to transforms with
positive determinant, and therefore exclude flips. They can be defined
as the component from the corresponding general group that contains the
identity transform.

Note that the Special Linear group (SL) can have negative determinants,
but is restricted to unit-norm determinants (+/- 1, i.e. volume
preserving).

All classical linear group can be extended with the translation group
(⋉ T) to define affine groups.

::note
    * The Special Euclidean group (SE) contains transforms that are
      classicaly referred to as "rigid-body" transforms. They preserve
      angles and volumes.

    * The Special Conformal Euclidean group (CSE) contains transforms 
      that are classicaly referred to as "similitude" (or conformal) 
      transforms. They only preserve angles.
"""
__all__ = [
    "Transformation", 
    "Bijection", 
    "Isomorphism", 
    "Diffeomorphism",                               # Diff
    "VolumePreservingDiffeomorphism",               # SDiff
    # LIE / MATRIX
    "MatrixTransformation", 
    "LieGroupTransformation", 
    "DiffeomorphicLieGroupTransformation",
    "ConnectedLieGroupTransformation", 
    "SimplyConnectedLieGroupTransformation", 
    "MatrixLieGroupTransformation", 
    "ConnectedMatrixLieGroupTransformation",
    # AFFINE
    "AffineTransformation",
    "InvertibleAffineTransformation",               # Aff              (det ≠ 0)
    "PositiveAffineTransformation",                 # Aff+             (det > 0)
    "SpecialAffineTransformation",                  # SAff = SL  ⋉ T   (det = 1)
    "VolumePreservingAffineTransformation",         # ^ alias
    "ConformalEuclideanTransformation",             # CE   = CO  ⋉ T   (det ≠ 0)
    "SpecialConformalEuclideanTransformation",      # CSE  = CSO ⋉ T   (det > 0)
    "Similitude",                                   # ^ alias
    "EuclideanTransformation",                      # E    = O   ⋉ T   (det =±1)
    "SpecialEuclideanTransformation",               # SE   = SO.  ⋉ T  (det = 1)
    "RigidTransformation",                          # ^ alias
    "Dilation",
    "PositiveDilation",
    "Translation",                                  # T                (det = 1)
    # LINEAR
    "LinearTransformation",
    "InvertibleLinearTransformation",               # GL               (det ≠ 0)
    "PositiveLinearTransformation",                 # GL+              (det > 0)    
    "SpecialLinearTransformation",                  # SL               (det = 1)
    "ConformalOrthogonalTransformation",            # CO  = O x R*     (det ≠ 0)
    "SpecialConformalOrthogonalTransformation",     # CSO = O x R+     (det > 0)
    "OrthogonalTransformation",                     # O                (det =±1)
    "SpecialOrthogonalTransformation",              # SO               (det = 1)
    "Rotation",                                     # ^ alias
    "Permutation",                                 
    "InvertiblePermutation",                        # S                (det ≠ 0)
    "DiagonalTransformation",
    "InvertibleDiagonalTransformation", 
    "PositiveDiagonalTransformation",
    "MultiplicativeTransformation",                 # R
    "InvertibleMultiplicativeTransformation",       # R*               (det ≠ 0)
    "PositiveMultiplicativeTransformation",         # R+               (det > 0)
    # IDENTITY
    "IdentityTransformation",                       # I                (det = 1)
]
from abc import ABC


class Transformation(ABC):
    """Any coordinate transformation."""

    SYMBOL: str
    FSYMBOL: str


class Bijection(Transformation):
    """An invertible transformation.
    
    wiki: https://en.wikipedia.org/wiki/Bijection
    """


class Isomorphism(Bijection):
    """
    In all useful senses, bijective morphisms are isomorphisms,
    but there are mathematical spaces on which this is not true.
    We introduce a separate class to make mathematicians happy.

    wiki: https://en.wikipedia.org/wiki/Isomorphism
    """


class Diffeomorphism(Isomorphism):
    """Smooth invertible transformation, with a smooth inverse.
    
    wiki: https://en.wikipedia.org/wiki/Diffeomorphism
    """

    SYMBOL = "Diff"


class VolumePreservingDiffeomorphism(Diffeomorphism):
    """A diffeomorphism that preserves volumes."""

    SYMBOL = "SDiff"


# ----------------------------------------------------------------------
#   L I E  /  M A T R I X
# ----------------------------------------------------------------------


class MatrixTransformation(Transformation):
    """Transformation that can be represented as a matrix.
    
    wiki: https://en.wikipedia.org/wiki/Transformation_matrix
    """


class LieGroupTransformation(Transformation):
    """Transformation that belong to a Lie group.
    
    wiki: https://en.wikipedia.org/wiki/Lie_group
    """


class DiffeomorphicLieGroupTransformation(
    LieGroupTransformation, 
    Diffeomorphism
):
    """Transformation that belong to a Diffeomorphic Lie group."""


class ConnectedLieGroupTransformation(DiffeomorphicLieGroupTransformation):
    """
    Transformation that belong to Lie groups with a single connected component.

    wiki: https://en.wikipedia.org/wiki/Connected_space
    """


class SimplyConnectedLieGroupTransformation(ConnectedLieGroupTransformation):
    """
    Transformation that belong to Lie groups with a single connected component
    and no holes.

    wiki: https://en.wikipedia.org/wiki/Simply_connected_space
    """


class MatrixLieGroupTransformation(
    DiffeomorphicLieGroupTransformation, 
    MatrixTransformation
):
    """
    A transformation that belongs to a Lie group and that can be 
    represented as a matrix.
    
    Most Lie groups are matrix groups (Lie groups oiginate in the study 
    of matrix subgroups), and all matrix Lie groups are subgroups of 
    the group of Diffeomorphisms.

    wiki: https://en.wikipedia.org/wiki/Lie_group#Matrix_Lie_groups
    """


class ConnectedMatrixLieGroupTransformation(
    MatrixLieGroupTransformation, 
    ConnectedLieGroupTransformation
):
    """
    Connected Lie groups have a single connected component.
    """


# ----------------------------------------------------------------------
#   A F F I N E
# ----------------------------------------------------------------------

class AffineTransformation(MatrixTransformation):
    """An affine transformation. May not be invertible.
    
    wiki: https://en.wikipedia.org/wiki/Affine_transformation
    """


class InvertibleAffineTransformation(
    AffineTransformation, 
    MatrixLieGroupTransformation
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
    FSYMBOL = "Aff({n})"


class PositiveAffineTransformation(
    InvertibleAffineTransformation, 
    ConnectedMatrixLieGroupTransformation
):
    """
    Affine transformation with a positive determinant.

    This group does not include reflections, and has therefore a 
    single connected component.

    symbol: Aff+ = GL+ ⋉ T
    """
    SYMBOL = "Aff+"
    FSYMBOL = "Aff+({n})"


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
    SYMBOL = 'SAff'
    FSYMBOL = 'SAff({n})'


VolumePreservingAffineTransformation = SpecialAffineTransformation


class ConformalEuclideanTransformation(InvertibleAffineTransformation):
    """
    An affine transformation that preserves angles (up to their sign).
    
    This group includes reflections, and has therefore two connected 
    components.

    symbol: CE = CO ⋉ T
    """
    SYMBOL = 'CE'
    FSYMBOL = 'CE({n})'


class SpecialConformalEuclideanTransformation(ConformalEuclideanTransformation):
    """
    An affine transformation that preserves angles
    
    This group does not include reflections, and has therefore a 
    single connected component.

    symbol: CSE = CSO ⋉ T

    alias: Similitude
    """
    SYMBOL = 'CSE'
    FSYMBOL = 'CSE({n})'


Similitude = SpecialConformalEuclideanTransformation


class EuclideanTransformation(ConformalEuclideanTransformation):
    """
    A euclidean transformation with positive determinant +/- 1
    
    symbol: E = O ⋉ T

    wiki: https://en.wikipedia.org/wiki/Euclidean_group
    """
    SYMBOL = 'E'
    FSYMBOL = 'E({n})'


class SpecialEuclideanTransformation(SpecialConformalEuclideanTransformation):
    """
    A euclidean transformation with determinant +1
    
    symbol: SE = SO ⋉ T

    alias: RigidTransformation

    wiki: https://en.wikipedia.org/wiki/Euclidean_group#Direct_and_indirect_isometries
    """
    SYMBOL = 'SE'
    FSYMBOL = 'SE({n})'


RigidTransformation = SpecialEuclideanTransformation


class Dilation(ConformalEuclideanTransformation):
    """
    A dilation is a similitude with no rotation, i.e. a scaling with
    translation.

    symbol: R* ⋉ T

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """
    SYMBOL = 'R* ⋉ T'
    FSYMBOL = 'R* ⋉ T({n})'


class PositiveDilation(Dilation, SpecialConformalEuclideanTransformation):
    """
    A dilation with a positive scaling factor.

    symbol: R+ ⋉ T

    wiki: https://en.wikipedia.org/wiki/Homothety
    wiki: https://en.wikipedia.org/wiki/Dilation_(metric_space)
    """
    SYMBOL = 'R+ ⋉ T'
    FSYMBOL = 'R+ ⋉ T({n})'


class Translation(PositiveDilation):
    """A translation.
    
    symbol: T

    wiki: https://en.wikipedia.org/wiki/Translation_(geometry)#As_a_group
    """
    SYMBOL = 'T'
    FSYMBOL = 'T({n})'


# ----------------------------------------------------------------------
#   L I N E A R
# ----------------------------------------------------------------------


class LinearTransformation(AffineTransformation):
    """A linear transformation. May not be invertible."""


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


class SpecialLinearTransformation(
    PositiveLinearTransformation,
    VolumePreservingDiffeomorphism
):
    """
    A linear transformation with determinant 1 -- preserves volumes .

    symbol: SL

    wiki: https://en.wikipedia.org/wiki/Special_linear_group
    """
    SYMBOL = 'SL'
    FSYMBOL = 'SL({n})'


class ConformalOrthogonalTransformation(InvertibleLinearTransformation):
    """
    A linear transformation that preserves (absolute) angles (CO)

    symbol: CO = O x R* (if n is even)
                 O x R+ (if n is odd)

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """
    SYMBOL = 'CO'
    FSYMBOL = 'CO({n})'


class SpecialConformalOrthogonalTransformation(
    ConformalOrthogonalTransformation, 
    SpecialLinearTransformation
):
    """
    An linear transformation that preserves angles (CSO)

    symbol: CSO = SO x R+

    alias: Dilation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Conformal_group
    """
    SYMBOL = 'CSO'
    FSYMBOL = 'CSO({n})'


class OrthogonalTransformation(ConformalOrthogonalTransformation, Euclidean):
    """
    A orthogonal matrix (AA' = I) with determinant +/- 1

    symbol: O

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group
    """
    SYMBOL = 'O'
    FSYMBOL = 'O({n})'


class SpecialOrthogonalTransformation(
    OrthogonalTransformation, 
    SpecialConformalOrthogonalTransformation, 
    SpecialEuclideanTransformation
):
    """
    A orthogonal matrix (AA' = I) with determinant +1 (SO)

    symbol: SO
    
    alias: Rotation

    wiki: https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group
    """
    SYMBOL = 'SO'
    FSYMBOL = 'SO({n})'


Rotation = SpecialOrthogonalTransformation


class Permutation(LinearTransformation):
    """A permutation, may not be invertible."""


class InvertiblePermutation(Permutation, OrthogonalTransformation):
    """An invertible permutation.
    
    Invertible permutation form the symmetric Lie group, named S

    Symbol: S

    wiki: https://en.wikipedia.org/wiki/Permutation_group
    """
    SYMBOL = 'S'
    FSYMBOL = 'S_{n}'


class DiagonalTransformation(LinearTransformation):
    """A diagonal matrix, may not be invertible.
    
    wiki: https://en.wikipedia.org/wiki/Diagonal_matrix
    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """


class InvertibleDiagonalTransformation(
    DiagonalTransformation, 
    InvertibleLinearTransformation
):
    """An invertible diagonal matrix.
    
    Invertible diagonal matrices form a Lie group.

    This group includes reflections, and has therefore two connected 
    components.
    """


class PositiveDiagonalTransformation(
    InvertibleDiagonalTransformation, 
    PositiveLinearTransformation
):
    """A diagonal matrix with positive entries.
    
    This group does not include reflections, and has therefore a
    single connected component.
    """


class MultiplicativeTransformation(
    DiagonalTransformation,
):
    """A scaling with the same factor in all dimensions.
    
    symbol: R
    """
    SYMBOL = 'R'
    FSYMBOL = 'R'


class InvertibleMultiplicativeTransformation(
    MultiplicativeTransformation, 
    InvertibleDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.
    
    symbol: R*

    alias: Homothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """
    SYMBOL = 'R*'
    FSYMBOL = 'R*'


Homothety = InvertibleMultiplicativeTransformation


class PositiveMultiplicativeTransformation(
    MultiplicativeTransformation, 
    PositiveDiagonalTransformation
):
    """A scaling with the same positive factor in all dimensions.
    
    symbol: R+

    alias: PositiveHomothety

    wiki: https://en.wikipedia.org/wiki/Scaling_(geometry)
    """
    SYMBOL = 'R+'
    FSYMBOL = 'R+'


PositiveHomothety = PositiveMultiplicativeTransformation


# ----------------------------------------------------------------------
#   I D E N T I T Y
# ----------------------------------------------------------------------

class IdentityTransformation(
    Translation,
    InvertiblePermutation,
    PositiveMultiplicativeTransformation,
    SpecialOrthogonalTransformation
):
    """The identity transformation.
    
    symbol: I

    wiki: https://en.wikipedia.org/wiki/Identity_function
    """
    SYMBOL = 'I'
    FSYMBOL = 'I({n})'