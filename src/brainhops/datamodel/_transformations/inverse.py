from numbers import Integral, Real

import typing_extensions as tx
from bagof.magic import NotKwOnly

from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.typing import (
    ArrayProtocol,
    Derived,
    npmatrix,
    npvector,
)
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

from .base import Transformation
from .concrete import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    StoreEnum,
    Translation,
    _alias,
    _data2coeffs,
    _data2values,
)
from .nocycles import register_operator
from .operators import Operation
from .tangents import (
    AffineExponential,
    LinearExponential,
    RotationExponential,
    ScalingExponential,
    StationaryVelocityField,
)

TRANSFORMATION = tx.TypeVar("TRANSFORMATION", bound=Transformation)

_TypeReference = tx.ClassVar[tx.Optional[tx.Type[Transformation]]]
_FieldNames = tx.ClassVar[tx.Tuple[str, ...]]
_Matrix: tx.TypeAlias = npmatrix[Real]
_Vector: tx.TypeAlias = npvector[Real]
_Perm: tx.TypeAlias = npmatrix[Integral]
_Array: tx.TypeAlias = ArrayProtocol
_OptionalMatrix: tx.TypeAlias = tx.Optional[_Matrix]
_OptionalVector: tx.TypeAlias = tx.Optional[_Vector]
_OptionalPerm: tx.TypeAlias = tx.Optional[_Perm]
_OptionalArray: tx.TypeAlias = tx.Optional[_Array]


# ----------------------------------------------------------------------
#       BASE
# ======================================================================


@register_operator("inverse")
class Inverse(Operation, tx.Generic[TRANSFORMATION], polymorphic=True):
    """Lazy inverse of a forward transformation, resolved on demand.

    The inverse is resolved when it is applied, computed or converted. When
    the inverse is placed next to its forward transformation in a
    [`Sequence`][brainhops.datamodel.transformations.Sequence], the two cancel
    and no inverse is computed.

    `Inverse(forward=t)` builds the inverse wrapper that belongs to the family
    of `t`, such as [`InverseAffine`][] for an affine transformation, and
    `t.inverse()` returns the same wrapper. Such a typed inverse is an instance
    of the family it inverts, so composition and the kind checks treat it like
    any other member of that family. Unlike the other kinds of
    [`Operation`][], an inverse reverses the direction of its forward
    transformation, so its input and output systems are those of the forward,
    swapped.
    """

    _operator: tx.ClassVar[str] = "inverse"
    _reverses: tx.ClassVar[bool] = True

    # The class of the family that a typed inverse inverts, which is the type
    # that evaluating the inverse produces. It is unset on the generic
    # `Inverse` class. The wrapper class is chosen by the `on={...}`
    # predicates of the subclasses, so nothing needs to be registered.
    _resultof: _TypeReference = None

    forward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """The transformation whose inverse this is."""

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation, with its endpoints restored.

        The inverse of an inverse is the forward. Endpoints edited on the
        wrapper are carried onto the forward; when there are none, the forward
        itself is returned. With `compute`, the other keyword arguments are
        passed to `compute()`.
        """
        return self._undo(compute, **kwargs)


# ----------------------------------------------------------------------
#       MIXINS
# ======================================================================


# A typed inverse derives its `data` from the forward and reads its views off
# that data. The convenience keywords of the family, such as `translation=`
# and `matrix=`, are deactivated, since a wrapper is built from its forward
# only.


class ConcreteInverseMixin:
    """Mixin that reads the parameter of a typed inverse from its forward.

    Each family computes the parameter of its inverse in the `_inverse`
    property of the forward transformation, and caches it there in the
    encoding of the forward. The wrapper holds no array of its own and reports
    the cached array as its `data`. The forward clears the cache when its
    `data` or a flag is assigned, so an edited forward never serves a stale
    inverse.
    """

    data = _alias("data", "forward._inverse", fset=False)


class FieldInverseMixin(ConcreteInverseMixin):
    """Mixin that decodes the views of an inverse field with the forward flags.

    The views are not cached, because a cache would live on the wrapper, which
    is not told when the forward is edited.
    """

    @property
    def values(self) -> _OptionalArray:
        """Inverse field as values."""
        return _data2values(self.data, self.store, self.degree, self.bound)

    @property
    def coefficients(self) -> _OptionalArray:
        """Inverse field as spline coefficients."""
        return _data2coeffs(self.data, self.store, self.degree, self.bound)


# ----------------------------------------------------------------------
#       CONCRETE INVERSES
# ======================================================================


class InverseTranslation(
    ConcreteInverseMixin,
    Inverse[Translation],
    Translation,
    on={"forward": partial(isinstance, PLACEHOLDER, Translation)},
):
    """Inverse of a [`Translation`][], resolved on demand."""

    _resultof: _TypeReference = Translation

    derived_fields: _FieldNames = "data", "translation"

    _data: Derived[_OptionalVector]
    _translation: Derived[_OptionalVector]


class InverseScaling(
    ConcreteInverseMixin,
    Inverse[Scaling],
    Scaling,
    on={"forward": partial(isinstance, PLACEHOLDER, Scaling)},
):
    """Inverse of a [`Scaling`][], resolved on demand."""

    _resultof: _TypeReference = Scaling

    derived_fields: _FieldNames = "data", "scale"

    _data: Derived[_OptionalVector]
    _scale: Derived[_OptionalVector]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InversePermutation(
    ConcreteInverseMixin,
    Inverse[Permutation],
    Permutation,
    on={"forward": partial(isinstance, PLACEHOLDER, Permutation)},
):
    """Inverse of a [`Permutation`][], resolved on demand."""

    _resultof: _TypeReference = Permutation

    derived_fields: _FieldNames = "data", "permutation"

    _data: Derived[_OptionalPerm]
    _permutation: Derived[_OptionalPerm]


class InverseRotation(
    ConcreteInverseMixin,
    Inverse[Rotation],
    Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # A rotation is a linear transformation, so `InverseLinear` matches it too.
    # This wrapper wins and inverts by transposing rather than by solving a
    # linear system.
    priority=1,
):
    """Inverse of a [`Rotation`][], resolved on demand."""

    _resultof: _TypeReference = Rotation

    derived_fields: _FieldNames = "data", "matrix"

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseLinear(
    ConcreteInverseMixin,
    Inverse[Linear],
    Linear,
    on={"forward": partial(isinstance, PLACEHOLDER, Linear)},
):
    """Inverse of a [`Linear`][] transformation, resolved on demand."""

    _resultof: _TypeReference = Linear

    derived_fields: _FieldNames = "data", "matrix"

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseAffine(
    ConcreteInverseMixin,
    Inverse[Affine],
    Affine,
    on={"forward": partial(isinstance, PLACEHOLDER, Affine)},
):
    """Inverse of an [`Affine`][] transformation, resolved on demand."""

    _resultof: _TypeReference = Affine

    derived_fields: _FieldNames = (
        "data",
        "matrix",
        "homogeneous_matrix",
    )

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseDisplacementField(
    FieldInverseMixin,
    Inverse[DisplacementField],
    DisplacementField,
    on={"forward": partial(isinstance, PLACEHOLDER, DisplacementField)},
):
    """Inverse of a [`DisplacementField`][], resolved on demand.

    The inverse reports the `degree`, `bound` and `store` of its forward. Its
    `data` is the inverse field in the same encoding. To build it, the values
    of the forward are inverted, and then refitted to coefficients if the
    forward holds coefficients. The `field` view holds the inverse as values
    in either case.

    !!! note "Accuracy"
        The inversion sees only the values at the grid nodes and inverts the
        piecewise-affine map that they define (see
        [`inverse`][brainhops._ext.invfield.inverse]), whatever the degree of
        the forward. The result is then interpolated at that degree. The
        inversion is exact only for the piecewise-affine map, so a cubic field
        is inverted about as accurately as a linear one, and the error grows
        near the border. For smooth fields with an amplitude of a few voxels,
        `fwd(inv(x)) - x` is typically a few hundredths of a voxel in the
        interior and a few tenths near the border.
    """

    _resultof: _TypeReference = DisplacementField

    derived_fields: _FieldNames = ("data", "field", "values", "coefficients")

    forward: NotKwOnly[tx.Optional[DisplacementField]] = None
    """The displacement field whose inverse this is."""

    # The fields below derive from the forward, so they are kept out of
    # `__init__`.

    _data: Derived[_OptionalArray]
    _field: Derived[_OptionalArray]
    _values: Derived[_OptionalArray]
    _coefficients: Derived[_OptionalArray]
    _degree: Derived[InterpolationOrder]
    _bound: Derived[tx.Union[BoundaryCondition, float]]
    _store: Derived[StoreEnum]
    _log: Derived[bool]

    degree = _alias("degree", "forward.degree", fset=False)
    bound = _alias("bound", "forward.bound", fset=False)
    store = _alias("store", "forward.store", fset=False)
    log = _alias("log", "forward.log", fset=False)


class InverseCoordinatesField(
    FieldInverseMixin,
    Inverse[CoordinatesField],
    CoordinatesField,
    on={"forward": partial(isinstance, PLACEHOLDER, CoordinatesField)},
):
    """Inverse of a [`CoordinatesField`][], resolved on demand.

    The inverse is encoded like an [`InverseDisplacementField`][]. When the
    inverse is placed next to the field it inverts in a
    [`Sequence`][brainhops.datamodel.transformations.Sequence], the two cancel
    without any computation. Otherwise, evaluating the inverse runs a mesh
    inversion, since a coordinates field has no closed-form inverse.

    !!! note "Accuracy"
        As for an [`InverseDisplacementField`][], only the piecewise-affine map
        defined by the values at the grid nodes is inverted, so the result is
        approximate between nodes and worse near the border.

    !!! warning "The coordinates must live on the grid they are sampled on"
        A coordinates field is inverted as an identity grid plus a
        displacement, which assumes coordinates in voxels of the grid.
        Coordinates in world units give a result that is well defined but
        useless, because it does not lie on the output grid. A world-to-voxel
        affine transformation should therefore be composed into the field
        first.
    """

    _resultof: _TypeReference = CoordinatesField

    derived_fields: _FieldNames = "data", "field", "values", "coefficients"

    _data: Derived[_OptionalArray]
    _field: Derived[_OptionalArray]
    _values: Derived[_OptionalArray]
    _coordinates: Derived[_OptionalArray]
    _degree: Derived[InterpolationOrder]
    _bound: Derived[tx.Union[BoundaryCondition, float]]
    _store: Derived[StoreEnum]

    degree = _alias("degree", "forward.degree", fset=False)
    bound = _alias("bound", "forward.bound", fset=False)
    store = _alias("store", "forward.store", fset=False)


# ----------------------------------------------------------------------
#       INVERSE TANGENTS
# ======================================================================


class InverseAffineExponential(
    ConcreteInverseMixin,
    Inverse[AffineExponential],
    AffineExponential,
    on={"forward": partial(isinstance, PLACEHOLDER, AffineExponential)},
):
    """Inverse of an [`AffineExponential`][], whose tangent is negated."""

    _resultof: _TypeReference = AffineExponential

    derived_fields: _FieldNames = (
        "data",
        "matrix",
        "logmatrix",
        "homogeneous_matrix",
        "homogeneous_logmatrix",
    )

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _logmatrix: Derived[_OptionalMatrix]
    _homogeneous_matrix: Derived[_OptionalMatrix]
    _homogeneous_logmatrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseLinearExponential(
    ConcreteInverseMixin,
    Inverse[LinearExponential],
    LinearExponential,
    on={"forward": partial(isinstance, PLACEHOLDER, LinearExponential)},
):
    """Inverse of a [`LinearExponential`][], whose tangent is negated."""

    _resultof: _TypeReference = LinearExponential

    derived_fields: _FieldNames = "data", "matrix", "logmatrix"

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _logmatrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseRotationExponential(
    ConcreteInverseMixin,
    Inverse[RotationExponential],
    RotationExponential,
    on={"forward": partial(isinstance, PLACEHOLDER, RotationExponential)},
    # `InverseRotation` matches a rotation exponential too, with the same
    # priority, and the deeper subclass wins the tie.
    priority=1,
):
    """Inverse of a [`RotationExponential`][], whose tangent is negated."""

    _resultof: _TypeReference = RotationExponential

    derived_fields: _FieldNames = "data", "matrix", "logmatrix"

    _data: Derived[_OptionalMatrix]
    _matrix: Derived[_OptionalMatrix]
    _logmatrix: Derived[_OptionalMatrix]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseScalingExponential(
    ConcreteInverseMixin,
    Inverse[ScalingExponential],
    ScalingExponential,
    on={"forward": partial(isinstance, PLACEHOLDER, ScalingExponential)},
):
    """Inverse of a [`ScalingExponential`][], whose tangent is negated."""

    _resultof: _TypeReference = ScalingExponential

    derived_fields: _FieldNames = "data", "scale"

    _data: Derived[tx.Optional[npvector[Real]]]
    _scale: Derived[tx.Optional[npvector[Real]]]
    _logscale: Derived[tx.Optional[npvector[Real]]]
    _log: Derived[bool]

    log = _alias("log", "forward.log", fset=False)


class InverseStationaryVelocityField(
    FieldInverseMixin,
    Inverse[StationaryVelocityField],
    StationaryVelocityField,
    on={"forward": partial(isinstance, PLACEHOLDER, StationaryVelocityField)},
):
    """Inverse of a [`StationaryVelocityField`][], resolved on demand.

    The inverse of the map `exp(v)` is `exp(-v)`. Its `data` is the negated
    velocity in the encoding of the forward, and it reports the flags of the
    forward, including `steps`. Its `field` view integrates the negated
    velocity in the same way as the forward integrates its own velocity. No
    mesh is inverted, so the accuracy caveat of [`InverseDisplacementField`][]
    does not apply.
    """

    _resultof: _TypeReference = StationaryVelocityField

    derived_fields: _FieldNames = "data", "field", "values", "coefficients"

    _data: Derived[_OptionalArray]
    _field: Derived[_OptionalArray]
    _values: Derived[_OptionalArray]
    _coefficients: Derived[_OptionalArray]
    _degree: Derived[InterpolationOrder]
    _bound: Derived[tx.Union[BoundaryCondition, float]]
    _store: Derived[StoreEnum]
    _log: Derived[bool]
    _steps: Derived[tx.Optional[int]]

    degree = _alias("degree", "forward.degree", fset=False)
    bound = _alias("bound", "forward.bound", fset=False)
    store = _alias("store", "forward.store", fset=False)
    log = _alias("log", "forward.log", fset=False)
    steps = _alias("steps", "forward.steps", fset=False)
