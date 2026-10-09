# stdlib
from numbers import Integral, Real

# dependencies
import typing_extensions as tx
from bagof.magic import NotKwOnly

# core
from brainhops._core.compat import PLACEHOLDER, partial
from brainhops._core.typing import (
    ArrayProtocol,
    Derived,
    npmatrix,
    npvector,
)

# api
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder

# internals
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

# typing
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
    """The inverse of a transformation, resolved on demand.

    An `Inverse` holds a forward transformation and represents its
    inverse. The inverse is not computed when the wrapper is built. It is
    computed only when the wrapper is applied, computed, or converted to a
    concrete type. Placed next to its forward transformation in a
    [`Sequence`][], the two cancel to the identity, and no inverse is ever
    computed.

    Constructing `Inverse(forward=t)` represents the inverse of any
    transformation `t`. Each family of transformations also has its own
    typed inverse, such as [`InverseAffine`][] or
    [`InverseDisplacementField`][], which a transformation returns from its
    `inverse()` method. A typed inverse remains an instance of the family
    it inverts, so composition and the kind checks treat it exactly like a
    forward transformation of that family.

    It is one of the two [`Operation`][]s: the one that *reverses* the
    direction its forward maps, which is what `_reverses` says and what
    everything the base does differently for it follows from.
    """

    # --- class attributes ---------------------------------------------

    _operator: tx.ClassVar[str] = "inverse"
    _reverses: tx.ClassVar[bool] = True

    # The forward transformation family a typed inverse inverts. It is
    # unset on the generic `Inverse` front-door and set on each typed
    # subclass, which is what the materialization rebuilds. Which wrapper
    # a given transform gets is decided polymorphically, from the
    # `on={...}` predicates below, so nothing is registered anywhere.
    _resultof: _TypeReference = None

    # --- attributes ---------------------------------------------------

    forward: NotKwOnly[tx.Optional[TRANSFORMATION]] = None
    """The forward transformation whose inverse this represents."""

    # --- methods ------------------------------------------------------

    def inverse(self, compute: bool = False, **kwargs) -> Transformation:
        """Return the forward transformation, with the endpoints restored.

        The inverse of an inverse is the original forward transformation.
        An endpoint edit made on the wrapper is carried onto it. With
        `compute`, the other keywords are passed on to `compute()`.
        """
        return self._undo(compute, **kwargs)


# ----------------------------------------------------------------------
#       MIXINS
# ======================================================================


# Each typed inverse derives its `data` from its forward transform, in
# the forward's encoding, and reads its views (`translation`, `matrix`,
# `field`, ...) off that `data` exactly as a forward transform does. The
# convenience keyword its family takes (`translation=`, `matrix=`, ...)
# is deactivated: a wrapper is built from its forward only.


class ConcreteInverseMixin:
    """The parameter of a typed inverse, read off its forward.

    Every forward family derives the parameter of its own inverse under
    `_inverse` -- negated, reciprocal, transposed, mesh-inverted -- and
    caches it there, in its own encoding. A wrapper holds no array: it
    reports that one. The forward clears the cache when its `data` or a
    flag is assigned, so a rebuilt wrapper never recomputes an inversion
    and an edited forward never serves a stale one.
    """

    data = _alias("data", "forward._inverse", fset=False)


class FieldInverseMixin(ConcreteInverseMixin):
    """The views of a typed inverse of a field.

    `values` and `coefficients` read `data` under the forward's flags.
    They are plain properties rather than cached ones: a cache would live
    on the wrapper, which does not hear of the forward being edited. The
    inversion itself is still run once, on the forward.
    """

    @property
    def values(self) -> _OptionalArray:
        """The inverse field, as values."""
        return _data2values(self.data, self.store, self.degree, self.bound)

    @property
    def coefficients(self) -> _OptionalArray:
        """The inverse field, as spline coefficients."""
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
    """The inverse of a [`Translation`][], resolved on demand."""

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
    """The inverse of a [`Scaling`][], resolved on demand."""

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
    """The inverse of a [`Permutation`][], resolved on demand."""

    _resultof: _TypeReference = Permutation

    derived_fields: _FieldNames = "data", "permutation"

    _data: Derived[_OptionalPerm]
    _permutation: Derived[_OptionalPerm]


class InverseRotation(
    ConcreteInverseMixin,
    Inverse[Rotation],
    Rotation,
    on={"forward": partial(isinstance, PLACEHOLDER, Rotation)},
    # A `Rotation` is a `Linear`, so `Inverse(forward=rotation)` matches
    # `InverseLinear` just as well. The more specific wrapper wins: it
    # inverts by transposing rather than by solving a linear system.
    priority=1,
):
    """The inverse of a [`Rotation`][], resolved on demand."""

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
    """The inverse of a [`Linear`][] transformation, resolved on demand."""

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
    """The inverse of an [`Affine`][] transformation, resolved on demand."""

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
    """The inverse of a [`DisplacementField`][], resolved on demand.

    The wrapper reports the `degree`, `bound` and `store` of the forward
    field, and its `data` is the inverse field in that same encoding: the
    forward field's values are inverted, and the result is fitted back to
    spline coefficients when the forward field holds coefficients. Its
    `field` view is the inverse field, as values, either way.

    !!! note "Accuracy"
        The inversion only sees the forward field's values at the grid
        nodes: it inverts the piecewise-affine map they define (see
        [`brainhops._ext.invfield.inverse`][]), whatever the forward's
        `degree`. The inverse is then interpolated with that degree. It
        is exact at the level of that piecewise-affine map only, so a
        cubic field is inverted about as accurately as a linear one,
        and the error grows near the border. On smooth fields of a few
        voxels' amplitude, `fwd(inv(x)) - x` is typically a few
        hundredths of a voxel in the interior, and a few tenths near
        the border.
    """

    # --- class attributes ---------------------------------------------

    _resultof: _TypeReference = DisplacementField

    derived_fields: _FieldNames = ("data", "field", "values", "coefficients")

    # --- attributes ---------------------------------------------------

    forward: NotKwOnly[tx.Optional[DisplacementField]] = None
    """The displacement field whose inverse this represents."""

    # --- derived attributes -------------------------------------------
    # Declare derived fields as classvar to exclude them from `__init__`

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
    """The inverse of a [`CoordinatesField`][], resolved on demand.

    The wrapper reports the `degree`, `bound` and `store` of the forward
    field, and its `data` is the inverse field in that same encoding: the
    forward field's values are inverted, and the result is fitted back to
    spline coefficients when the forward field holds coefficients. Its
    `field` view is the inverse field, as values, either way.

    !!! note "Accuracy"
        As for [`InverseDisplacementField`][], the inversion only sees
        the forward field's values at the grid nodes and inverts the
        piecewise-affine map they define, whatever the forward's
        `degree`; the result is approximate between nodes, and more so
        near the border.

    Placed next to the field it inverts in a [`Sequence`][], the two cancel
    and nothing is computed. That is the cheap path, and the one worth
    reaching for: a coordinate field has no closed-form inverse, so
    materializing this wrapper runs a mesh inversion.

    !!! warning "The coordinates must live on the grid they are sampled on"
        A coordinate field is inverted by reading it as the identity grid
        plus a displacement, inverting that displacement, and adding the
        grid back. The mesh inversion therefore assumes the coordinates are
        expressed in the units of the grid they are sampled on -- voxels,
        in practice. A field whose coordinates are in world units is
        inverted as though they were voxel coordinates, and the result,
        while well defined, falls outside the output lattice and is of
        little use. Compose the world-to-voxel affine into the field first.
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
    """The inverse of an [`AffineExponential`][]: the tangent negated."""

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
    """The inverse of a [`LinearExponential`][]: the tangent negated."""

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
    # `InverseRotation` matches a `RotationExponential` too, with its own
    # priority; the deeper subclass wins the tie.
    priority=1,
):
    """The inverse of a [`RotationExponential`][]: the tangent negated."""

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
    """The inverse of a [`ScalingExponential`][]: the tangent negated."""

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
    """The inverse of a [`StationaryVelocityField`][]: `exp(-v)`.

    Its `data` is the negated velocity, in the encoding of the forward
    field, whose flags (`steps` included) it reports. Its `field` view
    integrates that velocity -- exactly as the forward integrates its own,
    rather than by inverting the forward's displacement. The accuracy
    note of [`InverseDisplacementField`][] does not apply: no mesh is
    inverted, the inverse is exact in the tangent, and `exp(-v)` is
    integrated as accurately as `exp(v)` is.
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
