"""Tests for the `log` encoding flag: transformations stored as their tangent.

`log=True` says that `data` holds the tangent of the map about the
identity, and selects a subclass whose views read it as such: a
`StationaryVelocityField` (its velocity, integrated by scaling and
squaring), an `AffineExponential`, `LinearExponential`,
`RotationExponential` or `ScalingExponential` (the exponential of the
matrix or of the scales). These cover the views under every `log` and
`store` combination, the conversions, the exponential maps against
SciPy and against analytic flows, the exact tangent operations, the
containers, the setters and the kind checks. Fields use cubic splines:
linear ones would hide an encoding mixed up with another.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.linalg

from brainhops._core.bsplines import coeff2value_field, value2coeff_field
from brainhops.datamodel import kinds
from brainhops.datamodel._transformations import tangents as _tangents
from brainhops.datamodel.systems import CoordinateSystem, SpaceAxis
from brainhops.datamodel.transformations import (
    Affine,
    AffineExponential,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    LinearExponential,
    Permutation,
    Rotation,
    RotationExponential,
    Scaling,
    ScalingExponential,
    Sequence,
    StationaryVelocityField,
    SubspaceTransformation,
    Translation,
    is_identity,
    is_kind,
)
from brainhops.errors import (
    ConversionError,
    DomainError,
)
from brainhops.io.transformations.base.affines import VoxelToLPS

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# An affine tangent `[L, l]`, and the matrices it generates.
TANGENT = np.array(
    [[0.1, -0.3, 0.2, 1.0], [0.3, 0.05, -0.1, 2.0], [-0.2, 0.1, 0.0, -1.0]]
)
LINEAR_TANGENT = TANGENT[:, :-1].copy()
SKEW = np.array([[0.0, -0.4, 0.2], [0.4, 0.0, -0.3], [-0.2, 0.3, 0.0]])


def _homogeneous_tangent(tangent: np.ndarray) -> np.ndarray:
    n = tangent.shape[0]
    full = np.zeros((n + 1, n + 1))
    full[:n] = tangent
    return full


def _expm_affine(tangent: np.ndarray) -> np.ndarray:
    return scipy.linalg.expm(_homogeneous_tangent(tangent))[:-1]


SHAPE = (32, 32)
INTERIOR = (slice(10, 22), slice(10, 22))
DEGREE = 3


def _grid(shape: tuple = SHAPE) -> np.ndarray:
    axes = [np.arange(s, dtype=float) for s in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


# A linear velocity `x -> L x + l`, in the voxels of the grid, sampled as
# it is (the grid is not subtracted): its flow at time one is the affine
# `expm([[L, l], [0, 0]])`.
# The offset is chosen so that the velocity is small at the centre of the
# grid, whose interior the flow is checked on.
FIELD_GENERATOR = np.array([[0.05, -0.2], [0.2, 0.03]])
FIELD_OFFSET = np.array([0.3, -0.2]) - FIELD_GENERATOR @ np.full(2, 15.5)
FIELD_TANGENT = np.concatenate([FIELD_GENERATOR, FIELD_OFFSET[:, None]], 1)


def _linear_velocity() -> np.ndarray:
    return _grid() @ FIELD_GENERATOR.T + FIELD_OFFSET


def _linear_flow() -> np.ndarray:
    # The displacement of the flow of the linear velocity.
    matrix = _expm_affine(FIELD_TANGENT)
    grid = _grid()
    return grid @ matrix[:, :-1].T + matrix[:, -1] - grid


def _velocity_field(**kwargs) -> StationaryVelocityField:
    return StationaryVelocityField(
        data=_linear_velocity(), degree=DEGREE, **kwargs
    )


def _system(n: int) -> CoordinateSystem:
    return CoordinateSystem(
        axes=[SpaceAxis(name=f"a{i}", unit="voxel") for i in range(n)]
    )


# ----------------------------------------------------------------------
#   THE SUBCLASS IS SELECTED BY THE FLAG
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "base, tangent, data",
    [
        (Affine, AffineExponential, TANGENT),
        (Linear, LinearExponential, LINEAR_TANGENT),
        (Rotation, RotationExponential, SKEW),
        (Scaling, ScalingExponential, np.array([0.1, -0.2, 0.3])),
        (DisplacementField, StationaryVelocityField, np.zeros((4, 4, 2))),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_log_selects_the_tangent_subclass(
    base: type, tangent: type, data: np.ndarray
) -> None:
    built = base(data=data, log=True)
    assert type(built) is tangent and built.log is True
    direct = tangent(data=data)
    assert type(direct) is tangent and direct.log is True
    assert type(base(data=data)) is base and base(data=data).log is False
    # `.to(log=False)` builds the base class, not the subclass.
    assert type(built.to(log=False)) is base
    # A tangent subclass cannot be told it holds the map.
    with pytest.raises(ValueError):
        tangent(data=data, log=False)


def test_linear_log_builds_a_linear_tangent_not_a_rotation() -> None:
    assert type(Linear(data=LINEAR_TANGENT, log=True)) is LinearExponential
    assert type(Rotation(data=SKEW, log=True)) is RotationExponential


def test_steps_exist_only_on_a_velocity() -> None:
    velocity = DisplacementField(data=np.zeros((4, 4, 2)), log=True, steps=6)
    assert type(velocity) is StationaryVelocityField and velocity.steps == 6
    with pytest.raises(TypeError):
        DisplacementField(data=np.zeros((4, 4, 2)), steps=6)
    with pytest.raises(TypeError):
        Affine(data=TANGENT, log=True, steps=6)
    # `steps` is keyword-only: `data` stays the first positional argument.
    zeros = np.zeros((4, 4, 2))
    built = StationaryVelocityField(zeros)
    assert built.data is zeros
    assert built.steps is None
    # A zero velocity is its own flow: no squaring at all.
    assert built._compute_steps == 0


@pytest.mark.parametrize(
    "cls, data",
    [
        (Affine, TANGENT),
        (Linear, LINEAR_TANGENT),
        (Rotation, SKEW),
        (Scaling, np.ones(3)),
        (DisplacementField, np.zeros((4, 4, 2))),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_a_map_class_refuses_the_flag(cls: type, data: np.ndarray) -> None:
    # The flag selects the class, and an instance cannot change its own
    # class, so `log` is read-only: `t.to(log=True)` is the way.
    t = cls(data=data)
    with pytest.raises(AttributeError):
        t.log = True


def test_a_class_log_does_not_select_refuses_it() -> None:
    # An io subclass of `Affine` holds a map; it has no tangent subclass.
    with pytest.raises(TypeError):
        VoxelToLPS(data=TANGENT, log=True)


# ----------------------------------------------------------------------
#   VIEWS
# ----------------------------------------------------------------------


def test_affine_tangent_views() -> None:
    t = AffineExponential(data=TANGENT)
    np.testing.assert_allclose(t.matrix, _expm_affine(TANGENT), atol=1e-12)
    np.testing.assert_allclose(
        t.homogeneous_matrix,
        scipy.linalg.expm(_homogeneous_tangent(TANGENT)),
        atol=1e-12,
    )
    # A singular tangent: a pure translation is `[0, t]`.
    shift = np.zeros((3, 4))
    shift[:, -1] = [1.0, -2.0, 0.5]
    np.testing.assert_allclose(
        AffineExponential(data=shift).matrix,
        Translation(translation=shift[:, -1]).to(Affine).matrix,
        atol=1e-12,
    )


def test_linear_rotation_and_scaling_tangent_views() -> None:
    np.testing.assert_allclose(
        LinearExponential(data=LINEAR_TANGENT).matrix,
        scipy.linalg.expm(LINEAR_TANGENT),
        atol=1e-12,
    )
    rotation = RotationExponential(data=SKEW).matrix
    np.testing.assert_allclose(rotation, scipy.linalg.expm(SKEW), atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    s = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(ScalingExponential(data=s).scale, np.exp(s))


def test_a_tangent_identity_matrix_is_not_the_identity() -> None:
    # A tangent is never a matrix: the identity matrix, read as one, is
    # the scaling by e.
    t = LinearExponential(data=np.eye(3))
    np.testing.assert_allclose(t.matrix, np.e * np.eye(3), atol=1e-12)
    assert not is_identity(t, compute=True)


@pytest.mark.parametrize(
    "t, view, expected",
    [
        (AffineExponential(data=np.zeros((3, 4))), "matrix", np.eye(3, 4)),
        (LinearExponential(data=np.zeros((3, 3))), "matrix", np.eye(3)),
        (RotationExponential(data=np.zeros((3, 3))), "matrix", np.eye(3)),
        (ScalingExponential(data=np.zeros(3)), "scale", np.ones(3)),
        (
            StationaryVelocityField(data=np.zeros((8, 8, 2)), degree=3),
            "field",
            np.zeros((8, 8, 2)),
        ),
    ],
    ids=lambda x: getattr(type(x), "__name__", ""),
)
def test_a_zero_tangent_is_the_identity(
    t: object, view: str, expected: np.ndarray
) -> None:
    np.testing.assert_array_equal(getattr(t, view), expected)
    assert is_identity(t, compute=True)
    assert isinstance(t.simplify("numeric"), Identity)


@pytest.mark.parametrize(
    "cls",
    [
        AffineExponential,
        LinearExponential,
        RotationExponential,
        ScalingExponential,
        StationaryVelocityField,
    ],
)
def test_an_unset_tangent_is_the_identity(cls: type) -> None:
    t = cls()
    assert is_identity(t)
    assert isinstance(t.simplify(), Identity)
    # Unset is the identity under either value of the flag.
    assert t.to(log=False).data is None


def test_field_views_under_every_encoding() -> None:
    velocity = _linear_velocity()
    coefficients = value2coeff_field(velocity, degree=DEGREE, bound="nearest")
    displacement = DisplacementField(data=velocity, degree=DEGREE)
    # log=False, store="values": `data` is the displacement, as values.
    np.testing.assert_array_equal(displacement.field, velocity)
    # log=False, store="coefficients": `data` is the displacement's
    # coefficients.
    np.testing.assert_allclose(
        DisplacementField(
            data=coefficients, degree=DEGREE, store="coefficients"
        ).field,
        velocity,
        atol=1e-10,
    )
    # log=True, store="values": `data` is the velocity, as values, and the
    # view is the displacement of its flow.
    values = StationaryVelocityField(data=velocity, degree=DEGREE)
    np.testing.assert_allclose(
        values.field[INTERIOR], _linear_flow()[INTERIOR], atol=5e-3
    )
    # log=True, store="coefficients": `data` is the velocity's
    # coefficients (NiftyReg `-vel -cpp`); the same flow.
    coeffs = StationaryVelocityField(
        data=coefficients, degree=DEGREE, store="coefficients"
    )
    np.testing.assert_allclose(
        coeffs.field[INTERIOR], _linear_flow()[INTERIOR], atol=5e-3
    )


# ----------------------------------------------------------------------
#   EXPONENTIAL MAPS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("steps, atol", [(None, 5e-3), (10, 1e-3)])
def test_velocity_integrates_to_the_affine_flow(
    steps: object, atol: float
) -> None:
    # The velocity `x -> L x + l` is sampled without subtracting the grid,
    # and its flow is the exponential of the affine tangent. The default
    # number of steps bounds the first step by an eighth of a voxel, whose
    # first-order error more steps reduce.
    t = _velocity_field(steps=steps)
    np.testing.assert_allclose(
        t.field[INTERIOR], _linear_flow()[INTERIOR], atol=atol
    )


def test_the_refit_at_each_step_matches_the_values_path() -> None:
    values = _velocity_field(steps=8)
    coeffs = _velocity_field(steps=8).to(store="coefficients")
    assert coeffs.log and coeffs.store == "coefficients"
    np.testing.assert_allclose(
        coeffs.field[INTERIOR], values.field[INTERIOR], atol=1e-6
    )


def test_steps_default_rule_and_override() -> None:
    velocity = _linear_velocity()
    largest = np.linalg.norm(velocity, axis=-1).max()
    steps = _tangents._squaring_steps(velocity)
    assert largest / 2**steps <= 0.125 < largest / 2 ** (steps - 1)
    # No squaring at all is the first-order step `id + v`.
    np.testing.assert_array_equal(
        _velocity_field(steps=0).field, _linear_velocity()
    )
    with pytest.raises(ValueError, match="non-negative"):
        _velocity_field(steps=-1).field  # noqa: B018
    infinite = StationaryVelocityField(data=np.full(SHAPE + (2,), np.inf))
    with pytest.raises(DomainError, match="not finite"):
        infinite.field  # noqa: B018


def test_a_constant_velocity_is_a_translation() -> None:
    shift = np.array([3.3, -1.7])
    t = StationaryVelocityField(data=np.zeros(SHAPE + (2,)) + shift)
    np.testing.assert_allclose(t.field, np.zeros(SHAPE + (2,)) + shift)


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------

MATRIX = _expm_affine(TANGENT)


@pytest.mark.parametrize(
    "t",
    [
        Affine(matrix=MATRIX),
        Linear(matrix=MATRIX[:, :-1]),
        Rotation(matrix=scipy.linalg.expm(SKEW)),
        Scaling(scale=np.array([1.5, 0.5, 2.0])),
    ],
    ids=lambda t: type(t).__name__,
)
def test_matrices_round_trip_through_their_tangent(t: object) -> None:
    tangent = t.to(log=True)
    assert tangent.log and isinstance(tangent, type(t))
    back = tangent.to(log=False)
    assert type(back) is type(t)
    np.testing.assert_allclose(back.data, t.data, atol=1e-12)
    np.testing.assert_allclose(
        tangent.to(log=True).data, tangent.data, atol=1e-12
    )


def test_the_tangent_of_a_matrix_is_its_principal_logarithm() -> None:
    np.testing.assert_allclose(
        Affine(matrix=MATRIX).to(log=True).data, TANGENT, atol=1e-12
    )
    np.testing.assert_allclose(
        Rotation(matrix=scipy.linalg.expm(SKEW)).to(log=True).data,
        SKEW,
        atol=1e-12,
    )
    # `matrix=` is the map, encoded as the class says: as a tangent here.
    np.testing.assert_allclose(
        AffineExponential(matrix=MATRIX).data, TANGENT, atol=1e-12
    )
    np.testing.assert_allclose(
        AffineExponential(data=TANGENT).to(matrix=MATRIX).data,
        TANGENT,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "t",
    [
        # A reflection, a singular matrix and a negative scale factor:
        # each has an eigenvalue on the closed negative real axis, so
        # none has a principal logarithm.
        Affine(matrix=np.diag([1.0, -2.0, 1.0, 1.0])[:3]),
        Linear(matrix=np.diag([-1.0, 1.0, 1.0])),
        Linear(matrix=np.diag([1.0, 0.0, 1.0])),
        Scaling(scale=np.array([1.0, -2.0])),
    ],
    ids=lambda x: type(x).__name__,
)
def test_a_logarithm_outside_the_principal_domain_is_refused(
    t: object,
) -> None:
    with pytest.raises(DomainError):
        t.to(log=True)


def test_a_field_has_no_logarithm() -> None:
    with pytest.raises(NotImplementedError, match="logarithm"):
        DisplacementField(data=_linear_velocity()).to(log=True)
    with pytest.raises(NotImplementedError, match="logarithm"):
        DisplacementField(field=_linear_velocity(), log=True)
    # An unset field is the identity, whose velocity is zero ...
    assert type(DisplacementField().to(log=True)) is StationaryVelocityField
    # ... and a velocity given as `data=` is stored as it is.
    velocity = DisplacementField().to(log=True, data=_linear_velocity())
    np.testing.assert_array_equal(velocity.data, _linear_velocity())


def test_a_velocity_converts_to_its_displacement() -> None:
    velocity = _velocity_field()
    displacement = velocity.to(log=False)
    assert type(displacement) is DisplacementField
    assert (displacement.degree, displacement.store) == (
        DEGREE, "values"
    )
    np.testing.assert_array_equal(displacement.data, velocity.field)


def test_combined_flags_are_well_defined() -> None:
    values = _velocity_field()
    coeffs = values.to(store="coefficients")
    # `store` alone re-encodes the velocity: `log` stays.
    assert type(coeffs) is StationaryVelocityField
    np.testing.assert_allclose(
        coeff2value_field(coeffs.data, degree=DEGREE, bound="nearest"),
        values.data,
        atol=1e-10,
    )
    decoded = coeffs.to(store="values")
    assert decoded.log and decoded.store == "values"
    np.testing.assert_allclose(decoded.data, values.data, atol=1e-10)
    # `log=False` keeps `store`, and integrates.
    integrated = coeffs.to(log=False)
    assert type(integrated) is DisplacementField
    assert integrated.store == "coefficients"
    np.testing.assert_allclose(
        integrated.field[INTERIOR], coeffs.field[INTERIOR], atol=1e-10
    )
    # Both at once: the displacement, as values.
    both = coeffs.to(log=False, store="values")
    assert both.store == "values"
    np.testing.assert_array_equal(both.data, coeffs.field)


def test_a_velocity_converts_to_the_coordinates_of_its_flow() -> None:
    # The coordinates of a velocity are those of the integrated map, and
    # the result keeps only the spline encoding (#294).
    velocity = _velocity_field().to(store="coefficients")
    coordinates = velocity.to(CoordinatesField)
    assert type(coordinates) is CoordinatesField
    assert (coordinates.store, coordinates.degree) == (
        "coefficients", DEGREE
    )
    assert not hasattr(coordinates, "steps")
    np.testing.assert_allclose(
        coordinates.field[INTERIOR],
        (_grid() + velocity.field)[INTERIOR],
        atol=1e-8,
    )


def test_a_composition_reads_the_displacement_of_a_velocity() -> None:
    # A field composer samples the left operand's displacement, not the
    # velocity a velocity field stores.
    velocity = _velocity_field()
    plain = velocity.to(log=False)
    shift = DisplacementField(data=np.zeros(SHAPE + (2,)) + 0.5, degree=3)
    np.testing.assert_allclose(
        (shift @ velocity).compute().field,
        (shift @ plain).compute().field,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        (velocity @ shift).compute().field,
        (plain @ shift).compute().field,
        atol=1e-10,
    )


# ----------------------------------------------------------------------
#   TANGENT OPERATIONS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        AffineExponential(data=TANGENT),
        LinearExponential(data=LINEAR_TANGENT),
        RotationExponential(data=SKEW),
        ScalingExponential(data=np.array([0.1, -0.2, 0.3])),
        StationaryVelocityField(data=np.zeros((8, 8, 2)) + 0.3, degree=3),
    ],
    ids=lambda t: type(t).__name__,
)
def test_tangent_operations_are_exact(t: object) -> None:
    inverse = t.inverse()
    assert isinstance(inverse, Inverse) and isinstance(inverse, type(t))
    assert inverse.forward is t and inverse.log
    np.testing.assert_array_equal(inverse.data, -t.data)
    root = t.sqrt()
    assert type(root) is type(t)
    np.testing.assert_array_equal(root.data, t.data / 2)
    square = t.square()
    assert type(square) is type(t)
    np.testing.assert_array_equal(square.data, t.data * 2)
    # The inverse of a tangent cancels against it, and nothing is
    # integrated or exponentiated to find out.
    assert isinstance((inverse @ t).compute(), Identity)
    assert isinstance((t @ inverse).compute(), Identity)


def test_inverse_cancels_without_integrating() -> None:
    velocity = _velocity_field()
    computed = AssertionError("integrated")
    with mock.patch.object(
        _tangents, "_integrate_field", side_effect=computed
    ):
        inverse = velocity.inverse()
        assert isinstance((inverse @ velocity).compute(), Identity)


def test_the_inverse_of_a_velocity_integrates_its_negation() -> None:
    velocity = _velocity_field()
    inverse = velocity.inverse()
    negated = StationaryVelocityField(data=-_linear_velocity(), degree=DEGREE)
    np.testing.assert_array_equal(inverse.field, negated.field)
    materialized = inverse.compute()
    assert type(materialized) is StationaryVelocityField
    np.testing.assert_array_equal(materialized.data, -velocity.data)
    # Composed, the two are the identity up to the integration error.
    residual = (materialized @ velocity).compute(simplify=False)
    np.testing.assert_allclose(residual.field[INTERIOR], 0, atol=1e-2)


def test_the_square_root_of_a_velocity_drops_one_squaring() -> None:
    velocity = _velocity_field()
    steps = _tangents._squaring_steps(velocity.data)
    root = velocity.sqrt()
    assert _tangents._squaring_steps(root.data) == steps - 1
    # Squared, it is the field the velocity integrates to: the last
    # squaring of `exp(v)` is the square of `exp(v / 2)`.
    np.testing.assert_allclose(
        (root.to(log=False) @ root.to(log=False)).compute().field,
        velocity.field,
        atol=1e-12,
    )
    # An explicit number of steps follows: one fewer, one more.
    explicit = _velocity_field(steps=6)
    assert explicit.sqrt().steps == 5 and explicit.square().steps == 7


def test_the_tangent_of_an_inverse_is_lazy_and_exact() -> None:
    affine = Affine(matrix=MATRIX)
    tangent = affine.inverse().to(log=True)
    assert isinstance(tangent, Inverse) and tangent.log
    np.testing.assert_allclose(tangent.data, -TANGENT, atol=1e-12)
    back = tangent.to(log=False)
    assert isinstance(back, Inverse) and not back.log


# ----------------------------------------------------------------------
#   CONTAINERS
# ----------------------------------------------------------------------


def test_a_subspace_forwards_the_flag_to_its_inner_transformation() -> None:
    full = _system(4)
    sub = SubspaceTransformation(
        transformation=Affine(matrix=MATRIX),
        input_axes=[0, 1, 2],
        output_axes=[0, 1, 2],
        input=full,
        output=full,
    )
    tangent = sub.to(log=True)
    assert isinstance(tangent, SubspaceTransformation)
    assert type(tangent.transformation) is AffineExponential
    # Padding the map with the identity is padding the tangent with zero.
    np.testing.assert_allclose(
        tangent.to(Affine).matrix, sub.to(Affine).matrix, atol=1e-12
    )
    assert type(tangent.to(log=False).transformation) is Affine


def test_a_chain_composes_to_one_leaf_first() -> None:
    chain = Sequence(
        [Affine(matrix=MATRIX), Scaling(scale=np.array([1.2, 0.9, 1.1]))]
    )
    tangent = chain.to(log=True)
    assert type(tangent) is AffineExponential
    np.testing.assert_allclose(
        tangent.matrix, chain.compute().to(Affine).matrix, atol=1e-12
    )
    unreduced = Sequence(
        [
            Affine(matrix=np.array([[1.1, 0.0, 0.5], [0.0, 0.9, 0.0]])),
            StationaryVelocityField(data=np.zeros((4, 4, 2)) + 0.1),
            Translation(translation=np.ones(2)),
        ]
    )
    with pytest.raises(ConversionError, match="single transformation"):
        unreduced.to(log=True)


# ----------------------------------------------------------------------
#   SETTERS
# ----------------------------------------------------------------------


def test_assigning_data_refreshes_the_views() -> None:
    affine = AffineExponential(data=TANGENT)
    first = affine.matrix
    affine.data = 2 * TANGENT
    np.testing.assert_allclose(
        affine.matrix, _expm_affine(2 * TANGENT), atol=1e-12
    )
    assert affine.matrix is not first
    scaling = ScalingExponential(data=np.zeros(2))
    np.testing.assert_array_equal(scaling.scale, [1.0, 1.0])
    scaling.data = np.ones(2)
    np.testing.assert_allclose(scaling.scale, [np.e, np.e])
    velocity = _velocity_field()
    field = velocity.field
    velocity.data = np.zeros(SHAPE + (2,))
    np.testing.assert_array_equal(velocity.field, 0)
    assert velocity.field is not field


def test_assigning_steps_refreshes_the_views() -> None:
    # `field` integrates the velocity and caches the flow; assigning the
    # number of squaring steps clears it, so the next read integrates
    # again with the new count.
    velocity = _velocity_field(steps=0)
    np.testing.assert_array_equal(velocity.field, _linear_velocity())
    velocity.steps = 8
    np.testing.assert_allclose(
        velocity.field[INTERIOR], _linear_flow()[INTERIOR], atol=1e-3
    )


def test_assigning_data_forgets_a_cached_inverse() -> None:
    affine = AffineExponential(data=TANGENT)
    np.testing.assert_array_equal(affine.inverse().data, -TANGENT)
    affine.data = 2 * TANGENT
    np.testing.assert_array_equal(affine.inverse().data, -2 * TANGENT)


# ----------------------------------------------------------------------
#   KINDS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, node",
    [
        (AffineExponential(data=TANGENT), kinds.PositiveAffine),
        (LinearExponential(data=LINEAR_TANGENT), kinds.PositiveLinear),
        (RotationExponential(data=SKEW), kinds.SpecialOrthogonal),
        (
            ScalingExponential(data=np.array([0.1, -0.2])),
            kinds.PositiveDiagonal,
        ),
    ],
    ids=lambda x: getattr(type(x), "__name__", ""),
)
def test_the_exponential_of_a_real_tangent_is_positive(
    t: object, node: type
) -> None:
    assert is_kind(t, node)
    assert is_kind(t, node, compute=True)


# ----------------------------------------------------------------------
#   WRITERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "t",
    [
        AffineExponential(data=TANGENT),
        Affine(matrix=MATRIX).inverse(),
        Affine(matrix=MATRIX).sqrt(),
    ],
    ids=lambda t: type(t).__name__,
)
def test_a_format_copies_the_map_not_the_stored_data(t: object) -> None:
    # An affine format holds the map: a tangent is copied as its
    # exponential, and a lazy wrapper as the map it derives.
    from brainhops.io.transformations.niftyreg import NiftyRegAffine

    copied = NiftyRegAffine.from_any(t)
    assert not copied.log
    np.testing.assert_allclose(copied.matrix, t.matrix, atol=1e-12)


# ----------------------------------------------------------------------
#   REVIEW FOLLOW-UPS
# ----------------------------------------------------------------------


def test_an_unset_field_becomes_a_velocity_with_steps() -> None:
    velocity = DisplacementField(degree=DEGREE).to(log=True, steps=4)
    assert type(velocity) is StationaryVelocityField
    assert (velocity.steps, velocity.degree, velocity.data) == (4, 3, None)


def test_a_velocity_type_points_at_the_flag() -> None:
    with pytest.raises(ConversionError, match=r"to\(log=True\)"):
        DisplacementField(data=_linear_velocity()).to(StationaryVelocityField)
    velocity = _velocity_field()
    assert velocity.to(StationaryVelocityField) is velocity


@pytest.mark.parametrize(
    "t, value",
    [
        (StationaryVelocityField(data=np.zeros((4, 4, 2))), False),
        (AffineExponential(data=TANGENT), False),
        (ScalingExponential(data=np.zeros(2)), False),
        (Affine(data=MATRIX), True),
        (DisplacementField(data=np.zeros((4, 4, 2))), True),
    ],
    ids=lambda x: getattr(type(x), "__name__", str(x)),
)
def test_the_log_flag_cannot_change_in_place(t: object, value: bool) -> None:
    # Both directions refuse alike: the flag selects the class, and an
    # instance cannot change its own class. `t.to(log=...)` is the way.
    with pytest.raises(AttributeError):
        t.log = value


def test_the_resolved_steps() -> None:
    # `steps` is what the velocity was told, and `_compute_steps` is
    # what the integration uses: the rule fills in the one it was not.
    velocity = _velocity_field()
    assert velocity.steps is None
    assert velocity._compute_steps == _tangents._squaring_steps(
        velocity.data
    )
    declared = _velocity_field(steps=3)
    assert (declared.steps, declared._compute_steps) == (3, 3)
    assert StationaryVelocityField().steps is None
    assert StationaryVelocityField()._compute_steps is None


@pytest.mark.parametrize("cls", [Translation, Permutation])
def test_assigning_data_forgets_the_cached_inverse_and_root(
    cls: type,
) -> None:
    first, second = (
        (np.array([1.0, 2.0]), np.array([4.0, -2.0]))
        if cls is Translation
        else (np.array([1, 2, 0]), np.array([2, 0, 1]))
    )
    t = cls(data=first)
    t.inverse().data  # noqa: B018
    if cls is Translation:
        t.sqrt().data  # noqa: B018
    t.data = second
    expected = cls(data=second)
    np.testing.assert_array_equal(t.inverse().data, expected.inverse().data)
    if cls is Translation:
        np.testing.assert_array_equal(t.sqrt().data, second / 2)


@pytest.mark.parametrize(
    "cls, wrapper",
    [
        (
            Translation,
            Translation(translation=np.array([1.0, -2.0])).inverse(),
        ),
        (Permutation, Permutation(permutation=np.array([1, 2, 0])).inverse()),
        (AffineExponential, AffineExponential(data=TANGENT).inverse()),
        (
            DisplacementField,
            DisplacementField(
                data=_linear_velocity() * 0.01, degree=3
            ).inverse(),
        ),
        (StationaryVelocityField, _velocity_field(steps=3).inverse()),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_a_copy_of_a_lazy_wrapper_holds_its_map(
    cls: type, wrapper: object
) -> None:
    copied = cls.from_instance(wrapper)
    assert type(copied) is cls
    np.testing.assert_allclose(copied.data, wrapper.data, atol=1e-12)
    if cls is StationaryVelocityField:
        assert copied.log and copied.steps == 3


def test_a_copy_of_a_velocity_into_a_displacement_is_integrated() -> None:
    velocity = _velocity_field()
    copied = DisplacementField.from_instance(velocity)
    assert type(copied) is DisplacementField and not copied.log
    np.testing.assert_array_equal(copied.field, velocity.field)


# --- a chain between a change of coordinates --------------------------


def _framed(middle: object) -> Sequence:
    voxel_to_world = VoxelToLPS(matrix=np.diag([2.0, 3.0, 1.0])[:2])
    return Sequence([voxel_to_world.inverse(), middle, voxel_to_world])


def test_a_framed_velocity_converts_to_its_displacement() -> None:
    velocity = _velocity_field()
    chain = _framed(velocity)
    plain = chain.to(log=False)
    assert isinstance(plain, Sequence) and len(plain) == 3
    assert plain[0] is chain[0] and plain[2] is chain[2]
    assert type(plain[1]) is DisplacementField
    np.testing.assert_array_equal(plain[1].field, velocity.field)
    # And back: the displacement has no logarithm.
    with pytest.raises(NotImplementedError, match="logarithm"):
        plain.to(log=True)
    tangent = _framed(Affine(matrix=np.eye(2, 3) + 0.1)).to(log=True)
    assert type(tangent[1]) is AffineExponential


def test_a_chain_that_does_not_reduce_is_refused_before_computing() -> None:
    from brainhops.datamodel._transformations import sequence as _sequence

    chain = Sequence(
        [
            Translation(translation=np.ones(2)),
            _velocity_field(),
            DisplacementField(data=np.zeros(SHAPE + (2,))),
        ]
    )
    computed = AssertionError("composed")
    with mock.patch.object(_sequence, "compose", side_effect=computed):
        with mock.patch.object(_tangents, "compose", side_effect=computed):
            for log in (True, False):
                with pytest.raises(ConversionError, match="not composed"):
                    chain.to(log=log)
