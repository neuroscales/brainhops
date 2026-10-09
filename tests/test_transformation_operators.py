"""Tests for the unary operators of #47: square and principal root.

The exponential and logarithm are encodings, tested in
test_transformation_log.py.
"""

from unittest import mock

import numpy as np
import pytest

from brainhops.datamodel import kinds
from brainhops.datamodel._transformations import concrete as _concrete
from brainhops.datamodel.systems import (
    CoordinateSystem,
    SpaceAxis,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    UNARY_OPERATORS,
    Affine,
    AffineExponential,
    Bijection,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    ScalingExponential,
    Sequence,
    Sqrt,
    StationaryVelocityField,
    SubspaceTransformation,
    Transformation,
    Translation,
    is_kind,
)
from brainhops.errors import DomainError
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# This matrix has no eigenvalue on the negative real axis, so it has a
# principal root.
AFFINE = np.array(
    [[1.2, 0.1, -0.2, 3.0], [0.05, 0.9, 0.3, -1.0], [0.1, -0.1, 1.1, 2.0]]
)
LINEAR = AFFINE[:, :-1].copy()


def _rotation(angle: float, axis: int = 2) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    plane = [i for i in range(3) if i != axis]
    matrix = np.eye(3)
    matrix[np.ix_(plane, plane)] = [[c, -s], [s, c]]
    return matrix


ROTATION = _rotation(0.4, 0) @ _rotation(-0.7, 1) @ _rotation(1.1, 2)
CYCLE = np.array([1, 2, 0])  # a third of a turn


def _matrix(t: Transformation) -> np.ndarray:
    return np.asarray(t.to(Affine).matrix)


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    full = np.eye(n + 1)
    full[:n] = matrix
    return full


def _grid(shape: tuple) -> np.ndarray:
    axes = [np.arange(s, dtype=float) for s in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1)


def _system(n: int, name: str = "a") -> CoordinateSystem:
    return CoordinateSystem(
        axes=[SpaceAxis(name=f"{name}{i}", unit="voxel") for i in range(n)]
    )


def _transforms() -> list:
    return [
        Translation(translation=np.array([1.0, -2.0, 0.5])),
        Scaling(scale=np.array([2.0, 0.5, 1.5])),
        Permutation(permutation=CYCLE),
        Rotation(matrix=ROTATION),
        Linear(matrix=LINEAR),
        Affine(matrix=AFFINE),
    ]


def _rootable() -> list:
    # The principal root of a permutation is not a permutation, so
    # permutations are left out.
    return [t for t in _transforms() if not isinstance(t, Permutation)]


# ----------------------------------------------------------------------
#   API: LAZY, TYPED WRAPPERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, forward, wrapper, family",
    [
        ("sqrt", Translation(translation=np.ones(3)), Sqrt, Translation),
        ("sqrt", Scaling(scale=np.ones(3) * 2), Sqrt, Scaling),
        ("sqrt", Rotation(matrix=ROTATION), Sqrt, Rotation),
        ("sqrt", Linear(matrix=LINEAR), Sqrt, Linear),
        ("sqrt", Affine(matrix=AFFINE), Sqrt, Affine),
    ],
)
def test_operator_returns_typed_lazy_wrapper(
    method: str, forward: Transformation, wrapper: type, family: type
) -> None:
    result = getattr(forward, method)()
    assert isinstance(result, wrapper)
    # The compose engine and kind checks see through the wrapper.
    assert isinstance(result, family)
    assert result.forward is forward
    computed = getattr(forward, method)(compute=True)
    assert not isinstance(computed, wrapper)
    assert isinstance(computed, family)


def test_front_doors_build_typed_wrappers_and_refuse_the_rest() -> None:
    affine = Affine(matrix=AFFINE)
    assert type(Sqrt(affine)) is type(affine.sqrt())
    # A family without a typed wrapper is refused, not wrapped opaquely.
    with pytest.raises(TypeError):
        Sqrt(forward=Identity())
    with pytest.raises(TypeError):
        Sqrt(forward=DisplacementField(field=np.zeros((4, 4, 2))))


def test_wrapper_endpoints_are_the_forward_endpoints() -> None:
    vox = VoxelCoordinateSystem()
    t = Affine(matrix=AFFINE, input=vox, output=vox)
    result = t.sqrt()
    assert result.input == vox
    assert result.output == vox
    assert t.sqrt(compute=True).input == vox


def test_nothing_is_computed_until_the_parameter_is_read() -> None:
    affine = Affine(matrix=AFFINE)
    with mock.patch.object(
        _concrete, "affine_sqrtm", side_effect=AssertionError("computed")
    ) as compute:
        root = affine.sqrt()
        # None of these resolves the wrapper.
        assert is_kind(root, kinds.Affine)
        assert is_kind(root, kinds.InvertibleAffine)
        assert not is_kind(root, kinds.Translation)
        assert root.simplify() is root
        assert root.simplify("numeric") is root
        assert isinstance(root.to(input=VoxelCoordinateSystem()), Sqrt)
        assert root.compute(mode="translation") is root
        assert root.square() is affine
        compute.assert_not_called()


def test_operators_are_listed_by_name() -> None:
    assert set(UNARY_OPERATORS) == {"inverse", "square", "sqrt"}
    affine = Affine(matrix=AFFINE)
    for name, operator in UNARY_OPERATORS.items():
        result = operator(affine)
        expected = getattr(affine, name)()
        assert type(result) is type(expected)
    # A subclass override is honoured.
    identity = Identity()
    assert UNARY_OPERATORS["sqrt"](identity) is identity
    with pytest.raises(TypeError):
        UNARY_OPERATORS["cube"] = None  # type: ignore[index]


# ----------------------------------------------------------------------
#   ALGEBRAIC IDENTITIES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("t", _rootable(), ids=lambda t: type(t).__name__)
def test_sqrt_squared_is_the_transform(t: Transformation) -> None:
    root = t.sqrt(compute=True)
    np.testing.assert_allclose(
        _matrix((root @ root).compute()), _matrix(t), atol=1e-12
    )
    assert t.sqrt().square() is t


def test_a_permutation_has_no_root() -> None:
    # The principal root of this cycle, a sixth of a turn, is not a
    # permutation of the axes.
    p = Permutation(permutation=CYCLE)
    with pytest.raises(NotImplementedError):
        p.sqrt()
    root = p.to(Linear).sqrt(compute=True)
    np.testing.assert_allclose(
        _matrix((root @ root).compute()), _matrix(p), atol=1e-12
    )
    identity = Permutation()
    assert identity.sqrt() is identity


@pytest.mark.parametrize("t", _transforms(), ids=lambda t: type(t).__name__)
def test_square_is_the_composition_with_itself(t: Transformation) -> None:
    square = t.square()
    assert isinstance(square, Sequence)
    assert list(square) == [t, t]
    expected = (t @ t).compute()
    np.testing.assert_array_equal(_matrix(square.compute()), _matrix(expected))
    np.testing.assert_allclose(
        _homogeneous(_matrix(t.square(compute=True))),
        _homogeneous(_matrix(t)) @ _homogeneous(_matrix(t)),
        atol=1e-12,
    )


def test_sqrt_is_the_principal_root() -> None:
    # Of the two planar roots, the principal one is the half angle.
    root = Rotation(matrix=_rotation(2.0)).sqrt(compute=True)
    np.testing.assert_allclose(root.matrix, _rotation(1.0), atol=1e-12)
    half = Translation(translation=np.array([2.0, -4.0])).sqrt(compute=True)
    np.testing.assert_array_equal(half.translation, [1.0, -2.0])
    np.testing.assert_allclose(
        Scaling(scale=np.array([4.0, 0.25])).sqrt(compute=True).scale,
        [2.0, 0.5],
    )


def test_inverse_commutes_with_sqrt() -> None:
    affine = Affine(matrix=AFFINE)
    a = affine.inverse().sqrt(compute=True)
    b = affine.sqrt().inverse(compute=True)
    np.testing.assert_allclose(_matrix(a), _matrix(b), atol=1e-12)


# ----------------------------------------------------------------------
#   THE DISPLACEMENT READING OF EXP AND LOG
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["square", "sqrt"])
def test_identity_is_fixed(method: str) -> None:
    identity = Identity()
    assert getattr(identity, method)() is identity
    grid = CartesianField(shape=(3, 4))
    assert getattr(grid, method)() is grid
    for unset in (Affine(), Linear(), Scaling(), DisplacementField()):
        if method != "square":
            assert getattr(unset, method)() is unset


@pytest.mark.parametrize("method", ["sqrt"])
def test_operators_commute_with_a_change_of_coordinates(method: str) -> None:
    # The displacement reading does not depend on coordinates.
    change = Affine(
        matrix=np.array(
            [[0.0, 2.0, 0.0, 5.0], [1.5, 0.0, 0.0, -3.0], [0, 0.0, 0.5, 1.0]]
        )
    )
    affine = Affine(matrix=AFFINE)
    chain = Sequence([change, affine, change.inverse()])
    inside = getattr(chain, method)()
    assert isinstance(inside, Sequence)
    assert inside[0] is change and isinstance(inside[1], Sqrt)
    outside = getattr(chain.compute(), method)(compute=True)
    np.testing.assert_allclose(
        _matrix(inside.compute()), _matrix(outside), atol=1e-10
    )


@pytest.mark.parametrize("method", ["sqrt"])
def test_operators_act_on_a_subspace_inner(method: str) -> None:
    # The pass-through axes are fixed by every operator.
    full = _system(4)
    sub = SubspaceTransformation(
        transformation=Affine(matrix=AFFINE),
        input_axes=[0, 1, 2],
        output_axes=[0, 1, 2],
        input=full,
        output=full,
    )
    result = getattr(sub, method)()
    assert isinstance(result, SubspaceTransformation)
    assert isinstance(result.transformation, Sqrt)
    embedded = getattr(sub.to(Affine), method)(compute=True)
    np.testing.assert_allclose(_matrix(result), _matrix(embedded), atol=1e-12)


# ----------------------------------------------------------------------
#   CLOSURE OF KINDS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, t, family",
    [
        ("sqrt", Translation(translation=np.ones(3)), Translation),
        ("sqrt", Scaling(scale=np.array([2.0, 3.0, 4.0])), Scaling),
        ("sqrt", Rotation(matrix=ROTATION), Rotation),
        ("sqrt", Linear(matrix=LINEAR), Linear),
        ("sqrt", Affine(matrix=AFFINE), Affine),
        ("square", Translation(translation=np.ones(3)), Translation),
        ("square", Scaling(scale=np.array([2.0, -3.0, 4.0])), Scaling),
        ("square", Permutation(permutation=np.array([1, 0, 2])), Permutation),
        ("square", Rotation(matrix=ROTATION), Rotation),
        ("square", Linear(matrix=LINEAR), Linear),
        ("square", Affine(matrix=AFFINE), Affine),
    ],
)
def test_operator_result_kind(
    method: str, t: Transformation, family: type
) -> None:
    result = getattr(t, method)(compute=True)
    assert type(result) is family


def test_sqrt_of_a_rotation_is_a_rotation() -> None:
    root = Rotation(matrix=ROTATION).sqrt(compute=True)
    assert isinstance(root, Rotation)
    np.testing.assert_allclose(
        root.matrix.T @ root.matrix, np.eye(3), atol=1e-12
    )
    assert np.isclose(np.linalg.det(root.matrix), 1.0)


@pytest.mark.parametrize(
    "t, node, expected",
    [
        (Rotation(matrix=ROTATION).sqrt(), kinds.SpecialOrthogonal, True),
        (Scaling(scale=np.ones(3) * 2).sqrt(), kinds.Diagonal, True),
    ],
)
def test_kind_membership_is_read_from_the_forward(
    t: Transformation, node: type, expected: bool
) -> None:
    for compute in (False, True):
        assert is_kind(t, node, compute) is expected


def test_a_simplified_forward_rebuilds_a_cheaper_wrapper() -> None:
    # The root is rebuilt as the root of the simplified scaling.
    root = Affine(matrix=np.diag([4.0, 9.0, 1.0, 1.0])[:3]).sqrt()
    simplified = root.simplify("numeric")
    assert isinstance(simplified, Sqrt) and isinstance(simplified, Scaling)
    np.testing.assert_allclose(simplified.scale, [2.0, 3.0, 1.0])
    assert isinstance(
        Affine(matrix=np.eye(3, 4)).sqrt().simplify(True), Identity
    )


# ----------------------------------------------------------------------
#   REFUSALS
# ----------------------------------------------------------------------


def test_domain_error_is_a_value_error() -> None:
    assert issubclass(DomainError, ValueError)


@pytest.mark.parametrize(
    "t, reason",
    [
        (Rotation(matrix=_rotation(np.pi)), "negative real axis"),
        (Linear(matrix=np.diag([-1.0, 1.0, 1.0])), "negative real axis"),
        (Affine(matrix=np.diag([1.0, -2.0, 1.0, 1.0])[:3]), "negative"),
        (Linear(matrix=np.diag([1.0, 0.0, 1.0])), "singular"),
        (Scaling(scale=np.array([1.0, -2.0])), "positive"),
        (Scaling(scale=np.array([1.0, 0.0])), "positive"),
    ],
    ids=lambda x: getattr(type(x), "__name__", x),
)
def test_sqrt_refuses_outside_the_principal_domain(
    t: Transformation, reason: str
) -> None:
    # The refusal is raised when the lazy result is resolved.
    lazy = t.sqrt()
    with pytest.raises(DomainError, match=reason):
        lazy.compute()
    with pytest.raises(DomainError):
        t.sqrt(compute=True)


@pytest.mark.parametrize("method", ["square", "sqrt"])
def test_operators_refuse_a_map_between_different_spaces(method: str) -> None:
    voxel_to_world = VoxelToLPS(matrix=np.eye(3, 4))
    with pytest.raises(DomainError, match="maps a space to itself"):
        getattr(voxel_to_world, method)()
    rectangular = Linear(matrix=np.ones((2, 3)))
    with pytest.raises(DomainError, match="3 axes to 2 axes"):
        getattr(rectangular, method)()


def test_operators_refuse_what_they_do_not_compute() -> None:
    field = DisplacementField(field=np.zeros((4, 4, 2)) + 0.1)
    with pytest.raises(NotImplementedError, match="velocity"):
        field.sqrt()
    coords = CoordinatesField(field=_grid((4, 4)))
    with pytest.raises(NotImplementedError, match="coordinates field"):
        coords.sqrt()
    reindex = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0, 3.0])),
        input_axes=[0, 1],
        output_axes=[1, 0],
    )
    with pytest.raises(NotImplementedError, match="reindexes"):
        reindex.sqrt()
    with pytest.raises(NotImplementedError):
        Projection(dropped=[0]).sqrt()


def test_a_chain_that_does_not_reduce_is_refused() -> None:
    field = DisplacementField(field=np.zeros((4, 4, 2)) + 0.1)
    affine = Affine(matrix=np.array([[1.1, 0.0, 0.5], [0.0, 0.9, 0.0]]))
    with pytest.raises(NotImplementedError, match="single transformation"):
        Sequence([affine, field]).sqrt()


def test_a_simplifier_never_raises_on_an_undefined_operator() -> None:
    half_turn = Linear(matrix=_rotation(np.pi))
    root = half_turn.sqrt()
    assert root.simplify("numeric") is not None
    with pytest.raises(DomainError):
        root.compute()


# ----------------------------------------------------------------------
#   CHAINS AND WRAPPERS
# ----------------------------------------------------------------------


def test_a_chain_is_simplified_first() -> None:
    affine = Affine(matrix=AFFINE)
    other = Translation(translation=np.ones(3))
    chain = Sequence([other, other.inverse(), affine])
    root = chain.sqrt()
    assert isinstance(root, Sqrt) and root.forward is affine


def test_a_chain_that_composes_is_composed_first() -> None:
    a = Affine(matrix=AFFINE)
    b = Scaling(scale=np.array([1.5, 1.0, 0.5]))
    result = Sequence([a, b]).sqrt(compute=True)
    expected = (b @ a).compute().sqrt(compute=True)
    np.testing.assert_allclose(_matrix(result), _matrix(expected))


VOX2LPS = np.array(
    [[-2.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, -5.0], [0.0, 0.0, 4.0, 1.0]]
)


def test_a_change_of_coordinates_is_recognized_exactly() -> None:
    # The ends cancel only when their product is exactly the identity...
    exact = np.linalg.inv(_homogeneous(VOX2LPS))[:3]
    np.testing.assert_array_equal(
        _homogeneous(exact) @ _homogeneous(VOX2LPS), np.eye(4)
    )
    inner = Affine(matrix=AFFINE)
    chain = Sequence(
        [LPSToVoxel(matrix=exact), inner, VoxelToLPS(matrix=VOX2LPS)]
    )
    root = chain.sqrt()
    assert isinstance(root[1], Sqrt) and root[1].forward is inner
    # ...not up to rounding, in which case the chain is composed.
    rounded = exact.copy()
    rounded[0, -1] += 1e-12
    chain = Sequence(
        [LPSToVoxel(matrix=rounded), inner, VoxelToLPS(matrix=VOX2LPS)]
    )
    root = chain.sqrt()
    assert isinstance(root, Sqrt) and root.forward is not inner


def _velocity_3d() -> StationaryVelocityField:
    return StationaryVelocityField(data=np.zeros((4, 4, 4, 3)) + 0.2)


def test_a_world_space_velocity_halves_inside_its_frame() -> None:
    # The root halves the velocity inside the change of coordinates.
    voxel_to_world = VoxelToLPS(matrix=VOX2LPS)
    velocity = _velocity_3d()
    chain = Sequence([voxel_to_world.inverse(), velocity, voxel_to_world])
    root = chain.sqrt()
    assert isinstance(root, Sequence) and len(root) == 3
    assert root[0] is chain[0] and root[2] is chain[2]
    assert isinstance(root[1], StationaryVelocityField)
    np.testing.assert_array_equal(root[1].data, velocity.data / 2)
    # Rounded ends are not a change of coordinates.
    rounded = np.linalg.inv(_homogeneous(VOX2LPS))[:3]
    rounded[0, -1] += 1e-12
    chain = Sequence(
        [LPSToVoxel(matrix=rounded), velocity, VoxelToLPS(matrix=VOX2LPS)]
    )
    with pytest.raises(NotImplementedError, match="single transformation"):
        chain.sqrt()


def test_bijection_keeps_both_directions_of_its_root() -> None:
    forward = Affine(matrix=AFFINE)
    bijection = Bijection(forward=forward, backward=forward.inverse())
    root = bijection.sqrt()
    assert isinstance(root, Bijection)
    np.testing.assert_allclose(
        _matrix(root.backward.compute()),
        _matrix(root.forward.inverse(compute=True)),
        atol=1e-12,
    )


def test_operators_compose_lazily_in_a_sequence() -> None:
    a = Affine(matrix=AFFINE)
    b = Rotation(matrix=ROTATION)
    expression = a.inverse() @ b.sqrt() @ a.sqrt()
    expected = (
        a.inverse(compute=True) @ b.sqrt(compute=True) @ a.sqrt(compute=True)
    ).compute()
    np.testing.assert_allclose(
        _matrix(expression.compute()), _matrix(expected), atol=1e-12
    )
    partial = Sequence([b.sqrt(), b.sqrt()]).compute(mode="translation")
    assert all(isinstance(t, Sqrt) for t in partial)


# ----------------------------------------------------------------------
#   FIELDS: SCALING AND SQUARING
# ----------------------------------------------------------------------

SHAPE = (32, 32)
CENTER = (np.array(SHAPE) - 1) / 2
INTERIOR = (slice(10, 22), slice(10, 22))
GENERATOR = np.array([[0.05, -0.2], [0.2, 0.03]])


def _linear_velocity(**kwargs) -> DisplacementField:
    grid = _grid(SHAPE)
    return DisplacementField(field=(grid - CENTER) @ GENERATOR.T, **kwargs)


@pytest.mark.parametrize("operator", ["sqrt", "square"])
@pytest.mark.parametrize("cls", [AffineExponential, ScalingExponential])
def test_the_root_and_square_of_an_unset_tangent_are_unset(
    cls: type, operator: str
) -> None:
    # Regression test for #383: unset data stands for the identity, whose
    # square root and square are the identity itself.
    tangent = cls()
    result = getattr(tangent, operator)()
    assert type(result) is cls
    assert result.data is None
