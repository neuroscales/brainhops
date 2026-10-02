"""Tests for the unary operators of a transformation (issue #47).

These cover the square, the principal square root, the exponential and the
principal logarithm: their algebraic identities, the kind each one keeps,
their laziness, the cases they refuse, the displacement reading of the
exponential and logarithm (and the consistency it buys across
representations), and the scaling-and-squaring exponential of a field
against analytic flows.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.linalg

from brainhops.datamodel import kinds
from brainhops.datamodel._transformations import operators as _ops
from brainhops.datamodel.systems import (
    CoordinateSystem,
    SpaceAxis,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    UNARY_OPERATORS,
    Affine,
    Bijection,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    DomainError,
    Exp,
    Identity,
    Inverse,
    Linear,
    Log,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    Sqrt,
    SubspaceTransformation,
    Transformation,
    Translation,
    is_kind,
)
from brainhops.io.transformations.base.affines import LPSToVoxel, VoxelToLPS

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# A well-conditioned affine whose linear part has no eigenvalue on the
# negative real axis, so that it has a principal square root and logarithm.
AFFINE = np.array(
    [[1.2, 0.1, -0.2, 3.0], [0.05, 0.9, 0.3, -1.0], [0.1, -0.1, 1.1, 2.0]]
)
LINEAR = AFFINE[:, :-1].copy()


def _rotation(angle: float, axis: int = 2) -> np.ndarray:
    # A 3D rotation by `angle` about one axis.
    c, s = np.cos(angle), np.sin(angle)
    plane = [i for i in range(3) if i != axis]
    matrix = np.eye(3)
    matrix[np.ix_(plane, plane)] = [[c, -s], [s, c]]
    return matrix


ROTATION = _rotation(0.4, 0) @ _rotation(-0.7, 1) @ _rotation(1.1, 2)
CYCLE = np.array([1, 2, 0])  # a 3-cycle: a rotation by a third of a turn


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
        axes=[SpaceAxis(name=f"{name}{i}", unit="sample") for i in range(n)]
    )


def _transforms() -> list:
    # One parameterized transformation of every concrete matrix kind.
    return [
        Translation(translation=np.array([1.0, -2.0, 0.5])),
        Scaling(scale=np.array([2.0, 0.5, 1.5])),
        Permutation(permutation=CYCLE),
        Rotation(matrix=ROTATION),
        Linear(matrix=LINEAR),
        Affine(matrix=AFFINE),
    ]


# ----------------------------------------------------------------------
#   API: LAZY, TYPED WRAPPERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, forward, wrapper, family",
    [
        ("sqrt", Translation(translation=np.ones(3)), Sqrt, Translation),
        ("sqrt", Scaling(scale=np.ones(3) * 2), Sqrt, Scaling),
        ("sqrt", Permutation(permutation=CYCLE), Sqrt, Linear),
        ("sqrt", Rotation(matrix=ROTATION), Sqrt, Rotation),
        ("sqrt", Linear(matrix=LINEAR), Sqrt, Linear),
        ("sqrt", Affine(matrix=AFFINE), Sqrt, Affine),
        ("exp", Scaling(scale=np.ones(3) * 2), Exp, Scaling),
        ("exp", Permutation(permutation=CYCLE), Exp, Linear),
        ("exp", Rotation(matrix=ROTATION), Exp, Linear),
        ("exp", Linear(matrix=LINEAR), Exp, Linear),
        ("exp", Affine(matrix=AFFINE), Exp, Affine),
        (
            "exp",
            DisplacementField(field=np.zeros((4, 4, 2)) + 0.1),
            Exp,
            DisplacementField,
        ),
        ("log", Scaling(scale=np.ones(3) * 2), Log, Scaling),
        ("log", Permutation(permutation=CYCLE), Log, Linear),
        ("log", Rotation(matrix=ROTATION), Log, Linear),
        ("log", Linear(matrix=LINEAR), Log, Linear),
        ("log", Affine(matrix=AFFINE), Log, Affine),
    ],
)
def test_operator_returns_typed_lazy_wrapper(
    method: str, forward: Transformation, wrapper: type, family: type
) -> None:
    result = getattr(forward, method)()
    assert isinstance(result, wrapper)
    # It stays an instance of the family of its result, so the compose
    # engine and the kind checks treat it like any transform of that family.
    assert isinstance(result, family)
    assert result.forward is forward
    # Computing it gives a plain instance of that family.
    computed = getattr(forward, method)(compute=True)
    assert not isinstance(computed, wrapper)
    assert isinstance(computed, family)


def test_front_doors_build_typed_wrappers_and_refuse_the_rest() -> None:
    affine = Affine(matrix=AFFINE)
    assert type(Sqrt(affine)) is type(affine.sqrt())
    assert type(Exp(forward=affine)) is type(affine.exp())
    # A family with no typed wrapper is refused when it is constructed,
    # rather than building an opaque wrapper the compose engine cannot see
    # through. The methods are the general entry point.
    with pytest.raises(TypeError):
        Sqrt(forward=Identity())
    with pytest.raises(TypeError):
        Exp(forward=Translation(translation=np.ones(2)))


def test_wrapper_endpoints_are_the_forward_endpoints() -> None:
    vox = VoxelCoordinateSystem()
    t = Affine(matrix=AFFINE, input=vox, output=vox)
    for method in ("sqrt", "exp", "log"):
        result = getattr(t, method)()
        assert result.input == vox
        assert result.output == vox
        assert getattr(t, method)(compute=True).input == vox


def test_nothing_is_computed_until_the_parameter_is_read() -> None:
    affine = Affine(matrix=AFFINE)
    with mock.patch.object(
        _ops, "_affine_sqrt", side_effect=AssertionError("computed")
    ) as compute:
        root = affine.sqrt()
        # Neither a kind check, a simplification, an endpoint edit, nor a
        # compute under a mode that does not admit it resolves the wrapper.
        assert is_kind(root, kinds.Affine)
        assert is_kind(root, kinds.InvertibleAffine)
        assert not is_kind(root, kinds.Translation)
        assert root.simplify() is root
        assert root.simplify("numeric") is root
        assert isinstance(root.to(input=VoxelCoordinateSystem()), Sqrt)
        assert root.compute(mode="translation") is root
        assert root.square() is affine
        compute.assert_not_called()


def test_the_parameter_is_cached_across_rebuilds() -> None:
    affine = Affine(matrix=AFFINE)
    with mock.patch.object(
        _ops, "_affine_exp", wraps=_ops._affine_exp
    ) as compute:
        result = affine.exp()
        first = result.matrix
        rebuilt = result.to(input=VoxelCoordinateSystem())
        assert rebuilt.matrix is first
        assert affine.exp().matrix is first
        assert compute.call_count == 1


def test_lazy_operator_cancels_with_its_lazy_inverse() -> None:
    affine = Affine(matrix=AFFINE)
    with mock.patch.object(
        _ops, "_affine_log", side_effect=AssertionError("computed")
    ):
        log = affine.log()
        assert isinstance((log.inverse() @ log).compute(), Identity)


def test_operators_are_listed_by_name() -> None:
    assert set(UNARY_OPERATORS) == {"inverse", "square", "sqrt", "exp", "log"}
    affine = Affine(matrix=AFFINE)
    for name, operator in UNARY_OPERATORS.items():
        result = operator(affine)
        expected = getattr(affine, name)()
        assert type(result) is type(expected)
    # A subclass override is honoured, not bypassed.
    identity = Identity()
    assert UNARY_OPERATORS["sqrt"](identity) is identity
    with pytest.raises(TypeError):
        UNARY_OPERATORS["cube"] = None  # type: ignore[index]


# ----------------------------------------------------------------------
#   ALGEBRAIC IDENTITIES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("t", _transforms(), ids=lambda t: type(t).__name__)
def test_sqrt_squared_is_the_transform(t: Transformation) -> None:
    root = t.sqrt(compute=True)
    np.testing.assert_allclose(
        _matrix((root @ root).compute()), _matrix(t), atol=1e-12
    )
    # And the lazy square of a lazy root is the transform itself.
    assert t.sqrt().square() is t


@pytest.mark.parametrize("t", _transforms(), ids=lambda t: type(t).__name__)
def test_exp_of_log_is_the_transform(t: Transformation) -> None:
    log = t.log(compute=True)
    np.testing.assert_allclose(
        _matrix(log.exp(compute=True)), _matrix(t), atol=1e-12
    )
    # And the lazy exponential of a lazy logarithm is the transform itself.
    lazy = t.log()
    assert lazy is t or lazy.exp() is t


@pytest.mark.parametrize(
    "t",
    [
        Translation(translation=np.array([1.0, -2.0, 0.5])),
        Scaling(scale=np.array([1.2, 0.8, 1.1])),
        Linear(matrix=np.eye(3) + 0.1 * LINEAR),
        Affine(matrix=np.eye(3, 4) + 0.1 * AFFINE),
    ],
    ids=lambda t: type(t).__name__,
)
def test_log_of_exp_is_the_transform_near_the_identity(
    t: Transformation,
) -> None:
    # Near the identity, the exponential lies in the principal domain.
    roundtrip = t.exp(compute=True).log(compute=True)
    np.testing.assert_allclose(_matrix(roundtrip), _matrix(t), atol=1e-12)


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
    # A rotation by an angle has two square roots in the plane (by half the
    # angle, and by half the angle plus a half turn); the principal one is
    # the half rotation.
    root = Rotation(matrix=_rotation(2.0)).sqrt(compute=True)
    np.testing.assert_allclose(root.matrix, _rotation(1.0), atol=1e-12)
    # A translation's root is half of it, a positive scaling's its root.
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


def test_inverse_of_affine_exp_is_exp_of_the_negated_velocity() -> None:
    # The exponential does not commute with the inverse: its argument is
    # read as a velocity, and the inverse of the flow of `v` is the flow of
    # `-v`, not of the velocity of the inverse map. For `x -> M x + t`, the
    # negated velocity `x -> -(M - I) x - t` is the affine `[2I - M, -t]`.
    negated = Affine(matrix=2 * np.eye(3, 4) - AFFINE)
    np.testing.assert_allclose(
        _matrix(negated.exp(compute=True)),
        _matrix(Affine(matrix=AFFINE).exp().inverse(compute=True)),
        atol=1e-12,
    )


# ----------------------------------------------------------------------
#   THE DISPLACEMENT READING OF EXP AND LOG
# ----------------------------------------------------------------------


def test_exp_of_an_affine_is_the_flow_of_its_displacement() -> None:
    # `x -> M x + t` is read as the velocity `x -> (M - I) x + t`.
    generator = np.zeros((4, 4))
    generator[:3] = AFFINE
    generator[:3, :3] -= np.eye(3)
    expected = scipy.linalg.expm(generator)[:3]
    result = Affine(matrix=AFFINE).exp(compute=True)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12)
    linear = Linear(matrix=LINEAR).exp(compute=True)
    np.testing.assert_allclose(
        linear.matrix, scipy.linalg.expm(LINEAR - np.eye(3)), atol=1e-12
    )
    scale = np.array([2.0, 0.5, 1.0])
    scaling = Scaling(scale=scale).exp(compute=True)
    np.testing.assert_allclose(scaling.scale, np.exp(scale - 1))


def test_log_of_an_affine_is_the_identity_plus_its_velocity() -> None:
    velocity = scipy.linalg.logm(_homogeneous(AFFINE))[:3]
    expected = velocity + np.eye(3, 4)
    result = Affine(matrix=AFFINE).log(compute=True)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12)
    scale = np.array([2.0, 0.5, 1.0])
    np.testing.assert_allclose(
        Scaling(scale=scale).log(compute=True).scale, 1 + np.log(scale)
    )


@pytest.mark.parametrize("method", ["square", "sqrt", "exp", "log"])
def test_identity_is_fixed(method: str) -> None:
    identity = Identity()
    assert getattr(identity, method)() is identity
    grid = CartesianField(shape=(3, 4))
    assert getattr(grid, method)() is grid
    for unset in (Affine(), Linear(), Scaling(), DisplacementField()):
        if method != "square":
            assert getattr(unset, method)() is unset


@pytest.mark.parametrize("method", ["exp", "log"])
def test_a_translation_is_its_own_exp_and_log(method: str) -> None:
    t = Translation(translation=np.array([1.0, -2.0, 0.5]))
    assert getattr(t, method)() is t
    # The same holds for an affine that holds a translation.
    affine = t.to(Affine)
    np.testing.assert_allclose(
        getattr(affine, method)(compute=True).matrix, affine.matrix
    )


@pytest.mark.parametrize("method", ["sqrt", "exp", "log"])
def test_operators_commute_with_a_change_of_coordinates(method: str) -> None:
    # The displacement reading does not depend on the coordinates, so
    # computing a conjugated chain first gives the same answer as applying
    # the operator inside the change of coordinates.
    change = Affine(
        matrix=np.array(
            [[0.0, 2.0, 0.0, 5.0], [1.5, 0.0, 0.0, -3.0], [0, 0.0, 0.5, 1.0]]
        )
    )
    affine = Affine(matrix=AFFINE)
    chain = Sequence([change, affine, change.inverse()])
    inside = getattr(chain, method)()
    assert isinstance(inside, Sequence)
    assert inside[0] is change and isinstance(inside[1], (Sqrt, Exp, Log))
    outside = getattr(chain.compute(), method)(compute=True)
    np.testing.assert_allclose(
        _matrix(inside.compute()), _matrix(outside), atol=1e-10
    )


@pytest.mark.parametrize("method", ["sqrt", "exp", "log"])
def test_operators_act_on_a_subspace_inner(method: str) -> None:
    # The pass-through axes are the identity, which every operator fixes,
    # so the operator of an embedding is the embedding of the operator.
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
    assert isinstance(result.transformation, (Sqrt, Exp, Log))
    embedded = getattr(sub.to(Affine), method)(compute=True)
    np.testing.assert_allclose(_matrix(result), _matrix(embedded), atol=1e-12)


def test_exp_of_an_affine_matches_exp_of_its_field() -> None:
    # An affine and the field that samples it have the same exponential.
    shape = (24, 24)
    linear = np.array([[1.04, -0.1], [0.12, 0.97]])
    center = (np.array(shape) - 1) / 2
    offset = center - linear @ center + np.array([0.3, -0.2])
    affine = Affine(matrix=np.concatenate([linear, offset[:, None]], 1))
    grid = _grid(shape)
    field = DisplacementField(field=grid @ linear.T + offset - grid)
    expected = grid @ _matrix(affine.exp(compute=True))[:, :-1].T
    expected += _matrix(affine.exp(compute=True))[:, -1] - grid
    interior = (slice(8, 16), slice(8, 16))
    np.testing.assert_allclose(
        field.exp().field[interior], expected[interior], atol=1e-2
    )


# ----------------------------------------------------------------------
#   CLOSURE OF KINDS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, t, family",
    [
        ("sqrt", Translation(translation=np.ones(3)), Translation),
        ("sqrt", Scaling(scale=np.array([2.0, 3.0, 4.0])), Scaling),
        ("sqrt", Rotation(matrix=ROTATION), Rotation),
        ("sqrt", Permutation(permutation=CYCLE), Linear),
        ("sqrt", Linear(matrix=LINEAR), Linear),
        ("sqrt", Affine(matrix=AFFINE), Affine),
        ("exp", Scaling(scale=np.array([2.0, -3.0, 4.0])), Scaling),
        ("exp", Rotation(matrix=ROTATION), Linear),
        ("exp", Linear(matrix=LINEAR), Linear),
        ("exp", Affine(matrix=AFFINE), Affine),
        ("log", Scaling(scale=np.array([2.0, 3.0, 4.0])), Scaling),
        ("log", Rotation(matrix=ROTATION), Linear),
        ("log", Linear(matrix=LINEAR), Linear),
        ("log", Affine(matrix=AFFINE), Affine),
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
        (Permutation(permutation=CYCLE).sqrt(), kinds.Permutation, False),
        (Permutation(permutation=CYCLE).sqrt(), kinds.Linear, True),
        (Scaling(scale=np.ones(3) * 2).exp(), kinds.PositiveDiagonal, True),
        (Rotation(matrix=ROTATION).exp(), kinds.PositiveLinear, True),
        (Rotation(matrix=ROTATION).exp(), kinds.SpecialOrthogonal, False),
        (Affine(matrix=AFFINE).exp(), kinds.PositiveAffine, True),
        (Scaling(scale=np.ones(3) * 2).log(), kinds.Diagonal, True),
        # `1 + log(s)` vanishes at `s = 1 / e`: not established invertible.
        (Scaling(scale=np.ones(3) * 2).log(), kinds.InvertibleDiagonal, False),
        (Rotation(matrix=ROTATION).log(), kinds.SpecialOrthogonal, False),
    ],
)
def test_kind_membership_is_read_from_the_forward(
    t: Transformation, node: type, expected: bool
) -> None:
    for compute in (False, True):
        assert is_kind(t, node, compute) is expected


def test_a_simplified_forward_rebuilds_a_cheaper_wrapper() -> None:
    # An affine that holds a scaling is simplified to one numerically, and
    # its root is rebuilt as the root of the scaling.
    root = Affine(matrix=np.diag([4.0, 9.0, 1.0, 1.0])[:3]).sqrt()
    simplified = root.simplify("numeric")
    assert isinstance(simplified, Sqrt) and isinstance(simplified, Scaling)
    np.testing.assert_allclose(simplified.scale, [2.0, 3.0, 1.0])
    # An operator of an identity collapses to the identity.
    assert isinstance(
        Affine(matrix=np.eye(3, 4)).exp().simplify(True), Identity
    )


# ----------------------------------------------------------------------
#   REFUSALS
# ----------------------------------------------------------------------


def test_domain_error_is_a_value_error() -> None:
    assert issubclass(DomainError, ValueError)


@pytest.mark.parametrize("method", ["sqrt", "log"])
@pytest.mark.parametrize(
    "t, reason",
    [
        (Rotation(matrix=_rotation(np.pi)), "negative real axis"),
        (Linear(matrix=np.diag([-1.0, 1.0, 1.0])), "negative real axis"),
        (Affine(matrix=np.diag([1.0, -2.0, 1.0, 1.0])[:3]), "negative"),
        (Permutation(permutation=np.array([1, 0, 2])), "negative real axis"),
        (Linear(matrix=np.diag([1.0, 0.0, 1.0])), "singular"),
        (Scaling(scale=np.array([1.0, -2.0])), "positive"),
        (Scaling(scale=np.array([1.0, 0.0])), "positive"),
    ],
    ids=lambda x: getattr(type(x), "__name__", x),
)
def test_sqrt_and_log_refuse_outside_the_principal_domain(
    method: str, t: Transformation, reason: str
) -> None:
    # The refusal is a property of the value, so it is raised when the
    # lazy result is resolved, never as a complex or non-principal answer.
    lazy = getattr(t, method)()
    with pytest.raises(DomainError, match=reason):
        lazy.compute()
    with pytest.raises(DomainError):
        getattr(t, method)(compute=True)


def test_exp_has_no_principal_domain() -> None:
    # Every real matrix has a real exponential.
    for t in (
        Rotation(matrix=_rotation(np.pi)),
        Linear(matrix=np.diag([-1.0, 1.0, 1.0])),
        Linear(matrix=np.zeros((3, 3))),
        Scaling(scale=np.array([-1.0, 0.0])),
    ):
        assert np.isfinite(_matrix(t.exp(compute=True))).all()


@pytest.mark.parametrize("method", ["square", "sqrt", "exp", "log"])
def test_operators_refuse_a_map_between_different_spaces(method: str) -> None:
    voxel_to_world = VoxelToLPS(matrix=np.eye(3, 4))
    with pytest.raises(DomainError, match="maps a space to itself"):
        getattr(voxel_to_world, method)()
    rectangular = Linear(matrix=np.ones((2, 3)))
    with pytest.raises(DomainError, match="3 axes to 2 axes"):
        getattr(rectangular, method)()


def test_operators_refuse_what_they_do_not_compute() -> None:
    field = DisplacementField(field=np.zeros((4, 4, 2)) + 0.1)
    with pytest.raises(NotImplementedError, match="exponential"):
        field.sqrt()
    with pytest.raises(NotImplementedError, match="exponential"):
        field.log()
    coords = CoordinatesField(field=_grid((4, 4)))
    for method in ("sqrt", "exp", "log"):
        with pytest.raises(NotImplementedError, match="coordinates field"):
            getattr(coords, method)()
    reindex = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0, 3.0])),
        input_axes=[0, 1],
        output_axes=[1, 0],
    )
    with pytest.raises(NotImplementedError, match="reindexes"):
        reindex.sqrt()
    with pytest.raises(NotImplementedError):
        Projection(dropped=[0]).exp()


def test_a_chain_that_does_not_reduce_is_refused() -> None:
    field = DisplacementField(field=np.zeros((4, 4, 2)) + 0.1)
    affine = Affine(matrix=np.array([[1.1, 0.0, 0.5], [0.0, 0.9, 0.0]]))
    with pytest.raises(NotImplementedError, match="single transformation"):
        Sequence([affine, field]).exp()


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
    for method in ("sqrt", "exp", "log"):
        result = getattr(Sequence([a, b]), method)(compute=True)
        expected = getattr((b @ a).compute(), method)(compute=True)
        np.testing.assert_allclose(_matrix(result), _matrix(expected))


VOX2LPS = np.array(
    [[-2.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, -5.0], [0.0, 0.0, 4.0, 1.0]]
)


def _velocity_3d() -> DisplacementField:
    return DisplacementField(field=np.zeros((4, 4, 4, 3)) + 0.2)


def test_a_world_space_velocity_exponentiates_inside_its_frame() -> None:
    # A field stored in voxels between a voxel-to-world affine and its lazy
    # inverse -- the form the Zarr reader builds -- is a change of
    # coordinates, and its operators act on the field inside it.
    voxel_to_world = VoxelToLPS(matrix=VOX2LPS)
    velocity = _velocity_3d()
    chain = Sequence([voxel_to_world.inverse(), velocity, voxel_to_world])
    result = chain.exp()
    assert isinstance(result, Sequence) and len(result) == 3
    assert result[0] is chain[0] and result[2] is chain[2]
    assert isinstance(result[1], Exp) and result[1].forward is velocity
    root = result.sqrt()
    assert isinstance(root[1], Sqrt)
    assert root[0] is chain[0] and root[2] is chain[2]


def test_a_change_of_coordinates_is_recognized_exactly() -> None:
    # Two affines built separately undo each other when their product is
    # exactly the identity ...
    exact = np.linalg.inv(_homogeneous(VOX2LPS))[:3]
    np.testing.assert_array_equal(
        _homogeneous(exact) @ _homogeneous(VOX2LPS), np.eye(4)
    )
    chain = Sequence(
        [LPSToVoxel(matrix=exact), _velocity_3d(), VoxelToLPS(matrix=VOX2LPS)]
    )
    assert isinstance(chain.exp()[1], Exp)
    # ... but not when they are inverses only up to rounding: the operator
    # would then act on a chain that is not quite the one given, so the
    # chain is composed instead, and a field between two affines does not
    # compose to a single transformation.
    rounded = exact.copy()
    rounded[0, -1] += 1e-12
    chain = Sequence(
        [
            LPSToVoxel(matrix=rounded),
            _velocity_3d(),
            VoxelToLPS(matrix=VOX2LPS),
        ]
    )
    with pytest.raises(NotImplementedError, match="single transformation"):
        chain.exp()


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
    np.testing.assert_allclose(
        _matrix(bijection.exp(compute=True)),
        _matrix(forward.exp(compute=True)),
    )


def test_operators_compose_lazily_in_a_sequence() -> None:
    a = Affine(matrix=AFFINE)
    b = Rotation(matrix=ROTATION)
    expression = a.inverse() @ b.sqrt() @ a.exp()
    expected = (
        a.inverse(compute=True) @ b.sqrt(compute=True) @ a.exp(compute=True)
    ).compute()
    np.testing.assert_allclose(
        _matrix(expression.compute()), _matrix(expected), atol=1e-12
    )
    # A mode that does not admit the operators leaves them unresolved.
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


def test_exp_of_a_constant_velocity_is_a_translation() -> None:
    shift = np.array([3.3, -1.7])
    velocity = DisplacementField(field=np.zeros(SHAPE + (2,)) + shift)
    result = velocity.exp()
    assert isinstance(result, Exp) and isinstance(result, DisplacementField)
    np.testing.assert_allclose(result.field, np.zeros(SHAPE + (2,)) + shift)


def test_exp_of_a_linear_velocity_is_its_matrix_exponential() -> None:
    grid = _grid(SHAPE)
    expected = (grid - CENTER) @ scipy.linalg.expm(GENERATOR).T + CENTER
    for order in (1, 3):
        result = _linear_velocity(order=order).exp(compute=True)
        assert type(result) is DisplacementField
        np.testing.assert_allclose(
            (result.field + grid)[INTERIOR], expected[INTERIOR], atol=5e-3
        )


def test_exp_of_a_zero_velocity_is_the_identity() -> None:
    velocity = DisplacementField(field=np.zeros(SHAPE + (2,)))
    np.testing.assert_array_equal(velocity.exp().field, velocity.field)


def test_steps_default_and_override() -> None:
    velocity = _linear_velocity()
    largest = np.linalg.norm(velocity.field, axis=-1).max()
    steps = _ops._squaring_steps(velocity.field)
    assert largest / 2**steps <= 0.125 < largest / 2 ** (steps - 1)
    # No squaring at all is the first-order step `id + v`.
    np.testing.assert_array_equal(
        velocity.exp().to(steps=0).field, velocity.field
    )
    assert velocity.exp().to(steps=3).steps == 3
    with pytest.raises(ValueError, match="non-negative"):
        velocity.exp().to(steps=-1).field  # noqa: B018
    infinite = DisplacementField(field=np.full(SHAPE + (2,), np.inf))
    with pytest.raises(DomainError, match="not finite"):
        infinite.exp().compute()


def test_exp_keeps_the_field_metadata_and_endpoints() -> None:
    vox = VoxelCoordinateSystem()
    velocity = _linear_velocity(order=3, bound="reflect")
    velocity = velocity.to(input=vox, output=vox)
    result = velocity.exp()
    assert (result.order, result.bound, result.coeff) == (3, "reflect", False)
    assert result.input == vox and result.output == vox
    computed = result.compute()
    assert (computed.order, computed.bound, computed.coeff) == (
        3,
        "reflect",
        False,
    )


def test_exp_of_a_coefficient_velocity_is_a_coefficient_field() -> None:
    values = _linear_velocity(order=3)
    coefficients = values.to(coeff=True)
    result = coefficients.exp(compute=True)
    assert result.coeff
    np.testing.assert_allclose(
        result.to(coeff=False).field,
        values.exp().field,
        atol=1e-8,
    )


def test_inverse_of_exp_is_exp_of_the_negated_velocity() -> None:
    velocity = _linear_velocity()
    exp = velocity.exp()
    inverse = exp.inverse()
    assert isinstance(inverse, Inverse) and inverse.forward is exp
    negated = DisplacementField(field=-velocity.field).exp()
    np.testing.assert_array_equal(inverse.field, negated.field)
    assert type(inverse.compute()) is DisplacementField
    # Placed next to each other, the two cancel without computing anything.
    with mock.patch.object(
        _ops, "_exp_velocity", side_effect=AssertionError("computed")
    ):
        assert isinstance((inverse @ exp).compute(), Identity)
    # And composed, they are the identity up to the integration error.
    residual = (inverse.compute() @ exp.compute()).compute()
    np.testing.assert_allclose(residual.field[INTERIOR], 0, atol=1e-2)


def test_sqrt_of_exp_is_exp_of_half_the_velocity() -> None:
    velocity = _linear_velocity()
    exp = velocity.exp()
    root = exp.sqrt()
    assert isinstance(root, Sqrt) and isinstance(root, DisplacementField)
    # Its square is exactly the field `exp(v)` materializes: the root is
    # integrated with all of its squaring steps but the last.
    np.testing.assert_array_equal(
        (root.compute() @ root.compute()).compute().field, exp.field
    )
    half = DisplacementField(field=velocity.field / 2).exp()
    np.testing.assert_allclose(root.field, half.field, atol=1e-12)
    assert root.square() is exp


def test_log_of_exp_is_the_velocity() -> None:
    velocity = _linear_velocity()
    assert velocity.exp().log() is velocity


def test_exp_of_a_velocity_is_not_computed_until_read() -> None:
    velocity = _linear_velocity()
    with mock.patch.object(
        _ops, "_exp_velocity", side_effect=AssertionError("computed")
    ):
        exp = velocity.exp()
        root = exp.sqrt()
        assert is_kind(exp, "displacements")
        assert exp.simplify() is exp
        assert root.simplify() is root
        assert exp.log() is velocity


def test_a_chain_of_two_fields_composes_before_its_exp() -> None:
    a = DisplacementField(field=np.zeros(SHAPE + (2,)) + 0.25)
    b = DisplacementField(field=np.zeros(SHAPE + (2,)) - 0.5)
    result = Sequence([a, b]).exp(compute=True)
    np.testing.assert_allclose(result.field, -0.25)
