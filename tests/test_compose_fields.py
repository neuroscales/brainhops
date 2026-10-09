"""Tests of the composition of fields with affines and subspace transforms.

An affine folded into a coordinate or displacement field must keep the
degree, bound and store of the field, and a coefficient field must be
converted to values before the affine arithmetic and back afterwards. The
folded field, evaluated at interior points, must equal the affine applied
to the interpolated field.
"""

import numpy as np
import pytest

from brainhops.backends import (
    available_backends,
    backend,
    get_array_backend,
)
from brainhops.datamodel._transformations.compute.compose import compose
from brainhops.datamodel._transformations.sequence import (
    normalize_modes,
)
from brainhops.datamodel.axes import (
    A,
    R,
    S,
    TimeAxis,
)
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    InverseDisplacementField,
    Linear,
    Sequence,
    SubspaceTransformation,
    Transformation,
    is_identity,
)
from brainhops.errors import (
    CompositionError,
    ConversionError,
)

# Anisotropic, sheared and shifted, far from the identity, so that a dropped
#
# setting or arithmetic on coefficients gives a large error.
AFFINE_MATRIX = np.array([[1.7, 0.4, 2.0], [-0.3, 0.9, -1.5]])

# Large enough to keep the query points several nodes from every edge.
GRID_SHAPE = (14, 15)

# Offset from the nodes, so that a wrong spline degree changes the result.
QUERY_POINTS = np.array(
    [[5.5, 6.5], [7.2, 8.1], [6.3, 5.7], [8.0, 9.0], [5.8, 7.4]]
)

# Coefficients require a degree of at least 2.
DEGREE_STORE = [
    (1, "values"),
    (3, "values"),
    (3, "coefficients"),
]

ARRAY_BACKENDS = [
    "numpy",
    pytest.param(
        "dask",
        marks=pytest.mark.skipif(
            "dask" not in available_backends(),
            reason="dask is not installed",
        ),
    ),
]
"""Array backends on which a fold is checked."""


def _evaluate(field, points):  # noqa: ANN001, ANN202
    """Return the coordinates of a field evaluated at the given points."""
    computed = Sequence(
        transformations=[CoordinatesField(field=points), field]
    ).compute()
    return np.asarray(computed.to(CoordinatesField).field)


@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("degree, store", DEGREE_STORE)
def test_fold_affine_into_field_keeps_interpolation_settings(
    field_type: type,
    degree: int,
    store: str,
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale
    field = field_type(
        field=values, degree=degree, bound=BoundaryCondition.mirror
    ).to(store=store)

    folded = (Affine(matrix=AFFINE_MATRIX) @ field).compute()

    assert folded.degree == field.degree
    assert folded.bound == field.bound
    assert folded.store == field.store


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("degree, store", DEGREE_STORE)
def test_fold_affine_into_field_matches_inorder_reference(
    field_type: type,
    degree: int,
    store: str,
    array_backend: str,
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale

    # Under dask, the folded field is a dask array prefiltered chunk by chunk,
    #
    # with a halo wide enough to match the prefilter over a whole axis.
    with backend(array_backend):
        field = field_type(
            field=values, degree=degree, bound=BoundaryCondition.mirror
        ).to(store=store)

        matrix = AFFINE_MATRIX
        sampled = _evaluate(field, QUERY_POINTS)
        reference = sampled @ matrix[:, :-1].T + matrix[:, -1]

        folded = (Affine(matrix=matrix) @ field).compute()
        result = _evaluate(folded, QUERY_POINTS)

    # The fold of a coordinate field is exact. The fold of a displacement field
    #
    # of degree above 1 has an extra term, because the node grid that the
    #
    # representation subtracts is not exactly reproduced by cubic interpolation
    #
    # of a finite grid.
    interior_term = field_type is DisplacementField and degree > 1
    atol = 1e-3 if interior_term else 1e-10
    np.testing.assert_allclose(result, reference, atol=atol, rtol=0)


# ----------------------------------------------------------------------
#   SUBSPACE COMPOSERS: A 3D TRANSFORM ACROSS A 4D FIELD
# ----------------------------------------------------------------------


def _full4(name: str) -> CoordinateSystem:
    return CoordinateSystem(
        name=name, axes=[R(), A(), S(), TimeAxis(name="t")]
    )


def _sub3(name: str) -> CoordinateSystem:
    return CoordinateSystem(name=name, axes=[R(), A(), S()])


# Sheared and shifted, so that a dropped or misplaced component is visible.
SUB_AFFINE = np.array(
    [[1.3, 0.2, -0.1, 4.0], [0.0, 0.9, 0.3, -2.0], [0.1, 0.0, 1.1, 1.0]]
)


def _subspace_affine(positions) -> SubspaceTransformation:  # noqa: ANN001
    positions = np.asarray(positions, dtype=int)
    inner = Affine(matrix=SUB_AFFINE, input=_sub3("lps"), output=_sub3("lps"))
    return SubspaceTransformation(
        transformation=inner,
        input_axes=positions,
        output_axes=positions,
        input=_full4("in"),
        output=_full4("out"),
    )


def _coords_4d(seed: int = 0) -> CoordinatesField:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((5, 6, 4))
    return CoordinatesField(
        field=field,
        output=_full4("in"),
        degree=3,
        bound=BoundaryCondition.mirror,
    )


def test_subspace_coords_matches_affine_reduction() -> None:
    # Folding a subspace affine equals folding its reduction to an affine.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_subspace_coords_passthrough_is_bit_exact() -> None:
    # The time component is copied through bit for bit.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = np.asarray(compose(To, Ti).field)
    np.testing.assert_array_equal(got[..., 3], np.asarray(Ti.field)[..., 3])


def test_subspace_coords_permuted_positions_align() -> None:
    # Non-monotonic positions place the components where the axis vectors say.
    To = _subspace_affine([2, 0, 1])
    Ti = _coords_4d()
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(got.field)[..., 3], np.asarray(Ti.field)[..., 3]
    )


def test_subspace_coords_preserves_interpolation_settings() -> None:
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = compose(To, Ti)
    assert got.degree == Ti.degree
    assert got.bound == Ti.bound
    assert got.store == Ti.store


def test_subspace_coords_promotes_an_integer_domain() -> None:
    # An integer coordinate field is promoted to float.
    ab = get_array_backend()
    field = ab.asarray(np.arange(5 * 6 * 4).reshape(5, 6, 4), dtype="int64")
    Ti = CoordinatesField(field=field, output=_full4("in"))
    To = _subspace_affine([0, 1, 2])
    got = compose(To, Ti)
    assert np.asarray(got.field).dtype.kind == "f"


def test_subspace_disp_matches_affine_reduction() -> None:
    rng = np.random.default_rng(2)
    disp = rng.standard_normal((3, 4, 5, 2, 4)) * 0.1
    Ti = DisplacementField(field=disp, output=_full4("in"))
    To = _subspace_affine([0, 1, 2])
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    assert isinstance(got, DisplacementField)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def _empty_subspace_3d(input_axes, output_axes) -> SubspaceTransformation:  # noqa: ANN001
    # A subspace transform without an inner transform, over 3-D space.
    return SubspaceTransformation(
        transformation=None,
        input_axes=np.asarray(input_axes, dtype=int),
        output_axes=np.asarray(output_axes, dtype=int),
        input=_sub3("in"),
        output=_sub3("out"),
    )


def _coords_3d() -> CoordinatesField:
    # Each component has distinct values, so that a swap or drop is visible.
    field = np.stack(
        [
            np.full((2, 3, 4), 1.0) + np.arange(4),
            np.full((2, 3, 4), 10.0) + np.arange(4),
            np.full((2, 3, 4), 100.0) + np.arange(4),
        ],
        axis=-1,
    )
    return CoordinatesField(field=field, output=_sub3("in"))


def test_empty_subspace_coords_reindexes_like_the_affine_reduction() -> None:
    # Swapped axis vectors reindex the components, as the affine reduction
    #
    # does; relabelling alone would not pass.
    To = _empty_subspace_3d([0, 1], [1, 0])
    Ti = _coords_3d()
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    assert isinstance(got, CoordinatesField)
    np.testing.assert_array_equal(np.asarray(got.field), np.asarray(ref.field))
    x = np.asarray(Ti.field)
    np.testing.assert_array_equal(np.asarray(got.field), x[..., [1, 0, 2]])
    assert got.output == _sub3("out")


def test_empty_subspace_disp_reindexes_like_the_affine_reduction() -> None:
    rng = np.random.default_rng(3)
    Ti = DisplacementField(
        field=rng.standard_normal((2, 3, 4, 3)) * 0.1, output=_sub3("in")
    )
    To = _empty_subspace_3d([0, 1], [1, 0])
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    assert isinstance(got, DisplacementField)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_empty_subspace_coords_matching_axes_is_unchanged() -> None:
    # Matching axis vectors leave the field unchanged, except for its labels.
    To = _empty_subspace_3d([0, 1], [0, 1])
    Ti = _coords_3d()
    got = compose(To, Ti)
    assert isinstance(got, CoordinatesField)
    np.testing.assert_array_equal(np.asarray(got.field), np.asarray(Ti.field))
    assert got.output == _sub3("out")
    assert got.degree == Ti.degree
    assert got.bound == Ti.bound


def test_subspace_compose_subspace_matches_into_one_wrapper() -> None:
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    composed = compose(second, first)
    assert isinstance(composed, SubspaceTransformation)
    np.testing.assert_array_equal(composed.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(composed.output_axes, [0, 1, 2])


def test_subspace_compose_its_inverse_is_identity_without_inverting(
    monkeypatch,  # noqa: ANN001
) -> None:
    # The pair cancels without inverting the field numerically.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.random.default_rng(3).standard_normal((4, 4, 4, 3)) * 0.1,
        input=voxel,
        output=voxel,
    )
    wrapper = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )
    composed = compose(wrapper, wrapper.inverse())
    assert isinstance(composed, Identity)


def test_subspace_compose_subspace_mismatch_raises() -> None:
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([1, 2, 3])
    with pytest.raises(CompositionError):
        compose(second, first)


_DEFAULT_MODE = normalize_modes(None)


def test_merge_adjacent_subspaces_folds_a_matching_pair() -> None:
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    folded = compose(second, first)
    assert isinstance(folded, SubspaceTransformation)
    np.testing.assert_array_equal(folded.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(folded.output_axes, [0, 1, 2])


def _field_wrapper(seed: int) -> SubspaceTransformation:
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.random.default_rng(seed).standard_normal((4, 4, 4, 3)) * 0.1,
        input=voxel,
        output=voxel,
    )
    return SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )


def test_merge_adjacent_subspaces_drops_an_inverse_pair() -> None:
    # The pair cancels symbolically, leaving an identity rather than an empty
    #
    # sequence.
    wrapper = _field_wrapper(4)
    computed = Sequence([wrapper.inverse(), wrapper]).compute(
        mode=_DEFAULT_MODE
    )
    assert isinstance(computed, Identity)
    assert computed.input == _full4("s")
    assert computed.output == _full4("s")


def test_field_subspaces_are_not_composed_under_affine_mode(
    monkeypatch,  # noqa: ANN001
) -> None:
    # No field is resampled through another under the affine-only mode.
    from brainhops.datamodel._transformations.compute import composers

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("a field was composed numerically")

    monkeypatch.setattr(composers, "pull_field", _boom)
    first = _field_wrapper(4)
    second = _field_wrapper(5)
    result = Sequence([first, second]).compute(mode="Affine")
    assert isinstance(result, Sequence)
    assert len(result.transformations) == 2


def test_full_subspace_cancellation_computes_to_the_identity() -> None:
    wrapper = _field_wrapper(6)
    result = Sequence([wrapper.inverse(), wrapper]).compute()
    assert isinstance(result, Identity)
    assert result.input == _full4("s")
    assert result.output == _full4("s")


def _subspace_reindex(input_axes, output_axes) -> SubspaceTransformation:  # noqa: ANN001
    # A subspace transform whose axis vectors move components from
    #
    # `input_axes` to `output_axes`.
    return SubspaceTransformation(
        transformation=None,
        input_axes=np.asarray(input_axes, dtype=int),
        output_axes=np.asarray(output_axes, dtype=int),
        input=_full4("in"),
        output=_full4("out"),
    )


def test_subspace_compose_identity_inner_keeps_axis_reindex() -> None:
    # The inner transforms cancel, but the axis vectors form a real
    #
    # permutation, so the result is a reindex and not a bare identity.
    Ti = _subspace_reindex([0, 1, 2], [1, 2, 0])
    To = _subspace_reindex([1, 2, 0], [1, 2, 0])

    composed = compose(To, Ti)

    assert not isinstance(composed, Identity)
    assert isinstance(composed, SubspaceTransformation)
    assert not is_identity(composed, compute=True)
    np.testing.assert_array_equal(composed.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(composed.output_axes, [1, 2, 0])

    # Input axis i goes to output axis o for each (o, i) pair, and the time
    #
    # axis passes through: y = [x2, x0, x1, x3].
    matrix = np.asarray(composed.to(Affine).matrix)
    expected = np.zeros((4, 5))
    expected[1, 0] = 1.0
    expected[2, 1] = 1.0
    expected[0, 2] = 1.0
    expected[3, 3] = 1.0
    np.testing.assert_array_equal(matrix, expected)


def test_subspace_compose_identity_inner_matching_axes_is_identity() -> None:
    # With matching axis vectors, the result is a bare identity.
    Ti = _subspace_reindex([0, 1, 2], [0, 1, 2])
    To = _subspace_reindex([0, 1, 2], [0, 1, 2])

    composed = compose(To, Ti)

    assert isinstance(composed, Identity)
    assert composed.input == _full4("in")
    assert composed.output == _full4("out")


def test_subspace_to_affine_on_a_field_inner_raises() -> None:
    # A field inner cannot be reduced to an affine.
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.zeros((4, 4, 4, 3)), input=voxel, output=voxel
    )
    wrapper = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )
    with pytest.raises(ConversionError):
        wrapper.to(Affine)


# ----------------------------------------------------------------------
#   EMBED COMPOSERS AND THE ANALYTIC IDENTITY CANCEL
# ----------------------------------------------------------------------


def test_subspace_affine_embed_folds_a_non_interpolating_subspace() -> None:
    # Embedding an affine does not interpolate, so it folds into one affine.
    sub = _subspace_affine([0, 1, 2])
    aff = Affine(
        matrix=np.eye(4, 5), input=_full4("out"), output=_full4("world")
    )
    folded = compose(aff, sub)
    assert isinstance(folded, Affine)
    assert not isinstance(folded, SubspaceTransformation)
    ref = compose(aff, sub.to(Affine))
    np.testing.assert_allclose(
        np.asarray(folded.matrix), np.asarray(ref.matrix)
    )

    # The mirrored order folds as well.
    aff2 = Affine(
        matrix=np.eye(4, 5), input=_full4("world"), output=_full4("in")
    )
    folded2 = compose(sub, aff2)
    assert isinstance(folded2, Affine)
    assert not isinstance(folded2, SubspaceTransformation)


def test_interpolating_subspace_does_not_embed_into_an_affine() -> None:
    # A subspace wrapping a field interpolates, so it does not fold into an
    #
    # affine and survives in a sequence.
    wrapper = _field_wrapper(7)
    aff = Affine(matrix=np.eye(4, 5), input=_full4("s"), output=_full4("s"))
    with pytest.raises(CompositionError):
        compose(aff, wrapper)
    result = Sequence([wrapper, aff]).compute()
    leaves = list(result) if isinstance(result, Sequence) else [result]
    assert any(isinstance(t, SubspaceTransformation) for t in leaves)


def test_compose_cancels_inverse_by_identity_without_materializing(
    monkeypatch,  # noqa: ANN001
) -> None:
    # Cancellation runs before the numeric composers, so an affine composed
    #
    # with its inverse never computes a matrix inverse.
    real_inv = np.linalg.inv
    calls = {"n": 0}

    def counting(matrix):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return real_inv(matrix)

    monkeypatch.setattr(np.linalg, "inv", counting)
    affine = Affine(matrix=np.array([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]]))
    result = compose(affine, affine.inverse())
    assert isinstance(result, Identity)
    assert calls["n"] == 0

    # A field composed with its inverse is never inverted numerically.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    field = DisplacementField(field=np.zeros((5, 6, 2)))
    cancelled = compose(field, field.inverse())
    assert isinstance(cancelled, Identity)


def test_an_equal_but_distinct_transform_never_cancels() -> None:
    # Cancellation compares identity only, never values, which might raise.
    from brainhops.datamodel._transformations.compute import simplifiers

    affine = Affine(matrix=SUB_AFFINE)
    twin = Affine(matrix=SUB_AFFINE.copy())
    field = DisplacementField(field=np.zeros((5, 6, 2)))
    copy = DisplacementField(field=np.zeros((5, 6, 2)))
    for first, second in ((affine, twin), (field, copy)):
        assert simplifiers._cancels(first, first.inverse())
        assert not simplifiers._cancels(first, second.inverse())
        assert not simplifiers._cancels(second.inverse(), first)


def test_a_3d_affine_refuses_a_4d_field() -> None:
    # A 3-D affine applied to a 4-D field is refused, not partially applied.
    points = np.random.default_rng(0).standard_normal((5, 4))
    system = _full4("world")
    for transform in (
        Affine(matrix=SUB_AFFINE, input=system, output=system),
        Linear(matrix=SUB_AFFINE[:, :-1], input=system, output=system),
    ):
        with pytest.raises(ValueError):
            compose(transform, CoordinatesField(field=points))


def test_restrictive_mode_prevents_field_through_field_composition() -> None:
    # The affine-only mode keeps two field wrappers apart; the default merges
    #
    # them.
    first = _field_wrapper(4)
    second = _field_wrapper(5)
    unmerged = Sequence([first, second]).compute(mode="Affine")
    assert isinstance(unmerged, Sequence)
    assert len(unmerged.transformations) == 2
    merged = Sequence([first, second]).compute()
    assert isinstance(merged, SubspaceTransformation)


# ----------------------------------------------------------------------
#   DISPATCH ORDER AND THE ANALYTIC CANCEL TIER
# ----------------------------------------------------------------------


def test_compose_distinct_inverse_falls_through_to_affine() -> None:
    # Distinct affines do not cancel and go to the numeric affine composer.
    A = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]))
    B = Affine(matrix=np.array([[1.5, 0.0, -1.0], [0.0, 0.5, 4.0]]))
    result = compose(A, B.inverse())
    assert type(result) is Affine
    assert not isinstance(result, Identity)


def test_compose_identity_with_lazy_inverse_stays_unmaterialized(
    monkeypatch,  # noqa: ANN001
) -> None:
    # Composing with the identity returns the lazy inverse untouched.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    df = DisplacementField(field=np.zeros((5, 6, 2)))
    lazy = df.inverse()
    result = compose(Identity(), lazy)
    assert result is lazy
    assert isinstance(result, InverseDisplacementField)


def test_compose_tries_the_pair_simplifiers_before_any_composer() -> None:
    # No composer can materialize the inverse of a coordinate field, so only
    #
    # the pair simplifiers can cancel the pair.
    cf = CoordinatesField(field=np.zeros((5, 6, 2)))
    assert isinstance(compose(cf.inverse(), cf), Identity)


def test_dispatch_order_and_terminal_composition_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    # Only the most specific composer runs, and a `CompositionError` that it
    #
    # raises is final: there is no fallback to a less specific composer.
    from bagof.dispatchers import Function

    from brainhops.datamodel._transformations.compute import (
        compose as compose_mod,
    )

    a = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]))
    b = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    calls: list = []

    def make_composers(specific):  # noqa: ANN001, ANN202
        fn = Function("compose")

        def family(x1: Transformation, x2: Transformation) -> Transformation:
            calls.append("family")
            return Identity()

        fn.register(specific)
        fn.register(family)
        return fn

    def specific_identity(x1: Affine, x2: Affine) -> Transformation:
        calls.append("specific")
        return Identity()

    monkeypatch.setattr(
        compose_mod, "_compose", make_composers(specific_identity)
    )
    result = compose(a, b)
    assert isinstance(result, Identity)
    assert calls == ["specific"]

    calls.clear()

    def specific_raises(x1: Affine, x2: Affine) -> Transformation:
        calls.append("specific")
        raise CompositionError("right types, cannot combine")

    monkeypatch.setattr(
        compose_mod, "_compose", make_composers(specific_raises)
    )
    with pytest.raises(CompositionError):
        compose(a, b)
    assert calls == ["specific"]


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("affine", [True, False])
def test_matrix_that_changes_axes_is_not_folded_into_a_displacement(
    rows: int, affine: bool
) -> None:
    # A displacement field maps a space onto itself, so a matrix that changes
    #
    # the number of axes cannot be folded into it and stays in the sequence.
    rng = np.random.default_rng(0)
    field = DisplacementField(field=rng.normal(size=(4, 5, 3, 3)) * 0.3)
    if affine:
        matrix = Affine(matrix=np.eye(rows, 4))
    else:
        matrix = Linear(matrix=np.eye(rows, 3))
    with pytest.raises(CompositionError):
        compose(matrix, field)
    result = Sequence([field, matrix]).compute()
    assert [type(t) for t in result.transformations] == [
        DisplacementField,
        type(matrix),
    ]
