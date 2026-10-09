"""Tests for lazy inverses, their cancellation and materialization."""

import inspect
from unittest import mock

import numpy as np
import pytest

from brainhops._core.bsplines import coeff2value_field
from brainhops._ext.invfield import inverse as inverse_disp
from brainhops.datamodel import transformations as _xf
from brainhops.datamodel._transformations import concrete as _concrete
from brainhops.datamodel.systems import (
    CoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    InverseAffine,
    InverseCoordinatesField,
    InverseDisplacementField,
    InverseLinear,
    InversePermutation,
    InverseRotation,
    InverseScaling,
    InverseTranslation,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Sequence,
    Transformation,
    Translation,
    is_identity,
)
from brainhops.io.transformations.base.affines import (
    LPSToVoxel,
    RASToVoxel,
    VoxelToLPS,
    VoxelToRAS,
)


def _small_field(shape: tuple = (6, 7, 2), seed: int = 0) -> np.ndarray:
    # Small displacements keep the mesh inversion stable.
    rng = np.random.RandomState(seed)
    return rng.randn(*shape) * 0.05


# ----------------------------------------------------------------------
#   LAZY VS EAGER SELECTION
# ----------------------------------------------------------------------


def test_displacement_field_inverse_is_lazy() -> None:
    df = DisplacementField(field=_small_field())
    inv = df.inverse()
    assert isinstance(inv, InverseDisplacementField)
    # The kind checks of the compose engine see through the wrapper.
    assert isinstance(inv, DisplacementField)
    assert inv.forward is df


def test_coordinates_field_inverse_is_lazy() -> None:
    cf = CoordinatesField(field=_small_field())
    inv = cf.inverse()
    assert isinstance(inv, InverseCoordinatesField)
    assert isinstance(inv, CoordinatesField)
    assert inv.forward is cf


def test_empty_field_inverse_stays_eager() -> None:
    # Without a field, the inverse is an eager endpoint-swapped transform.
    df = DisplacementField()
    assert not isinstance(df.inverse(), Inverse)
    assert type(df.inverse()) is DisplacementField
    cf = CoordinatesField()
    assert not isinstance(cf.inverse(), Inverse)
    assert type(cf.inverse()) is CoordinatesField


def test_meta_families_stay_eager() -> None:
    # Identity is its own inverse, and a Bijection carries its own inverse.
    cases = [
        Identity(),
        Bijection(
            forward=Translation(translation=[1.0]),
            backward=Translation(translation=[-1.0]),
        ),
    ]
    for t in cases:
        inv = t.inverse()
        assert not isinstance(inv, Inverse), type(t).__name__
        assert type(inv) is type(t), type(t).__name__


def test_trivial_families_are_lazy() -> None:
    cases = [
        (Translation(translation=[1.0, 2.0]), InverseTranslation, Translation),
        (Scaling(scale=[2.0, 3.0]), InverseScaling, Scaling),
        (Permutation(permutation=[1, 0]), InversePermutation, Permutation),
        (
            Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]]),
            InverseRotation,
            Rotation,
        ),
    ]
    for fwd, wrapper, family in cases:
        inv = fwd.inverse()
        assert isinstance(inv, wrapper), type(fwd).__name__
        assert isinstance(inv, family), type(fwd).__name__
        assert inv.forward is fwd, type(fwd).__name__


def test_affine_and_linear_inverse_is_lazy() -> None:
    affine = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    inv = affine.inverse()
    assert isinstance(inv, InverseAffine)
    assert isinstance(inv, Affine)
    assert inv.forward is affine

    linear = Linear(matrix=[[2.0, 0.0], [0.0, 4.0]])
    inv = linear.inverse()
    assert isinstance(inv, InverseLinear)
    assert isinstance(inv, Linear)
    assert inv.forward is linear


def test_affine_materialization_matches_matrix_inverse() -> None:
    matrix = np.array([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    affine = Affine(matrix=matrix)
    inv = affine.inverse()
    homogeneous = np.concatenate([matrix, [[0.0, 0.0, 1.0]]], axis=0)
    expected = np.linalg.inv(homogeneous)[:-1]
    np.testing.assert_allclose(np.asarray(inv.matrix), expected)


def test_affine_cancels_symbolically_with_zero_matrix_inversions() -> None:
    # The product A^-1 @ A cancels without inverting any matrix.
    affine = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    calls = {"n": 0}
    real_inv = np.linalg.inv

    def counting_inv(m: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real_inv(m)

    with mock.patch.object(np.linalg, "inv", counting_inv):
        result = Sequence(transformations=[affine, affine.inverse()]).compute()
    assert isinstance(result, Identity)
    assert calls["n"] == 0


# ----------------------------------------------------------------------
#   ATTRIBUTE PRESERVATION (the "C2 inverse half")
# ----------------------------------------------------------------------


def test_displacement_inverse_preserves_degree_store_bound() -> None:
    df = DisplacementField(
        data=_small_field(), degree=3, bound=2.0, store="coefficients"
    )
    inv = df.inverse()
    assert inv.degree == 3
    assert inv.bound == 2.0
    assert inv.store == "coefficients"
    assert inv.input is df.output
    assert inv.output is df.input


def test_coordinates_inverse_preserves_degree_store_bound() -> None:
    cf = CoordinatesField(
        data=_small_field(), degree=2, bound=1.0, store="coefficients"
    )
    inv = cf.inverse()
    assert inv.degree == 2
    assert inv.bound == 1.0
    assert inv.store == "coefficients"


# ----------------------------------------------------------------------
#   MATERIALIZATION EQUALS THE FORMER EAGER INVERSE
# ----------------------------------------------------------------------


def test_materialized_field_matches_eager_inverse() -> None:
    values = _small_field(seed=3)
    df = DisplacementField(
        field=values, degree=1, bound="nearest", store="values"
    )
    inv = df.inverse()
    np.testing.assert_allclose(np.asarray(inv.field), inverse_disp(values))


def test_materialized_field_is_cached() -> None:
    df = DisplacementField(field=_small_field(seed=4))
    inv = df.inverse()
    first = inv.field
    assert inv.field is first


def test_coefficient_inverse_is_a_coefficient_field() -> None:
    # The metadata carries over without materializing.
    df = DisplacementField(
        data=_small_field(), degree=3, bound=2.0, store="coefficients"
    )
    inv = df.inverse()
    assert inv.store == "coefficients"
    assert inv.degree == 3
    assert inv.bound == 2.0


@pytest.mark.xfail(
    reason="#63: bsplines coeff2value/value2coeff are broken on this base; "
    "the coefficient re-fit round-trip is fixed in a separate session.",
    strict=False,
)
def test_coefficient_inverse_refits_to_coefficients() -> None:
    # This test depends on the bsplines fix (#63).
    degree, bound = 3, "nearest"
    values = _small_field(seed=7)
    df = DisplacementField(
        field=values, degree=degree, bound=bound, store="values"
    )
    coeff = df.to(store="coefficients")
    materialized = coeff.inverse().data
    expected = inverse_disp(values)
    from brainhops._core.bsplines import coeff2value_field

    recovered = coeff2value_field(
        np.asarray(materialized), degree=degree, bound=bound
    )
    np.testing.assert_allclose(np.asarray(recovered), expected, atol=1e-6)


def _coordinate_field(seed: int = 0) -> tuple:
    # A coordinate field made of its own grid plus a small displacement.
    values = _small_field(seed=seed)
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in values.shape[:-1]], indexing="ij"),
        -1,
    )
    return CoordinatesField(field=grid + values), grid, values


def test_coordinate_inverse_matches_the_displacement_inverse() -> None:
    # The inverse of grid + d is grid + inverse(d).
    cf, grid, values = _coordinate_field()
    np.testing.assert_allclose(
        np.asarray(cf.inverse().field), grid + inverse_disp(values)
    )


def test_coordinate_inverse_materializes_to_a_plain_instance() -> None:
    cf, grid, values = _coordinate_field(seed=1)
    computed = cf.inverse().compute()
    assert type(computed) is CoordinatesField
    np.testing.assert_allclose(
        np.asarray(computed.field), grid + inverse_disp(values)
    )


def test_coordinate_inverse_of_coefficients_stays_coefficients() -> None:
    cf, _grid, _values = _coordinate_field(seed=2)
    coeffs = cf.to(degree=3).to(store="coefficients")
    inverse = coeffs.inverse()
    assert inverse.store == "coefficients"
    assert inverse.degree == coeffs.degree
    recovered = coeff2value_field(
        np.asarray(inverse.data), degree=coeffs.degree, bound=coeffs.bound
    )
    np.testing.assert_allclose(
        recovered, np.asarray(cf.inverse().field), atol=1e-6
    )


def test_coordinate_inverse_cancels_rather_than_inverting() -> None:
    # The pair cancels, so the mesh inversion never runs.
    cf, _grid, _values = _coordinate_field(seed=3)
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        result = Sequence(transformations=[cf, cf.inverse()]).compute()
    assert isinstance(result, Identity)
    assert calls["n"] == 0


# ----------------------------------------------------------------------
#   DOUBLE INVERSE
# ----------------------------------------------------------------------


def test_double_inverse_returns_operand() -> None:
    df = DisplacementField(
        data=_small_field(), degree=3, store="coefficients", bound=2.0
    )
    assert df.inverse().inverse() is df


def test_invert_operator_matches_inverse() -> None:
    df = DisplacementField(field=_small_field())
    assert isinstance(~df, InverseDisplacementField)
    assert (~df).forward is df


# ----------------------------------------------------------------------
#   CANCELLATION IN A SEQUENCE
# ----------------------------------------------------------------------


def test_cancellation_transform_then_inverse() -> None:
    df = DisplacementField(field=_small_field())
    result = Sequence(transformations=[df, df.inverse()]).compute()
    assert isinstance(result, Identity)


def test_cancellation_inverse_then_transform() -> None:
    df = DisplacementField(field=_small_field())
    result = Sequence(transformations=[df.inverse(), df]).compute()
    assert isinstance(result, Identity)


def test_cancellation_does_not_materialize() -> None:
    # Materializing this inverse would raise, so an Identity shows that
    # cancellation comes first.
    df = DisplacementField(data=_small_field(), degree=3, store="coefficients")
    result = Sequence(transformations=[df, df.inverse()]).compute()
    assert isinstance(result, Identity)


def test_cancellation_of_coordinate_field() -> None:
    cf = CoordinatesField(field=_small_field())
    result = Sequence(transformations=[cf, cf.inverse()]).compute()
    assert isinstance(result, Identity)


def test_nested_cancellation_collapses_to_identity() -> None:
    x = DisplacementField(field=_small_field(seed=1))
    y = DisplacementField(field=_small_field(seed=2))
    seq = Sequence(transformations=[y, x, x.inverse(), y.inverse()])
    assert isinstance(seq.compute(), Identity)


def test_partial_cancellation_keeps_survivors() -> None:
    df = DisplacementField(field=_small_field())
    t = Translation(translation=[1.0, 2.0])
    left = Sequence(transformations=[t, df, df.inverse()]).compute()
    assert isinstance(left, Translation)
    right = Sequence(transformations=[df, df.inverse(), t]).compute()
    assert isinstance(right, Translation)


def test_lazy_inverse_of_lazy_inverse_cancels() -> None:
    df = DisplacementField(field=_small_field())
    inv = df.inverse()
    result = Sequence(transformations=[inv, inv.inverse()]).compute()
    assert isinstance(result, Identity)


# ----------------------------------------------------------------------
#   TYPE TRANSPARENCY (composition materializes the same result)
# ----------------------------------------------------------------------


def test_composed_lazy_inverse_equals_composed_eager_inverse() -> None:
    values = _small_field(seed=5)
    df = DisplacementField(
        field=values, degree=1, bound="nearest", store="values"
    )
    t = Translation(translation=[1.0, 2.0])

    lazy = Sequence(transformations=[df.inverse(), t]).compute()
    eager = DisplacementField(
        field=inverse_disp(values),
        degree=1,
        bound="nearest",
        store="values",
    )
    expected = Sequence(transformations=[eager, t]).compute()

    assert isinstance(lazy, DisplacementField)
    np.testing.assert_allclose(
        np.asarray(lazy.field), np.asarray(expected.field)
    )


def test_standalone_lazy_inverse_survives_compute() -> None:
    df = DisplacementField(field=_small_field())
    result = Sequence(transformations=[df.inverse()]).compute()
    assert isinstance(result, InverseDisplacementField)


# ----------------------------------------------------------------------
#   THE ARCHETYPAL CASE: level^-1 @ level  (#57)
# ----------------------------------------------------------------------


def _displacement_level() -> tuple:
    # A displacement level as built by the OME-Zarr reader (#57).
    voxel2world = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    world2voxel = voxel2world.inverse()
    field = DisplacementField(field=_small_field())
    level = Sequence(transformations=[world2voxel, field, voxel2world])
    return level, field


def test_level_inv_at_level_cancels_zero_field_inversions() -> None:
    # The affines and the field cancel before any field is inverted.
    level, _ = _displacement_level()
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        result = (level.inverse() @ level).compute()

    assert isinstance(result, Identity)
    assert calls["n"] == 0


def test_level_at_level_inv_cancels_zero_field_inversions() -> None:
    level, _ = _displacement_level()
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        result = (level @ level.inverse()).compute()

    assert isinstance(result, Identity)
    assert calls["n"] == 0


# ----------------------------------------------------------------------
#   is_identity IS COST-FREE ON A LAZY INVERSE  (#2)
# ----------------------------------------------------------------------


def test_is_identity_compute_false_does_not_materialize() -> None:
    # The answer comes from the operand; the coefficient inverse could not be
    # materialized here anyway (#63).
    for operand in (
        DisplacementField(data=_small_field(), degree=3, store="coefficients"),
        CoordinatesField(field=_small_field()),
    ):
        inv = operand.inverse()
        assert is_identity(inv, compute=False) is False


def test_is_identity_recognizes_an_empty_lazy_inverse() -> None:
    # An inverse of an empty field is recognized from its operand.
    inv = InverseDisplacementField(forward=DisplacementField())
    assert is_identity(inv, compute=False) is True


# ----------------------------------------------------------------------
#   CANCELLATION AFTER GRID DROP AND TO A FIXPOINT  (#4)
# ----------------------------------------------------------------------


def test_cancellation_after_interior_grid_drop() -> None:
    # The grid is dropped first, which makes the field pair adjacent.
    from brainhops.datamodel.transformations import CartesianField

    df = DisplacementField(field=_small_field())
    grid = CartesianField(shape=(6, 7))
    a = Translation(translation=[1.0, 2.0])
    b = Translation(translation=[3.0, 4.0])
    seq = Sequence(transformations=[a, df, grid, df.inverse(), b])
    result = seq.compute()
    # The translations a and b combine, and the field pair cancels across the
    # grid.
    assert isinstance(result, (Translation, Identity))


def test_cancellation_reaches_a_fixpoint() -> None:
    from brainhops.datamodel.transformations import CartesianField

    x = DisplacementField(field=_small_field(seed=1))
    y = DisplacementField(field=_small_field(seed=2))
    grid = CartesianField(shape=(6, 7))
    seq = Sequence(transformations=[y, x, grid, x.inverse(), y.inverse()])
    assert isinstance(seq.compute(), Identity)


# ----------------------------------------------------------------------
#   COLLAPSED IDENTITY TAKES THE ENDPOINTS  (#5)
# ----------------------------------------------------------------------


def test_collapsed_identity_takes_first_input_and_last_output() -> None:
    from brainhops.datamodel.systems import CoordinateSystem

    world = CoordinateSystem(name="world")
    voxel = CoordinateSystem(name="voxel")
    df = DisplacementField(field=_small_field(), input=world, output=voxel)
    inv = df.inverse()
    result = Sequence(transformations=[df, inv]).compute()
    assert isinstance(result, Identity)
    # The identity spans the first input to the last output, not the unset
    # endpoints of the sequence.
    assert result.input is world
    assert result.output is world


# ----------------------------------------------------------------------
#   PUBLIC Inverse RECONCILIATION  (#6)
# ----------------------------------------------------------------------


def test_public_inverse_wrapper_cancels_in_a_sequence() -> None:
    # A generic Inverse expands to the typed inverse and cancels.
    df = DisplacementField(field=_small_field())
    result = Sequence(transformations=[df, Inverse(forward=df)]).compute()
    assert isinstance(result, Identity)
    result = Sequence(transformations=[Inverse(forward=df), df]).compute()
    assert isinstance(result, Identity)


def test_public_inverse_of_affine_cancels() -> None:
    affine = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    seq = Sequence(transformations=[affine, Inverse(forward=affine)])
    assert isinstance(seq.compute(), Identity)


# ----------------------------------------------------------------------
#   ENDPOINT EDITS AND CACHING  (#7, #8, #10)
# ----------------------------------------------------------------------


def test_inverse_preserves_endpoint_edits() -> None:
    from brainhops.datamodel.systems import CoordinateSystem

    world = CoordinateSystem(name="world")
    df = DisplacementField(field=_small_field())
    inv = df.inverse()
    edited = inv.to(output=world)
    assert edited.output is world
    assert isinstance(edited, InverseDisplacementField)
    assert edited.forward is df
    back = edited.inverse()
    assert back.input is world


def test_materialization_cached_across_replace_and_to() -> None:
    df = DisplacementField(field=_small_field(seed=8))
    inv = df.inverse()
    first = np.asarray(inv.field)
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        rebuilt = inv.to(input=None)
        again = np.asarray(rebuilt.field)
    # The rebuilt wrapper reuses the cached materialization.
    assert calls["n"] == 0
    np.testing.assert_array_equal(first, again)


def test_compute_materializes_to_a_plain_instance() -> None:
    values = _small_field(seed=9)
    df = DisplacementField(
        field=values, degree=1, bound="nearest", store="values"
    )
    computed = df.inverse().compute()
    assert type(computed) is DisplacementField
    np.testing.assert_allclose(
        np.asarray(computed.field), inverse_disp(values)
    )


def test_to_plain_type_materializes() -> None:
    values = _small_field(seed=10)
    df = DisplacementField(
        field=values, degree=1, bound="nearest", store="values"
    )
    plain = df.inverse().to(DisplacementField)
    assert type(plain) is DisplacementField
    np.testing.assert_allclose(np.asarray(plain.field), inverse_disp(values))


# ----------------------------------------------------------------------
#   TRIVIAL INVERSES CANCEL AND MATERIALIZE
# ----------------------------------------------------------------------


def _trivial_cases() -> list:
    return [
        Translation(translation=[1.0, 2.0]),
        Scaling(scale=[2.0, 3.0]),
        Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]]),
        Permutation(permutation=[1, 0]),
    ]


@pytest.mark.parametrize("t", _trivial_cases())
def test_trivial_inverse_cancels_both_orders(t: Transformation) -> None:
    assert isinstance(
        Sequence(transformations=[t, t.inverse()]).compute(), Identity
    )
    assert isinstance(
        Sequence(transformations=[t.inverse(), t]).compute(), Identity
    )


@pytest.mark.parametrize("t", _trivial_cases())
def test_trivial_inverse_cancels_through_generic_inverse(
    t: Transformation,
) -> None:
    assert isinstance(
        Sequence(transformations=[t, Inverse(forward=t)]).compute(), Identity
    )
    assert isinstance(
        Sequence(transformations=[Inverse(forward=t), t]).compute(), Identity
    )


def test_trivial_inverse_materializes_to_closed_form() -> None:
    # Each materializes to the closed-form inverse of its parameter.
    negated = Translation(translation=[1.0, -2.0]).inverse().compute()
    assert type(negated) is Translation
    np.testing.assert_allclose(np.asarray(negated.translation), [-1.0, 2.0])

    reciprocal = Scaling(scale=[2.0, 4.0]).inverse().compute()
    assert type(reciprocal) is Scaling
    np.testing.assert_allclose(np.asarray(reciprocal.scale), [0.5, 0.25])

    transposed = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]]).inverse().compute()
    assert type(transposed) is Rotation
    np.testing.assert_allclose(
        np.asarray(transposed.matrix), [[0.0, 1.0], [-1.0, 0.0]]
    )

    argsorted = Permutation(permutation=[2, 0, 1]).inverse().compute()
    assert type(argsorted) is Permutation
    np.testing.assert_array_equal(np.asarray(argsorted.permutation), [1, 2, 0])


def test_distinct_trivial_inverse_does_not_falsely_cancel() -> None:
    # An inverse cancels only the exact transform it wraps.
    t1 = Translation(translation=[1.0, 2.0])
    t2 = Translation(translation=[10.0, 20.0])
    result = Sequence(transformations=[t1, t2.inverse()]).compute()
    assert isinstance(result, Translation)
    np.testing.assert_allclose(np.asarray(result.translation), [-9.0, -18.0])


# ----------------------------------------------------------------------
#   PUBLIC API
# ----------------------------------------------------------------------


def test_inverse_classes_are_public() -> None:
    for name in (
        "Inverse",
        "InverseTranslation",
        "InverseScaling",
        "InverseRotation",
        "InversePermutation",
        "InverseLinear",
        "InverseAffine",
        "InverseDisplacementField",
        "InverseCoordinatesField",
    ):
        assert name in _xf.__all__, name
        assert isinstance(getattr(_xf, name), type), name


def test_subclass_inherits_the_inverse_of_its_base() -> None:
    # Only base types are paired, so a subclass such as a format-specific
    # affine uses the inherited pairing.
    class MyAffine(Affine):
        pass

    class MyRefinedAffine(MyAffine):
        pass

    t = MyRefinedAffine(matrix=np.diag([2.0, 4.0, 1.0])[:2])
    inv = t.inverse()
    assert isinstance(inv, InverseAffine)
    np.testing.assert_allclose(
        np.asarray(inv.compute().matrix), [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0]]
    )


def test_inverse_comes_from_the_most_derived_paired_base() -> None:
    # Attribute lookup picks the nearest paired base.
    class MyRotation(Rotation):
        pass

    theta = np.pi / 3
    matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    inv = MyRotation(matrix=matrix).inverse()
    assert isinstance(inv, InverseRotation)
    np.testing.assert_allclose(
        np.asarray(inv.compute().matrix), np.transpose(matrix), atol=1e-12
    )


# ----------------------------------------------------------------------
#   PAIRED TRANSFORMATION TYPES
# ----------------------------------------------------------------------
# Paired types join the same spaces in opposite directions, and the class name
# states the direction. Inverting one must give the other on every path, or the
# name reads backwards.


PAIRS = [
    (VoxelToRAS, RASToVoxel, VoxelCoordinateSystem, RASCoordinateSystem),
    (RASToVoxel, VoxelToRAS, RASCoordinateSystem, VoxelCoordinateSystem),
    (VoxelToLPS, LPSToVoxel, VoxelCoordinateSystem, LPSCoordinateSystem),
    (LPSToVoxel, VoxelToLPS, LPSCoordinateSystem, VoxelCoordinateSystem),
]


@pytest.mark.parametrize("cls, reverse, source, target", PAIRS)
def test_materialized_inverse_is_the_paired_type(
    cls: type, reverse: type, source: type, target: type
) -> None:
    # Only the materialized result names a direction.
    matrix = np.diag([2.0, 4.0, 8.0, 1.0])[:3]
    inv = cls(matrix=matrix).inverse().compute()

    assert type(inv) is reverse
    assert isinstance(inv.input, target)
    assert isinstance(inv.output, source)
    np.testing.assert_allclose(
        np.asarray(inv.matrix), np.diag([0.5, 0.25, 0.125, 1.0])[:3]
    )


@pytest.mark.parametrize("cls, reverse, source, target", PAIRS)
def test_unset_parameter_inverse_is_the_paired_type(
    cls: type, reverse: type, source: type, target: type
) -> None:
    # Without a matrix the inverse is eager and must also name the swapped
    # direction.
    t = cls()
    assert isinstance(t.input, source)
    assert isinstance(t.output, target)

    inv = t.inverse()
    assert type(inv) is reverse
    assert isinstance(inv.input, target)
    assert isinstance(inv.output, source)


@pytest.mark.parametrize("cls, reverse, source, target", PAIRS)
def test_lazy_inverse_of_a_paired_type_keeps_its_endpoints(
    cls: type, reverse: type, source: type, target: type
) -> None:
    # The lazy wrapper has no direction of its own.
    matrix = np.diag([2.0, 4.0, 8.0, 1.0])[:3]
    t = cls(matrix=matrix)
    inv = t.inverse()

    assert isinstance(inv, InverseAffine)
    assert inv.forward is t
    assert inv.forward.matrix is matrix
    assert isinstance(inv.input, target)
    assert isinstance(inv.output, source)


@pytest.mark.parametrize("cls, reverse, source, target", PAIRS)
def test_paired_type_round_trips(
    cls: type, reverse: type, source: type, target: type
) -> None:
    matrix = np.diag([2.0, 4.0, 8.0, 1.0])[:3]
    there = cls(matrix=matrix).inverse().compute()
    back = there.inverse().compute()

    assert type(back) is cls
    assert isinstance(back.input, source)
    assert isinstance(back.output, target)
    np.testing.assert_allclose(np.asarray(back.matrix), matrix)


def test_a_pair_is_declared_once_and_resolved_both_ways() -> None:
    # Only the later-defined class can name the other...
    assert LPSToVoxel.__dict__["_reverseof"] is VoxelToLPS
    # ...and a hook writes the reverse onto VoxelToLPS.
    assert VoxelToLPS._reverseof is LPSToVoxel


def test_a_refinement_inherits_the_pairing_of_its_base() -> None:
    # The hook reads _reverseof only from a class's own body, so a refinement
    # does not steal the pairing.
    class MyVoxelToLPS(VoxelToLPS):
        pass

    assert MyVoxelToLPS._reverseof is LPSToVoxel
    assert LPSToVoxel._reverseof is VoxelToLPS

    inv = MyVoxelToLPS(matrix=np.diag([2.0, 4.0, 8.0, 1.0])[:3])
    assert type(inv.inverse().compute()) is LPSToVoxel


def test_an_unpaired_pinned_type_keeps_its_own_class() -> None:
    # Pairing is opt-in; without it the class inverts to itself.
    class PinnedRotation(Rotation):
        _input: CoordinateSystem = VoxelCoordinateSystem()
        _output: CoordinateSystem = RASCoordinateSystem()

    assert PinnedRotation._reverseof is None

    empty = PinnedRotation().inverse()
    assert type(empty) is PinnedRotation
    assert isinstance(empty.input, RASCoordinateSystem)
    assert isinstance(empty.output, VoxelCoordinateSystem)

    matrix = np.diag([0.0, -1.0, 1.0])[[1, 0, 2]]
    filled = PinnedRotation(matrix=matrix).inverse().compute()
    assert type(filled) is PinnedRotation
    assert isinstance(filled.input, RASCoordinateSystem)
    assert isinstance(filled.output, VoxelCoordinateSystem)
    np.testing.assert_allclose(np.asarray(filled.matrix), np.transpose(matrix))


@pytest.mark.parametrize("cls", [Affine, Rotation, Linear, Translation])
def test_an_unpaired_type_is_unchanged_on_every_path(cls: type) -> None:
    lps, ras = LPSCoordinateSystem(), RASCoordinateSystem()
    assert cls._reverseof is None

    empty = cls(input=lps, output=ras).inverse()
    assert type(empty) is cls
    assert empty.input == ras
    assert empty.output == lps

    values = {
        Affine: np.diag([2.0, 4.0, 8.0, 1.0])[:3],
        Rotation: np.diag([1.0, -1.0, -1.0]),
        Linear: np.diag([2.0, 4.0, 8.0]),
        Translation: np.asarray([1.0, 2.0, 3.0]),
    }
    t = cls(input=lps, output=ras, **{cls.data_fields[0]: values[cls]})

    lazy = t.inverse()
    assert isinstance(lazy, Inverse)
    assert lazy.forward is t
    assert lazy.input == ras
    assert lazy.output == lps

    materialized = lazy.compute()
    assert type(materialized) is cls
    assert materialized.input == ras
    assert materialized.output == lps


# ----------------------------------------------------------------------
#   A PRODUCT OF SUBSPACES (A SPACE-AND-TIME GEOMETRY)
# ----------------------------------------------------------------------


def _space_and_time() -> Sequence:
    # An affine over (x, y, z), then a scaling and a translation over t.
    from brainhops.datamodel.transformations import SubspaceTransformation

    spatial = Affine(
        matrix=np.array(
            [[0.0, -2.0, 0.0, 1.0], [1.5, 0.0, 0.0, 2.0], [0.0, 0.0, 3, 0]]
        )
    )
    temporal = Sequence(
        transformations=[
            Scaling(scale=[2.0]),
            Translation(translation=[0.5]),
        ]
    )
    full = CoordinateSystem().expand(4)
    return Sequence(
        transformations=[
            SubspaceTransformation(
                transformation=spatial,
                input_axes=[0, 1, 2],
                output_axes=[0, 1, 2],
                input=full,
                output=full,
            ),
            SubspaceTransformation(
                transformation=temporal,
                input_axes=[3],
                output_axes=[3],
                input=full,
                output=full,
            ),
        ]
    )


def test_a_subspace_product_cancels_its_inverse_by_identity() -> None:
    # The product cancels by object identity, without inverting anything.
    product = _space_and_time()
    with mock.patch.object(
        _concrete, "affine_inv", side_effect=AssertionError
    ):
        for chain in (
            [product, product.inverse()],
            [product.inverse(), product],
        ):
            result = Sequence(transformations=chain).compute(mode=False)
            assert isinstance(result, Identity)


def test_an_equal_subspace_product_does_not_cancel() -> None:
    # A distinct but equal product is not recognized (comparing values would
    # raise), so the chain is composed numerically.
    product, twin = _space_and_time(), _space_and_time()
    chain = Sequence(transformations=[product, twin.inverse()])
    assert not isinstance(chain.compute(mode=False), Identity)
    matrix = np.asarray(chain.to(Affine).matrix)
    assert np.allclose(matrix, np.eye(4, 5))


# ----------------------------------------------------------------------
#   INVERSES OF FIELDS AND OF IDENTITIES
# ----------------------------------------------------------------------


def test_the_coordinates_inverse_has_the_displacement_signature() -> None:
    # The inverse of a coordinates field takes only `forward`, `input`
    # and `output`, like the other lazy inverses.
    coords = inspect.signature(InverseCoordinatesField).parameters
    disp = inspect.signature(InverseDisplacementField).parameters
    assert list(coords) == list(disp)


def test_a_grid_inverse_keeps_the_flags_of_the_grid() -> None:
    # The inverse of a grid encodes its `data` with the same flags as
    # the grid.
    grid = CartesianField(
        (4, 5, 6), degree=3, bound="mirror", store="coefficients"
    )
    inverse = grid.inverse()
    assert inverse.degree == grid.degree
    assert inverse.bound == grid.bound
    assert inverse.store == grid.store
    np.testing.assert_allclose(inverse.data, grid.data)


def test_the_inverse_of_an_unset_coordinates_field_is_unset() -> None:
    # The inverse of a coordinates field without data has no data
    # either.
    inverse = InverseCoordinatesField(CoordinatesField())
    assert inverse.data is None


@pytest.mark.parametrize("cls", [DisplacementField, CoordinatesField])
def test_a_computed_identity_inverse_is_computed(cls: type) -> None:
    # The call `inverse(compute=True)` computes the inverse of an
    # identity, as `inverse().compute()` does.
    forward = cls()
    expected = type(forward.inverse().compute())
    assert type(forward.inverse(compute=True)) is expected
    assert expected is Identity


def test_a_computed_grid_inverse_passes_the_options_to_compute() -> None:
    # The inverse of a grid passes `compute` and its options on to
    # `compute()`.
    grid = CartesianField((4, 5, 6))
    with mock.patch.object(
        CartesianField, "compute", autospec=True, return_value=grid
    ) as compute:
        assert grid.inverse(compute=True, simplify="analytic") is grid
    compute.assert_called_once()
    assert compute.call_args.kwargs == {"simplify": "analytic"}
    assert type(grid.inverse(compute=True)) is CartesianField
