"""Tests for the lazy inverse of transformations.

These cover which families defer their inversion to a type-transparent
lazy wrapper and which stay eager, the preservation of ``order``,
``bound`` and ``coeff`` across an inversion, double-inverse cancellation,
adjacent transform/inverse cancellation in a ``Sequence``, and the
equivalence of a materialized lazy inverse with the former eager inverse.
"""

from unittest import mock

import numpy as np
import pytest

from brainhops._ext.invfield import inverse as inverse_disp
from brainhops.datamodel import transformations as _xf
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
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


def _small_field(shape: tuple = (6, 7, 2), seed: int = 0) -> np.ndarray:
    # A small, low-amplitude displacement field whose mesh inversion is
    # well behaved.
    rng = np.random.RandomState(seed)
    return rng.randn(*shape) * 0.05


# ----------------------------------------------------------------------
#   LAZY VS EAGER SELECTION
# ----------------------------------------------------------------------


def test_displacement_field_inverse_is_lazy() -> None:
    df = DisplacementField(field=_small_field())
    inv = df.inverse()
    assert isinstance(inv, InverseDisplacementField)
    # It stays an instance of its family, so the compose engine and the
    # kind checks treat it transparently.
    assert isinstance(inv, DisplacementField)
    assert inv.forward is df


def test_coordinates_field_inverse_is_lazy() -> None:
    cf = CoordinatesField(field=_small_field())
    inv = cf.inverse()
    assert isinstance(inv, InverseCoordinatesField)
    assert isinstance(inv, CoordinatesField)
    assert inv.forward is cf


def test_empty_field_inverse_stays_eager() -> None:
    # With no field there is nothing to invert, so the inverse is the
    # eager endpoint-swapped transform, not a lazy wrapper.
    df = DisplacementField()
    assert not isinstance(df.inverse(), Inverse)
    assert type(df.inverse()) is DisplacementField
    cf = CoordinatesField()
    assert not isinstance(cf.inverse(), Inverse)
    assert type(cf.inverse()) is CoordinatesField


def test_meta_families_stay_eager() -> None:
    # Identity is its own inverse, and a Bijection carries its own inverse,
    # so neither defers to an `Inverse` wrapper.
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
    # A translation, scaling, rotation and permutation each defer their
    # inverse to a type-transparent wrapper. The inverse remains an
    # instance of the family it inverts, so it cancels symbolically next to
    # the forward transform instead of composing numerically.
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
    # The general affine/linear family defers its inverse to a lazy,
    # type-transparent wrapper, so an affine placed next to its own inverse
    # cancels symbolically instead of composing to a numerically-identity
    # matrix.
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
    # A[-1] @ A collapses to the identity by symbolic cancellation, without
    # ever computing the matrix inverse.
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


def test_displacement_inverse_preserves_order_coeff_bound() -> None:
    df = DisplacementField(
        field=_small_field(), order=3, bound=2.0, coeff=True
    )
    inv = df.inverse()
    assert inv.order == 3
    assert inv.bound == 2.0
    assert inv.coeff is True
    # The endpoints are swapped.
    assert inv.input is df.output
    assert inv.output is df.input


def test_coordinates_inverse_preserves_order_coeff_bound() -> None:
    cf = CoordinatesField(field=_small_field(), order=2, bound=1.0, coeff=True)
    inv = cf.inverse()
    assert inv.order == 2
    assert inv.bound == 1.0
    assert inv.coeff is True


# ----------------------------------------------------------------------
#   MATERIALIZATION EQUALS THE FORMER EAGER INVERSE
# ----------------------------------------------------------------------


def test_materialized_field_matches_eager_inverse() -> None:
    values = _small_field(seed=3)
    df = DisplacementField(field=values, order=1, bound="nearest", coeff=False)
    inv = df.inverse()
    np.testing.assert_allclose(np.asarray(inv.field), inverse_disp(values))


def test_materialized_field_is_cached() -> None:
    df = DisplacementField(field=_small_field(seed=4))
    inv = df.inverse()
    first = inv.field
    assert inv.field is first


def test_coefficient_inverse_is_a_coefficient_field() -> None:
    # The inverse of a coefficient field is itself a coefficient field. The
    # metadata carries across without materializing anything.
    df = DisplacementField(
        field=_small_field(), order=3, bound=2.0, coeff=True
    )
    inv = df.inverse()
    assert inv.coeff is True
    assert inv.order == 3
    assert inv.bound == 2.0


@pytest.mark.xfail(
    reason="#63: bsplines coeff2value/value2coeff are broken on this base; "
    "the coeff re-fit round-trip is fixed in a separate session.",
    strict=False,
)
def test_coefficient_inverse_refits_to_coefficients() -> None:
    # A coefficient field is inverted by re-fitting: coeff -> value ->
    # inverse -> coeff. The numeric round-trip depends on the corrected
    # bsplines conversions (#63).
    order, bound = 3, "nearest"
    values = _small_field(seed=7)
    df = DisplacementField(field=values, order=order, bound=bound, coeff=False)
    coeff = df.to(coeff=True)
    materialized = coeff.inverse().field
    expected = inverse_disp(values)
    # Reading the coefficient inverse back as values should recover the
    # inverse displacement field.
    from brainhops._core.bsplines import coeff2value_field

    recovered = coeff2value_field(
        np.asarray(materialized), order=order, bound=bound
    )
    np.testing.assert_allclose(np.asarray(recovered), expected, atol=1e-6)


def test_coordinate_inverse_is_not_materialized() -> None:
    cf = CoordinatesField(field=_small_field())
    with pytest.raises(NotImplementedError):
        _ = cf.inverse().field


# ----------------------------------------------------------------------
#   DOUBLE INVERSE
# ----------------------------------------------------------------------


def test_double_inverse_returns_operand() -> None:
    df = DisplacementField(
        field=_small_field(), order=3, coeff=True, bound=2.0
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
    # A coefficient field cannot be inverted without approximation, so if
    # cancellation touched the field it would raise. It collapses to the
    # identity instead, which proves the pair is removed before any
    # numeric inversion.
    df = DisplacementField(field=_small_field(), order=3, coeff=True)
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
    # inverse(inverse(X)) is X, so the pair [inverse(X), inverse(inverse(X))]
    # is [inverse(X), X] and cancels.
    df = DisplacementField(field=_small_field())
    inv = df.inverse()
    result = Sequence(transformations=[inv, inv.inverse()]).compute()
    assert isinstance(result, Identity)


# ----------------------------------------------------------------------
#   TYPE TRANSPARENCY (composition materializes the same result)
# ----------------------------------------------------------------------


def test_composed_lazy_inverse_equals_composed_eager_inverse() -> None:
    values = _small_field(seed=5)
    df = DisplacementField(field=values, order=1, bound="nearest", coeff=False)
    t = Translation(translation=[1.0, 2.0])

    lazy = Sequence(transformations=[df.inverse(), t]).compute()
    eager = DisplacementField(
        field=inverse_disp(values), order=1, bound="nearest", coeff=False
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
    # A displacement level as built by the OME-Zarr reader (#57): the
    # world-to-voxel affine, the displacement in voxel units, and the
    # voxel-to-world affine.
    voxel2world = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    world2voxel = voxel2world.inverse()
    field = DisplacementField(field=_small_field())
    level = Sequence(transformations=[world2voxel, field, voxel2world])
    return level, field


def test_level_inv_at_level_cancels_zero_field_inversions() -> None:
    # level^-1 @ level must collapse to the identity with ZERO field
    # inversions: the affines and the field each cancel against their own
    # lazy inverse symbolically, before any numeric inversion runs.
    level, _ = _displacement_level()
    calls = {"n": 0}
    real = _xf.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_xf, "inverse_disp", counting):
        result = (level.inverse() @ level).compute()

    assert isinstance(result, Identity)
    assert calls["n"] == 0


def test_level_at_level_inv_cancels_zero_field_inversions() -> None:
    # The other order, level @ level^-1, cancels the same way.
    level, _ = _displacement_level()
    calls = {"n": 0}
    real = _xf.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_xf, "inverse_disp", counting):
        result = (level @ level.inverse()).compute()

    assert isinstance(result, Identity)
    assert calls["n"] == 0


# ----------------------------------------------------------------------
#   is_identity IS COST-FREE ON A LAZY INVERSE  (#2)
# ----------------------------------------------------------------------


def test_is_identity_compute_false_does_not_materialize() -> None:
    # A coefficient inverse cannot be materialized on this base (its
    # re-fit goes through the broken bsplines, #63), and a coordinate
    # inverse never can. is_identity(compute=False) must answer from the
    # operand without reading the lazy field, so it must not raise.
    for operand in (
        DisplacementField(field=_small_field(), order=3, coeff=True),
        CoordinatesField(field=_small_field()),
    ):
        inv = operand.inverse()
        assert is_identity(inv, compute=False) is False


def test_is_identity_recognizes_an_empty_lazy_inverse() -> None:
    # The inverse of an identity-valued field is the identity. With no
    # field there is no wrapper, but a wrapper whose operand is empty is
    # recognized from the operand.
    inv = InverseDisplacementField(forward=DisplacementField())
    assert is_identity(inv, compute=False) is True


# ----------------------------------------------------------------------
#   CANCELLATION AFTER GRID DROP AND TO A FIXPOINT  (#4)
# ----------------------------------------------------------------------


def test_cancellation_after_interior_grid_drop() -> None:
    # An interior CartesianField grid sits between a field and its inverse.
    # The grid is dropped first, which makes the pair adjacent, and then
    # they cancel. Cancellation must run after the grid drop.
    from brainhops.datamodel.transformations import CartesianField

    df = DisplacementField(field=_small_field())
    grid = CartesianField(shape=(6, 7))
    # Two outer transforms keep the grid strictly interior.
    a = Translation(translation=[1.0, 2.0])
    b = Translation(translation=[3.0, 4.0])
    seq = Sequence(transformations=[a, df, grid, df.inverse(), b])
    result = seq.compute()
    # a and b survive and combine; the field pair cancels across the grid.
    assert isinstance(result, (Translation, Identity))


def test_cancellation_reaches_a_fixpoint() -> None:
    # Nested inverse pairs collapse fully in one compute, even when an
    # interior grid separates a pair.
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
    inv = df.inverse()  # input=voxel, output=world
    result = Sequence(transformations=[df, inv]).compute()
    assert isinstance(result, Identity)
    # The identity runs from the first element's input to the last
    # element's output, not from the (unset) sequence endpoints.
    assert result.input is world
    assert result.output is world


# ----------------------------------------------------------------------
#   PUBLIC Inverse RECONCILIATION  (#6)
# ----------------------------------------------------------------------


def test_public_inverse_wrapper_cancels_in_a_sequence() -> None:
    # A generic Inverse(forward=X) placed next to X is expanded into X's
    # typed inverse and cancels.
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
    # The endpoint edit survives, and the wrapper stays lazy.
    assert edited.output is world
    assert isinstance(edited, InverseDisplacementField)
    assert edited.forward is df
    # Its own inverse reflects the edited endpoint rather than dropping it.
    back = edited.inverse()
    assert back.input is world


def test_materialization_cached_across_replace_and_to() -> None:
    df = DisplacementField(field=_small_field(seed=8))
    inv = df.inverse()
    first = np.asarray(inv.field)
    calls = {"n": 0}
    real = _xf.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_xf, "inverse_disp", counting):
        rebuilt = inv.to(input=None)  # a plain endpoint edit
        again = np.asarray(rebuilt.field)
    # The rebuilt wrapper reused the cached materialization on the operand.
    assert calls["n"] == 0
    np.testing.assert_array_equal(first, again)


def test_compute_materializes_to_a_plain_instance() -> None:
    values = _small_field(seed=9)
    df = DisplacementField(field=values, order=1, bound="nearest", coeff=False)
    computed = df.inverse().compute()
    assert type(computed) is DisplacementField
    np.testing.assert_allclose(
        np.asarray(computed.field), inverse_disp(values)
    )


def test_to_plain_type_materializes() -> None:
    values = _small_field(seed=10)
    df = DisplacementField(field=values, order=1, bound="nearest", coeff=False)
    plain = df.inverse().to(DisplacementField)
    assert type(plain) is DisplacementField
    np.testing.assert_allclose(np.asarray(plain.field), inverse_disp(values))


def test_coordinate_inverse_reports_a_clear_message() -> None:
    cf = CoordinatesField(field=_small_field())
    with pytest.raises(NotImplementedError, match="coordinate field"):
        cf.inverse().compute()


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
    # A translation, scaling, rotation or permutation placed next to its
    # own inverse collapses to the identity by symbolic cancellation, the
    # same way an affine does.
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
    # The same cancellation holds for the generic Inverse(forward=X) front
    # door placed next to X.
    assert isinstance(
        Sequence(transformations=[t, Inverse(forward=t)]).compute(), Identity
    )
    assert isinstance(
        Sequence(transformations=[Inverse(forward=t), t]).compute(), Identity
    )


def test_trivial_inverse_materializes_to_closed_form() -> None:
    # Each trivial inverse materializes cheaply to the closed-form inverse
    # of its parameter, and the materialized transform is a plain instance
    # of the forward family.
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
    # An inverse only cancels against the exact transform it wraps. Two
    # different translations do not cancel, and their sequence composes to
    # the difference of the two.
    t1 = Translation(translation=[1.0, 2.0])
    t2 = Translation(translation=[10.0, 20.0])
    result = Sequence(transformations=[t1, t2.inverse()]).compute()
    assert isinstance(result, Translation)
    np.testing.assert_allclose(np.asarray(result.translation), [-9.0, -18.0])


# ----------------------------------------------------------------------
#   PUBLIC API
# ----------------------------------------------------------------------


def test_inverse_classes_are_public() -> None:
    # Every typed inverse, and the generic front door, are part of the
    # module's public API and reachable by name.
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
