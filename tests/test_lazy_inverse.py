"""Tests for the lazy inverse of transformations.

These cover which families defer their inversion to a type-transparent
lazy wrapper and which stay eager, the preservation of ``order``,
``bound`` and ``coeff`` across an inversion, double-inverse cancellation,
adjacent transform/inverse cancellation in a ``Sequence``, and the
equivalence of a materialized lazy inverse with the former eager inverse.
"""

import numpy as np
import pytest

from brainhops._ext.invfield import inverse as inverse_disp
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Sequence,
    Translation,
    _LazyInverse,
    _LazyInverseCoordinatesField,
    _LazyInverseDisplacementField,
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
    assert isinstance(inv, _LazyInverseDisplacementField)
    # It stays an instance of its family, so the compose engine and the
    # kind checks treat it transparently.
    assert isinstance(inv, DisplacementField)
    assert inv.operand is df


def test_coordinates_field_inverse_is_lazy() -> None:
    cf = CoordinatesField(field=_small_field())
    inv = cf.inverse()
    assert isinstance(inv, _LazyInverseCoordinatesField)
    assert isinstance(inv, CoordinatesField)
    assert inv.operand is cf


def test_empty_field_inverse_stays_eager() -> None:
    # With no field there is nothing to invert, so the inverse is the
    # eager endpoint-swapped transform, not a lazy wrapper.
    df = DisplacementField()
    assert not isinstance(df.inverse(), _LazyInverse)
    assert type(df.inverse()) is DisplacementField
    cf = CoordinatesField()
    assert not isinstance(cf.inverse(), _LazyInverse)
    assert type(cf.inverse()) is CoordinatesField


def test_trivial_families_stay_eager() -> None:
    cases = [
        Identity(),
        Translation(translation=[1.0, 2.0]),
        Scaling(scale=[2.0, 3.0]),
        Permutation(permutation=[1, 0]),
        Linear(matrix=[[2.0, 0.0], [0.0, 4.0]]),
        Affine(matrix=[[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]]),
        Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]]),
    ]
    for t in cases:
        inv = t.inverse()
        assert not isinstance(inv, _LazyInverse), type(t).__name__
        assert type(inv) is type(t), type(t).__name__


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


def test_coefficient_inverse_is_not_materialized() -> None:
    # Inverting spline coefficients directly would approximate the field,
    # so a forced materialization is refused rather than approximated.
    df = DisplacementField(field=_small_field(), order=3, coeff=True)
    with pytest.raises(NotImplementedError):
        _ = df.inverse().field


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
    assert isinstance(~df, _LazyInverseDisplacementField)
    assert (~df).operand is df


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
    assert isinstance(result, _LazyInverseDisplacementField)
