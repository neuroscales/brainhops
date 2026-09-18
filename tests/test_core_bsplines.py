"""Regression tests for the b-spline coefficient conversions.

The value/coefficient round trip in :mod:`brainhops._core.bsplines` was
broken in two ways that no test covered:

* ``value2coeff`` forwarded a ``cval`` keyword to ``spline_filter``, which
  does not accept it (``TypeError``).
* ``coeff2value`` built its sampling grid from the *batch* dimensions
  instead of the *spatial* ones, so ``map_coordinates`` rejected the
  coordinate array (``RuntimeError``).

Together these made ``DisplacementField.to(coeff=...)`` and
``CoordinatesField.to(coeff=...)`` raise for any field carrying data, and
therefore made composing any ``coeff=True`` field fail (the composers call
``Ti.to(coeff=False)``).
"""

import numpy as np
import pytest

from brainhops.datamodel._transformations.compose import compose
from brainhops.datamodel.transformations import (
    CoordinatesField,
    DisplacementField,
)

BOUNDS = ["nearest", "reflect", "mirror", "grid-wrap", "wrap"]
ORDERS = [2, 3]
NDIMS = [2, 3]
FIELD_TYPES = [DisplacementField, CoordinatesField]


def _random_field(rng, ndim):  # noqa: ANN001, ANN202
    """A small, smooth displacement field of shape (*spatial, ndim)."""
    spatial = (6, 7, 5)[:ndim]
    return rng.standard_normal((*spatial, ndim)) * 0.05


@pytest.mark.parametrize("field_type", FIELD_TYPES)
@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("bound", BOUNDS)
def test_value_coeff_round_trip(
    field_type,  # noqa: ANN001
    ndim,  # noqa: ANN001
    order,  # noqa: ANN001
    bound,  # noqa: ANN001
) -> None:
    # value -> coeff -> value must recover the original samples exactly (up
    # to spline-filter numerical error) for every string boundary condition.
    rng = np.random.default_rng(0)
    values = _random_field(rng, ndim)
    field = field_type(field=values.copy(), order=order, bound=bound)

    coeffs = field.to(coeff=True)
    assert coeffs.coeff is True

    recovered = coeffs.to(coeff=False)
    assert recovered.coeff is False
    np.testing.assert_allclose(np.asarray(recovered.field), values, atol=1e-6)


def test_repro_from_report() -> None:
    # Verbatim reproduction from the bug report.
    f = np.random.RandomState(0).randn(6, 7, 2) * 0.05
    D = DisplacementField(field=f, order=3)
    C = D.to(coeff=True)
    V = C.to(coeff=False)
    assert np.allclose(np.asarray(V.field), f, atol=1e-6)


@pytest.mark.parametrize("order", ORDERS)
def test_compose_coeff_fields_does_not_raise(order) -> None:  # noqa: ANN001
    # Composition routes through ``Ti.to(coeff=False)``, so a broken
    # coefficient conversion made composing any ``coeff=True`` field fail.
    rng = np.random.default_rng(1)
    d1 = DisplacementField(field=_random_field(rng, 2), order=order).to(
        coeff=True
    )
    d2 = DisplacementField(field=_random_field(rng, 2), order=order).to(
        coeff=True
    )

    out = compose(d1, d2)
    assert isinstance(out, DisplacementField)
