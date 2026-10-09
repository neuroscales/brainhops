"""Regression tests for the b-spline coefficient conversions.

The value/coefficient round trip in :mod:`brainhops._core.bsplines` was
broken in two ways that no test covered:

* ``value2coeff`` forwarded a ``cval`` keyword to ``spline_filter``, which
  does not accept it (``TypeError``).
* ``coeff2value`` built its sampling grid from the *batch* dimensions
  instead of the *spatial* ones, so ``map_coordinates`` rejected the
  coordinate array (``RuntimeError``).

Together these made ``DisplacementField.to(store=...)`` and
``CoordinatesField.to(store=...)`` raise for any field carrying data, and
therefore made composing any field of coefficients fail (the composers
call ``Ti.to(store="values")``).
"""

import numpy as np
import pytest

from brainhops.datamodel._transformations.compute.compose import compose
from brainhops.datamodel.transformations import (
    CoordinatesField,
    DisplacementField,
)

BOUNDS = ["nearest", "reflect", "mirror", "grid-wrap", "wrap"]
DEGREES = [2, 3]
NDIMS = [2, 3]
FIELD_TYPES = [DisplacementField, CoordinatesField]


def _random_field(rng, ndim):  # noqa: ANN001, ANN202
    """A small, smooth displacement field of shape (*spatial, ndim)."""
    spatial = (6, 7, 5)[:ndim]
    return rng.standard_normal((*spatial, ndim)) * 0.05


@pytest.mark.parametrize("field_type", FIELD_TYPES)
@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("bound", BOUNDS)
def test_value_coeff_round_trip(
    field_type,  # noqa: ANN001
    ndim,  # noqa: ANN001
    degree,  # noqa: ANN001
    bound,  # noqa: ANN001
) -> None:
    # value -> coeff -> value must recover the original samples exactly (up
    # to spline-filter numerical error) for every string boundary condition.
    rng = np.random.default_rng(0)
    values = _random_field(rng, ndim)
    field = field_type(field=values.copy(), degree=degree, bound=bound)

    coeffs = field.to(store="coefficients")
    assert coeffs.store == "coefficients"

    recovered = coeffs.to(store="values")
    assert recovered.store == "values"
    np.testing.assert_allclose(np.asarray(recovered.field), values, atol=1e-6)


def test_repro_from_report() -> None:
    # Verbatim reproduction from the bug report.
    f = np.random.RandomState(0).randn(6, 7, 2) * 0.05
    D = DisplacementField(field=f, degree=3)
    C = D.to(store="coefficients")
    V = C.to(store="values")
    assert np.allclose(np.asarray(V.field), f, atol=1e-6)


@pytest.mark.parametrize("degree", DEGREES)
def test_compose_coeff_fields_does_not_raise(degree) -> None:  # noqa: ANN001
    # Composition routes through ``Ti.to(store="values")``, so a broken
    # coefficient conversion made composing any field of coefficients
    # fail.
    rng = np.random.default_rng(1)
    d1 = DisplacementField(field=_random_field(rng, 2), degree=degree).to(
        store="coefficients"
    )
    d2 = DisplacementField(field=_random_field(rng, 2), degree=degree).to(
        store="coefficients"
    )

    out = compose(d1, d2)
    assert isinstance(out, DisplacementField)


# ----------------------------------------------------------------------
#   value2coeff INVERTS coeff2value, FOR EVERY BOUND AND DEGREE (#250)
# ----------------------------------------------------------------------

ALL_BOUNDS = [*BOUNDS, "constant", 0.0, 2.5]
ALL_DEGREES = [2, 3, 4, 5]


def _backends() -> list:
    from brainhops.backends import available_backends

    return [
        "numpy",
        pytest.param(
            "dask",
            marks=pytest.mark.skipif(
                "dask" not in available_backends(),
                reason="dask is not installed",
            ),
        ),
    ]


@pytest.mark.parametrize("array_backend", _backends())
@pytest.mark.parametrize(
    "shape", [(5,), (12, 14, 16), (97, 3), (230, 7)], ids=str
)
@pytest.mark.parametrize("degree", ALL_DEGREES)
@pytest.mark.parametrize("bound", ALL_BOUNDS, ids=str)
def test_value2coeff_inverts_coeff2value(
    bound,  # noqa: ANN001
    degree: int,
    shape: tuple,
    array_backend: str,
) -> None:
    """
    The coefficients `value2coeff` returns are those `coeff2value`
    interpolates back to the values, under every bound, at every degree,
    on axes shorter and longer than the band near each end that a bound
    changes.

    scipy's prefilter extends the values the way its evaluation extends
    the coefficients for `mirror`, `grid-wrap` and `wrap` only: for
    `nearest` and constant bounds the round trip was off by up to 10 at
    degree five, and for `reflect` scipy approximates it, by up to 1e-4.
    """
    from brainhops._core.bsplines import coeff2value, value2coeff

    values = np.random.default_rng(14).standard_normal(shape)
    array = values
    if array_backend == "dask":
        import dask.array as da

        array = da.from_array(values, chunks=[max(1, s // 3) for s in shape])
    coeff = value2coeff(array, degree, bound)
    if array_backend == "dask":
        assert coeff.chunks == array.chunks
    back = coeff2value(coeff, degree, bound)
    np.testing.assert_allclose(np.asarray(back), values, atol=1e-9)


@pytest.mark.parametrize("degree", ALL_DEGREES)
def test_a_nearest_spline_takes_its_edge_coefficient_past_the_edge(
    degree: int,
) -> None:
    """
    Past the edges a `nearest` spline is its edge coefficient, which is
    what interpolates the edge value: the clamped coefficients, evaluated
    at the edge sample and beyond, agree with it there.
    """
    from brainhops._core.bsplines import pull, value2coeff

    values = np.random.default_rng(15).standard_normal(20)
    coeff = value2coeff(values, degree, "nearest")
    far = pull(coeff, np.array([[-30.0], [50.0]]), degree, "nearest", True)
    np.testing.assert_allclose(far, coeff[[0, -1]], atol=1e-12)
    edges = pull(coeff, np.array([[0.0], [19.0]]), degree, "nearest", True)
    np.testing.assert_allclose(edges, values[[0, -1]], atol=1e-12)
