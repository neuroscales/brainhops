"""Regression tests for the conversion between B-spline values and
coefficients.

The conversion used to raise for any field holding data, in both
directions, so that `DisplacementField.to(store=...)` and
`CoordinatesField.to(store=...)` failed, and with them the composition of
coefficient fields.
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
    """Return a small smooth displacement field."""
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
    # Converting values to coefficients and back recovers the samples, for
    # every bound.
    rng = np.random.default_rng(0)
    values = _random_field(rng, ndim)
    field = field_type(field=values.copy(), degree=degree, bound=bound)

    coeffs = field.to(store="coefficients")
    assert coeffs.store == "coefficients"

    recovered = coeffs.to(store="values")
    assert recovered.store == "values"
    np.testing.assert_allclose(np.asarray(recovered.field), values, atol=1e-6)


def test_repro_from_report() -> None:
    # The original bug report, verbatim.
    f = np.random.RandomState(0).randn(6, 7, 2) * 0.05
    D = DisplacementField(field=f, degree=3)
    C = D.to(store="coefficients")
    V = C.to(store="values")
    assert np.allclose(np.asarray(V.field), f, atol=1e-6)


@pytest.mark.parametrize("degree", DEGREES)
def test_compose_coeff_fields_does_not_raise(degree) -> None:  # noqa: ANN001
    # Composition converts its inputs with `to(store="values")`.
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
    """Interpolating the coefficients gives back the values.

    The check covers every bound and degree, with axes both shorter and longer
    than the band that a bound alters at each end. The scipy prefilter matches
    its own evaluation only for mirror, grid-wrap and wrap; it was off by up to
    10 for nearest and constant at degree 5, and approximate for reflect.
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
    """Past the edges, a nearest spline equals its edge coefficient."""
    from brainhops._core.bsplines import pull, value2coeff

    values = np.random.default_rng(15).standard_normal(20)
    coeff = value2coeff(values, degree, "nearest")
    far = pull(coeff, np.array([[-30.0], [50.0]]), degree, "nearest", True)
    np.testing.assert_allclose(far, coeff[[0, -1]], atol=1e-12)
    edges = pull(coeff, np.array([[0.0], [19.0]]), degree, "nearest", True)
    np.testing.assert_allclose(edges, values[[0, -1]], atol=1e-12)
