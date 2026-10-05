"""Tests for the stored `data` of concrete transformations and their views.

Every concrete transformation stores one array, `data`, plus the flags
that say how it is encoded. Its named views (`field`, `matrix`, `scale`,
...) are always the map, as values. The views are checked under every
encoding, with cubic splines for the fields: at the default linear
degree, spline coefficients equal the values and would hide a view that
reads the stored array as values.
"""

import numpy as np
import pytest

from brainhops._core.bsplines import value2coeff_field
from brainhops.datamodel.transformations import (
    CoordinatesField,
    DisplacementField,
)

DEGREE = 3
BOUND = "nearest"
FIELDS = (DisplacementField, CoordinatesField)


def _values(shape: tuple = (12, 13, 2), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _small(shape: tuple = (8, 9, 2), seed: int = 0) -> np.ndarray:
    # Small enough for the mesh inversion to be well behaved.
    return 0.1 * np.random.default_rng(seed).normal(size=shape)


def _coefficients(values: np.ndarray, **options) -> np.ndarray:
    options = {"degree": DEGREE, "bound": BOUND, **options}
    return np.asarray(value2coeff_field(values, **options))


def _grid(shape: tuple) -> np.ndarray:
    return np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1
    )


# ----------------------------------------------------------------------
#   DISPLACEMENTS TO COORDINATES  (#294)
# ----------------------------------------------------------------------


def test_coordinates_from_coefficients_match_those_from_values() -> None:
    # The reproducer of #294: on main the two differed by 10.7, and the
    # result lost its encoding.
    u = np.random.default_rng(0).normal(size=(12, 13, 2))
    d = DisplacementField(field=u, degree=3)
    c = d.to(coeff=True)
    a = d.to(CoordinatesField)
    b = c.to(CoordinatesField)
    np.testing.assert_allclose(
        np.asarray(b.field), np.asarray(a.field), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(a.field), u + _grid((12, 13)))
    assert (a.coeff, a.degree, a.bound) == (False, d.degree, d.bound)
    assert (b.coeff, b.degree, b.bound) == (True, c.degree, c.bound)


@pytest.mark.parametrize("bound", ["nearest", "constant", "reflect"])
def test_coordinates_keep_the_encoding_of_the_displacements(
    bound: str,
) -> None:
    u = _values()
    c = DisplacementField(field=u, degree=DEGREE, bound=bound).to(coeff=True)
    b = c.to(CoordinatesField)
    assert type(b) is CoordinatesField
    assert (b.coeff, b.degree, b.bound) == (True, DEGREE, c.bound)
    np.testing.assert_allclose(
        np.asarray(b.field), u + _grid(u.shape[:-1]), atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(b.data),
        _coefficients(u + _grid(u.shape[:-1]), bound=bound),
        atol=1e-10,
    )


def test_an_unset_displacement_converts_to_unset_coordinates() -> None:
    d = DisplacementField(degree=DEGREE, coeff=True)
    c = d.to(CoordinatesField)
    assert c.data is None
    assert (c.coeff, c.degree) == (True, DEGREE)
