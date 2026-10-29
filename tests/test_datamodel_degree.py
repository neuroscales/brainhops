"""Tests that the spline degree of fields and reslices is called `degree`.

See #283.
"""

import numpy as np
import pytest

from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import (
    CartesianField,
    CoordinateField,
    DisplacementField,
)
from brainhops.io.transformations.base.fields import (
    LPSCoordinateField,
    RASCoordinateField,
)

FIELDS = [
    DisplacementField,
    CoordinateField,
    CartesianField,
    RASCoordinateField,
    LPSCoordinateField,
]


@pytest.mark.parametrize("cls", FIELDS)
def test_a_field_takes_a_degree(cls: type) -> None:
    assert cls(degree=3).degree == 3
    assert cls().degree == 1


def test_a_field_is_rebuilt_with_a_degree() -> None:
    field = DisplacementField(field=np.zeros((3, 4, 2)), degree=3)
    assert field.to(degree=1).degree == 1


def test_the_degree_is_shown_in_the_repr() -> None:
    assert "degree=" in repr(DisplacementField(degree=3))


def test_a_reslice_takes_a_degree() -> None:
    image = SingleScaleImage(data=np.zeros((4, 5)))
    assert image.reslice(degree=0).data.shape == (4, 5)
    MultiScaleImage(images=[image]).reslice(degree=0)


@pytest.mark.parametrize("degree", [6, 7])
def test_degrees_six_and_seven_are_accepted(degree: int) -> None:
    # The spline coefficients of a field of degree 6 or 7 give back its
    # values, and a reslice at that degree keeps the data of an image.
    rng = np.random.default_rng(degree)
    values = rng.standard_normal((5, 6, 2))
    field = DisplacementField(field=values, degree=degree)
    coefficients = field.to(store="coefficients")
    assert coefficients.degree == degree
    np.testing.assert_allclose(coefficients.field, values, atol=1e-10)
    data = rng.standard_normal((4, 5))
    image = SingleScaleImage(data=data)
    np.testing.assert_allclose(image.reslice(degree=degree).data, data)
