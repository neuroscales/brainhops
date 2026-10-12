"""Tests that the spline degree of fields and reslices is called `degree`.

See #283.
"""

import numpy as np
import pytest

from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import (
    CartesianField,
    CoordinatesField,
    DisplacementField,
)
from brainhops.io.transformations.base.fields import (
    LPSCoordinatesField,
    RASCoordinatesField,
)

FIELDS = [
    DisplacementField,
    CoordinatesField,
    CartesianField,
    RASCoordinatesField,
    LPSCoordinatesField,
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
