"""The spline degree is `degree`; `order` was its name before #283.

A field and a reslice refuse an integer `order` with a pointer to the new
name. Any other `order` is refused as the unknown keyword it is: `order`
now only names the memory order of an array coordinate system.
"""

import numpy as np
import pytest
from bagof.magic import replace

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

RENAMED = "`order` was renamed to `degree`"


@pytest.mark.parametrize("cls", FIELDS)
def test_a_field_takes_a_degree(cls: type) -> None:
    assert cls(degree=3).degree == 3
    assert cls().degree == 1


@pytest.mark.parametrize("cls", FIELDS)
@pytest.mark.parametrize("order", [0, 3, np.int64(2)])
def test_a_field_refuses_an_integer_order(cls: type, order: int) -> None:
    with pytest.raises(TypeError, match=RENAMED):
        cls(order=order)


@pytest.mark.parametrize("cls", FIELDS)
def test_a_field_refuses_a_memory_order_as_unknown(cls: type) -> None:
    with pytest.raises(TypeError, match="unexpected keyword") as info:
        cls(order="F")
    assert "renamed" not in str(info.value)


def test_a_field_rebuilt_with_an_order_is_refused() -> None:
    # `replace` and `to` check the names against the fields first.
    field = DisplacementField(field=np.zeros((3, 4, 2)), degree=3)
    with pytest.raises(TypeError, match="no field named 'order'"):
        replace(field, order=1)
    with pytest.raises(TypeError, match="no field named 'order'"):
        field.to(order=1)
    assert field.to(degree=1).degree == 1


def test_the_degree_is_shown_in_the_repr() -> None:
    assert "degree=" in repr(DisplacementField(degree=3))
    assert "order" not in repr(DisplacementField(degree=3))


def test_a_reslice_refuses_an_integer_order() -> None:
    image = SingleScaleImage(data=np.zeros((4, 5)))
    assert image.reslice(degree=0).data.shape == (4, 5)
    with pytest.raises(TypeError, match=RENAMED):
        image.reslice(order=0)
    pyramid = MultiScaleImage(images=[image])
    with pytest.raises(TypeError, match=RENAMED):
        pyramid.reslice(order=0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        image.reslice(order="C")
