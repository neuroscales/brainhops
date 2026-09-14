"""Tests for the vector-axis classification of a field's axes."""

import pytest

from brainhops.datamodel.axes import (
    Axis,
    AxisError,
    CoordinateAxis,
    DisplacementAxis,
    vector_axis,
)


def test_displacement_axis_has_a_fixed_type() -> None:
    assert DisplacementAxis().type == "displacement"


def test_coordinate_axis_has_a_fixed_type() -> None:
    assert CoordinateAxis().type == "coordinate"


def test_a_single_displacement_axis_is_classified_and_located() -> None:
    axes = [Axis(type="space"), Axis(type="space"), Axis(type="displacement")]
    assert vector_axis(axes) == ("displacement", 2)


def test_a_single_coordinate_axis_is_classified_and_located() -> None:
    axes = [Axis(type="space"), Axis(type="space"), Axis(type="coordinate")]
    assert vector_axis(axes) == ("coordinate", 2)


def test_the_vector_axis_need_not_be_last() -> None:
    axes = [Axis(type="displacement"), Axis(type="space"), Axis(type="space")]
    assert vector_axis(axes) == ("displacement", 0)


def test_axis_subclasses_are_recognized_by_type() -> None:
    axes = [Axis(type="space"), Axis(type="space"), DisplacementAxis()]
    assert vector_axis(axes) == ("displacement", 2)


def test_mixed_displacement_and_coordinate_axes_are_refused() -> None:
    axes = [
        Axis(type="space"),
        Axis(type="displacement"),
        Axis(type="coordinate"),
    ]
    with pytest.raises(AxisError):
        vector_axis(axes)


def test_axes_with_no_vector_axis_are_refused() -> None:
    with pytest.raises(AxisError):
        vector_axis([Axis(type="space"), Axis(type="space")])


def test_repeated_vector_axes_are_refused() -> None:
    axes = [
        Axis(type="displacement"),
        Axis(type="space"),
        Axis(type="displacement"),
    ]
    with pytest.raises(AxisError):
        vector_axis(axes)
