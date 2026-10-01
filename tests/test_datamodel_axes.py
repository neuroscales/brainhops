"""Tests for axes, and for the vector-axis classification of a field's
axes."""

import re

import pytest
from bagof.converters import ConversionError

from brainhops.datamodel.axes import (
    Axis,
    AxisError,
    CoordinateAxis,
    DisplacementAxis,
    LeftToRightAxis,
    SpatialAxis,
    vector_axis,
)
from brainhops.datamodel.orientation import RightToLeft
from brainhops.datamodel.systems import (
    CoordinateSystem3D,
    SpatialCoordinateSystem3D,
)


def _messages(error: BaseException) -> list:
    # Every message in the chain of causes of `error`, including the
    # failed branches that a union keeps in `causes`.
    seen, todo, out = set(), [error], []
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        out.append(str(e))
        todo.append(e.__cause__)
        todo.extend(getattr(e, "causes", None) or ())
    return out


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


def test_ellipsis_is_not_an_axis() -> None:
    with pytest.raises(TypeError, match=re.escape("`...` is not an axis")):
        Axis(...)
    with pytest.raises(TypeError, match=re.escape("`...` is not an axis")):
        SpatialAxis(name=...)


def test_a_fixed_dimension_system_refuses_ellipsis_as_an_axis() -> None:
    # Regression: the `...` was taken as the name of a third axis, and
    # refused with a message about strings.
    with pytest.raises(ConversionError) as info:
        CoordinateSystem3D(axes=[Axis(), Axis(), ...])
    assert any("`...` is not an axis" in m for m in _messages(info.value))


def test_generic_axes_are_read_as_spatial_axes() -> None:
    # Regression: each generic `Axis` was taken as the name of a
    # `SpatialAxis`, so the system could not be built.
    system = SpatialCoordinateSystem3D(
        axes=[Axis(name="x"), Axis(name="y"), {"name": "z"}]
    )
    assert all(type(axis) is SpatialAxis for axis in system.axes)
    assert [axis.name for axis in system.axes] == ["x", "y", "z"]


def test_an_axis_with_another_orientation_is_not_read_as_oriented() -> None:
    with pytest.raises(ValueError, match="always LeftToRight"):
        LeftToRightAxis.from_other(Axis(name="x", orientation=RightToLeft()))


def test_an_axis_of_another_type_is_not_read_as_spatial() -> None:
    with pytest.raises(ValueError, match="always 'space'"):
        SpatialAxis.from_other(Axis(name="t", type="time"))
