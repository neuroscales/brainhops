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
    SpaceAxis,
    TimeAxis,
    vector_axis,
)
from brainhops.datamodel.orientation import (
    LeftToRight,
    Orientation,
    RightToLeft,
)
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
        SpaceAxis(name=...)


def test_a_fixed_dimension_system_refuses_ellipsis_as_an_axis() -> None:
    # Regression: the `...` was taken as the name of a third axis, and
    # refused with a message about strings.
    with pytest.raises(ConversionError) as info:
        CoordinateSystem3D(axes=[Axis(), Axis(), ...])
    assert any("`...` is not an axis" in m for m in _messages(info.value))


def test_generic_axes_are_read_as_spatial_axes() -> None:
    # Regression: each generic `Axis` was taken as the name of a
    # `SpaceAxis`, so the system could not be built.
    system = SpatialCoordinateSystem3D(
        axes=[Axis(name="x"), Axis(name="y"), {"name": "z"}]
    )
    assert all(type(axis) is SpaceAxis for axis in system.axes)
    assert [axis.name for axis in system.axes] == ["x", "y", "z"]
    # A generic axis leaves its unit unset, so the spatial default holds:
    # a spatial axis claims no unit unless it is given one.
    assert all(axis.unit is None for axis in system.axes)


def test_an_axis_with_the_same_orientation_is_read_as_oriented() -> None:
    # The generic axis has no name: the oriented axis keeps its own.
    axis = LeftToRightAxis.from_other(Axis(orientation=LeftToRight()))
    assert axis == LeftToRightAxis()


def test_an_axis_with_another_orientation_is_not_read_as_oriented() -> None:
    with pytest.raises(ValueError, match="always LeftToRight"):
        LeftToRightAxis.from_other(Axis(name="x", orientation=RightToLeft()))


def test_an_axis_of_another_type_is_not_read_as_spatial() -> None:
    with pytest.raises(ValueError, match="always 'space'"):
        SpaceAxis.from_other(Axis(name="t", type="time"))


def test_an_axis_of_another_unit_kind_is_not_read_as_spatial() -> None:
    # A spatial axis measured in seconds is a contradiction. It is refused
    # wherever it is written -- not quietly built as a generic `Axis`, and
    # not quietly read as the sample either (a second instance used to
    # fall through the `Union[SpaceUnit, SampleUnit]` to `SampleUnit`).
    from brainhops.datamodel.units import Second, Unit

    for build in (
        lambda: Axis(name="x", type="space", unit="s"),
        lambda: SpaceAxis(name="x", unit="s"),
        lambda: SpaceAxis(name="x", unit=Unit("s")),
        lambda: SpaceAxis(name="x", unit=Second()),
    ):
        with pytest.raises(ConversionError, match="SpaceAxis.unit"):
            build()
    for build in (
        lambda: Axis(name="t", type="time", unit="mm"),
        lambda: TimeAxis(name="t", unit=Unit("mm")),
    ):
        with pytest.raises(ConversionError, match="TimeAxis.unit"):
            build()


def test_a_sibling_axis_is_read_field_by_field() -> None:
    # Calling `Axis` builds the subclass its arguments select, so an axis
    # that only carries an orientation is an `OrientedAxis` -- a sibling of
    # `SpaceAxis`, not a parent. It is still read field by field, rather
    # than taken as the name of the new axis.
    orientation = Orientation(value="toward-the-light")
    axis = Axis(name="x", orientation=orientation, unit="mm")
    assert not isinstance(axis, SpaceAxis)
    read = SpaceAxis.from_other(axis)
    assert isinstance(read, SpaceAxis)
    assert (read.name, read.unit) == ("x", axis.unit)
    assert read.orientation == orientation
