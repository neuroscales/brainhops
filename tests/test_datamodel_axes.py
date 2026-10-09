"""Tests of axes: the vector axis of a field, and the compatibility and
merging of two descriptions of the same axis.
"""

import re

import pytest
from bagof.converters import ConversionError

from brainhops.datamodel._sugar import vector_axis
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    LeftToRightAxis,
    RightToLeftAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.orientations import (
    LeftToRight,
    Orientation,
    RightToLeft,
)
from brainhops.datamodel.systems import (
    CoordinateSystem3D,
    SpatialCoordinateSystem3D,
)
from brainhops.errors import AxisError


def _messages(error: BaseException) -> list:
    # Every message in the chain of causes, including the failed union
    #
    # branches kept in `causes`.
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
    # `...` used to be read as the name of a third axis.
    with pytest.raises(ConversionError) as info:
        CoordinateSystem3D(axes=[Axis(), Axis(), ...])
    assert any("`...` is not an axis" in m for m in _messages(info.value))


def test_generic_axes_are_read_as_spatial_axes() -> None:
    # Each generic axis used to be read as the name of a spatial axis.
    system = SpatialCoordinateSystem3D(
        axes=[Axis(name="x"), Axis(name="y"), {"name": "z"}]
    )
    assert all(type(axis) is SpaceAxis for axis in system.axes)
    assert [axis.name for axis in system.axes] == ["x", "y", "z"]
    # A spatial axis states no unit unless one is given.
    assert all(axis.unit is None for axis in system.axes)


def test_an_axis_with_the_same_orientation_is_read_as_oriented() -> None:
    # The generic axis has no name, so the oriented axis keeps its own.
    axis = LeftToRightAxis.from_any(Axis(orientation=LeftToRight()))
    assert axis == LeftToRightAxis()


def test_an_axis_with_another_orientation_is_not_read_as_oriented() -> None:
    with pytest.raises(ValueError, match="always LeftToRight"):
        LeftToRightAxis.from_any(Axis(name="x", orientation=RightToLeft()))


def test_an_axis_of_another_type_is_not_read_as_spatial() -> None:
    with pytest.raises(ValueError, match="always 'space'"):
        SpaceAxis.from_any(Axis(name="t", type="time"))


def test_an_axis_of_another_unit_kind_is_not_read_as_spatial() -> None:
    # A spatial axis measured in seconds is refused however it is written.
    from brainhops.datamodel.units import Unit

    for build in (
        lambda: Axis(name="x", type="space", unit="s"),
        lambda: SpaceAxis(name="x", unit="s"),
        lambda: SpaceAxis(name="x", unit=Unit("s")),
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
    # An axis with an arbitrary orientation is a sibling of `SpaceAxis`, and
    #
    # is still read field by field rather than as the name of the new axis.
    orientation = Orientation(value="toward-the-light")
    axis = Axis(name="x", orientation=orientation, unit="mm")
    assert not isinstance(axis, SpaceAxis)
    read = SpaceAxis.from_any(axis)
    assert isinstance(read, SpaceAxis)
    assert (read.name, read.unit) == ("x", axis.unit)
    assert read.orientation == orientation


# ----------------------------------------------------------------------
#   COMPATIBLE AND MERGE
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second",
    [
        (Axis(), Axis()),
        (Axis(), SpaceAxis()),
        (Axis(), LeftToRightAxis(name="x")),
        (Axis(name="x"), Axis(name="x", unit="mm")),
        (Axis(type="space"), SpaceAxis(name="x")),
        (SpaceAxis(), Axis(unit="millimeter")),
        (LeftToRightAxis(), Axis(orientation=LeftToRight())),
        (Axis(discrete=True), Axis(name="c", discrete=True)),
    ],
)
def test_axes_that_agree_on_every_shared_field_are_compatible(
    first: Axis, second: Axis
) -> None:
    assert first.compatible_with(second)
    assert second.compatible_with(first)


@pytest.mark.parametrize(
    "first, second",
    [
        (Axis(name="x"), Axis(name="y")),
        (Axis(type="space"), TimeAxis()),
        (SpaceAxis(unit="mm"), Axis(unit="micrometer")),
        (LeftToRightAxis(), RightToLeftAxis()),
        (Axis(discrete=True), Axis(discrete=False)),
        (TimeAxis(), ChannelAxis()),
    ],
)
def test_axes_that_disagree_on_a_shared_field_are_not_compatible(
    first: Axis, second: Axis
) -> None:
    assert not first.compatible_with(second)
    assert not second.compatible_with(first)


def test_compatibility_is_not_transitive() -> None:
    x, y = Axis(name="x"), Axis(name="y")
    assert Axis().compatible_with(x) and Axis().compatible_with(y)
    assert not x.compatible_with(y)


@pytest.mark.parametrize("other", [None, ..., "x", {"name": "x"}])
def test_compatible_refuses_a_non_axis(other: object) -> None:
    with pytest.raises(TypeError):
        Axis().compatible_with(other)  # type: ignore[arg-type]


def test_an_axis_with_defaults_is_not_unknown() -> None:
    # Only `Axis()` is unknown. A `SpaceAxis` states its type, but its unit
    #
    # stays None, and thus compatible with any unit, until one is given.
    assert SpaceAxis() != Axis()
    assert SpaceAxis(unit=None) != Axis()
    assert not SpaceAxis().compatible_with(Axis(type="time"))
    assert SpaceAxis().compatible_with(Axis(unit="micrometer"))
    assert not SpaceAxis(unit="mm").compatible_with(Axis(unit="micrometer"))


def test_merge_combines_the_known_fields() -> None:
    merged = Axis(name="x", discrete=False).merge_with(Axis(unit="mm"))
    assert merged == Axis(name="x", unit="mm", discrete=False)


@pytest.mark.parametrize(
    "axis", [Axis(name="x"), SpaceAxis(), LeftToRightAxis(name="x")]
)
def test_merge_with_the_unknown_axis_is_the_other_axis(axis: Axis) -> None:
    assert Axis().merge_with(axis) == axis
    assert axis.merge_with(Axis()) == axis


def test_merge_keeps_the_more_derived_class() -> None:
    merged = Axis(name="x").merge_with(SpaceAxis(unit="micrometer"))
    assert type(merged) is SpaceAxis
    assert merged == SpaceAxis(name="x", unit="micrometer")
    merged = LeftToRightAxis().merge_with(Axis(type="space", discrete=False))
    assert type(merged) is LeftToRightAxis
    assert merged.discrete is False


def test_merge_is_symmetric_on_compatible_axes() -> None:
    a = Axis(name="x", discrete=True)
    b = SpaceAxis(unit="micrometer")
    assert a.merge_with(b) == b.merge_with(a)


def test_merge_does_not_change_its_operands() -> None:
    a, b = Axis(name="x"), Axis(unit="mm")
    a.merge_with(b)
    assert a == Axis(name="x") and b == Axis(unit="mm")


@pytest.mark.parametrize(
    "first, second, field",
    [
        (Axis(name="x"), Axis(name="y"), "name"),
        (SpaceAxis(unit="mm"), Axis(unit="micrometer"), "unit"),
        (Axis(type="space"), TimeAxis(), "type"),
        (LeftToRightAxis(name="x"), RightToLeftAxis(name="x"), "orientation"),
    ],
)
def test_merge_refuses_a_conflict(
    first: Axis, second: Axis, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        first.merge_with(second)


def test_merge_refuses_unrelated_classes() -> None:
    class FirstAxis(Axis):
        """A test-local axis class."""

    class SecondAxis(Axis):
        """An unrelated test-local axis class."""

    with pytest.raises(ValueError, match="neither class derives"):
        FirstAxis(name="x").merge_with(SecondAxis(name="x"))


def test_merge_refuses_a_non_axis() -> None:
    with pytest.raises(TypeError):
        Axis().merge_with(...)  # type: ignore[arg-type]
