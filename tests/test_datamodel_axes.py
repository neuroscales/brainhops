"""Tests for axes: the vector-axis classification of a field's axes, and
the compatibility and merging of two descriptions of one axis."""

import pytest

from brainhops.datamodel.axes import (
    Axis,
    AxisError,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    LeftToRightAxis,
    RightToLeftAxis,
    SpatialAxis,
    TimeAxis,
    vector_axis,
)
from brainhops.datamodel.orientation import LeftToRight


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


# ----------------------------------------------------------------------
#   COMPATIBLE AND MERGE
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second",
    [
        (Axis(), Axis()),
        (Axis(), SpatialAxis()),
        (Axis(), LeftToRightAxis(name="x")),
        (Axis(name="x"), Axis(name="x", unit="mm")),
        (Axis(type="space"), SpatialAxis(name="x")),
        (SpatialAxis(), Axis(unit="millimeter")),
        (LeftToRightAxis(), Axis(orientation=LeftToRight())),
        (Axis(discrete=True), Axis(name="c", discrete=True)),
    ],
)
def test_axes_that_agree_on_every_shared_field_are_compatible(
    first: Axis, second: Axis
) -> None:
    assert first.compatible(second)
    assert second.compatible(first)


@pytest.mark.parametrize(
    "first, second",
    [
        (Axis(name="x"), Axis(name="y")),
        (Axis(type="space"), TimeAxis()),
        (SpatialAxis(), Axis(unit="micrometer")),
        (LeftToRightAxis(), RightToLeftAxis()),
        (Axis(discrete=True), Axis(discrete=False)),
        (TimeAxis(), ChannelAxis()),
    ],
)
def test_axes_that_disagree_on_a_shared_field_are_not_compatible(
    first: Axis, second: Axis
) -> None:
    assert not first.compatible(second)
    assert not second.compatible(first)


def test_compatibility_is_not_transitive() -> None:
    x, y = Axis(name="x"), Axis(name="y")
    assert Axis().compatible(x) and Axis().compatible(y)
    assert not x.compatible(y)


@pytest.mark.parametrize("other", [None, ..., "x", {"name": "x"}])
def test_compatible_refuses_a_non_axis(other: object) -> None:
    with pytest.raises(TypeError):
        Axis().compatible(other)  # type: ignore[arg-type]


def test_an_axis_with_defaults_is_not_unknown() -> None:
    # Only a plain `Axis()` is unknown. A `SpatialAxis` states its type and
    # its default unit, so it is neither equal to `Axis()` nor compatible
    # with an axis of another type or unit.
    assert SpatialAxis() != Axis()
    assert SpatialAxis(unit=None) != Axis()
    assert not SpatialAxis().compatible(Axis(type="time"))
    assert not SpatialAxis().compatible(Axis(unit="micrometer"))


def test_merge_combines_the_known_fields() -> None:
    merged = Axis(name="x", discrete=False).merge(Axis(unit="mm"))
    assert merged == Axis(name="x", unit="mm", discrete=False)


@pytest.mark.parametrize(
    "axis", [Axis(name="x"), SpatialAxis(), LeftToRightAxis(name="x")]
)
def test_merge_with_the_unknown_axis_is_the_other_axis(axis: Axis) -> None:
    assert Axis().merge(axis) == axis
    assert axis.merge(Axis()) == axis


def test_merge_keeps_the_more_derived_class() -> None:
    merged = Axis(name="x").merge(SpatialAxis(unit="micrometer"))
    assert type(merged) is SpatialAxis
    assert merged == SpatialAxis(name="x", unit="micrometer")
    merged = LeftToRightAxis().merge(Axis(type="space", discrete=False))
    assert type(merged) is LeftToRightAxis
    assert merged.discrete is False


def test_merge_is_symmetric_on_compatible_axes() -> None:
    a = Axis(name="x", discrete=True)
    b = SpatialAxis(unit="micrometer")
    assert a.merge(b) == b.merge(a)


def test_merge_does_not_change_its_operands() -> None:
    a, b = Axis(name="x"), Axis(unit="mm")
    a.merge(b)
    assert a == Axis(name="x") and b == Axis(unit="mm")


@pytest.mark.parametrize(
    "first, second, field",
    [
        (Axis(name="x"), Axis(name="y"), "name"),
        (SpatialAxis(), Axis(unit="micrometer"), "unit"),
        (Axis(type="space"), TimeAxis(), "type"),
        (LeftToRightAxis(name="x"), RightToLeftAxis(name="x"), "orientation"),
    ],
)
def test_merge_refuses_a_conflict(
    first: Axis, second: Axis, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        first.merge(second)


def test_merge_refuses_unrelated_classes() -> None:
    class FirstAxis(Axis):
        """An axis class of its own."""

    class SecondAxis(Axis):
        """Another one, unrelated to the first."""

    with pytest.raises(ValueError, match="neither class derives"):
        FirstAxis(name="x").merge(SecondAxis(name="x"))


def test_merge_refuses_a_non_axis() -> None:
    with pytest.raises(TypeError):
        Axis().merge(...)  # type: ignore[arg-type]
