"""Tests for the polymorphic dispatch of axes, orientations and systems.

Calling a polymorphic data model class builds the most specific subclass
its arguments describe. These tests pin down, for every concrete class,
from which classes it is reached -- the root and the intermediate bases --
and that no call is ambiguous.
"""

import pytest
import typing_extensions as tx
from bagof.converters import ConversionError
from bagof.magic import PolymorphError

from brainhops.datamodel import axes as ax
from brainhops.datamodel import systems as cs
from brainhops.datamodel.orientation import (
    AnatomicalOrientation,
    AnteriorToPosterior,
    LeftToRight,
    Orientation,
    RightToLeft,
)
from brainhops.datamodel.units import MilliMeter, SampleUnit, Second

# ----------------------------------------------------------------------
#   AXES
# ----------------------------------------------------------------------

_ORIENTED = [
    (LeftToRight(), ax.LeftToRightAxis),
    (RightToLeft(), ax.RightToLeftAxis),
    (AnteriorToPosterior(), ax.AnteriorToPosteriorAxis),
]


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"type": "space"}, ax.SpaceAxis),
        ({"type": "space", "unit": "mm"}, ax.SpaceAxis),
        ({"type": "space", "unit": "sample"}, ax.SpaceAxis),
        ({"type": "time"}, ax.TimeAxis),
        ({"type": "time", "unit": "s"}, ax.TimeAxis),
        ({"type": "channel"}, ax.ChannelAxis),
        ({"type": "displacement"}, ax.DisplacementAxis),
        ({"type": "coordinate"}, ax.CoordinateAxis),
        ({"orientation": Orientation(value="up")}, ax.OrientedAxis),
        (
            {"type": "time", "orientation": Orientation(value="up")},
            ax.OrientedTimeAxis,
        ),
        (
            {"type": "space", "orientation": Orientation(value="up")},
            ax.OrientedSpaceAxis,
        ),
        *[({"orientation": o}, cls) for o, cls in _ORIENTED],
        *[({"type": "space", "orientation": o}, cls) for o, cls in _ORIENTED],
        ({"name": "x"}, ax.Axis),
    ],
)
def test_axis_root_reaches_every_axis(kwargs: dict, expected: type) -> None:
    assert type(ax.Axis(**kwargs)) is expected


@pytest.mark.parametrize(
    "base, kwargs, expected",
    [
        (
            ax.SpaceAxis,
            {"orientation": Orientation(value="up")},
            ax.OrientedSpaceAxis,
        ),
        (ax.SpaceAxis, {"orientation": LeftToRight()}, ax.LeftToRightAxis),
        (
            ax.TimeAxis,
            {"orientation": Orientation(value="up")},
            ax.OrientedTimeAxis,
        ),
        (
            ax.OrientedAxis,
            {"type": "space", "orientation": Orientation(value="up")},
            ax.OrientedSpaceAxis,
        ),
        (
            ax.OrientedSpaceAxis,
            {"orientation": RightToLeft()},
            ax.RightToLeftAxis,
        ),
        (
            ax.AnatomicalAxis,
            {"orientation": LeftToRight()},
            ax.LeftToRightAxis,
        ),
    ],
)
def test_intermediate_axes_reach_their_subclasses(
    base: type, kwargs: dict, expected: type
) -> None:
    assert type(base(**kwargs)) is expected


def test_an_anatomical_orientation_makes_a_spatial_axis() -> None:
    # An anatomical direction is a direction in space, so an axis that only
    # names one is spatial -- also when it counts samples (a voxel axis).
    axis = ax.Axis(orientation=LeftToRight(), unit="sample")
    assert type(axis) is ax.LeftToRightAxis
    assert axis.type == "space"
    assert isinstance(axis.unit, SampleUnit)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ax.Axis(type="space", unit="s"),
        lambda: ax.SpaceAxis(unit="s"),
        lambda: ax.Axis(type="time", unit="mm"),
        lambda: ax.TimeAxis(unit="mm"),
        lambda: ax.SpaceAxis(unit=Second()),
        lambda: ax.TimeAxis(unit=MilliMeter()),
        lambda: ax.SpaceAxis(type="time"),
        lambda: ax.ChannelAxis(type="space"),
        lambda: ax.LeftToRightAxis(orientation=RightToLeft()),
        # An anatomical orientation makes a spatial axis.
        lambda: ax.Axis(type="time", orientation=LeftToRight()),
        lambda: ax.Axis(type="channel", orientation=LeftToRight()),
        # A class selected on a value refuses one it is not selected on.
        lambda: ax.OrientedAxis(),
        lambda: ax.OrientedSpaceAxis(orientation=None),
        lambda: ax.OrientedTimeAxis(),
        lambda: ax.AnatomicalAxis(orientation=Orientation(value="up")),
    ],
)
def test_contradicting_axes_are_refused(build: tx.Callable) -> None:
    # A unit of the wrong kind fails the conversion of the field (a
    # `ConversionError`); a contradicting discriminant fails as a value.
    with pytest.raises((ValueError, ConversionError)):
        build()


def test_a_singleton_orientation_cannot_be_given_another_value() -> None:
    with pytest.raises(ValueError, match="left-to-right"):
        LeftToRight(value="right-to-left")
    # ... and the singleton was not changed on the way.
    assert LeftToRight().value == "left-to-right"
    assert ax.R().orientation.value == "left-to-right"


def test_a_singleton_orientation_cannot_be_subclassed() -> None:
    with pytest.raises(TypeError, match="singleton"):

        class _Other(LeftToRight):
            pass


def test_orientation_root_reaches_the_singletons() -> None:
    built = Orientation(type="anatomical", value="left-to-right")
    assert built is LeftToRight()
    assert type(AnatomicalOrientation(value="right-to-left")) is RightToLeft


# ----------------------------------------------------------------------
#   COORDINATE SYSTEMS
# ----------------------------------------------------------------------


def _space(n: int, unit: object = None) -> list:
    return [ax.SpaceAxis(name=f"x{i}", unit=unit) for i in range(n)]


def _plain(n: int, unit: object = None) -> list:
    return [ax.Axis(name=f"a{i}", unit=unit) for i in range(n)]


def _oriented(code: str, unit: object = None) -> list:
    return [getattr(ax, letter)(unit=unit) for letter in code]


@pytest.mark.parametrize(
    "axes, expected",
    [
        (_plain(2), cs.CoordinateSystem2D),
        (_plain(3), cs.CoordinateSystem3D),
        (_plain(4), cs.CoordinateSystem),
        (_space(2), cs.SpatialCoordinateSystem2D),
        (_space(3), cs.SpatialCoordinateSystem3D),
        (_space(4), cs.SpatialCoordinateSystem),
        (_plain(2, "sample"), cs.ArrayCoordinateSystem2D),
        (_plain(3, "sample"), cs.ArrayCoordinateSystem3D),
        (_space(2, "sample"), cs.PixelCoordinateSystem),
        (_space(3, "sample"), cs.VoxelCoordinateSystem),
        (_oriented("RAS"), cs.RASCoordinateSystem),
        (_oriented("LPS"), cs.LPSCoordinateSystem),
        (_oriented("RSA"), cs.RSACoordinateSystem),
        # An anatomical system says nothing about the metric: sampled axes
        # pointing R, A, S are still an RAS system.
        (_oriented("RAS", "sample"), cs.RASCoordinateSystem),
        (_oriented("RAS", "mm"), cs.RASmm),
        (_oriented("LPS", "mm"), cs.LPSmm),
        (_oriented("RSA", "mm"), cs.RSAmm),
        (_oriented("LAS"), cs.SpatialCoordinateSystem3D),
    ],
)
def test_the_root_reaches_every_generic_system(
    axes: list, expected: type
) -> None:
    assert type(cs.CoordinateSystem(axes=axes)) is expected


@pytest.mark.parametrize(
    "base, axes, expected",
    [
        (cs.CoordinateSystem2D, _space(2), cs.SpatialCoordinateSystem2D),
        (
            cs.CoordinateSystem2D,
            _plain(2, "sample"),
            cs.ArrayCoordinateSystem2D,
        ),
        (cs.CoordinateSystem3D, _oriented("RAS", "mm"), cs.RASmm),
        (cs.SpatialCoordinateSystem, _space(3), cs.SpatialCoordinateSystem3D),
        (cs.SpatialCoordinateSystem, _oriented("LPS"), cs.LPSCoordinateSystem),
        (
            cs.SpatialCoordinateSystem3D,
            _space(3, "sample"),
            cs.VoxelCoordinateSystem,
        ),
        (cs.SpatialCoordinateSystem3D, _oriented("RSA", "mm"), cs.RSAmm),
        (cs.RASCoordinateSystem, _oriented("RAS", "mm"), cs.RASmm),
        (cs.PhysicalCoordinateSystem, _oriented("RAS", "mm"), cs.RASmm),
        (cs.PhysicalCoordinateSystem, _oriented("LPS", "mm"), cs.LPSmm),
        (cs.ArrayCoordinateSystem, _plain(2), cs.ArrayCoordinateSystem2D),
        (cs.ArrayCoordinateSystem, _plain(3), cs.ArrayCoordinateSystem3D),
        (
            cs.ArrayCoordinateSystem2D,
            _space(2, "sample"),
            cs.PixelCoordinateSystem,
        ),
        (
            cs.ArrayCoordinateSystem3D,
            _space(3, "sample"),
            cs.VoxelCoordinateSystem,
        ),
        # The memory order is not written on the axes: the C- and
        # F-ordered classes are reached from their ordered bases only.
        (cs.CArrayCoordinateSystem, _plain(2), cs.CArrayCoordinateSystem2D),
        (cs.CArrayCoordinateSystem, _plain(3), cs.CArrayCoordinateSystem3D),
        (cs.FArrayCoordinateSystem, _plain(2), cs.FArrayCoordinateSystem2D),
        (cs.FArrayCoordinateSystem, _plain(3), cs.FArrayCoordinateSystem3D),
        (cs.CArrayCoordinateSystem2D, _space(2), cs.CPixelCoordinateSystem),
        (cs.FArrayCoordinateSystem2D, _space(2), cs.FPixelCoordinateSystem),
        (cs.CArrayCoordinateSystem3D, _space(3), cs.CVoxelCoordinateSystem),
        (cs.FArrayCoordinateSystem3D, _space(3), cs.FVoxelCoordinateSystem),
        (cs.FVoxelCoordinateSystem, _oriented("RAS"), cs.FRASCoordinateSystem),
        (cs.FVoxelCoordinateSystem, _oriented("LPS"), cs.FLPSCoordinateSystem),
        (cs.FVoxelCoordinateSystem, _oriented("RSA"), cs.FRSACoordinateSystem),
        # A C-ordered grid lists its axes z, y, x.
        (cs.CVoxelCoordinateSystem, _oriented("SAR"), cs.CRASCoordinateSystem),
        (cs.CVoxelCoordinateSystem, _oriented("SPL"), cs.CLPSCoordinateSystem),
        (cs.CVoxelCoordinateSystem, _oriented("ASR"), cs.CRSACoordinateSystem),
        (cs.FArrayCoordinateSystem, _oriented("RAS"), cs.FRASCoordinateSystem),
        (cs.CArrayCoordinateSystem, _oriented("SAR"), cs.CRASCoordinateSystem),
    ],
)
def test_intermediate_systems_reach_their_subclasses(
    base: type, axes: list, expected: type
) -> None:
    assert type(base(axes=axes)) is expected


# Every system class with a default, which `cs.__all__` lists among the
# other names it exports (the axis containers).
_DEFAULT_SYSTEMS = [
    name
    for name in cs.__all__
    if isinstance(getattr(cs, name), type)
    and issubclass(getattr(cs, name), cs.CoordinateSystem)
    and name != "PhysicalCoordinateSystem"
]


@pytest.mark.parametrize("name", _DEFAULT_SYSTEMS)
def test_every_system_builds_itself_by_default(name: str) -> None:
    # No class is ambiguous with another when it is built from its own
    # defaults: it is what it says it is.
    cls = getattr(cs, name)
    assert type(cls()) is cls


@pytest.mark.parametrize(
    "name",
    [
        "ArrayCoordinateSystem2D",
        "ArrayCoordinateSystem3D",
        "CArrayCoordinateSystem2D",
        "CArrayCoordinateSystem3D",
        "FArrayCoordinateSystem2D",
        "FArrayCoordinateSystem3D",
        "PixelCoordinateSystem",
        "VoxelCoordinateSystem",
        "CPixelCoordinateSystem",
        "FPixelCoordinateSystem",
        "CVoxelCoordinateSystem",
        "FVoxelCoordinateSystem",
        "FRASCoordinateSystem",
        "FLPSCoordinateSystem",
        "FRSACoordinateSystem",
        "CRASCoordinateSystem",
        "CLPSCoordinateSystem",
        "CRSACoordinateSystem",
    ],
)
def test_array_systems_count_samples_by_default(name: str) -> None:
    system = getattr(cs, name)()
    assert all(isinstance(axis.unit, SampleUnit) for axis in system.axes)


def test_ras_dispatch_follows_the_orientations() -> None:
    assert type(cs.CoordinateSystem(axes=[ax.R(), ax.A(), ax.S()])) is (
        cs.RASCoordinateSystem
    )
    # A system that is not RAS-oriented is not read as one, even in mm.
    with pytest.raises(ValueError, match="RASCoordinateSystem.axes"):
        cs.RASCoordinateSystem(axes=_oriented("LPS", "mm"))


def test_a_physical_system_needs_physical_axes() -> None:
    with pytest.raises(ValueError, match="must have axes"):
        cs.PhysicalCoordinateSystem()
    with pytest.raises(ValueError, match="must have axes"):
        cs.PhysicalCoordinateSystem(axes=[])
    with pytest.raises(ValueError, match="unit that measures"):
        cs.RASmm(axes=_oriented("RAS"))
    with pytest.raises(ValueError, match="unit that measures"):
        cs.RASmm(axes=_oriented("RAS", "sample"))


def test_no_system_is_ambiguous() -> None:
    # Every combination of axis kinds the classes select on is decided.
    kinds = [
        _plain(2),
        _plain(3),
        _space(2),
        _space(3),
        _plain(2, "sample"),
        _plain(3, "sample"),
        _space(2, "sample"),
        _space(3, "sample"),
        _space(3, "mm"),
        _oriented("RAS", "sample"),
        _oriented("SAR"),
        _oriented("RAS", "mm"),
    ]
    roots = [
        getattr(cs, name)
        for name in cs.__all__
        if name != "PhysicalCoordinateSystem"
    ]
    for root in roots:
        for axes in kinds:
            try:
                root(axes=axes)
            except PolymorphError as e:  # pragma: no cover
                pytest.fail(f"{root.__name__}: {e}")
            except (TypeError, ValueError):
                # A fixed-arity or oriented class refuses foreign axes.
                pass


# ----------------------------------------------------------------------
#   NO SHARED AXIS INSTANCES
# ----------------------------------------------------------------------

_SHORT = {
    "R": ax.LeftToRightAxis,
    "LR": ax.LeftToRightAxis,
    "L": ax.RightToLeftAxis,
    "RL": ax.RightToLeftAxis,
    "A": ax.PosteriorToAnteriorAxis,
    "PA": ax.PosteriorToAnteriorAxis,
    "P": ax.AnteriorToPosteriorAxis,
    "AP": ax.AnteriorToPosteriorAxis,
    "S": ax.InferiorToSuperiorAxis,
    "IS": ax.InferiorToSuperiorAxis,
    "I": ax.SuperiorToInferiorAxis,
    "SI": ax.SuperiorToInferiorAxis,
}


@pytest.mark.parametrize("short, cls", _SHORT.items())
def test_short_axis_names_are_classes(short: str, cls: type) -> None:
    # `axes.R` is the class, not a shared instance: each call builds a new
    # axis, and `isinstance(axis, R)` reads as it should.
    assert getattr(ax, short) is cls
    first, second = cls(), cls(unit="mm")
    assert first is not second
    assert isinstance(first, getattr(ax, short))


def test_the_axes_module_holds_no_axis_instance() -> None:
    assert not [
        name for name, value in vars(ax).items() if isinstance(value, ax.Axis)
    ]


@pytest.mark.parametrize("name", _DEFAULT_SYSTEMS)
def test_two_default_systems_share_no_axis_with_another_class(
    name: str,
) -> None:
    # A default system holds axes of its own class's making: no axis
    # object is shared with the default of another system class.
    system = getattr(cs, name)()
    # (`...`, which an open system defaults to, is no axis but the one
    # `Ellipsis`.)
    mine = {id(axis) for axis in system.axes if axis is not ...}
    for other in _DEFAULT_SYSTEMS:
        if other == name:
            continue
        theirs = getattr(cs, other)().axes
        assert not mine & {id(axis) for axis in theirs if axis is not ...}


def test_the_singleton_orientations_are_frozen() -> None:
    # Every axis that points left-to-right holds the one `LeftToRight`, so
    # it cannot be changed through any of them.
    with pytest.raises(AttributeError):
        ax.R().orientation.value = "right-to-left"
    with pytest.raises(AttributeError):
        cs.RASmm().axes[0].orientation.value = "right-to-left"
    assert LeftToRight().value == "left-to-right"
