"""Tests of the polymorphic dispatch of axes, orientations and systems.

Calling a polymorphic class builds the most specific subclass that its
arguments describe, and no call may be ambiguous.
"""

import pytest
import typing_extensions as tx
from bagof.converters import ConversionError
from bagof.magic import PolymorphError

from brainhops.datamodel import axes as ax
from brainhops.datamodel import systems as cs
from brainhops.datamodel.orientations import (
    AnatomicalOrientation,
    AnteriorToPosterior,
    LeftToRight,
    Orientation,
    RightToLeft,
)
from brainhops.datamodel.units import IndexUnit, Unit

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
        ({"type": "space", "unit": "index"}, ax.SpaceAxis),
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
    # An anatomical direction is a direction in space, so an axis that names
    #
    # one is spatial, even when the axis counts samples.
    axis = ax.Axis(orientation=LeftToRight(), unit="index")
    assert type(axis) is ax.LeftToRightAxis
    assert axis.type == "space"
    assert isinstance(axis.unit, IndexUnit)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ax.Axis(type="space", unit="s"),
        lambda: ax.SpaceAxis(unit="s"),
        lambda: ax.Axis(type="time", unit="mm"),
        lambda: ax.TimeAxis(unit="mm"),
        lambda: ax.SpaceAxis(unit=Unit("s")),
        lambda: ax.TimeAxis(unit=Unit("mm")),
        lambda: ax.SpaceAxis(type="time"),
        lambda: ax.ChannelAxis(type="space"),
        lambda: ax.LeftToRightAxis(orientation=RightToLeft()),
        lambda: ax.Axis(type="time", orientation=LeftToRight()),
        lambda: ax.Axis(type="channel", orientation=LeftToRight()),
        # A class selected on a value refuses any other value.
        lambda: ax.OrientedAxis(),
        lambda: ax.OrientedSpaceAxis(orientation=None),
        lambda: ax.OrientedTimeAxis(),
        lambda: ax.AnatomicalAxis(orientation=Orientation(value="up")),
    ],
)
def test_contradicting_axes_are_refused(build: tx.Callable) -> None:
    # A unit of the wrong kind fails the conversion of the field.
    with pytest.raises((ValueError, ConversionError)):
        build()


def test_a_singleton_orientation_cannot_be_given_another_value() -> None:
    with pytest.raises(ValueError, match="left-to-right"):
        LeftToRight(value="right-to-left")
    # The singleton is not changed in passing.
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
        (_plain(2, "index"), cs.ArrayCoordinateSystem2D),
        (_plain(3, "index"), cs.ArrayCoordinateSystem3D),
        (_space(2, "index"), cs.PixelCoordinateSystem),
        (_space(3, "index"), cs.VoxelCoordinateSystem),
        (_oriented("RAS"), cs.RASCoordinateSystem),
        (_oriented("LPS"), cs.LPSCoordinateSystem),
        (_oriented("RSA"), cs.RSACoordinateSystem),
        # An anatomical system ignores the metric of its axes.
        (_oriented("RAS", "index"), cs.RASCoordinateSystem),
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
            _plain(2, "index"),
            cs.ArrayCoordinateSystem2D,
        ),
        (cs.CoordinateSystem3D, _oriented("RAS", "mm"), cs.RASmm),
        (cs.SpatialCoordinateSystem, _space(3), cs.SpatialCoordinateSystem3D),
        (cs.SpatialCoordinateSystem, _oriented("LPS"), cs.LPSCoordinateSystem),
        (
            cs.SpatialCoordinateSystem3D,
            _space(3, "index"),
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
            _space(2, "index"),
            cs.PixelCoordinateSystem,
        ),
        (
            cs.ArrayCoordinateSystem3D,
            _space(3, "index"),
            cs.VoxelCoordinateSystem,
        ),
        # An ordered base reaches the classes of its order from the axes alone.
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
        # A C-ordered grid lists its axes as z, y, x.
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


# Every system class that has a default.
_DEFAULT_SYSTEMS = [
    name
    for name in cs.__all__
    if isinstance(getattr(cs, name), type)
    and issubclass(getattr(cs, name), cs.CoordinateSystem)
    and name != "PhysicalCoordinateSystem"
]


@pytest.mark.parametrize("name", _DEFAULT_SYSTEMS)
def test_every_system_builds_itself_by_default(name: str) -> None:
    # No class built from its own defaults is ambiguous with another.
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
    assert all(isinstance(axis.unit, IndexUnit) for axis in system.axes)


def test_ras_dispatch_follows_the_orientations() -> None:
    assert type(cs.CoordinateSystem(axes=[ax.R(), ax.A(), ax.S()])) is (
        cs.RASCoordinateSystem
    )
    # A system that is not oriented as RAS is not read as RAS, even in mm.
    with pytest.raises(ValueError, match="RASCoordinateSystem.axes"):
        cs.RASCoordinateSystem(axes=_oriented("LPS", "mm"))


def test_a_physical_system_needs_physical_axes() -> None:
    # A physical system may be open or leave its unit unspecified (#112), so
    #
    # its defaults are not refused.
    assert cs.PhysicalCoordinateSystem().axes == [...]
    assert cs.PhysicalCoordinateSystem(axes=[]).ndim == 0
    # A physical axis never counts samples.
    with pytest.raises(ValueError, match="counts samples"):
        cs.PhysicalCoordinateSystem(axes=_oriented("RAS", "index"))
    # A millimeter system accepts only millimeters.
    with pytest.raises(ValueError, match="in millimetres"):
        cs.RASmm(axes=_oriented("RAS"))
    with pytest.raises(ValueError, match="counts samples"):
        cs.RASmm(axes=_oriented("RAS", "index"))


def test_no_system_is_ambiguous() -> None:
    # Every combination of the axis kinds that the classes select on.
    kinds = [
        _plain(2),
        _plain(3),
        _space(2),
        _space(3),
        _plain(2, "index"),
        _plain(3, "index"),
        _space(2, "index"),
        _space(3, "index"),
        _space(3, "mm"),
        _oriented("RAS", "index"),
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
    # Each short name is a class, so each call builds a new axis.
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
    # A default system holds axes of its own, shared with no other default.
    system = getattr(cs, name)()
    # The ellipsis of an open system is not an axis.
    mine = {id(axis) for axis in system.axes if axis is not ...}
    for other in _DEFAULT_SYSTEMS:
        if other == name:
            continue
        theirs = getattr(cs, other)().axes
        assert not mine & {id(axis) for axis in theirs if axis is not ...}


def test_the_singleton_orientations_are_frozen() -> None:
    # Every left-to-right axis holds the one `LeftToRight`, which no axis can
    #
    # change.
    with pytest.raises(AttributeError):
        ax.R().orientation.value = "right-to-left"
    with pytest.raises(AttributeError):
        cs.RASmm().axes[0].orientation.value = "right-to-left"
    assert LeftToRight().value == "left-to-right"


# ----------------------------------------------------------------------
#   MEMORY ORDER
# ----------------------------------------------------------------------
# `order` selects the C- and F-ordered classes together with the axes.

_ORDERED = [
    cs.CArrayCoordinateSystem2D,
    cs.CArrayCoordinateSystem3D,
    cs.FArrayCoordinateSystem2D,
    cs.FArrayCoordinateSystem3D,
    cs.CPixelCoordinateSystem,
    cs.FPixelCoordinateSystem,
    cs.CVoxelCoordinateSystem,
    cs.FVoxelCoordinateSystem,
    cs.FRASCoordinateSystem,
    cs.FLPSCoordinateSystem,
    cs.FRSACoordinateSystem,
    cs.CRASCoordinateSystem,
    cs.CLPSCoordinateSystem,
    cs.CRSACoordinateSystem,
]


def _bases(cls: type) -> tx.List[type]:
    # Every system class above `cls`, the root included, except an anatomical
    #
    # parent that cannot hold the S, A, R axes of a C-ordered grid.
    anatomical = {
        cs.RASCoordinateSystem,
        cs.LPSCoordinateSystem,
        cs.RSACoordinateSystem,
    }
    c_ordered = cls.__name__.startswith("C") and cls.__mro__[1] in anatomical
    return [
        base
        for base in cls.__mro__[1:]
        if isinstance(base, type)
        and issubclass(base, cs.CoordinateSystem)
        and not (c_ordered and base in anatomical)
    ]


@pytest.mark.parametrize("cls", _ORDERED, ids=lambda c: c.__name__)
def test_an_ordered_class_has_its_order(cls: type) -> None:
    order = "C" if issubclass(cls, cs.CArrayCoordinateSystem) else "F"
    assert cls().order == order and cls(axes=None).order == order
    other = "F" if order == "C" else "C"
    with pytest.raises((TypeError, ValueError)):
        cls(order=other)


@pytest.mark.parametrize("cls", _ORDERED, ids=lambda c: c.__name__)
def test_an_ordered_class_is_reached_from_every_base(cls: type) -> None:
    default = cls()
    bases = _bases(cls)
    assert cs.CoordinateSystem in bases
    assert cs.ArrayCoordinateSystem in bases
    for base in bases:
        built = base(axes=list(default.axes), order=default.order)
        assert type(built) is cls, base.__name__


@pytest.mark.parametrize(
    "base, axes, order, expected",
    [
        (cs.CoordinateSystem, _plain(4), "C", cs.CArrayCoordinateSystem),
        (
            cs.CoordinateSystem,
            [ax.Axis(), ...],
            "F",
            cs.FArrayCoordinateSystem,
        ),
        (cs.CoordinateSystem, _plain(2), "C", cs.CArrayCoordinateSystem2D),
        (cs.CoordinateSystem, _plain(3), "F", cs.FArrayCoordinateSystem3D),
        # An order says that the axes index an array, so unitless spatial axes
        #
        # suffice.
        (cs.CoordinateSystem, _space(2), "C", cs.CPixelCoordinateSystem),
        (
            cs.SpatialCoordinateSystem,
            _space(2),
            "F",
            cs.FPixelCoordinateSystem,
        ),
        (cs.CoordinateSystem, _space(3), "C", cs.CVoxelCoordinateSystem),
        (cs.ArrayCoordinateSystem, _space(3), "F", cs.FVoxelCoordinateSystem),
        (cs.CoordinateSystem, _oriented("RAS"), "F", cs.FRASCoordinateSystem),
        (
            cs.CoordinateSystem,
            _oriented("SPL", "index"),
            "C",
            cs.CLPSCoordinateSystem,
        ),
        (
            cs.ArrayCoordinateSystem,
            _oriented("ASR"),
            "C",
            cs.CRSACoordinateSystem,
        ),
    ],
)
def test_the_order_and_the_axes_select_the_class(
    base: type, axes: list, order: str, expected: type
) -> None:
    built = base(axes=axes, order=order)
    assert type(built) is expected and built.order == order


@pytest.mark.parametrize(
    "base, axes, expected",
    [
        (cs.ArrayCoordinateSystem, _plain(3), cs.ArrayCoordinateSystem3D),
        (cs.ArrayCoordinateSystem, [ax.Axis(), ...], cs.ArrayCoordinateSystem),
        (cs.ArrayCoordinateSystem, _plain(4), cs.ArrayCoordinateSystem),
        (cs.CoordinateSystem, _space(3, "index"), cs.VoxelCoordinateSystem),
        (cs.CoordinateSystem, _space(2, "index"), cs.PixelCoordinateSystem),
        (
            cs.CoordinateSystem,
            _oriented("RAS", "index"),
            cs.RASCoordinateSystem,
        ),
    ],
)
def test_an_unspecified_order_builds_the_generic_class(
    base: type, axes: list, expected: type
) -> None:
    for built in (base(axes=axes), base(axes=axes, order=None)):
        assert type(built) is expected and built.order is None


@pytest.mark.parametrize(
    "build",
    [
        lambda: cs.RASmm(order="F"),
        lambda: cs.LPSmm(order="C"),
        lambda: cs.RSAmm(axes=_oriented("RSA", "mm"), order="F"),
        lambda: cs.PhysicalCoordinateSystem(axes=[ax.Axis()], order="C"),
        lambda: cs.SpatialCoordinateSystem(axes=_space(4), order="C"),
    ],
)
def test_only_an_array_system_has_an_order(build: tx.Callable) -> None:
    with pytest.raises(ValueError, match="indexes no array"):
        build()


def test_no_ordered_call_is_ambiguous() -> None:
    kinds = [
        _plain(2),
        _plain(3),
        _plain(4),
        _space(2),
        _space(3),
        _space(2, "index"),
        _space(3, "index"),
        _oriented("RAS"),
        _oriented("RAS", "index"),
        _oriented("SAR", "index"),
        _oriented("LPS"),
        _oriented("RSA", "index"),
        [ax.Axis(), ...],
    ]
    roots = [
        getattr(cs, name)
        for name in cs.__all__
        if isinstance(getattr(cs, name), type)
        and issubclass(getattr(cs, name), cs.CoordinateSystem)
    ]
    for root in roots:
        for axes in kinds:
            for order in ("C", "F", None):
                try:
                    root(axes=axes, order=order)
                except PolymorphError as e:  # pragma: no cover
                    pytest.fail(f"{root.__name__}: {e}")
                except (TypeError, ValueError):
                    # The class refuses axes or an order that it cannot hold.
                    pass
