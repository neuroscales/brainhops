"""Tests for coordinate systems: open systems (`...`), positional and
named access to their axes, and the operations that close, restrict and
place them."""

import pytest
import typing_extensions as tx

from brainhops.datamodel import systems as S
from brainhops.datamodel.axes import (
    A,
    Axis,
    R,
    SpatialAxis,
    TimeAxis,
)
from brainhops.datamodel.systems import (
    ArrayCoordinateSystem,
    CoordinateSystem,
    CoordinateSystem2D,
    CoordinateSystem3D,
    FRASCoordinateSystem,
    PixelCoordinateSystem,
    RASCoordinateSystem,
    SpatialCoordinateSystem,
    SpatialCoordinateSystem3D,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import Identity

CS = CoordinateSystem
X, Y, Z = Axis(name="x"), Axis(name="y"), Axis(name="z")
T = TimeAxis(name="t")

# The two spellings of "nothing is known about the axes".
UNKNOWN_SPELLINGS = {"None": None, "[...]": [...]}

# Where `...` sits among two explicit axes, `X` and `T`.
OPEN_LAYOUTS = {
    "start": [..., X, T],
    "middle": [X, ..., T],
    "end": [X, T, ...],
}

# Every fixed-dimension class, with the number of axes it holds.
FIXED_CLASSES = [
    (CoordinateSystem2D, 2),
    (PixelCoordinateSystem, 2),
    (CoordinateSystem3D, 3),
    (SpatialCoordinateSystem3D, 3),
    (VoxelCoordinateSystem, 3),
    (RASCoordinateSystem, 3),
    (FRASCoordinateSystem, 3),
]


@pytest.fixture(params=list(UNKNOWN_SPELLINGS), ids=list(UNKNOWN_SPELLINGS))
def unknown_axes(request: pytest.FixtureRequest) -> tx.Optional[list]:
    """The axes of a system about which nothing is known, both spellings."""
    return UNKNOWN_SPELLINGS[request.param]


@pytest.fixture(params=list(OPEN_LAYOUTS), ids=list(OPEN_LAYOUTS))
def layout(request: pytest.FixtureRequest) -> str:
    """Where `...` sits in an open system."""
    return request.param


# ----------------------------------------------------------------------
#   VALIDATION
# ----------------------------------------------------------------------


def test_none_is_stored_as_given(unknown_axes: tx.Optional[list]) -> None:
    system = CS(axes=unknown_axes)
    assert system.axes == unknown_axes


def test_ellipsis_may_sit_anywhere(layout: str) -> None:
    assert CS(axes=OPEN_LAYOUTS[layout]).axes == OPEN_LAYOUTS[layout]


@pytest.mark.parametrize("axes", [[..., ...], [..., X, ...], [X, ..., Y, ...]])
def test_at_most_one_ellipsis(axes: list) -> None:
    with pytest.raises(ValueError, match="at most one"):
        CS(axes=axes)


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
@pytest.mark.parametrize("where", ["None", "start", "middle", "end"])
def test_fixed_dimension_classes_refuse_an_open_system(
    cls: type, ndim: int, where: str
) -> None:
    explicit = [Axis()] * ndim
    axes = {
        "None": None,
        "start": [..., *explicit],
        "middle": [explicit[0], ..., *explicit[1:]],
        "end": [*explicit, ...],
    }[where]
    with pytest.raises(ValueError, match=f"exactly {ndim} axes"):
        cls(axes=axes)


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_fixed_dimension_classes_are_closed(cls: type, ndim: int) -> None:
    assert cls().ndim == ndim


def test_open_spatial_system() -> None:
    # A system whose number of axes is not fixed by its class may be open.
    system = SpatialCoordinateSystem(axes=[SpatialAxis(name="x"), ...])
    assert system.ndim is None


# ----------------------------------------------------------------------
#   NDIM
# ----------------------------------------------------------------------


def test_ndim_of_a_closed_system() -> None:
    assert CS(axes=[]).ndim == 0
    assert CS(axes=[X, Y, Z]).ndim == 3


def test_ndim_of_an_open_system(layout: str) -> None:
    assert CS(axes=OPEN_LAYOUTS[layout]).ndim is None


def test_ndim_of_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    assert CS(axes=unknown_axes).ndim is None


def test_ndim_of_a_missing_system() -> None:
    assert S._ndim_of(None) is None


# ----------------------------------------------------------------------
#   EQUALITY
# ----------------------------------------------------------------------


def test_none_equals_ellipsis() -> None:
    assert CS(axes=None) == CS(axes=[...])
    assert CS(name="s", axes=None) == CS(name="s", axes=[...])
    assert not CS(axes=None) != CS(axes=[...])
    assert SpatialCoordinateSystem() == SpatialCoordinateSystem(axes=[...])


def test_equality_stays_structural(layout: str) -> None:
    axes = OPEN_LAYOUTS[layout]
    assert CS(axes=axes) == CS(axes=list(axes))
    assert CS(axes=axes) != CS(axes=[...])
    assert CS(axes=axes) != CS(name="s", axes=axes)
    assert CS(axes=axes) != SpatialCoordinateSystem(axes=axes)
    # Compatible, but not equal: equality does not expand `...`.
    assert CS(axes=[X, ...]) != CS(axes=[X, Axis()])


def test_equality_of_closed_systems_is_unchanged() -> None:
    assert CS(axes=[X, Y]) == CS(axes=(X, Y))
    assert CS(axes=[X, Y]) != CS(axes=[Y, X])
    assert CS(axes=[X]) != CS(axes=[Axis()])
    assert RASCoordinateSystem() == RASCoordinateSystem()
    assert RASCoordinateSystem() != CoordinateSystem3D(axes=(R, A, Axis()))


def test_a_system_that_says_nothing_equals_a_missing_system(
    unknown_axes: tx.Optional[list],
) -> None:
    # A missing endpoint and a plain, unnamed system with no known axis
    # say the same thing, so they compare equal both ways round.
    system = CS(axes=unknown_axes)
    assert system == None  # noqa: E711
    assert None == system  # noqa: E711
    assert Identity(input=system) == Identity()
    assert Identity() == Identity(input=system)


def test_a_system_that_says_something_differs_from_a_missing_system(
    unknown_axes: tx.Optional[list],
) -> None:
    assert CS(name="s", axes=unknown_axes) != None  # noqa: E711
    assert CS(axes=[X, ...]) != None  # noqa: E711
    assert SpatialCoordinateSystem(axes=unknown_axes) != None  # noqa: E711
    assert ArrayCoordinateSystem(axes=unknown_axes) != None  # noqa: E711


def test_systems_stay_unhashable() -> None:
    with pytest.raises(TypeError):
        hash(CS())


# ----------------------------------------------------------------------
#   POSITION AND AXIS
# ----------------------------------------------------------------------


def test_position_of_a_name_in_a_closed_system() -> None:
    system = CS(axes=[X, Y, T])
    assert [system.position(n) for n in ("x", "y", "t")] == [0, 1, 2]


@pytest.mark.parametrize(
    "layout, x, t",
    [("start", -2, -1), ("middle", 0, -1), ("end", 0, 1)],
)
def test_position_of_a_name_in_an_open_system(
    layout: str, x: int, t: int
) -> None:
    # A name after `...` resolves to a negative position, because its
    # distance from the start is unknown.
    system = CS(axes=OPEN_LAYOUTS[layout])
    assert system.position("x") == x
    assert system.position("t") == t
    assert system.axis(system.position("x")) is X
    assert system.axis(system.position("t")) is T


@pytest.mark.parametrize(
    "axes", [[X, Y], [X, ..., Y], [..., X, Y], [X, Y, ...]]
)
def test_position_of_a_missing_name(axes: list) -> None:
    with pytest.raises(ValueError, match="no axis of the system"):
        CS(axes=axes).position("z")


@pytest.mark.parametrize(
    "axes", [[X, Axis(name="x")], [X, ..., Axis(name="x")], [..., X, X]]
)
def test_position_of_a_duplicated_name(axes: list) -> None:
    with pytest.raises(ValueError, match="2 axes"):
        CS(axes=axes).position("x")


def test_position_of_a_name_in_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    for system in (CS(axes=unknown_axes), None):
        with pytest.raises(ValueError, match="without a system that names"):
            S._position_in(system, "x")
    with pytest.raises(ValueError, match="without a system that names"):
        CS(axes=unknown_axes).position("x")


def test_position_of_an_index_in_a_closed_system() -> None:
    system = CS(axes=[X, Y, Z])
    assert [system.position(i) for i in range(3)] == [0, 1, 2]
    assert [system.position(i) for i in (-3, -2, -1)] == [0, 1, 2]
    for i in (3, -4, 100):
        with pytest.raises(IndexError, match="out of range"):
            system.position(i)
    with pytest.raises(IndexError):
        CS(axes=[]).position(0)


def test_position_of_an_index_in_an_open_system(layout: str) -> None:
    # `...` stands for any number of axes, so every position is valid, and
    # is returned as given: never clamped, never normalized.
    system = CS(axes=OPEN_LAYOUTS[layout])
    for i in (0, 1, 2, 5, 100, -1, -2, -3, -100):
        assert system.position(i) == i


def test_position_of_an_index_in_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    for i in (0, 3, -1):
        assert CS(axes=unknown_axes).position(i) == i
        assert S._position_in(None, i) == i


def test_position_accepts_numpy_integers() -> None:
    np = pytest.importorskip("numpy")
    assert CS(axes=[X, Y]).position(np.int64(-1)) == 1


@pytest.mark.parametrize("ref", [True, 1.0, None, b"x", ["x"]])
def test_position_refuses_other_references(ref: object) -> None:
    with pytest.raises(TypeError):
        CS(axes=[X, Y]).position(ref)  # type: ignore[arg-type]


def test_axis_of_a_closed_system() -> None:
    system = CS(axes=[X, Y, T])
    assert system.axis(0) is X and system.axis(-1) is T
    assert system.axis("y") is Y
    with pytest.raises(IndexError):
        system.axis(3)


@pytest.mark.parametrize(
    "layout, known",
    [
        # position -> axis, for the positions that an explicit axis holds
        ("start", {-2: X, -1: T}),
        ("middle", {0: X, -1: T}),
        ("end", {0: X, 1: T}),
    ],
)
def test_axis_of_an_open_system(layout: str, known: dict) -> None:
    # A non-negative position reads the part before `...`, a negative one
    # the part after it; any other position is unknown.
    system = CS(axes=OPEN_LAYOUTS[layout])
    for position in range(-5, 5):
        expected = known.get(position, Axis())
        assert system.axis(position) == expected


def test_axis_of_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    for position in (0, 2, -1):
        assert CS(axes=unknown_axes).axis(position) == Axis()
        assert S._axis_in(None, position) == Axis()


# ----------------------------------------------------------------------
#   EXPAND
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("start", [Axis(), Axis(), X, T]),
        ("middle", [X, Axis(), Axis(), T]),
        ("end", [X, T, Axis(), Axis()]),
    ],
)
def test_expand_an_open_system(layout: str, expected: list) -> None:
    system = CS(name="s", axes=OPEN_LAYOUTS[layout])
    assert system.expand(4) == CS(name="s", axes=expected)
    assert system.expand(2) == CS(name="s", axes=[X, T])


def test_expand_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    system = CS(axes=unknown_axes)
    assert system.expand(0) == CS(axes=[])
    assert system.expand(2) == CS(axes=[Axis(), Axis()])


def test_expand_keeps_the_class_and_the_name() -> None:
    system = ArrayCoordinateSystem(axes=[Axis(name="i"), ...])
    expanded = system.expand(2)
    assert type(expanded) is ArrayCoordinateSystem
    assert expanded.name == "array"
    assert expanded.axes == [Axis(name="i"), Axis()]


def test_expand_does_not_change_the_system() -> None:
    system = CS(axes=[X, ...])
    system.expand(3)
    assert system.axes == [X, ...]


def test_expand_a_closed_system_to_its_size_is_itself() -> None:
    system = CS(axes=[X, Y])
    assert system.expand(2) is system
    ras = RASCoordinateSystem()
    assert ras.expand(3) is ras


@pytest.mark.parametrize("ndim", [0, 1, 3])
def test_expand_a_closed_system_to_another_size(ndim: int) -> None:
    with pytest.raises(ValueError, match="closed system of 2 axes"):
        CS(axes=[X, Y]).expand(ndim)


def test_expand_below_the_explicit_axes(layout: str) -> None:
    with pytest.raises(ValueError, match="2 explicit axes to 1"):
        CS(axes=OPEN_LAYOUTS[layout]).expand(1)


@pytest.mark.parametrize("ndim", [2.0, "2", None, True])
def test_expand_refuses_a_non_integer(ndim: object) -> None:
    with pytest.raises(TypeError):
        CS(axes=[X, ...]).expand(ndim)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
#   TAKE
# ----------------------------------------------------------------------


def test_take_from_a_closed_system() -> None:
    system = CS(name="s", axes=[X, Y, T])
    assert system.take([2, 0]) == CS(axes=[T, X])
    assert system.take(["t", -2]) == CS(axes=[T, Y])
    assert system.take([]) == CS(axes=[])


@pytest.mark.parametrize(
    "layout, refs, expected",
    [
        ("start", [-1, 0, -2], [T, Axis(), X]),
        ("middle", [-1, 0, 1], [T, X, Axis()]),
        ("end", [1, 0, 2], [T, X, Axis()]),
    ],
)
def test_take_from_an_open_system(
    layout: str, refs: list, expected: list
) -> None:
    system = CS(axes=OPEN_LAYOUTS[layout])
    taken = system.take(refs)
    assert taken == CS(axes=expected)
    assert taken.ndim == len(refs)
    assert system.take(["t", "x"]) == CS(axes=[T, X])


def test_take_from_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    taken = CS(axes=unknown_axes).take([0, 3])
    assert taken == CS(axes=[Axis(), Axis()])


def test_take_refuses_a_repeated_axis() -> None:
    for refs in ([0, 0], [0, -2], ["x", 0]):
        with pytest.raises(ValueError, match="more than once"):
            CS(axes=[X, Y]).take(refs)


def test_take_refuses_a_string() -> None:
    with pytest.raises(TypeError, match="single string"):
        CS(axes=[X, Y]).take("x")  # type: ignore[arg-type]


def test_take_raises_as_position_does() -> None:
    with pytest.raises(IndexError):
        CS(axes=[X, Y]).take([2])
    with pytest.raises(ValueError):
        CS(axes=[X, ...]).take(["y"])


# ----------------------------------------------------------------------
#   PLACE
# ----------------------------------------------------------------------


def test_place_with_an_unknown_size() -> None:
    placed = CS(name="s", axes=[X, Y]).place([2, 0])
    assert placed == CS(axes=[Y, Axis(), X, ...])
    assert placed.ndim is None


def test_place_with_a_known_size() -> None:
    placed = CS(axes=[X, Y]).place([2, 0], ndim=4)
    assert placed == CS(axes=[Y, Axis(), X, Axis()])
    assert placed.ndim == 4


def test_place_is_the_inverse_of_take() -> None:
    system = CS(axes=[X, Y, T])
    positions = [4, 1, 2]
    assert system.place(positions).take(positions) == system
    assert system.place(positions, ndim=6).take(positions) == system


def test_place_keeps_the_positions_take_reads() -> None:
    # The placed system reads back, at each known position, the axis that
    # was placed there: the composers check discrete axes this way.
    placed = CS(axes=[X, Axis(name="c", discrete=True)]).place([0, 3])
    assert placed.axis(3).discrete is True
    assert placed.axis(1) == Axis()
    assert placed.axis(7) == Axis()


def test_place_an_open_system(layout: str) -> None:
    # An open system is first closed to one axis per position: wherever
    # `...` sits, two positions close it to `[X, T]`.
    system = CS(axes=OPEN_LAYOUTS[layout])
    assert system.place([2, 0], ndim=3) == CS(axes=[T, Axis(), X])
    assert system.place([2, 0]) == CS(axes=[T, Axis(), X, ...])


def test_place_an_open_system_with_room_for_unknown_axes() -> None:
    placed = CS(axes=[X, ...]).place([1, 0, 3], ndim=4)
    assert placed == CS(axes=[Axis(), X, Axis(), Axis()])


def test_place_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    placed = CS(axes=unknown_axes).place([1])
    assert placed == CS(axes=[Axis(), Axis(), ...])
    assert CS(axes=unknown_axes).place([]) == CS(axes=[...])


def test_place_nothing_with_a_known_size() -> None:
    assert CS(axes=[]).place([], ndim=2) == CS(axes=[Axis(), Axis()])


def test_place_refuses_a_count_mismatch() -> None:
    with pytest.raises(ValueError, match="system of 2 axes at 1"):
        CS(axes=[X, Y]).place([0])
    with pytest.raises(ValueError, match="system of 2 axes at 3"):
        CS(axes=[X, Y]).place([0, 1, 2])
    with pytest.raises(ValueError, match="2 explicit axes at 1"):
        CS(axes=[X, ..., Y]).place([0])


def test_place_refuses_bad_positions() -> None:
    with pytest.raises(ValueError, match="negative"):
        CS(axes=[X]).place([-1])
    with pytest.raises(ValueError, match="more than once"):
        CS(axes=[X, Y]).place([1, 1])
    with pytest.raises(TypeError):
        CS(axes=[X]).place(["x"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        CS(axes=[X]).place(0)  # type: ignore[arg-type]


def test_place_refuses_a_size_that_is_too_small() -> None:
    with pytest.raises(ValueError, match="position 2 of a space of 2"):
        CS(axes=[X]).place([2], ndim=2)


# ----------------------------------------------------------------------
#   COMPATIBLE
# ----------------------------------------------------------------------


def test_compatible_closed_systems() -> None:
    assert CS(axes=[X, Y]).compatible(CS(axes=[X, Y]))
    assert CS(axes=[X, Axis()]).compatible(CS(axes=[Axis(), Y]))
    assert not CS(axes=[X, Y]).compatible(CS(axes=[Y, X]))
    assert not CS(axes=[X, Y]).compatible(CS(axes=[X, Y, Z]))
    assert CS(axes=[]).compatible(CS(axes=[]))


@pytest.mark.parametrize(
    "layout, closed, ok",
    [
        ("start", [X, T], True),
        ("start", [Y, Z, X, T], True),
        ("start", [X, T, Y], False),
        ("middle", [X, T], True),
        ("middle", [X, Y, Z, T], True),
        ("middle", [Y, X, T], False),
        ("end", [X, T], True),
        ("end", [X, T, Y, Z], True),
        ("end", [Y, X, T], False),
        ("start", [T], False),
        ("middle", [X], False),
        ("end", [], False),
    ],
)
def test_compatible_open_and_closed(
    layout: str, closed: list, ok: bool
) -> None:
    system = CS(axes=OPEN_LAYOUTS[layout])
    assert system.compatible(CS(axes=closed)) is ok
    assert CS(axes=closed).compatible(system) is ok


@pytest.mark.parametrize(
    "first, second, ok",
    [
        ([X, ...], [X, Y, ...], True),
        ([X, ...], [Y, ...], False),
        ([X, ...], [..., X], True),
        ([X, ...], [..., Y], True),
        ([..., T], [X, ..., T], True),
        ([..., T], [..., X], False),
        ([X, ..., T], [X, Y, ..., Z, T], True),
        ([X, ..., T], [Y, ..., T], False),
    ],
)
def test_compatible_open_systems(first: list, second: list, ok: bool) -> None:
    assert CS(axes=first).compatible(CS(axes=second)) is ok
    assert CS(axes=second).compatible(CS(axes=first)) is ok


def test_an_unknown_system_is_compatible_with_every_system(
    unknown_axes: tx.Optional[list],
) -> None:
    unknown = CS(axes=unknown_axes)
    for axes in (None, [...], [], [X], [X, ...], [..., T], [X, Y, Z]):
        assert unknown.compatible(CS(axes=axes))
        assert CS(axes=axes).compatible(unknown)
        assert CS(axes=axes).compatible(None)
        assert S._compatible(None, CS(axes=axes))


def test_compatible_ras_with_time() -> None:
    # The example of the design: `[RAS axes, ...]` is compatible with a
    # closed RAS and time system.
    ras = list(RASCoordinateSystem().axes)
    assert CS(axes=[*ras, ...]).compatible(CS(axes=[*ras, TimeAxis()]))
    assert CS(axes=[*ras, ...]).compatible(RASCoordinateSystem())


def test_compatible_ignores_the_names_of_the_systems() -> None:
    assert CS(name="a", axes=[X]).compatible(CS(name="b", axes=[X]))


def test_compatible_refuses_a_non_system() -> None:
    with pytest.raises(TypeError):
        CS().compatible([X])  # type: ignore[arg-type]
