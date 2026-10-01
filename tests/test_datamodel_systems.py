"""Tests for coordinate systems: open systems (`...`), positional and
named access to their axes, and the operations that close, restrict and
embed them."""

import pytest
import typing_extensions as tx

from brainhops.datamodel.axes import (
    A,
    Axis,
    R,
    SpatialAxis,
    TimeAxis,
)
from brainhops.datamodel.systems import (
    ArrayCoordinateSystem,
    AxisList,
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

# The same two axes, closed or with `...` anywhere.
ANY_LAYOUTS = {"closed": [X, T], **OPEN_LAYOUTS}

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


@pytest.fixture(params=list(ANY_LAYOUTS), ids=list(ANY_LAYOUTS))
def any_layout(request: pytest.FixtureRequest) -> str:
    """A closed list of `X` and `T`, or `...` somewhere among them."""
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
    assert AxisList.of(None).ndim is None


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
    spatial = [SpatialAxis(), ...]
    assert CS(axes=spatial) != SpatialCoordinateSystem(axes=spatial)
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


def test_the_hand_written_equality_is_the_one_of_every_subclass() -> None:
    # Magic writes a field-wise `__eq__` into a subclass that does not
    # define its own, unless `eq=False` is inherited. That one would tell
    # `axes=None` from `axes=[...]`.
    class Mine(CoordinateSystem):
        pass

    for cls in (CS, SpatialCoordinateSystem, ArrayCoordinateSystem, Mine):
        assert cls.__eq__ is CS.__eq__
        assert cls(axes=None) == cls(axes=[...])
    assert RASCoordinateSystem.__eq__ is CS.__eq__


# ----------------------------------------------------------------------
#   AXIS LIST
# ----------------------------------------------------------------------


def test_axes_are_stored_as_an_axis_list(layout: str) -> None:
    for given in (OPEN_LAYOUTS[layout], tuple(OPEN_LAYOUTS[layout])):
        axes = CS(axes=given).axes
        assert type(axes) is AxisList
        assert axes == OPEN_LAYOUTS[layout]
    assert type(SpatialCoordinateSystem(axes=[...]).axes) is AxisList


def test_the_axes_are_converted_to_the_type_of_the_field() -> None:
    # A list is converted item by item, as a tuple always was.
    for sequence in (list, tuple):
        given = sequence([SpatialAxis(name="x"), ...])
        assert SpatialCoordinateSystem(axes=given).axes == list(given)
        with pytest.raises(TypeError):
            SpatialCoordinateSystem(axes=sequence([TimeAxis(), ...]))
    assert CS(axes=["x", ...]).axes == [Axis(name="x"), ...]
    # An axis of the right type is kept as it is.
    assert CS(axes=[X, ...]).axes[0] is X


def test_an_axis_list_is_a_plain_list(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    assert isinstance(axes, list)
    assert axes == entries
    assert repr(axes) == repr(entries)
    assert type(axes[1:]) is AxisList and axes[1:] == entries[1:]


def test_axis_list_of_a_system(any_layout: str) -> None:
    system = CS(axes=ANY_LAYOUTS[any_layout])
    axes = AxisList.of(system)
    assert type(axes) is AxisList and axes == ANY_LAYOUTS[any_layout]
    # A new list: changing it leaves the system as it is.
    axes.append(Z)
    assert system.axes == ANY_LAYOUTS[any_layout]


def test_axis_list_of_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    for system in (CS(axes=unknown_axes), None):
        assert AxisList.of(system) == [...]
        assert type(AxisList.of(system)) is AxisList
    # The system keeps what was given.
    assert CS(axes=unknown_axes).axes == unknown_axes


def test_axis_list_of_a_fixed_dimension_system() -> None:
    axes = AxisList.of(RASCoordinateSystem())
    assert type(axes) is AxisList
    assert axes == list(RASCoordinateSystem().axes)
    assert axes.ndim == 3


@pytest.mark.parametrize("other", [[X], (X,), "x", 0])
def test_axis_list_of_refuses_a_non_system(other: object) -> None:
    with pytest.raises(TypeError, match="CoordinateSystem or None"):
        AxisList.of(other)  # type: ignore[arg-type]


def test_is_open_and_ndim(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    assert axes.is_open is (any_layout != "closed")
    assert axes.ndim == (2 if any_layout == "closed" else None)
    assert AxisList([...]).is_open and AxisList([...]).ndim is None
    assert not AxisList([]).is_open and AxisList([]).ndim == 0


def test_more_than_one_ellipsis_describes_no_axes() -> None:
    axes = AxisList([X, ..., ...])
    for read in (
        lambda: axes.ndim,
        lambda: axes.is_open,
        lambda: axes.restrict([0]),
        lambda: axes.expand(3),
    ):
        with pytest.raises(ValueError, match="at most one"):
            read()


# ----------------------------------------------------------------------
#   NAMES, ENTRIES AND INDEX
# ----------------------------------------------------------------------


def test_names(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    assert axes.names == tuple(... if a is ... else a.name for a in axes)
    assert AxisList([Axis(), ...]).names == (None, ...)
    assert AxisList([]).names == ()


def test_an_axis_by_its_name(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    assert axes["x"] is X and axes["t"] is T
    assert axes["t"] is axes[axes.index("t")]


def test_a_missing_name(any_layout: str) -> None:
    # A name never matches one of the axes that `...` stands for.
    with pytest.raises(KeyError, match="'y'"):
        AxisList(ANY_LAYOUTS[any_layout])["y"]
    with pytest.raises(KeyError):
        AxisList([...])["x"]


def test_a_shared_name_is_ambiguous(any_layout: str) -> None:
    axes = AxisList([*ANY_LAYOUTS[any_layout], Axis(name="x")])
    with pytest.raises(ValueError, match="2 axes"):
        axes["x"]
    # `index` and `in` find the first one, as for any list.
    assert axes.index("x") == ANY_LAYOUTS[any_layout].index(X)
    assert "x" in axes


def test_a_name_is_in_the_list(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    assert "x" in axes and "t" in axes
    assert "y" not in axes
    assert "x" not in AxisList([...])


def test_anything_else_is_in_the_list_as_an_entry(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    assert X in axes and T in axes and Y not in axes
    assert (... in axes) is (any_layout != "closed")


def test_an_integer_indexes_the_entries(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    for i in range(-len(entries), len(entries)):
        assert axes[i] is entries[i]
    with pytest.raises(IndexError):
        axes[len(entries)]


def test_entries_and_axes_differ_in_an_open_list() -> None:
    # In `[x, ..., t]`, entry 2 is `t`, which is the last axis, and the
    # axis at position 2 is one of the axes that `...` stands for.
    axes = AxisList([X, ..., T])
    assert len(axes) == 3 and axes.ndim is None
    assert axes[2] is T and axes.index("t") == 2
    assert axes.restrict([2]) == [Axis()]
    assert axes.restrict([-1]) == [T]
    # Walking the entries walks `...`, not the axes it stands for.
    assert [axes[i] for i in range(len(axes))] == [X, ..., T]


def test_index_of_a_name(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    assert axes.index("x") == entries.index(X)
    assert axes.index("t") == entries.index(T)


def test_index_of_an_axis(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    # An axis matches itself, and a query that sets fewer fields.
    assert axes.index(T) == entries.index(T)
    assert axes.index(TimeAxis()) == entries.index(T)
    assert axes.index(Axis(unit="second")) == entries.index(T)
    assert axes.index(Axis(name="x")) == entries.index(X)
    # The empty query matches the first axis.
    assert axes.index(Axis()) == (1 if any_layout == "start" else 0)


def test_index_compares_only_the_fields_the_query_sets() -> None:
    axes = AxisList([Axis(), Axis(name="x"), Axis(name="x", unit="mm")])
    assert axes.index("x") == 1
    assert axes.index(Axis(name="x", unit="mm")) == 2
    # An unset field of the entry does not match a field the query sets.
    assert axes.index(Axis(unit="mm")) == 2


def test_index_asks_for_the_class_of_the_query() -> None:
    plain = Axis(name="x", unit="mm", type="space")
    spatial = SpatialAxis(name="x")
    axes = AxisList([plain, spatial])
    # A plain `Axis` query matches any axis; a `SpatialAxis` query only a
    # `SpatialAxis` (or a subclass), whatever the fields of the others.
    assert axes.index(Axis(name="x")) == 0
    assert axes.index(SpatialAxis(name="x")) == 1
    assert AxisList([R]).index(SpatialAxis()) == 0
    with pytest.raises(ValueError):
        AxisList([plain]).index(SpatialAxis())


def test_index_never_matches_ellipsis(unknown_axes: tx.Optional[list]) -> None:
    for system in (CS(axes=unknown_axes), None):
        axes = AxisList.of(system)
        for query in (Axis(), "x", TimeAxis()):
            with pytest.raises(ValueError, match="is not in list"):
                axes.index(query)


def test_index_of_a_missing_axis(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    with pytest.raises(ValueError, match=r"Axis\(name='y'\) is not in list"):
        axes.index("y")
    with pytest.raises(ValueError):
        axes.index(SpatialAxis())


def test_index_between_start_and_stop() -> None:
    axes = AxisList([X, ..., X, T])
    assert axes.index("x") == 0
    assert axes.index("x", 1) == 2
    assert axes.index("x", -2) == 2
    assert axes.index("t", 0, 4) == 3
    with pytest.raises(ValueError):
        axes.index("t", 0, 3)
    with pytest.raises(ValueError):
        axes.index("x", 3)


@pytest.mark.parametrize("query", [0, None, b"x", ["x"]])
def test_index_refuses_other_queries(query: object) -> None:
    with pytest.raises(TypeError):
        AxisList([X, Y]).index(query)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
#   POSITIONS (private helpers)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("closed", [0, 1]),
        ("start", [None, -2, -1]),
        ("middle", [0, None, -1]),
        ("end", [0, 1, None]),
    ],
)
def test_the_position_of_an_entry(layout: str, expected: list) -> None:
    # An entry after `...` is counted from the end; any other entry is its
    # own position. `...` is no axis, and has no position.
    axes = AxisList(ANY_LAYOUTS[layout])
    for entry, position in enumerate(expected):
        if position is None:
            with pytest.raises(ValueError):
                axes._position_of_entry(entry)
        else:
            assert axes._position_of_entry(entry) == position
            assert axes._position_of_entry(entry - len(axes)) == position
            assert axes._axis_at(position) is axes[entry]


def test_a_position_in_a_closed_list() -> None:
    axes = AxisList([X, Y, Z])
    assert [axes._position(i) for i in range(3)] == [0, 1, 2]
    assert [axes._position(i) for i in (-3, -2, -1)] == [0, 1, 2]
    for i in (3, -4, 100):
        with pytest.raises(IndexError, match="out of range"):
            axes._position(i)
    with pytest.raises(IndexError):
        AxisList([])._position(0)


def test_a_position_in_an_open_list(layout: str) -> None:
    # `...` stands for any number of axes, so every position is valid, and
    # is returned as given: never clamped, never normalized.
    axes = AxisList(OPEN_LAYOUTS[layout])
    for i in (0, 1, 2, 5, 100, -1, -2, -3, -100):
        assert axes._position(i) == i


def test_a_position_in_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    for system in (CS(axes=unknown_axes), None):
        for i in (0, 3, -1):
            assert AxisList.of(system)._position(i) == i


def test_a_position_may_be_a_numpy_integer() -> None:
    np = pytest.importorskip("numpy")
    assert AxisList([X, Y])._position(np.int64(-1)) == 1


@pytest.mark.parametrize("ref", [True, 1.0, None, b"x", ["x"]])
def test_a_position_is_an_integer(ref: object) -> None:
    with pytest.raises(TypeError):
        AxisList([X, Y])._position(ref)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "layout, known",
    [
        # position -> axis, for the positions that an explicit axis holds
        ("start", {-2: X, -1: T}),
        ("middle", {0: X, -1: T}),
        ("end", {0: X, 1: T}),
    ],
)
def test_the_axis_at_a_position_of_an_open_list(
    layout: str, known: dict
) -> None:
    # A non-negative position reads the part before `...`, a negative one
    # the part after it; any other position is unknown.
    axes = AxisList(OPEN_LAYOUTS[layout])
    for position in range(-5, 5):
        expected = known.get(position, Axis())
        assert axes._axis_at(position) == expected
        assert axes.restrict([position]) == [expected]


def test_the_axis_at_a_position_of_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    for system in (CS(axes=unknown_axes), None):
        for position in (0, 2, -1):
            assert AxisList.of(system)._axis_at(position) == Axis()


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
    with pytest.raises(ValueError, match="closed list of 2 axes"):
        CS(axes=[X, Y]).expand(ndim)


def test_expand_below_the_explicit_axes(layout: str) -> None:
    with pytest.raises(ValueError, match="2 explicit axes to 1"):
        CS(axes=OPEN_LAYOUTS[layout]).expand(1)


@pytest.mark.parametrize("ndim", [2.0, "2", None, True])
def test_expand_refuses_a_non_integer(ndim: object) -> None:
    with pytest.raises(TypeError):
        CS(axes=[X, ...]).expand(ndim)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
#   RESTRICT
# ----------------------------------------------------------------------


def test_restrict_from_a_closed_system() -> None:
    system = CS(name="s", axes=[X, Y, T])
    assert system.restrict([2, 0]) == CS(axes=[T, X])
    assert system.restrict(["t", -2]) == CS(axes=[T, Y])
    assert system.restrict([]) == CS(axes=[])


@pytest.mark.parametrize(
    "layout, refs, expected",
    [
        ("start", [-1, 0, -2], [T, Axis(), X]),
        ("middle", [-1, 0, 1], [T, X, Axis()]),
        ("end", [1, 0, 2], [T, X, Axis()]),
    ],
)
def test_restrict_from_an_open_system(
    layout: str, refs: list, expected: list
) -> None:
    system = CS(axes=OPEN_LAYOUTS[layout])
    restricted = system.restrict(refs)
    assert restricted == CS(axes=expected)
    assert restricted.ndim == len(refs)
    assert system.restrict(["t", "x"]) == CS(axes=[T, X])


def test_restrict_from_an_unknown_system(
    unknown_axes: tx.Optional[list],
) -> None:
    restricted = CS(axes=unknown_axes).restrict([0, 3])
    assert restricted == CS(axes=[Axis(), Axis()])


def test_restrict_refuses_a_repeated_axis() -> None:
    for refs in ([0, 0], [0, -2], ["x", 0]):
        with pytest.raises(ValueError, match="more than once"):
            CS(axes=[X, Y]).restrict(refs)


def test_restrict_refuses_a_string() -> None:
    with pytest.raises(TypeError, match="single string"):
        CS(axes=[X, Y]).restrict("x")  # type: ignore[arg-type]


def test_restrict_refuses_an_unknown_reference() -> None:
    with pytest.raises(IndexError, match="out of range"):
        CS(axes=[X, Y]).restrict([2])
    with pytest.raises(KeyError, match="'y'"):
        CS(axes=[X, ...]).restrict(["y"])
    with pytest.raises(ValueError, match="2 axes"):
        CS(axes=[X, ..., X]).restrict(["x"])
    with pytest.raises(TypeError):
        CS(axes=[X, Y]).restrict([1.0])


# ----------------------------------------------------------------------
#   EMBED
# ----------------------------------------------------------------------


def test_embed_with_an_unknown_size() -> None:
    embedded = CS(name="s", axes=[X, Y]).embed([2, 0])
    assert embedded == CS(axes=[Y, Axis(), X, ...])
    assert embedded.ndim is None


def test_embed_with_a_known_size() -> None:
    embedded = CS(axes=[X, Y]).embed([2, 0], ndim=4)
    assert embedded == CS(axes=[Y, Axis(), X, Axis()])
    assert embedded.ndim == 4


def test_embed_is_the_inverse_of_restrict() -> None:
    system = CS(axes=[X, Y, T])
    positions = [4, 1, 2]
    assert system.embed(positions).restrict(positions) == system
    assert system.embed(positions, ndim=6).restrict(positions) == system


def test_embed_keeps_the_positions_restrict_reads() -> None:
    # The embedded system reads back, at each known position, the axis that
    # was embedded there: the composers check discrete axes this way.
    embedded = CS(axes=[X, Axis(name="c", discrete=True)]).embed([0, 3])
    axes = AxisList.of(embedded)
    assert axes._axis_at(3).discrete is True
    assert axes._axis_at(1) == Axis()
    assert axes._axis_at(7) == Axis()


def test_embed_an_open_system(layout: str) -> None:
    # An open system is first closed to one axis per position: wherever
    # `...` sits, two positions close it to `[X, T]`.
    system = CS(axes=OPEN_LAYOUTS[layout])
    assert system.embed([2, 0], ndim=3) == CS(axes=[T, Axis(), X])
    assert system.embed([2, 0]) == CS(axes=[T, Axis(), X, ...])


def test_embed_an_open_system_with_room_for_unknown_axes() -> None:
    embedded = CS(axes=[X, ...]).embed([1, 0, 3], ndim=4)
    assert embedded == CS(axes=[Axis(), X, Axis(), Axis()])


def test_embed_an_unknown_system(unknown_axes: tx.Optional[list]) -> None:
    embedded = CS(axes=unknown_axes).embed([1])
    assert embedded == CS(axes=[Axis(), Axis(), ...])
    assert CS(axes=unknown_axes).embed([]) == CS(axes=[...])


def test_embed_nothing_with_a_known_size() -> None:
    assert CS(axes=[]).embed([], ndim=2) == CS(axes=[Axis(), Axis()])


def test_embed_refuses_a_count_mismatch() -> None:
    with pytest.raises(ValueError, match="list of 2 axes at 1"):
        CS(axes=[X, Y]).embed([0])
    with pytest.raises(ValueError, match="list of 2 axes at 3"):
        CS(axes=[X, Y]).embed([0, 1, 2])
    with pytest.raises(ValueError, match="2 explicit axes at 1"):
        CS(axes=[X, ..., Y]).embed([0])


def test_embed_refuses_bad_positions() -> None:
    with pytest.raises(ValueError, match="negative"):
        CS(axes=[X]).embed([-1])
    with pytest.raises(ValueError, match="more than once"):
        CS(axes=[X, Y]).embed([1, 1])
    with pytest.raises(TypeError):
        CS(axes=[X]).embed(["x"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        CS(axes=[X]).embed(0)  # type: ignore[arg-type]


def test_embed_refuses_a_size_that_is_too_small() -> None:
    with pytest.raises(ValueError, match="position 2 of a space of 2"):
        CS(axes=[X]).embed([2], ndim=2)


# ----------------------------------------------------------------------
#   COMPATIBLE WITH
# ----------------------------------------------------------------------


def test_compatible_closed_systems() -> None:
    assert CS(axes=[X, Y]).compatible_with(CS(axes=[X, Y]))
    assert CS(axes=[X, Axis()]).compatible_with(CS(axes=[Axis(), Y]))
    assert not CS(axes=[X, Y]).compatible_with(CS(axes=[Y, X]))
    assert not CS(axes=[X, Y]).compatible_with(CS(axes=[X, Y, Z]))
    assert CS(axes=[]).compatible_with(CS(axes=[]))


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
    assert system.compatible_with(CS(axes=closed)) is ok
    assert CS(axes=closed).compatible_with(system) is ok


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
    assert CS(axes=first).compatible_with(CS(axes=second)) is ok
    assert CS(axes=second).compatible_with(CS(axes=first)) is ok


def test_an_unknown_system_is_compatible_with_every_system(
    unknown_axes: tx.Optional[list],
) -> None:
    unknown = CS(axes=unknown_axes)
    for axes in (None, [...], [], [X], [X, ...], [..., T], [X, Y, Z]):
        assert unknown.compatible_with(CS(axes=axes))
        assert CS(axes=axes).compatible_with(unknown)
        assert CS(axes=axes).compatible_with(None)
        assert AxisList.of(None).compatible_with(AxisList.of(CS(axes=axes)))


def test_compatible_ras_with_time() -> None:
    # The example of the design: `[RAS axes, ...]` is compatible with a
    # closed RAS and time system.
    ras = list(RASCoordinateSystem().axes)
    assert CS(axes=[*ras, ...]).compatible_with(CS(axes=[*ras, TimeAxis()]))
    assert CS(axes=[*ras, ...]).compatible_with(RASCoordinateSystem())


def test_compatible_ignores_the_names_of_the_systems() -> None:
    assert CS(name="a", axes=[X]).compatible_with(CS(name="b", axes=[X]))


def test_compatible_refuses_a_non_system() -> None:
    with pytest.raises(TypeError):
        CS().compatible_with([X])  # type: ignore[arg-type]
