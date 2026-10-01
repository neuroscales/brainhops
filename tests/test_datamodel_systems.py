"""Tests for coordinate systems: open systems (`...`), positional and
named access to their axes, and the operations that close, restrict and
embed them."""

import collections.abc

import pytest
import typing_extensions as tx
from bagof.converters import ConversionError
from bagof.magic import replace

from brainhops.datamodel import systems as _systems
from brainhops.datamodel.axes import (
    A,
    Axis,
    L,
    P,
    R,
    S,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.systems import (
    ArrayCoordinateSystem,
    ArrayCoordinateSystem2D,
    ArrayCoordinateSystem3D,
    AxisList,
    AxisSequence,
    AxisTuple,
    CoordinateSystem,
    CoordinateSystem2D,
    CoordinateSystem3D,
    FRASCoordinateSystem,
    FVoxelCoordinateSystem,
    LPSmm,
    PhysicalCoordinateSystem,
    PixelCoordinateSystem,
    RASCoordinateSystem,
    RASmm,
    SpatialCoordinateSystem,
    SpatialCoordinateSystem2D,
    SpatialCoordinateSystem3D,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import Identity

CS = CoordinateSystem
X, Y, Z = Axis(name="x"), Axis(name="y"), Axis(name="z")
# A time axis leaves its unit unspecified unless it is given one.
T = TimeAxis(name="t", unit="second")

# "Nothing is known about the axes", given as a list or as a tuple. Both
# are stored as `[...]`, the default; `axes=None` is refused.
UNKNOWN_SPELLINGS = {"[...]": [...], "(...,)": (...,)}

# Where `...` sits among two explicit axes, `X` and `T`.
OPEN_LAYOUTS = {
    "start": [..., X, T],
    "middle": [X, ..., T],
    "end": [X, T, ...],
}

# The same two axes, closed or with `...` anywhere.
ANY_LAYOUTS = {"closed": [X, T], **OPEN_LAYOUTS}

# Every coordinate system class the module exports.
SYSTEM_CLASSES = [
    getattr(_systems, name)
    for name in _systems.__all__
    if isinstance(getattr(_systems, name), type)
    and issubclass(getattr(_systems, name), CoordinateSystem)
]

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
def unknown_axes(request: pytest.FixtureRequest) -> tx.Sequence:
    """The axes of a system about which nothing is known."""
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


def test_unknown_axes_are_stored_as_the_default(
    unknown_axes: tx.Sequence,
) -> None:
    system = CS(axes=unknown_axes)
    assert type(system.axes) is AxisList and system.axes == [...]
    assert system == CS()


OPEN_CLASSES = [
    CS,
    SpatialCoordinateSystem,
    ArrayCoordinateSystem,
    PhysicalCoordinateSystem,
]


@pytest.mark.parametrize("cls", OPEN_CLASSES, ids=lambda c: c.__name__)
def test_axes_none_is_refused(cls: type) -> None:
    # One spelling of unknown axes: `[...]`, which the error names.
    with pytest.raises(TypeError, match=r"cannot be None: use `\[\.\.\.\]`"):
        cls(axes=None)


def test_axes_default_to_ellipsis() -> None:
    assert CS().axes == [...] and type(CS().axes) is AxisList
    assert SpatialCoordinateSystem().axes == [...]
    assert ArrayCoordinateSystem().axes == [...]
    # Each system has a list of its own.
    first, second = CS(), CS()
    first.axes.append(X)
    assert second.axes == [...] and CS().axes == [...]


def test_ellipsis_may_sit_anywhere(layout: str) -> None:
    assert CS(axes=OPEN_LAYOUTS[layout]).axes == OPEN_LAYOUTS[layout]


@pytest.mark.parametrize("axes", [[..., ...], [..., X, ...], [X, ..., Y, ...]])
def test_at_most_one_ellipsis(axes: list) -> None:
    with pytest.raises(ValueError, match="at most one"):
        CS(axes=axes)


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
@pytest.mark.parametrize(
    "where, error, match",
    [
        # The type of the field, a tuple of `ndim` axes, refuses them: it
        # is not optional, it has a fixed length, and `...` is no axis.
        ("None", TypeError, "lists every one of them"),
        ("start", ValueError, "Expected iterable of length"),
        ("middle", ValueError, "Expected iterable of length"),
        ("end", ValueError, "Expected iterable of length"),
        ("instead of an axis", ConversionError, "`...` is not an axis"),
    ],
)
def test_fixed_dimension_classes_refuse_an_open_system(
    cls: type, ndim: int, where: str, error: type, match: str
) -> None:
    explicit = list(cls().axes)
    axes = {
        "None": None,
        "start": [..., *explicit],
        "middle": [explicit[0], ..., *explicit[1:]],
        "end": [*explicit, ...],
        "instead of an axis": [*explicit[:-1], ...],
    }[where]
    with pytest.raises(error, match=match):
        cls(axes=axes)


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_fixed_dimension_classes_store_a_tuple(cls: type, ndim: int) -> None:
    # Their axes are an `AxisTuple` of `ndim` axes, which is a tuple, with
    # the API of every axis sequence.
    for given in (list, tuple, AxisList, AxisTuple):
        axes = cls(axes=given(cls().axes)).axes
        assert type(axes) is AxisTuple and len(axes) == ndim
        assert isinstance(axes, tuple) and axes.ndim == ndim


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_fixed_dimension_classes_are_closed(cls: type, ndim: int) -> None:
    assert cls().ndim == ndim


def test_open_spatial_system() -> None:
    # A system whose number of axes is not fixed by its class may be open.
    system = SpatialCoordinateSystem(axes=[SpaceAxis(name="x"), ...])
    assert system.ndim is None


# ----------------------------------------------------------------------
#   NDIM
# ----------------------------------------------------------------------


def test_ndim_of_a_closed_system() -> None:
    assert CS(axes=[]).ndim == 0
    assert CS(axes=[X, Y, Z]).ndim == 3


def test_ndim_of_an_open_system(layout: str) -> None:
    assert CS(axes=OPEN_LAYOUTS[layout]).ndim is None


def test_ndim_of_an_unknown_system(unknown_axes: tx.Sequence) -> None:
    assert CS(axes=unknown_axes).ndim is None


def test_ndim_of_a_missing_system() -> None:
    assert _systems._axes_or_unknown(None).ndim is None


# ----------------------------------------------------------------------
#   EQUALITY
# ----------------------------------------------------------------------


def test_the_default_equals_ellipsis() -> None:
    assert CS() == CS(axes=[...])
    assert CS(name="s") == CS(name="s", axes=[...])
    assert not CS() != CS(axes=[...])
    assert SpatialCoordinateSystem() == SpatialCoordinateSystem(axes=[...])


def test_equality_stays_structural(layout: str) -> None:
    axes = OPEN_LAYOUTS[layout]
    assert CS(axes=axes) == CS(axes=list(axes))
    assert CS(axes=axes) != CS(axes=[...])
    assert CS(axes=axes) != CS(name="s", axes=axes)
    spatial = [SpaceAxis(), ...]
    assert CS(axes=spatial) != SpatialCoordinateSystem(axes=spatial)
    # Compatible, but not equal: equality does not expand `...`.
    assert CS(axes=[X, ...]) != CS(axes=[X, Axis()])


def test_equality_of_closed_systems_is_unchanged() -> None:
    assert CS(axes=[X, Y]) == CS(axes=(X, Y))
    assert CS(axes=[X, Y]) != CS(axes=[Y, X])
    assert CS(axes=[X]) != CS(axes=[Axis()])
    assert RASCoordinateSystem() == RASCoordinateSystem()
    assert RASCoordinateSystem() != CoordinateSystem3D(axes=(R(), A(), Axis()))


def test_no_system_equals_a_missing_system(unknown_axes: tx.Sequence) -> None:
    # Equality is ordinary: a plain system that says nothing is still a
    # system, and `None` is not one. Whether an endpoint tells anything is
    # what `_is_informative` answers (see below).
    system = CS(axes=unknown_axes)
    assert system != None  # noqa: E711
    assert None != system  # noqa: E711
    assert not (system == None)  # noqa: E711
    # So two transformations that spell a missing endpoint differently are
    # told apart, field by field, as any other two are.
    assert Identity(input=system) != Identity()
    assert Identity(input=system) == Identity(input=CS())


def test_a_system_that_says_something_differs_from_a_missing_system(
    unknown_axes: tx.Sequence,
) -> None:
    assert CS(name="s", axes=unknown_axes) != None  # noqa: E711
    assert CS(axes=[X, ...]) != None  # noqa: E711
    assert SpatialCoordinateSystem(axes=unknown_axes) != None  # noqa: E711
    assert ArrayCoordinateSystem(axes=unknown_axes) != None  # noqa: E711


@pytest.mark.parametrize(
    "system, informative",
    [
        (None, False),
        (CS(), False),
        (CS(axes=[...]), False),
        (CS(axes=(...,)), False),
        (CS(name="s"), True),
        (CS(axes=[X, ...]), True),
        (CS(axes=[..., X]), True),
        (CS(axes=[]), True),
        (CS(axes=[X]), True),
        (CS(axes=[X, Y]), True),  # a CoordinateSystem2D
        (SpatialCoordinateSystem(), True),
        (ArrayCoordinateSystem(), True),
        (RASCoordinateSystem(), True),
    ],
    ids=lambda v: repr(v) if not isinstance(v, bool) else str(v),
)
def test_is_informative(system: tx.Optional[CS], informative: bool) -> None:
    # A missing system, and a plain unnamed `CoordinateSystem` whose axes
    # are `[...]`, tell nothing; any name, axis or class of its own tells
    # something.
    assert _systems._is_informative(system) is informative


def test_systems_stay_unhashable() -> None:
    with pytest.raises(TypeError):
        hash(CS())


def test_equality_is_field_wise_in_a_new_subclass() -> None:
    class Mine(CoordinateSystem):
        pass

    assert Mine() == Mine(axes=[...])
    assert Mine(axes=[X, ...]) == Mine(axes=(X, ...))
    assert Mine(axes=[X, ...]) != Mine(axes=[Y, ...])
    assert Mine(name="a") != Mine(name="b")
    assert Mine() != CS()


@pytest.mark.parametrize("cls", SYSTEM_CLASSES, ids=lambda c: c.__name__)
def test_every_system_class_compares_field_by_field(cls: type) -> None:
    # No class writes an equality of its own: bagof's field-wise one,
    # which every class gets -- whether it is registered with a decorator
    # (the C- and F-ordered ones), selected on a narrowed constraint
    # (`RASmm`), or both -- is the right one, now that the axes have one
    # spelling of "unknown".
    if cls is PhysicalCoordinateSystem:
        system = cls(axes=[R(unit="mm")])
        other = cls(axes=[R(unit="cm")])
    else:
        system = cls()
        other = replace(system, name="something else")
    assert system == replace(system)
    assert system != other
    assert cls.__hash__ is None


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
        given = sequence([SpaceAxis(name="x"), ...])
        assert SpatialCoordinateSystem(axes=given).axes == list(given)
        with pytest.raises(TypeError):
            SpatialCoordinateSystem(axes=sequence([TimeAxis(), ...]))
    assert CS(axes=["x", ...]).axes == [Axis(name="x"), ...]
    # An axis of the right type is kept as it is.
    assert CS(axes=[X, ...]).axes[0] is X
    # So is an `AxisList`: the field is not optional, so no union takes
    # one as it is without looking at its items.
    given = AxisList([Axis(name="x"), ...])
    assert type(SpatialCoordinateSystem(axes=given).axes[0]) is SpaceAxis
    with pytest.raises(TypeError):
        SpatialCoordinateSystem(axes=AxisList([TimeAxis(), ...]))


def test_an_axis_list_is_a_plain_list(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    assert isinstance(axes, list)
    assert axes == entries
    assert repr(axes) == repr(entries)
    assert type(axes[1:]) is AxisList and axes[1:] == entries[1:]


def test_the_axes_of_a_system(any_layout: str) -> None:
    # A closed system of two axes is a `CoordinateSystem2D`, which stores
    # them as an `AxisTuple`; an open one stores an `AxisList`.
    system = CS(axes=ANY_LAYOUTS[any_layout])
    closed = any_layout == "closed"
    assert type(system.axes) is (AxisTuple if closed else AxisList)
    assert list(system.axes) == ANY_LAYOUTS[any_layout]
    # A copy, which a standard constructor makes, leaves the system as it
    # is.
    axes = AxisList(system.axes)
    axes.append(Z)
    assert list(system.axes) == ANY_LAYOUTS[any_layout]


def test_the_axes_of_a_missing_system_are_unknown() -> None:
    # Only a missing endpoint (`None`) has no `axes` to read, and reads as
    # `[...]`; any system reads as its own axes.
    assert _systems._axes_or_unknown(None) == [...]
    assert type(_systems._axes_or_unknown(None)) is AxisList
    for system in (CS(), CS(axes=[X, ...]), RASCoordinateSystem()):
        assert _systems._axes_or_unknown(system) is system.axes


def test_the_axes_of_a_fixed_dimension_system() -> None:
    axes = RASCoordinateSystem().axes
    assert type(axes) is AxisTuple and axes.ndim == 3
    assert AxisList(axes) == list(axes)


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
    # (`Axis(type="space")` builds a `SpaceAxis`, so the plain axis is one
    # that names no type.)
    plain = Axis(name="x", unit="mm")
    assert type(plain) is Axis
    spatial = SpaceAxis(name="x")
    axes = AxisList([plain, spatial])
    # A plain `Axis` query matches any axis; a `SpaceAxis` query only a
    # `SpaceAxis` (or a subclass), whatever the fields of the others.
    assert axes.index(Axis(name="x")) == 0
    assert axes.index(SpaceAxis(name="x")) == 1
    assert AxisList([R()]).index(SpaceAxis()) == 0
    with pytest.raises(ValueError):
        AxisList([plain]).index(SpaceAxis())


def test_index_never_matches_ellipsis(unknown_axes: tx.Sequence) -> None:
    for system in (CS(axes=unknown_axes), None):
        axes = _systems._axes_or_unknown(system)
        for query in (Axis(), "x", TimeAxis()):
            with pytest.raises(ValueError, match="is not in list"):
                axes.index(query)


def test_index_of_a_missing_axis(any_layout: str) -> None:
    axes = AxisList(ANY_LAYOUTS[any_layout])
    with pytest.raises(ValueError, match=r"Axis\(name='y'\) is not in list"):
        axes.index("y")
    with pytest.raises(ValueError):
        axes.index(SpaceAxis())


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
            assert axes.at(position) is axes[entry]


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
    unknown_axes: tx.Sequence,
) -> None:
    for system in (CS(axes=unknown_axes), None):
        for i in (0, 3, -1):
            assert _systems._axes_or_unknown(system)._position(i) == i


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
        assert axes.at(position) == expected
        assert axes.restrict([position]) == [expected]


def test_the_axis_at_a_position_of_an_unknown_system(
    unknown_axes: tx.Sequence,
) -> None:
    for system in (CS(axes=unknown_axes), None):
        for position in (0, 2, -1):
            assert _systems._axes_or_unknown(system).at(position) == Axis()


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


def test_expand_an_unknown_system(unknown_axes: tx.Sequence) -> None:
    system = CS(axes=unknown_axes)
    assert system.expand(0) == CS(axes=[])
    assert system.expand(2) == CS(axes=[Axis(), Axis()])


def test_expand_keeps_the_class_and_the_name() -> None:
    # The class is called again on the closed axes, so it keeps the name,
    # and builds the subclass that two axes select from it.
    system = ArrayCoordinateSystem(axes=[Axis(name="i"), ...])
    expanded = system.expand(2)
    assert type(expanded) is ArrayCoordinateSystem2D
    assert isinstance(expanded, ArrayCoordinateSystem)
    assert expanded.name == "array"
    assert list(expanded.axes) == [Axis(name="i"), Axis()]
    # A number of axes no subclass is selected on keeps the class itself.
    expanded = system.expand(4)
    assert type(expanded) is ArrayCoordinateSystem
    assert expanded.name == "array"
    assert expanded.axes == [Axis(name="i"), Axis(), Axis(), Axis()]


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
    unknown_axes: tx.Sequence,
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
    axes = embedded.axes
    assert axes.at(3).discrete is True
    assert axes.at(1) == Axis()
    assert axes.at(7) == Axis()


def test_embed_an_open_system(layout: str) -> None:
    # An open system is first closed to one axis per position: wherever
    # `...` sits, two positions close it to `[X, T]`.
    system = CS(axes=OPEN_LAYOUTS[layout])
    assert system.embed([2, 0], ndim=3) == CS(axes=[T, Axis(), X])
    assert system.embed([2, 0]) == CS(axes=[T, Axis(), X, ...])


def test_embed_an_open_system_with_room_for_unknown_axes() -> None:
    embedded = CS(axes=[X, ...]).embed([1, 0, 3], ndim=4)
    assert embedded == CS(axes=[Axis(), X, Axis(), Axis()])


def test_embed_an_unknown_system(unknown_axes: tx.Sequence) -> None:
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
    unknown_axes: tx.Sequence,
) -> None:
    unknown = CS(axes=unknown_axes)
    for axes in ([...], [], [X], [X, ...], [..., T], [X, Y, Z]):
        assert unknown.compatible_with(CS(axes=axes))
        assert CS(axes=axes).compatible_with(unknown)
        assert CS(axes=axes).compatible_with(None)
        assert AxisList([...]).compatible_with(CS(axes=axes).axes)


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


# ----------------------------------------------------------------------
#   DISPATCH OF OPEN SYSTEMS
# ----------------------------------------------------------------------
# Every dispatch predicate says something about all the axes of a system
# (how many there are, or what each one is), which an open system does
# not know. So no predicate holds of one, and calling a class with open
# axes builds that class itself.

MM = "mm"
SAMPLE = "sample"


def _ras(unit: tx.Optional[str] = None) -> list:
    return [R(unit=unit), A(unit=unit), S(unit=unit)]


@pytest.mark.parametrize(
    "axes",
    [
        # two entries, but an unknown number of axes
        [X, ...],
        [..., X],
        # three spatial entries
        [SpaceAxis(), SpaceAxis(), ...],
        # RAS, in millimetres, then anything
        [*_ras(MM), ...],
        [..., *_ras(MM)],
        # sampled axes, then anything
        [Axis(unit=SAMPLE), Axis(unit=SAMPLE), ...],
        [...],
    ],
    ids=[
        "x...",
        "...x",
        "space",
        "ras-mm...",
        "...ras-mm",
        "array",
        "...",
    ],
)
def test_an_open_system_selects_no_subclass(axes: list) -> None:
    assert type(CS(axes=axes)) is CS


def test_the_default_system_selects_no_subclass() -> None:
    assert type(CS()) is CS


@pytest.mark.parametrize(
    "cls, axes",
    [
        (SpatialCoordinateSystem, [SpaceAxis(), SpaceAxis(), ...]),
        (SpatialCoordinateSystem, [*_ras(SAMPLE), ...]),
        (ArrayCoordinateSystem, [Axis(unit=SAMPLE), Axis(unit=SAMPLE), ...]),
    ],
)
def test_an_open_system_keeps_the_class_it_was_called_as(
    cls: type, axes: list
) -> None:
    assert type(cls(axes=axes)) is cls


@pytest.mark.parametrize(
    "predicate",
    [
        _systems._is2d,
        _systems._is3d,
        _systems._is_spatial,
        _systems._is_array,
        _systems._is_physical,
        _systems._is_anat("RAS"),
    ],
    ids=lambda p: p.__name__,
)
@pytest.mark.parametrize(
    "axes",
    [None, [...], [*_ras(MM), ...], [..., *_ras(SAMPLE)], [X, ..., T]],
    ids=["none", "...", "ras-mm...", "...ras-sample", "x...t"],
)
def test_no_dispatch_predicate_holds_of_an_open_list(
    predicate: tx.Callable, axes: tx.Optional[list]
) -> None:
    # Not an error, and not a match, wherever `...` sits.
    assert predicate(axes) is False
    if axes is not None:
        assert predicate(AxisList(axes)) is False


def test_a_closed_list_still_selects_its_class() -> None:
    assert type(CS(axes=[X, Y])) is CoordinateSystem2D
    assert type(CS(axes=[SpaceAxis(), SpaceAxis()])) is (
        SpatialCoordinateSystem2D
    )
    assert type(CS(axes=_ras())) is RASCoordinateSystem
    assert type(CS(axes=_ras(MM))) is RASmm
    assert type(ArrayCoordinateSystem(axes=[X, Y, Z])) is (
        ArrayCoordinateSystem3D
    )


@pytest.mark.parametrize(
    "cls, axes, ndim, closed",
    [
        (CS, [X, ...], 2, CoordinateSystem2D),
        (CS, [X, ...], 4, CS),
        (CS, [*_ras(MM), ...], 3, RASmm),
        (CS, [..., *_ras()], 3, RASCoordinateSystem),
        # The unknown axes are read as the spatial axes the class declares.
        (
            SpatialCoordinateSystem,
            [SpaceAxis(), ...],
            3,
            (SpatialCoordinateSystem3D),
        ),
        (SpatialCoordinateSystem, [...], 2, SpatialCoordinateSystem2D),
        (
            ArrayCoordinateSystem,
            [Axis(unit=SAMPLE), ...],
            3,
            (ArrayCoordinateSystem3D),
        ),
    ],
)
def test_expand_selects_the_class_of_the_closed_system(
    cls: type, axes: list, ndim: int, closed: type
) -> None:
    # Closing a system builds the class that listing the closed axes
    # does: the one its axes select. The name is the system's own.
    system = cls(axes=axes)
    expanded = system.expand(ndim)
    assert type(expanded) is closed
    assert type(cls(axes=list(expanded.axes))) is closed
    assert expanded.name == system.name


def test_an_axis_list_given_to_expand_is_converted_item_by_item() -> None:
    # Regression: `expand` handed the class an `AxisList`, which the
    # optional field takes as it is, so the unknown `Axis()` were not read
    # as spatial axes, and the closed system was not dispatched.
    expanded = SpatialCoordinateSystem(axes=[SpaceAxis(), ...]).expand(3)
    assert all(type(axis) is SpaceAxis for axis in expanded.axes)


def test_restrict_and_embed_build_what_their_axes_select() -> None:
    # The result describes another space, so neither the class nor the
    # name of the system is carried over: it is what `CoordinateSystem`
    # builds from the axes.
    ras = RASmm()
    assert ras.restrict([0, 1, 2]) == CS(axes=list(ras.axes))
    assert type(ras.restrict([0, 2])) is SpatialCoordinateSystem2D
    voxel = FVoxelCoordinateSystem()
    assert type(voxel.embed([0, 1, 2])) is CS
    assert voxel.embed([0, 1, 2]).ndim is None
    assert voxel.embed([0, 1, 2], ndim=3) == CS(axes=list(voxel.axes))
    assert type(voxel.embed([1, 2, 3], ndim=4)) is CS


# ----------------------------------------------------------------------
#   PHYSICAL SYSTEMS
# ----------------------------------------------------------------------
# A physical system vouches for a physical unit on every one of its
# axes. `...` is not an axis without a unit: it stands for axes about
# which nothing is known, their units included, so a physical system is
# always closed, and an open one is refused for being open.


@pytest.mark.parametrize(
    "axes",
    [
        [*_ras(MM), ...],
        [..., TimeAxis(unit="s")],
        [R(unit=MM), ..., S(unit=MM)],
    ],
    ids=["end", "start", "middle"],
)
def test_a_physical_system_refuses_ellipsis_as_an_open_system(
    axes: list,
) -> None:
    with pytest.raises(ValueError, match="lists every one of its axes") as e:
        PhysicalCoordinateSystem(axes=axes)
    # It is refused as an open system, not as an axis without a unit.
    assert "carries" not in str(e.value)
    assert "close the system first" in str(e.value)


def test_a_physical_system_refuses_unknown_axes() -> None:
    # `[...]`, the default, states no axis, and is refused as a system
    # without axes, as `[]` is.
    for axes in ([...], [], (...,)):
        with pytest.raises(ValueError, match="must have axes"):
            PhysicalCoordinateSystem(axes=axes)
    with pytest.raises(ValueError, match="must have axes"):
        PhysicalCoordinateSystem()


def test_a_physical_system_still_refuses_an_axis_without_a_unit() -> None:
    for unit in (None, SAMPLE):
        with pytest.raises(ValueError, match="carries"):
            PhysicalCoordinateSystem(axes=[R(unit=MM), A(unit=unit)])


def test_a_closed_physical_system_is_built() -> None:
    system = PhysicalCoordinateSystem(axes=[R(unit=MM), TimeAxis(unit="s")])
    assert system.ndim == 2
    assert type(PhysicalCoordinateSystem(axes=_ras(MM))) is RASmm


def test_an_open_system_in_millimetres_closes_to_a_physical_one() -> None:
    # An open system is not physical, whatever its explicit axes say; it
    # becomes physical once closed, if every axis then carries a unit.
    system = CS(axes=[..., L(unit=MM), P(unit=MM), S(unit=MM)])
    assert type(system) is CS
    assert type(system.expand(3)) is LPSmm
    # The axes `...` closes to carry no unit, so no physical system holds
    # them.
    assert type(CS(axes=[R(unit=MM), ...]).expand(3)) is CoordinateSystem3D


# ----------------------------------------------------------------------
#   AXIS CONTAINERS
# ----------------------------------------------------------------------
# `AxisSequence` is the read-only base; `AxisTuple` (a tuple) and
# `AxisList` (a list) share all of its API.

CONTAINERS = [AxisList, AxisTuple]


def test_the_hierarchy_of_the_axis_containers() -> None:
    assert issubclass(AxisSequence, collections.abc.Sequence)
    assert not issubclass(AxisSequence, (list, tuple))
    assert issubclass(AxisList, AxisSequence) and issubclass(AxisList, list)
    assert issubclass(AxisTuple, AxisSequence)
    assert issubclass(AxisTuple, tuple)
    assert not issubclass(AxisList, tuple)
    assert not issubclass(AxisTuple, list)
    # `AxisSequence` is abstract: it says what a sequence of axes reads,
    # not how it stores them.
    with pytest.raises(TypeError):
        AxisSequence()  # type: ignore[abstract]


@pytest.mark.parametrize("cls", CONTAINERS, ids=lambda c: c.__name__)
def test_the_axis_sequence_reads_win_over_the_builtins(cls: type) -> None:
    # An item, a name and a query are read as `AxisSequence` reads them,
    # whichever of it and the builtin comes first in the bases.
    for name in ("__getitem__", "__contains__", "index"):
        assert getattr(cls, name) is getattr(AxisSequence, name)
    # Everything else is the builtin's, not the generic mixins of
    # `collections.abc.Sequence`.
    builtin = list if cls is AxisList else tuple
    for name in ("__len__", "__iter__", "count", "__eq__", "__repr__"):
        assert getattr(cls, name) is getattr(builtin, name)


@pytest.mark.parametrize("cls", CONTAINERS, ids=lambda c: c.__name__)
def test_the_shared_api(cls: type) -> None:
    axes = cls([X, ..., T])
    assert axes["x"] is X and "t" in axes and "y" not in axes
    assert X in axes and ... in axes
    assert axes.index("t") == 2 and axes.index(TimeAxis()) == 2
    assert axes.names == ("x", ..., "t")
    assert axes.ndim is None and axes.is_open
    assert axes[2] is T and axes.at(-1) is T and axes.at(2) == Axis()
    assert axes.compatible_with([X, Y, T])
    assert axes.compatible_with(AxisTuple([X, Y, T]))
    assert len(axes) == 3 and list(axes) == [X, ..., T]
    assert list(reversed(axes)) == [T, ..., X]


@pytest.mark.parametrize("cls", CONTAINERS, ids=lambda c: c.__name__)
def test_a_derived_sequence_is_of_the_same_type(cls: type) -> None:
    axes = cls([X, ..., T])
    for derived in (
        axes[1:],
        axes.expand(3),
        axes.restrict([0, -1]),
        axes.embed([2, 0]),
        axes.embed([2, 0], ndim=4),
    ):
        assert type(derived) is cls
    assert list(axes.expand(3)) == [X, Axis(), T]


@pytest.mark.parametrize("cls", CONTAINERS, ids=lambda c: c.__name__)
def test_at_is_the_axis_at_a_position(cls: type) -> None:
    closed = cls([X, Y, Z])
    assert [closed.at(i) for i in range(-3, 3)] == [X, Y, Z, X, Y, Z]
    for i in (3, -4):
        with pytest.raises(IndexError, match="out of range"):
            closed.at(i)
    with pytest.raises(TypeError):
        closed.at("x")  # type: ignore[arg-type]
    # Entries and positions differ in an open sequence.
    open_ = cls([X, ..., T])
    assert open_[2] is T
    assert open_.at(2) == Axis() and open_.at(-1) is T


def test_an_axis_tuple_is_immutable() -> None:
    axes = AxisTuple([X, Y])
    with pytest.raises(TypeError):
        axes[0] = Z  # type: ignore[index]
    assert not hasattr(axes, "append")
    # A fixed-dimension system's axes are one.
    with pytest.raises(TypeError):
        RASCoordinateSystem().axes[0] = R()  # type: ignore[index]


def test_an_axis_list_is_mutable() -> None:
    axes = AxisList([X, ...])
    axes.append(T)
    axes[0] = Y
    assert axes == [Y, ..., T] and axes.at(-1) is T


def test_the_containers_compare_as_their_builtins() -> None:
    assert AxisList([X, Y]) == [X, Y]
    assert AxisTuple([X, Y]) == (X, Y)
    assert repr(AxisTuple([X, Y])) == repr((X, Y))
    assert repr(AxisList([X, ...])) == repr([X, ...])


def test_every_system_stores_an_axis_sequence() -> None:
    for cls in SYSTEM_CLASSES:
        if cls is PhysicalCoordinateSystem:
            continue
        axes = cls().axes
        assert isinstance(axes, AxisSequence)
        fixed = issubclass(cls, (CoordinateSystem2D, CoordinateSystem3D))
        assert type(axes) is (AxisTuple if fixed else AxisList)
    # An open-capable class stores a tuple it is given as an `AxisList`.
    assert type(CS(axes=AxisTuple([X, ...])).axes) is AxisList


@pytest.mark.parametrize(
    "axes, error, match",
    [
        (None, TypeError, "lists every one of them"),
        ([SpaceAxis(), SpaceAxis()], ConversionError, "length 3, got 2"),
        (
            [SpaceAxis(), SpaceAxis(), SpaceAxis(), SpaceAxis()],
            ConversionError,
            "length 3, got 4",
        ),
        ([SpaceAxis(), SpaceAxis(), ...], ConversionError, "not an axis"),
        (
            [SpaceAxis(), SpaceAxis(), TimeAxis()],
            ConversionError,
            "always 'space'",
        ),
    ],
    ids=["none", "too-short", "too-long", "ellipsis", "wrong-kind"],
)
def test_a_fixed_axis_tuple_field_refuses(
    axes: tx.Optional[list], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        SpatialCoordinateSystem3D(axes=axes)


def test_a_fixed_axis_tuple_field_converts_each_position() -> None:
    system = SpatialCoordinateSystem3D(
        axes=AxisTuple([Axis(name="x"), {"name": "y"}, "z"])
    )
    assert type(system.axes) is AxisTuple
    assert [type(axis) for axis in system.axes] == [SpaceAxis] * 3
    assert system.axes.names == ("x", "y", "z")
    # Each position to its own type.
    ras = RASCoordinateSystem(axes=[Axis(name="r"), Axis(), Axis()])
    assert [type(axis) for axis in ras.axes] == [R, A, S]
