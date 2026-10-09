"""Tests for coordinate systems, open systems and their axis sequences."""

import collections.abc

import pytest
import typing_extensions as tx
from bagof.converters import ConversionError
from bagof.magic import replace

from brainhops.datamodel import systems as _systems
from brainhops.datamodel._sugar import get_axes
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
    RSAmm,
    SpatialCoordinateSystem,
    SpatialCoordinateSystem2D,
    SpatialCoordinateSystem3D,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import Identity

CS = CoordinateSystem
X, Y, Z = Axis(name="x"), Axis(name="y"), Axis(name="z")
# A time axis has no unit unless one is given.
T = TimeAxis(name="t", unit="second")

# Unknown axes spelled as a list or as a tuple. Both spellings are stored
# as the default [...].
UNKNOWN_SPELLINGS = {"[...]": [...], "(...,)": (...,)}

# Where `...` sits among the explicit axes X and T.
OPEN_LAYOUTS = {
    "start": [..., X, T],
    "middle": [X, ..., T],
    "end": [X, T, ...],
}

# The same two axes, closed or with `...` anywhere.
ANY_LAYOUTS = {"closed": [X, T], **OPEN_LAYOUTS}

# Every exported coordinate system class.
SYSTEM_CLASSES = [
    getattr(_systems, name)
    for name in _systems.__all__
    if isinstance(getattr(_systems, name), type)
    and issubclass(getattr(_systems, name), CoordinateSystem)
]

# Fixed-dimension classes with their axis count.
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
    """Axes of a system about which nothing is known."""
    return UNKNOWN_SPELLINGS[request.param]


@pytest.fixture(params=list(OPEN_LAYOUTS), ids=list(OPEN_LAYOUTS))
def layout(request: pytest.FixtureRequest) -> str:
    """Where `...` sits in an open system."""
    return request.param


@pytest.fixture(params=list(ANY_LAYOUTS), ids=list(ANY_LAYOUTS))
def any_layout(request: pytest.FixtureRequest) -> str:
    """A closed list of X and T, or `...` somewhere among them."""
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
def test_axes_none_reads_as_the_default(cls: type) -> None:
    # None is stored as [...], so the axes are always an axis sequence.
    system = cls(axes=None)
    assert type(system) is cls
    assert type(system.axes) is AxisList and system.axes == [...]
    assert system == cls() == cls(axes=[...])
    # Each system gets its own list.
    assert cls(axes=None).axes is not cls(axes=None).axes


def test_axes_none_reads_as_ellipsis_in_any_spelling() -> None:
    assert CS(axes=None) == CS() == CS(axes=[...])
    assert CS("s", None) == CS(name="s")
    assert CS.from_dict({"axes": None}) == CS()


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_axes_none_is_the_default_of_a_fixed_dimension_class(
    cls: type, ndim: int
) -> None:
    # A fixed-dimension class cannot store [...], so None gives its defaults.
    system = cls(axes=None)
    assert type(system) is cls and type(system.axes) is AxisTuple
    assert system == cls() and system.ndim == ndim


def test_axes_default_to_ellipsis() -> None:
    assert CS().axes == [...] and type(CS().axes) is AxisList
    assert SpatialCoordinateSystem().axes == [...]
    assert ArrayCoordinateSystem().axes == [...]
    # Each system has its own list.
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
        # A fixed-length tuple of axes refuses `...`, which is not an axis.
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
        "start": [..., *explicit],
        "middle": [explicit[0], ..., *explicit[1:]],
        "end": [*explicit, ...],
        "instead of an axis": [*explicit[:-1], ...],
    }[where]
    with pytest.raises(error, match=match):
        cls(axes=axes)


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_fixed_dimension_classes_store_a_tuple(cls: type, ndim: int) -> None:
    # The axes are an AxisTuple of ndim axes.
    for given in (list, tuple, AxisList, AxisTuple):
        axes = cls(axes=given(cls().axes)).axes
        assert type(axes) is AxisTuple and len(axes) == ndim
        assert isinstance(axes, tuple) and axes.ndim == ndim


@pytest.mark.parametrize("cls, ndim", FIXED_CLASSES)
def test_fixed_dimension_classes_are_closed(cls: type, ndim: int) -> None:
    assert cls().ndim == ndim


def test_open_spatial_system() -> None:
    # A system whose class does not fix the axis count may be open.
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
    assert get_axes(None).ndim is None


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
    # Compatible but not equal: equality does not expand `...`.
    assert CS(axes=[X, ...]) != CS(axes=[X, Axis()])


def test_equality_of_closed_systems_is_unchanged() -> None:
    assert CS(axes=[X, Y]) == CS(axes=(X, Y))
    assert CS(axes=[X, Y]) != CS(axes=[Y, X])
    assert CS(axes=[X]) != CS(axes=[Axis()])
    assert RASCoordinateSystem() == RASCoordinateSystem()
    assert RASCoordinateSystem() != CoordinateSystem3D(axes=(R(), A(), Axis()))


def test_no_system_equals_a_missing_system(unknown_axes: tx.Sequence) -> None:
    # An empty system is a system, while None is not.
    system = CS(axes=unknown_axes)
    assert system != None  # noqa: E711
    assert None != system  # noqa: E711
    assert not (system == None)  # noqa: E711
    # A missing endpoint differs from an explicit system with unknown axes.
    assert Identity(input=system).input != Identity().input
    assert Identity(input=system).input == Identity(input=CS()).input


def test_a_system_that_says_something_differs_from_a_missing_system(
    unknown_axes: tx.Sequence,
) -> None:
    assert CS(name="s", axes=unknown_axes) != None  # noqa: E711
    assert CS(axes=[X, ...]) != None  # noqa: E711
    assert SpatialCoordinateSystem(axes=unknown_axes) != None  # noqa: E711
    assert ArrayCoordinateSystem(axes=unknown_axes) != None  # noqa: E711


def test_an_explicit_endpoint_is_kept_and_a_missing_one_defers() -> None:
    # An endpoint is either None, which defers to the context, or a system.
    for system in (CS(), CS(axes=[...]), CS(name="s"), RASCoordinateSystem()):
        assert Identity(input=system).input is system
    assert Identity().input is None
    assert Identity(input=CS()).input != Identity().input


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
    # The field-wise equality of bagof suits every class, however registered.
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
    # An axis of the right type is kept as is.
    assert CS(axes=[X, ...]).axes[0] is X
    # An AxisList is converted too, since the field is not optional.
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
    # A closed system stores an AxisTuple and an open one an AxisList.
    system = CS(axes=ANY_LAYOUTS[any_layout])
    closed = any_layout == "closed"
    assert type(system.axes) is (AxisTuple if closed else AxisList)
    assert list(system.axes) == ANY_LAYOUTS[any_layout]
    # A copy made with the standard constructor leaves the system unchanged.
    axes = AxisList(system.axes)
    axes.append(Z)
    assert list(system.axes) == ANY_LAYOUTS[any_layout]


def test_the_axes_of_a_missing_system_are_unknown() -> None:
    # Only a missing endpoint reads [...]; a system reads its own axes.
    assert get_axes(None) == [...]
    assert type(get_axes(None)) is AxisList
    for system in (CS(), CS(axes=[X, ...]), RASCoordinateSystem()):
        assert get_axes(system) is system.axes


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
    # A name never matches the axes that `...` stands for.
    with pytest.raises(KeyError, match="'y'"):
        AxisList(ANY_LAYOUTS[any_layout])["y"]
    with pytest.raises(KeyError):
        AxisList([...])["x"]


def test_a_shared_name_is_ambiguous(any_layout: str) -> None:
    axes = AxisList([*ANY_LAYOUTS[any_layout], Axis(name="x")])
    with pytest.raises(ValueError, match="2 axes"):
        axes["x"]
    # Both `index` and `in` find the first match, as they do for lists.
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
    # Entry 2 is T, but the axis at position 2 is one that `...` stands for.
    axes = AxisList([X, ..., T])
    assert len(axes) == 3 and axes.ndim is None
    assert axes[2] is T and axes.index("t") == 2
    assert axes.restrict([2]) == [Axis()]
    assert axes.restrict([-1]) == [T]
    # Iterating over the entries yields `...` itself.
    assert [axes[i] for i in range(len(axes))] == [X, ..., T]


def test_index_of_a_name(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    assert axes.index("x") == entries.index(X)
    assert axes.index("t") == entries.index(T)


def test_index_of_an_axis(any_layout: str) -> None:
    entries = ANY_LAYOUTS[any_layout]
    axes = AxisList(entries)
    # An axis matches itself and any query that sets fewer fields.
    assert axes.index(T) == entries.index(T)
    assert axes.index(TimeAxis()) == entries.index(T)
    assert axes.index(Axis(unit="second")) == entries.index(T)
    assert axes.index(Axis(name="x")) == entries.index(X)
    # An empty query matches the first axis.
    assert axes.index(Axis()) == (1 if any_layout == "start" else 0)


def test_index_compares_only_the_fields_the_query_sets() -> None:
    axes = AxisList([Axis(), Axis(name="x"), Axis(name="x", unit="mm")])
    assert axes.index("x") == 1
    assert axes.index(Axis(name="x", unit="mm")) == 2
    # An unset field of an entry does not match a field that the query sets.
    assert axes.index(Axis(unit="mm")) == 2


def test_index_asks_for_the_class_of_the_query() -> None:
    # Axis(type='space') builds a SpaceAxis, so a plain axis has no type.
    plain = Axis(name="x", unit="mm")
    assert type(plain) is Axis
    spatial = SpaceAxis(name="x")
    axes = AxisList([plain, spatial])
    # A plain Axis query matches any axis; a SpaceAxis query only spatial ones.
    assert axes.index(Axis(name="x")) == 0
    assert axes.index(SpaceAxis(name="x")) == 1
    assert AxisList([R()]).index(SpaceAxis()) == 0
    with pytest.raises(ValueError):
        AxisList([plain]).index(SpaceAxis())


def test_index_never_matches_ellipsis(unknown_axes: tx.Sequence) -> None:
    for system in (CS(axes=unknown_axes), None):
        axes = get_axes(system)
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
    # Entries after `...` count from the end, and `...` has no position.
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
    # `...` stands for any count, so every position is returned as given.
    axes = AxisList(OPEN_LAYOUTS[layout])
    for i in (0, 1, 2, 5, 100, -1, -2, -3, -100):
        assert axes._position(i) == i


def test_a_position_in_an_unknown_system(
    unknown_axes: tx.Sequence,
) -> None:
    for system in (CS(axes=unknown_axes), None):
        for i in (0, 3, -1):
            assert get_axes(system)._position(i) == i


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
        # The axis at each explicit position.
        ("start", {-2: X, -1: T}),
        ("middle", {0: X, -1: T}),
        ("end", {0: X, 1: T}),
    ],
)
def test_the_axis_at_a_position_of_an_open_list(
    layout: str, known: dict
) -> None:
    # Non-negative positions read before `...` and negative ones after it.
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
            assert get_axes(system).at(position) == Axis()


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
    # The class is called again on the closed axes and keeps the name.
    system = ArrayCoordinateSystem(axes=[Axis(name="i"), ...])
    expanded = system.expand(2)
    assert type(expanded) is ArrayCoordinateSystem2D
    assert isinstance(expanded, ArrayCoordinateSystem)
    assert expanded.name == "array"
    assert list(expanded.axes) == [Axis(name="i"), Axis()]
    # An axis count with no subclass keeps the class itself.
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
    # Each known position reads back the axis embedded there.
    embedded = CS(axes=[X, Axis(name="c", discrete=True)]).embed([0, 3])
    axes = embedded.axes
    assert axes.at(3).discrete is True
    assert axes.at(1) == Axis()
    assert axes.at(7) == Axis()


def test_embed_an_open_system(layout: str) -> None:
    # An open system is closed to one axis per position first.
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
    # [RAS axes, ...] is compatible with a closed RAS and time system.
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
# Every dispatch predicate makes a claim about all axes, which an open
# system does not know. No predicate therefore holds, and a class called
# with open axes builds that class itself.

MM = "mm"
SAMPLE = "index"


def _ras(unit: tx.Optional[str] = None) -> list:
    return [R(unit=unit), A(unit=unit), S(unit=unit)]


@pytest.mark.parametrize(
    "axes",
    [
        # Two entries and an unknown axis count.
        [X, ...],
        [..., X],
        # Three spatial entries.
        [SpaceAxis(), SpaceAxis(), ...],
        # RAS in mm, then anything.
        [*_ras(MM), ...],
        [..., *_ras(MM)],
        # Sampled axes, then anything.
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
        _systems._is_mm,
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
    # The predicate neither raises nor matches, wherever `...` sits.
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
        # The unknown axes read as the declared spatial axes of the class.
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
    # Closing builds the class that the closed axes select, with the name.
    system = cls(axes=axes)
    expanded = system.expand(ndim)
    assert type(expanded) is closed
    assert type(cls(axes=list(expanded.axes))) is closed
    assert expanded.name == system.name


def test_an_axis_list_given_to_expand_is_converted_item_by_item() -> None:
    # Regression: unknown axes were not read as spatial, so no dispatch.
    expanded = SpatialCoordinateSystem(axes=[SpaceAxis(), ...]).expand(3)
    assert all(type(axis) is SpaceAxis for axis in expanded.axes)


def test_restrict_and_embed_build_what_their_axes_select() -> None:
    # The result describes another space, so neither class nor name is kept.
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
# A generic physical system gives each axis the physical unit of its kind,
# or none, and may be open. A millimetre system takes millimetres only.


@pytest.mark.parametrize(
    "axes",
    [
        [...],
        [],
        [*_ras(MM), ...],
        [..., TimeAxis(unit="s")],
        [R(unit=MM), ..., S(unit=MM)],
        [R(), A(unit="cm")],
        [Axis(unit="s"), SpaceAxis(unit="um")],
        [R(), ..., TimeAxis()],
    ],
    ids=[
        "unknown",
        "empty",
        "end",
        "start",
        "middle",
        "unspecified-unit",
        "other-units",
        "open-and-unspecified",
    ],
)
def test_a_physical_system_takes_open_axes_and_unspecified_units(
    axes: list,
) -> None:
    system = PhysicalCoordinateSystem(axes=axes)
    assert type(system) is PhysicalCoordinateSystem
    assert list(system.axes) == axes


def test_a_physical_system_with_no_axes_given() -> None:
    for system in (
        PhysicalCoordinateSystem(),
        PhysicalCoordinateSystem(axes=None),
    ):
        assert system.axes == [...] and system.ndim is None


@pytest.mark.parametrize(
    "axes",
    [
        [R(unit=MM), A(unit=SAMPLE)],
        [..., SpaceAxis(unit=SAMPLE)],
        [TimeAxis(unit=SAMPLE)],
    ],
)
def test_a_physical_system_refuses_the_sample(axes: list) -> None:
    with pytest.raises(ValueError, match="counts samples"):
        PhysicalCoordinateSystem(axes=axes)


def test_a_physical_system_takes_a_unit_of_the_axis_kind_only() -> None:
    # The axis type refuses a unit of the wrong kind first.
    with pytest.raises(ConversionError, match="SpaceAxis.unit"):
        PhysicalCoordinateSystem(axes=[SpaceAxis(unit="s")])
    with pytest.raises(ConversionError, match="TimeAxis.unit"):
        PhysicalCoordinateSystem(axes=[TimeAxis(unit="mm")])


def test_no_array_pixel_or_voxel_system_is_physical() -> None:
    for cls in SYSTEM_CLASSES:
        if issubclass(cls, ArrayCoordinateSystem):
            assert not issubclass(cls, PhysicalCoordinateSystem), cls
    physical = {
        cls
        for cls in SYSTEM_CLASSES
        if issubclass(cls, PhysicalCoordinateSystem)
    }
    assert physical == {PhysicalCoordinateSystem, RASmm, LPSmm, RSAmm}


@pytest.mark.parametrize(
    "cls", [RASmm, LPSmm, RSAmm], ids=lambda c: c.__name__
)
@pytest.mark.parametrize(
    "unit", [None, "m", "cm", "um", SAMPLE], ids=lambda u: str(u)
)
def test_a_millimetre_system_refuses_any_other_unit(
    cls: type, unit: tx.Optional[str]
) -> None:
    axes = [type(axis)(unit=unit) for axis in cls().axes]
    with pytest.raises(ValueError, match="in millimetres|counts samples"):
        cls(axes=axes)
    # A single axis in another unit is refused.
    mixed = list(cls().axes)
    mixed[1] = type(mixed[1])(unit=unit)
    with pytest.raises(ValueError, match="in millimetres|counts samples"):
        cls(axes=mixed)


@pytest.mark.parametrize(
    "cls", [RASmm, LPSmm, RSAmm], ids=lambda c: c.__name__
)
def test_a_millimetre_system_defaults_to_millimetres(cls: type) -> None:
    for system in (cls(), cls(axes=None)):
        assert all(str(axis.unit) == "millimeter" for axis in system.axes)
    # A millimetre system has three axes, so `...` is refused.
    with pytest.raises(ConversionError):
        cls(axes=[*cls().axes[:2], ...])


@pytest.mark.parametrize(
    "unit, expected",
    [
        (MM, RASmm),
        (None, RASCoordinateSystem),
        ("m", RASCoordinateSystem),
        ("cm", RASCoordinateSystem),
        (SAMPLE, RASCoordinateSystem),
    ],
    ids=lambda v: str(v) if not isinstance(v, type) else v.__name__,
)
def test_only_millimetres_select_a_millimetre_system(
    unit: tx.Optional[str], expected: type
) -> None:
    # RAS axes in another unit give a RASCoordinateSystem.
    assert type(CS(axes=_ras(unit))) is expected
    assert type(RASCoordinateSystem(axes=_ras(unit))) is expected
    # Mixed units select no millimetre system.
    mixed = [R(unit=MM), A(unit=MM), S(unit=unit)]
    assert type(CS(axes=mixed)) is expected


def test_a_physical_system_in_another_unit_stays_physical() -> None:
    system = PhysicalCoordinateSystem(axes=_ras("cm"))
    assert type(system) is PhysicalCoordinateSystem
    assert type(PhysicalCoordinateSystem(axes=_ras(MM))) is RASmm


def test_an_open_system_in_millimetres_closes_to_a_physical_one() -> None:
    # An open system is not physical, but it can close to a physical one.
    system = CS(axes=[..., L(unit=MM), P(unit=MM), S(unit=MM)])
    assert type(system) is CS
    assert type(system.expand(3)) is LPSmm
    # The axes that `...` closes to have no unit.
    assert type(CS(axes=[R(unit=MM), ...]).expand(3)) is CoordinateSystem3D


# ----------------------------------------------------------------------
#   AXIS CONTAINERS
# ----------------------------------------------------------------------
# AxisSequence is a read-only base whose API AxisTuple and AxisList share.

CONTAINERS = [AxisList, AxisTuple]


def test_the_hierarchy_of_the_axis_containers() -> None:
    assert issubclass(AxisSequence, collections.abc.Sequence)
    assert not issubclass(AxisSequence, (list, tuple))
    assert issubclass(AxisList, AxisSequence) and issubclass(AxisList, list)
    assert issubclass(AxisTuple, AxisSequence)
    assert issubclass(AxisTuple, tuple)
    assert not issubclass(AxisList, tuple)
    assert not issubclass(AxisTuple, list)
    # AxisSequence is abstract and says nothing about storage.
    with pytest.raises(TypeError):
        AxisSequence()  # type: ignore[abstract]


@pytest.mark.parametrize("cls", CONTAINERS, ids=lambda c: c.__name__)
def test_the_axis_sequence_reads_win_over_the_builtins(cls: type) -> None:
    # The item, name and query reads come from AxisSequence.
    for name in ("__getitem__", "__contains__", "index"):
        assert getattr(cls, name) is getattr(AxisSequence, name)
    # Everything else comes from the builtin type.
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
    # The axes of a fixed-dimension system are an immutable tuple.
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
    # A class that may be open stores a given tuple as an AxisList.
    assert type(CS(axes=AxisTuple([X, ...])).axes) is AxisList


@pytest.mark.parametrize(
    "axes, error, match",
    [
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
    ids=["too-short", "too-long", "ellipsis", "wrong-kind"],
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
    # Each position converts to its own axis type.
    ras = RASCoordinateSystem(axes=[Axis(name="r"), Axis(), Axis()])
    assert [type(axis) for axis in ras.axes] == [R, A, S]
