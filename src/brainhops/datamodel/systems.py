"""Coordinate systems, from unitless arrays to anatomical spaces.

Calling [`CoordinateSystem`][] builds the most specific system its axes
describe -- the dispatch is bagof's polymorphism, driven by the `on=`
constraint of each class:

| The axes are...                               | ...so the system is        |
| --------------------------------------------- | -------------------------- |
| two or three of anything                      | `CoordinateSystem2D`/`3D`  |
| all spatial                                   | `SpatialCoordinateSystem*` |
| two or three, all measured in samples         | `ArrayCoordinateSystem*`   |
| spatial *and* measured in samples             | `Pixel`/`VoxelCoordinate…` |
| oriented right, anterior, superior (in order) | `RASCoordinateSystem`      |
| ... and in a physical unit                    | `RASmm`                    |

and likewise for LPS and RSA. A class that inherits from two dispatch
targets -- `SpatialCoordinateSystem3D` from `CoordinateSystem3D` and
`SpatialCoordinateSystem` -- is selected on what both stand for, with no
constraint of its own. The C- and F-ordered variants cannot be told apart
from the axes (the memory order is not written on them), so they are
reached only from their own ordered base: `FVoxelCoordinateSystem(axes=<RAS
axes>)` builds an `FRASCoordinateSystem`.

Every row of the table says something about *all* the axes, so only a
closed system is dispatched. An open system -- one whose axes hold `...`
([`AxisList`][]) -- does not know all its axes, so it is built as the
class it was called as: `CoordinateSystem(axes=[x, ...])` is
not two-dimensional, and `CoordinateSystem(axes=[R(), A(), S(), ...])` is
not an `RASCoordinateSystem`. [`CoordinateSystem.expand`][] closes it,
and the closed system is dispatched like any other.
"""

__all__ = [
    "AxisList",
    "CoordinateSystem",
    "CoordinateSystem2D",
    "CoordinateSystem3D",
    "ArrayCoordinateSystem",
    "CArrayCoordinateSystem",
    "FArrayCoordinateSystem",
    "ArrayCoordinateSystem2D",
    "ArrayCoordinateSystem3D",
    "CArrayCoordinateSystem2D",
    "CArrayCoordinateSystem3D",
    "FArrayCoordinateSystem2D",
    "FArrayCoordinateSystem3D",
    "SpatialCoordinateSystem",
    "SpatialCoordinateSystem2D",
    "SpatialCoordinateSystem3D",
    "PixelCoordinateSystem",
    "VoxelCoordinateSystem",
    "CPixelCoordinateSystem",
    "FPixelCoordinateSystem",
    "CVoxelCoordinateSystem",
    "FVoxelCoordinateSystem",
    "RASCoordinateSystem",
    "LPSCoordinateSystem",
    "RSACoordinateSystem",
    "FRASCoordinateSystem",
    "FLPSCoordinateSystem",
    "FRSACoordinateSystem",
    "CRASCoordinateSystem",
    "CLPSCoordinateSystem",
    "CRSACoordinateSystem",
    "PhysicalCoordinateSystem",
    "RASmm",
    "LPSmm",
    "RSAmm",
]
# stdlib
import sys
from numbers import Integral

# externals
import typing_extensions as tx
from bagof.magic import fields, replace

# internals
from . import axes as _axes
from .axes import Axis, SpaceAxis
from .base import DataModelBase
from .units import is_physicalunit, is_sampleunit

_Ellipsis = type(Ellipsis)
# The type of `...`. Python 3.10 names it `types.EllipsisType`.

AXIS = tx.TypeVar("AXIS")
# The type of the items of an `AxisList`.

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis]
_3SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis, SpaceAxis]

_SAMPLE = "sample"


class AxisList(list, tx.Generic[AXIS]):
    """The axes of a coordinate system, which may leave some unknown.

    An `AxisList` is a `list` of [`Axis`][brainhops.datamodel.axes.Axis]
    that may hold one `...` (`Ellipsis`), anywhere in the list. `...`
    stands for *zero or more axes about which nothing is known*.

    * A list that holds `...` is *open*: its number of axes is unknown.
    * A list without it is *closed*: it lists every axis.
    * `...` is an entry of the list, but never counts as an axis.
    * A list that holds `...` more than once describes no axes: every
      method that reads the axes raises a `ValueError` on it. A
      coordinate system refuses such a list when it is built.

    `[..., TimeAxis()]` says that the last axis is time, and nothing
    about the others. `[Axis(name="x"), ...]` says that the first axis
    is `x`. `[...]` says nothing at all.

    !!! note "Entries and axes"
        An `AxisList` is the `list` it stores. `len()`, iteration,
        equality, `repr`, indexing with an integer or a slice (which
        gives an `AxisList`), and [`index`][] are about its *entries*,
        `...` included.

        [`ndim`][] counts the *axes* the list describes, and
        [`expand`][], [`restrict`][], [`embed`][] and
        [`compatible_with`][] place them in the space, where `...` stands
        for as many axes as needed. A position in the space is counted
        from the first axis when it is non-negative, and from the last
        one when it is negative.

        In a closed list, the entries are the axes, in order. In an open
        list, they are not: in `[x, ..., t]`, entry 2 is `t`, which is
        the last axis, and the axis at position 2 is one of the axes that
        `...` stands for, which has no entry. So
        `for i in range(len(axes)): axes[i]` walks the entries, not the
        axes.

    !!! note "Finding an axis"
        [`index`][] finds the first entry that matches a query, as
        `list.index` finds the first entry equal to a value. The query is
        an [`Axis`][brainhops.datamodel.axes.Axis], or a name, which
        stands for `Axis(name=...)`. An entry matches when it is an
        instance of the class of the query, and has every field that the
        query sets (not `None`), with the same value. The fields that the
        query leaves unset are not compared. So:

        * `axes.index("t")` finds the first axis named `"t"`;
        * `axes.index(TimeAxis())` finds the first time axis, whatever
          its unit: a `TimeAxis()` sets its type, and leaves its unit
          unspecified;
        * `axes.index(Axis())` finds the first axis;
        * `...` matches nothing. An unknown `Axis()` matches no query
          that sets a field, so neither does any of the axes that `...`
          stands for.

    !!! note "Names"
        An axis can also be read by its name, as in a `dict`:
        `axes["t"]` is the explicit axis named `"t"`, `"t" in axes` says
        whether there is one, and [`names`][] lists the names. A name
        only ever matches an explicit axis, never one of the axes that
        `...` stands for. `axes["t"]` is `axes[axes.index("t")]`, but it
        also refuses a name that more than one axis has.

        There is no `keys()`, `values()`, `items()`, `update()` or
        `pop()` by name: an axis may be unnamed, a name may be shared,
        and `...` has no name, so a mapping view would misrepresent the
        list, and changing an axis by its name would be a trap.

    The axes of a
    [`CoordinateSystem`][brainhops.datamodel.systems.CoordinateSystem]
    whose number of axes is not fixed by its class are stored as an
    `AxisList`: a list or a tuple given to the system is converted to
    one, item by item, to the type of axis the class declares. Its
    default, `[...]`, says nothing about the axes; `axes=None` is
    refused. [`of`][brainhops.datamodel.systems.AxisList.of] reads the
    axes of any system, or of a missing one, as an `AxisList`.

    The type parameter is the type of the items:
    `AxisList[Union[Axis, EllipsisType]]` may be open, and
    `AxisList[Axis]` is closed.

    !!! example
        ```pycon
        >>> x, t = Axis(name="x"), TimeAxis(name="t")
        >>> axes = AxisList([x, ..., t])
        >>> axes.ndim is None, axes.is_open
        (True, True)
        >>> axes.index("t"), axes.index(TimeAxis()), axes["t"] is t
        (2, 2, True)
        >>> "x" in axes, axes.names
        (True, ('x', Ellipsis, 't'))
        >>> axes.expand(4)[1:]
        [Axis(), Axis(), TimeAxis(name='t')]
        >>> axes.restrict([-1, 0, 1])[1:]
        [Axis(name='x'), Axis()]
        ```
    """

    @classmethod
    def of(cls, system: tx.Optional["CoordinateSystem"]) -> tx.Self:
        """The axes of a coordinate system that may be missing.

        This is the one way to read the axes of a system whatever it is:

        * a missing system (`None`) says nothing about its axes, and
          reads as `[...]`;
        * any other system reads as its axes, whether its class stores
          them as an `AxisList` or, for a fixed number of axes, as a
          tuple.

        Parameters
        ----------
        system : CoordinateSystem or None
            The system whose axes are read.

        Returns
        -------
        AxisList
            A new list. Changing it does not change the system.

        Raises
        ------
        TypeError
            If `system` is neither a
            [`CoordinateSystem`][brainhops.datamodel.systems.CoordinateSystem]
            nor `None`.

        !!! example
            ```pycon
            >>> AxisList.of(None), AxisList.of(CoordinateSystem())
            ([Ellipsis], [Ellipsis])
            >>> AxisList.of(RASCoordinateSystem()).ndim
            3
            ```
        """
        if system is None:
            return cls([...])
        if isinstance(system, CoordinateSystem):
            return cls(system.axes)
        raise TypeError(
            f"Expected a CoordinateSystem or None, not a "
            f"{type(system).__name__}."
        )

    # --- entries ------------------------------------------------------

    @tx.overload
    def __getitem__(self, key: tx.SupportsIndex) -> AXIS: ...

    @tx.overload
    def __getitem__(self, key: slice) -> tx.Self: ...

    @tx.overload
    def __getitem__(self, key: str) -> Axis: ...

    def __getitem__(self, key: tx.Any) -> tx.Any:
        """An entry (`int`), some entries (`slice`), or the axis with a
        name (`str`).

        An integer or a slice indexes the *entries* of the list, as in
        any `list`, and a slice gives an `AxisList`. A name gives the one
        explicit axis that has it.

        Raises
        ------
        KeyError
            If no explicit axis has the name.
        ValueError
            If more than one explicit axis has the name.
        IndexError, TypeError
            As `list` indexing does.

        !!! example
            ```pycon
            >>> x, t = Axis(name="x"), TimeAxis(name="t")
            >>> AxisList([x, ..., t])["t"] is t
            True
            >>> AxisList([x, ..., t])[2] is t
            True
            ```
        """
        if isinstance(key, str):
            return list.__getitem__(self, self._entry_named(key))
        if isinstance(key, slice):
            return type(self)(list.__getitem__(self, key))
        return list.__getitem__(self, key)

    def __contains__(self, item: object) -> bool:
        """Whether an explicit axis has a name (`str`), or whether an
        entry equals `item` (anything else, as in any `list`)."""
        if isinstance(item, str):
            return bool(self._entries_named(item))
        return list.__contains__(self, item)

    def index(
        self,
        query: tx.Union[Axis, str],
        start: tx.SupportsIndex = 0,
        stop: tx.SupportsIndex = sys.maxsize,
    ) -> int:
        """The first entry that matches an axis, or a name.

        This is `list.index`, with a looser test than equality: an entry
        *matches* the query when it is an instance of the class of the
        query, and has every field that the query sets (not `None`), with
        the same value. The fields that the query leaves unset are not
        compared. A name stands for `Axis(name=...)`, so it matches the
        axes with that name, whatever their class. `...` matches nothing.

        Parameters
        ----------
        query : Axis or str
            The axis to find, or its name.
        start, stop : int, optional
            Only the entries `start` to `stop` are searched, as in
            `list.index`.

        Returns
        -------
        int
            The index of the first entry that matches. It is an entry
            index, which is a position in the space only in a closed list
            (see the class notes).

        Raises
        ------
        ValueError
            If no entry matches.
        TypeError
            If `query` is neither an `Axis` nor a string.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis(name="t", unit="s")
            >>> axes = AxisList([x, ..., t])
            >>> axes.index("t"), axes.index(SpaceAxis())
            (2, 0)
            >>> axes.index(Axis(unit="second")), axes.index(Axis())
            (2, 0)
            >>> axes.index("y")
            Traceback (most recent call last):
              ...
            ValueError: Axis(name='y') is not in list
            ```
        """
        if isinstance(query, str):
            query = Axis(name=query)
        elif not isinstance(query, Axis):
            raise TypeError(
                f"An axis is found by an Axis or by its name (str), not by "
                f"a {type(query).__name__}."
            )
        entries = list(self)
        for i in range(*slice(start, stop).indices(len(entries))):
            if entries[i] is not ... and _matches(entries[i], query):
                return i
        raise ValueError(f"{query!r} is not in list")

    @property
    def names(self) -> tx.Tuple[tx.Union[str, None, _Ellipsis], ...]:
        """The name of each entry: `None` for an unnamed axis, and `...`
        in the place of `...`.

        !!! example
            ```pycon
            >>> AxisList([Axis(name="x"), Axis(), ...]).names
            ('x', None, Ellipsis)
            ```
        """
        return tuple(... if axis is ... else _name(axis) for axis in self)

    # --- axes ---------------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` when the list is open.

        A closed list has one axis per entry. An open list has an
        unknown number of axes.

        !!! example
            ```pycon
            >>> AxisList([Axis(), Axis()]).ndim
            2
            >>> AxisList([Axis(), ...]).ndim is None
            True
            ```
        """
        prefix, suffix = self._split()
        return len(prefix) if suffix is None else None

    @property
    def is_open(self) -> bool:
        """Whether the list holds `...`, so that its number of axes is
        unknown.

        `AxisList([...])`, which says nothing at all, is open. The empty
        list is closed: it has no axis.
        """
        return self._split()[1] is not None

    def expand(self, ndim: int) -> tx.Self:
        """The closed list of `ndim` axes that this list describes.

        In an open list, `...` is replaced with as many unknown `Axis()`
        as needed to reach `ndim` axes. Use it once the number of axes is
        known, for instance from the shape of the data.

        Parameters
        ----------
        ndim : int
            The number of axes.

        Returns
        -------
        AxisList
            A new, closed list of `ndim` axes. A closed list is returned
            as a copy of itself.

        Raises
        ------
        ValueError
            If `ndim` is less than the number of explicit axes of an open
            list, or differs from the number of axes of a closed one.
        TypeError
            If `ndim` is not an integer.

        !!! example
            ```pycon
            >>> AxisList([Axis(name="x"), ...]).expand(3)
            [Axis(name='x'), Axis(), Axis()]
            >>> AxisList([...]).expand(2)
            [Axis(), Axis()]
            ```
        """
        ndim = _as_int(ndim, "ndim")
        prefix, suffix = self._split()
        if suffix is None:
            if ndim != len(prefix):
                raise ValueError(
                    f"Cannot expand a closed list of {len(prefix)} axes to "
                    f"{ndim} axes."
                )
            return type(self)(prefix)
        explicit = len(prefix) + len(suffix)
        if ndim < explicit:
            raise ValueError(
                f"Cannot expand an open list with {explicit} explicit axes "
                f"to {ndim} axes."
            )
        fill = [Axis() for _ in range(ndim - explicit)]
        return type(self)(prefix + fill + suffix)

    def restrict(self, refs: tx.Iterable[tx.Union[int, str]]) -> tx.Self:
        """The axes at some positions, or with some names, of this list.

        A reference is the position of an axis in the space (`int`), or
        the name of an explicit axis (`str`), as `axes[name]` reads it.

        * In a closed list of `n` axes, a position lies in `[-n, n)`.
        * In an open list, every position is valid, because `...` stands
          for any number of axes. A non-negative position reads the
          explicit axes before `...`, and a negative one the explicit
          axes after it. Any other position falls among the axes that
          `...` stands for, and gives an unknown `Axis()`.

        The axes are listed in the order of `refs`.

        Parameters
        ----------
        refs : iterable of int or str
            The positions or names of the axes to keep.

        Returns
        -------
        AxisList
            A new, closed list of `len(refs)` axes.

        Raises
        ------
        IndexError
            If a position lies outside a closed list.
        KeyError
            If no explicit axis has a name.
        ValueError
            If more than one explicit axis has a name, or if two
            references name the same axis.
        TypeError
            If a reference is neither an integer nor a string, or if
            `refs` is a string rather than a list of references.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> AxisList([x, y, z]).restrict(["z", 0])
            [Axis(name='z'), Axis(name='x')]
            >>> AxisList([x, ...]).restrict([0, 1])
            [Axis(name='x'), Axis()]
            ```
        """
        positions = []
        for ref in _as_list(refs, "refs"):
            if isinstance(ref, str):
                entry = self._entry_named(ref)
                positions.append(self._position_of_entry(entry))
            else:
                positions.append(self._position(ref))
        _check_unique(positions, "refs")
        return type(self)(self._axis_at(p) for p in positions)

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> tx.Self:
        """The axes of a larger space in which this list's axes sit.

        This is the inverse of [`restrict`][]: axis `j` of this list sits
        at `positions[j]` of the result, and every other position holds
        an unknown `Axis()`. An open list is first closed to
        `len(positions)` axes, as by [`expand`][].

        The positions are absolute positions in the larger space, which
        does not exist yet, so they cannot be names.

        Parameters
        ----------
        positions : iterable of int
            The non-negative position of each axis in the larger space.
        ndim : int, optional
            The number of axes of the larger space. When it is not given,
            the number is unknown, and the result ends with `...` after
            the last embedded axis.

        Returns
        -------
        AxisList
            A new list, closed when `ndim` is given and open otherwise.

        Raises
        ------
        ValueError
            If a position is negative or repeated, if `ndim` does not
            exceed every position, or if this list cannot be closed to
            `len(positions)` axes.
        TypeError
            If a position or `ndim` is not an integer.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> AxisList([x]).embed([1])
            [Axis(), Axis(name='x'), Ellipsis]
            >>> AxisList([x]).embed([1], ndim=3)
            [Axis(), Axis(name='x'), Axis()]
            ```
        """
        positions = [
            _as_int(p, "positions") for p in _as_list(positions, "positions")
        ]
        if any(p < 0 for p in positions):
            raise ValueError(
                f"Positions in the larger space count from its first "
                f"axis, so they cannot be negative: {positions}."
            )
        _check_unique(positions, "positions")
        prefix, suffix = self._split()
        count = len(prefix) + len(suffix or [])
        if (suffix is None and count != len(positions)) or (
            count > len(positions)
        ):
            raise ValueError(
                f"Cannot embed a list of {count}"
                f"{'' if suffix is None else ' explicit'} axes at "
                f"{len(positions)} positions."
            )
        axes = self.expand(len(positions))
        size = max(positions) + 1 if positions else 0
        if ndim is not None:
            ndim = _as_int(ndim, "ndim")
            if ndim < size:
                raise ValueError(
                    f"Cannot embed an axis at position {size - 1} of a "
                    f"space of {ndim} axes."
                )
            size = ndim
        full: tx.List[tx.Any] = [Axis() for _ in range(size)]
        for axis, p in zip(axes, positions):
            full[p] = axis
        if ndim is None:
            full.append(...)
        return type(self)(full)

    def compatible_with(self, other: tx.Sequence[tx.Any]) -> bool:
        """Whether `self` and `other` could describe the same axes.

        Two lists are compatible when some choice of the axes that each
        `...` stands for makes them match axis by axis, each pair being
        [`compatible`][brainhops.datamodel.axes.Axis.compatible_with].

        For two closed lists, this asks for the same number of axes,
        pairwise compatible. Unlike `==`, an unknown `Axis()` matches
        any axis, and `[...]` matches every list. The relation is
        symmetric, but not transitive.

        Parameters
        ----------
        other : AxisList, or list or tuple of Axis
            The axes to compare with. A plain list or tuple is read as
            an `AxisList`.

        Returns
        -------
        bool
            Whether the two lists could describe the same axes.

        Raises
        ------
        TypeError
            If `other` is not a list or a tuple.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis()
            >>> AxisList([x, ...]).compatible_with([x, Axis(), t])
            True
            >>> AxisList([..., t]).compatible_with([x])
            False
            ```
        """
        if not isinstance(other, (list, tuple)):
            raise TypeError(
                f"A list of axes is compatible only with another list of "
                f"axes, not with {type(other).__name__}."
            )
        p1, s1 = self._split()
        p2, s2 = AxisList(other)._split()
        if s1 is None and s2 is None:
            return len(p1) == len(p2) and _pairwise(p1, p2)
        if s1 is None:
            # Let the first list be the open one.
            (p1, s1), (p2, s2) = (p2, s2), (p1, s1)
        assert s1 is not None
        if s2 is None:
            # Open against closed: the explicit axes of the open list must
            # fit, and match the axes at the start and at the end.
            n = len(p2)
            if len(p1) + len(s1) > n:
                return False
            return _pairwise(p1, p2[: len(p1)]) and _pairwise(
                s1, p2[n - len(s1) :]
            )
        # Open against open: with enough axes in each `...`, only the axes
        # that both lists state at the start, or both at the end, meet.
        k = min(len(p1), len(p2))
        m = min(len(s1), len(s2))
        return _pairwise(p1[:k], p2[:k]) and _pairwise(
            s1[len(s1) - m :], s2[len(s2) - m :]
        )

    # --- private helpers ----------------------------------------------
    # Positions in the space, as opposed to entries of the list, are only
    # handled here. A position is counted from the first axis when it is
    # non-negative, and from the last one when it is negative.

    def _split(self) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
        # The explicit axes before and after `...`. The second list is
        # `None` for a closed list, whose axes are then all in the first.
        entries = list(self)
        ellipses = [i for i, axis in enumerate(entries) if axis is ...]
        if not ellipses:
            return entries, None
        if len(ellipses) > 1:
            raise ValueError(
                "A list of axes holds at most one `...`, which stands for "
                "all the axes about which nothing is known."
            )
        i = ellipses[0]
        return entries[:i], entries[i + 1 :]

    def _position(self, position: int) -> int:
        # Check a position against the list, and normalize it. In a closed
        # list of `n` axes, it must lie in `[-n, n)`, and a negative one is
        # returned as its non-negative equivalent. In an open list, every
        # position is valid, and is returned as given: never clamped.
        if isinstance(position, bool) or not isinstance(position, Integral):
            raise TypeError(
                f"An axis is referred to by its position (int) or its name "
                f"(str), not by a {type(position).__name__}."
            )
        position = int(position)
        prefix, suffix = self._split()
        if suffix is not None:
            return position
        n = len(prefix)
        if not -n <= position < n:
            raise IndexError(
                f"Axis position {position} is out of range for a list of "
                f"{n} axes."
            )
        return position + n if position < 0 else position

    def _position_of_entry(self, entry: int) -> int:
        # The position of the axis that an entry holds. An entry after
        # `...` is counted from the end, because its distance from the
        # start is unknown. Any other entry is its own position.
        entries = list(self)
        entry = range(len(entries))[entry]
        if entries[entry] is ...:
            raise ValueError("`...` stands for axes, and is not one.")
        after = ... in entries[:entry]
        return entry - len(entries) if after else entry

    def _axis_at(self, position: int) -> Axis:
        # The axis at a position, checked as by `_position`: an explicit
        # axis, or a new unknown `Axis()` for a position among the axes
        # that `...` stands for, which has no entry.
        position = self._position(position)
        prefix, suffix = self._split()
        if suffix is None or 0 <= position < len(prefix):
            return prefix[position]
        if -len(suffix) <= position < 0:
            return suffix[position]
        return Axis()

    def _entries_named(self, name: str) -> tx.List[int]:
        # The entries of the explicit axes that have exactly this name.
        return [i for i, axis in enumerate(self) if _name(axis) == name]

    def _entry_named(self, name: str) -> int:
        # The entry of the one explicit axis that has exactly this name.
        entries = self._entries_named(name)
        if not entries:
            raise KeyError(f"No axis of the list is named {name!r}.")
        if len(entries) > 1:
            raise ValueError(
                f"Cannot read the axis named {name!r}: {len(entries)} axes "
                f"of the list have that name."
            )
        return entries[0]


def _matches(candidate: tx.Any, query: Axis) -> bool:
    # Whether an entry matches a query of `AxisList.index`: it is an
    # instance of the class of the query, and has every field that the
    # query sets, with the same value.
    if not isinstance(candidate, type(query)):
        return False
    for field in fields(type(query)):
        wanted = getattr(query, field.name, None)
        if wanted is not None and getattr(candidate, field.name) != wanted:
            return False
    return True


def _pairwise(first: tx.List[Axis], second: tx.List[Axis]) -> bool:
    return all(a.compatible_with(b) for a, b in zip(first, second))


def _name(axis: tx.Any) -> tx.Optional[str]:
    return getattr(axis, "name", None)


def _as_int(value: tx.Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"Expected an integer for {what}, not a {type(value).__name__}."
        )
    return int(value)


def _as_list(values: tx.Any, what: str) -> tx.List[tx.Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{what} must be a list of axis references, not a single string."
        )
    try:
        return list(values)
    except TypeError:
        raise TypeError(
            f"{what} must be a list, not a {type(values).__name__}."
        ) from None


def _check_unique(positions: tx.List[int], what: str) -> None:
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"{what} names the same axis more than once: {positions}."
        )


# ----------------------------------------------------------------------
#   DISPATCH PREDICATES
# ----------------------------------------------------------------------
# Every predicate below says something about *all* the axes of a system:
# how many there are, or what each one is. An open system (one whose
# axes hold `...`) does not know all its axes -- `...`
# may stand for none, or for axes of any kind -- so no predicate holds
# of it, and calling a class with open axes builds that class itself:
# `CoordinateSystem(axes=[SpaceAxis(), ...])` is a `CoordinateSystem`,
# not a `SpatialCoordinateSystem`, and `CoordinateSystem(axes=[x, ...])`
# is not two-dimensional although its list has two entries. Only once
# it is closed (see `CoordinateSystem.expand`) does a system reach the
# class its axes describe.


def _closed(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Optional[tx.List[Axis]]:
    """The axes, when they list every axis of the system; else `None`."""
    if axes is None:
        return None
    axes = list(axes)
    if any(axis is ... for axis in axes):
        return None
    return axes


def _ndim(n: int) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _closed(axes)
        return closed is not None and len(closed) == n

    check.__name__ = check.__qualname__ = f"_is{n}d"
    return check


_is2d = _ndim(2)
_is3d = _ndim(3)


def _all(
    test: tx.Callable[[Axis], bool], name: str
) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Whether the axes are closed, not empty, and all pass `test`."""

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _closed(axes)
        return bool(closed) and all(test(axis) for axis in closed)

    check.__name__ = check.__qualname__ = name
    return check


_is_spatial = _all(lambda axis: axis.type == "space", "_is_spatial")
_is_array = _all(lambda axis: is_sampleunit(axis.unit), "_is_array")
_is_physical = _all(lambda axis: is_physicalunit(axis.unit), "_is_physical")


def _both(
    *tests: tx.Callable[[tx.Any], bool],
) -> tx.Callable[[tx.Any], bool]:
    def check(axes: tx.Any) -> bool:
        return all(test(axes) for test in tests)

    check.__name__ = check.__qualname__ = "_and_".join(
        test.__name__.lstrip("_") for test in tests
    )
    check.__name__ = check.__qualname__ = "_" + check.__name__
    return check


def _is_anat(code: str) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Whether the axes point, in order, the way the letters of `code` say.

    `code` is spelled with the letters of [`brainhops.datamodel.axes`][]:
    `"RAS"` is a left-to-right, a posterior-to-anterior and an
    inferior-to-superior axis, in that order. Only the orientations are
    compared; the names and units of the axes are free.
    """
    expected = tuple(
        getattr(_axes, letter)().orientation.value for letter in code
    )

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _closed(axes)
        if closed is None or len(closed) != len(expected):
            return False
        return all(
            getattr(getattr(axis, "orientation", None), "value", None) == value
            for axis, value in zip(closed, expected)
        )

    check.__name__ = check.__qualname__ = f"_is_{code}"
    return check


# ----------------------------------------------------------------------
#   GENERIC COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class CoordinateSystem(DataModelBase, polymorphic=True):
    """A coordinate system defines the meaning of coordinates in a space.

    It describes each axis in the system (name, unit and/or other properties),
    and can be named.

    !!! note "Open systems"
        The axes of a system are an [`AxisList`][], which may hold at
        most one `...` (`Ellipsis`), anywhere in the list, that stands
        for *zero or more axes about which nothing is known*. A system
        with `...` is *open*: its number of axes is unknown. A system
        without it is *closed*.

        * `[..., TimeAxis()]` says that the last axis is time, and nothing
          about the others.
        * `[Axis(name="x"), ...]` says that the first axis is `x`.
        * `[...]`, the default, says nothing at all. It is the one
          spelling of unknown axes: `axes=None` is refused.

        A list or a tuple given as `axes` is stored as an [`AxisList`][].
        [`AxisList.of`][] reads the axes of any system, or of a missing
        one (`None`), as an [`AxisList`][], with `[...]` for a missing
        system. An axis is read by its name as `system.axes["x"]`, and
        found by [`index`][brainhops.datamodel.systems.AxisList.index].

        Classes with a fixed number of axes, such as
        [`CoordinateSystem3D`][], are always closed. They store their
        axes as a tuple, whose type fixes the number of axes and the
        class of each one, so they reject `...`.

        Calling a class builds the most specific system its axes
        describe (see the module), and only a closed system is
        dispatched: `CoordinateSystem(axes=[x, y])` is a
        [`CoordinateSystem2D`][], whose axes are a tuple, but
        `CoordinateSystem(axes=[x, ...])` -- whose `...` may stand for
        no axis, or for many -- stays a `CoordinateSystem`. Closing an
        open system with [`expand`][] dispatches it again.

    !!! note "Equality"
        Equality is field by field: two systems are equal when they are
        of the same class, have the same name, and have equal axes. A
        system is never equal to `None`, not even a plain
        `CoordinateSystem()` that says nothing at all; whether a system
        tells anything, as a missing endpoint of a transformation does
        not, is what [`_is_informative`][] answers.
        [`compatible_with`][] is the looser question of whether two
        systems could describe the same space.
    """

    name: tx.Optional[str] = None
    """The name of the coordinate system."""

    axes: AxisList[tx.Union[Axis, _Ellipsis]] = [...]
    """The axes of the coordinate system, in order. `[...]`, the default,
    says nothing about them."""

    # --- validation ---------------------------------------------------

    def __pre_init__(self, arguments: tx.Any) -> None:
        # The converter would refuse `None` too, but with a message about
        # iterables: say what to write instead.
        if arguments.get("axes", ...) is not None:
            return
        name = type(self).__name__
        if isinstance(self, (CoordinateSystem2D, CoordinateSystem3D)):
            raise TypeError(
                f"{name}.axes cannot be None: a {name} has a fixed number "
                f"of axes, and lists every one of them."
            )
        raise TypeError(
            f"{name}.axes cannot be None: use `[...]` for axes about which "
            f"nothing is known."
        )

    def __post_init__(self) -> None:
        if sum(a is ... for a in self.axes) > 1:
            raise ValueError(
                "The axes of a coordinate system hold at most one `...`, "
                "which stands for all the axes about which nothing is known."
            )

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` when the system is open.

        A closed system has exactly `len(axes)` axes. An open system,
        whose axes hold `...`, has an unknown number of axes, and its
        `ndim` is `None`. This is
        [`AxisList.ndim`][brainhops.datamodel.systems.AxisList.ndim]
        read through [`AxisList.of`][].

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(), Axis()]).ndim
            2
            >>> CoordinateSystem(axes=[Axis(), ...]).ndim is None
            True
            >>> CoordinateSystem().ndim is None
            True
            ```
        """
        return AxisList.of(self).ndim

    # --- operations ---------------------------------------------------

    def expand(self, ndim: int) -> tx.Self:
        """The closed system of `ndim` axes that this system describes.

        The axes are expanded by
        [`AxisList.expand`][brainhops.datamodel.systems.AxisList.expand]:
        in an open system, `...` is replaced with as many unknown
        `Axis()` as needed to reach `ndim` axes. The class is called
        again with the closed axes, and the other fields, the name
        included, are kept: the result is of this class, or of the
        subclass that the closed axes select from it (an
        `ArrayCoordinateSystem` closed to two axes is an
        `ArrayCoordinateSystem2D`). Each unknown `Axis()` is first read as
        the type of axis the class declares, so a `SpatialCoordinateSystem`
        closed to three axes has three spatial axes, and is a
        `SpatialCoordinateSystem3D`. Use it once the number of axes is
        known, for instance from the shape of the data.

        Parameters
        ----------
        ndim : int
            The number of axes.

        Returns
        -------
        CoordinateSystem
            A closed system of `ndim` axes, of this class or of one of its
            subclasses. A closed system is returned as itself.

        Raises
        ------
        ValueError
            If `ndim` is less than the number of explicit axes of an open
            system, or differs from the number of axes of a closed one.
        TypeError
            If `ndim` is not an integer.

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(name="x"), ...]).expand(4)
            CoordinateSystem(axes=[Axis(name='x'), Axis(), Axis(), Axis()])
            >>> CoordinateSystem().expand(2)
            CoordinateSystem2D(axes=(Axis(), Axis()))
            ```
        """
        axes = AxisList.of(self)
        expanded = axes.expand(ndim)
        if not axes.is_open:
            return self
        # A plain list, which the field converts item by item (and which
        # the class dispatches on, once converted). An `AxisList` would be
        # taken as it is, unknown `Axis()` and all, because a union -- the
        # field is optional -- does not look inside an instance of one of
        # its members.
        return replace(self, axes=list(expanded))

    def restrict(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """The system of the axes at some positions of this system.

        The axes are restricted by
        [`AxisList.restrict`][brainhops.datamodel.systems.AxisList.restrict]:
        a reference is a position in the space or a name, and a position
        of an open system that falls among the axes that `...` stands
        for gives an unknown `Axis()`. The axes are listed in the order
        of `refs`. The result describes a different space, so the class
        and the name of this system are not carried over: it is the
        closed system that `CoordinateSystem(axes=...)` builds from the
        restricted axes, which is a [`CoordinateSystem2D`][], an
        [`RASCoordinateSystem`][], ... when the axes select one.

        Parameters
        ----------
        refs : iterable of int or str
            The positions or names of the axes to keep.

        Returns
        -------
        CoordinateSystem
            A closed system of `len(refs)` axes.

        Raises
        ------
        ValueError, IndexError, TypeError
            As
            [`AxisList.restrict`][brainhops.datamodel.systems.AxisList.restrict]
            does.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> CoordinateSystem(axes=[x, y, z]).restrict(["z", 0])
            CoordinateSystem2D(axes=(Axis(name='z'), Axis(name='x')))
            >>> CoordinateSystem(axes=[x, ...]).restrict([0])
            CoordinateSystem(axes=[Axis(name='x')])
            ```
        """
        # A plain list, as in `expand`.
        return CoordinateSystem(axes=list(AxisList.of(self).restrict(refs)))

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """The system of a larger space in which this system's axes sit.

        This is the inverse of [`restrict`][]. The axes are embedded by
        [`AxisList.embed`][brainhops.datamodel.systems.AxisList.embed]:
        axis `j` of this system sits at `positions[j]` of the result,
        and every other position holds an unknown `Axis()`. The result
        describes a different space, so the class and the name of this
        system are not carried over: it is the system that
        `CoordinateSystem(axes=...)` builds from the embedded axes -- a
        plain, open `CoordinateSystem` when `ndim` is not given, and the
        closed system the axes select when it is.

        Parameters
        ----------
        positions : iterable of int
            The non-negative position of each axis in the larger space.
        ndim : int, optional
            The number of axes of the larger space. When it is not given,
            the number is unknown, and the result ends with `...` after
            the last embedded axis.

        Returns
        -------
        CoordinateSystem
            A system that is closed when `ndim` is given, and open
            otherwise.

        Raises
        ------
        ValueError, TypeError
            As [`AxisList.embed`][brainhops.datamodel.systems.AxisList.embed]
            does.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> CoordinateSystem(axes=[x]).embed([1])
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Ellipsis])
            >>> CoordinateSystem(axes=[x]).embed([1], ndim=4)
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Axis(), Axis()])
            ```
        """
        # A plain list, as in `expand`.
        return CoordinateSystem(
            axes=list(AxisList.of(self).embed(positions, ndim=ndim))
        )

    def compatible_with(self, other: tx.Optional["CoordinateSystem"]) -> bool:
        """Whether `self` and `other` could describe the same space.

        Two systems are compatible when their axes are
        [`AxisList.compatible_with`][brainhops.datamodel.systems.AxisList.compatible_with]
        each other: some choice of the axes that each `...` stands for
        makes them match axis by axis, each pair being
        [`Axis.compatible_with`][brainhops.datamodel.axes.Axis.compatible_with].
        Only the axes are compared, not the names of the systems. `None`
        is read as a system about which nothing is known, which is
        compatible with every system.

        For two closed systems, this asks for the same number of axes,
        pairwise compatible. Unlike `==`, an unknown `Axis()` matches
        any axis. The relation is symmetric, but not transitive.

        Parameters
        ----------
        other : CoordinateSystem or None
            The system to compare with.

        Returns
        -------
        bool
            Whether the two systems could describe the same space.

        Raises
        ------
        TypeError
            If `other` is neither a [`CoordinateSystem`][] nor `None`.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis()
            >>> CoordinateSystem(axes=[x, ...]).compatible_with(
            ...     CoordinateSystem(axes=[x, Axis(), t])
            ... )
            True
            >>> CoordinateSystem(axes=[..., t]).compatible_with(
            ...     CoordinateSystem(axes=[x])
            ... )
            False
            ```
        """
        if other is not None and not isinstance(other, CoordinateSystem):
            raise TypeError(
                f"A coordinate system is compatible only with another "
                f"CoordinateSystem or None, not with {type(other).__name__}."
            )
        return AxisList.of(self).compatible_with(AxisList.of(other))


def _is_informative(system: tx.Optional[CoordinateSystem]) -> bool:
    """Whether `system` tells anything about a space.

    A missing system (`None`) tells nothing, and neither does a plain,
    unnamed `CoordinateSystem` whose axes are `[...]`: an endpoint of a
    transformation that is either one is read as undeclared, and is
    derived, propagated or replaced as a missing one is. Any other
    system -- one with a name, an axis, or a class of its own -- tells
    something. This is not equality: no system equals `None`.
    """
    if system is None:
        return False
    return not (
        type(system) is CoordinateSystem
        and system.name is None
        and list(system.axes) == [...]
    )


class CoordinateSystem2D(CoordinateSystem, on={"axes": _is2d}):
    """A coordinate systems with exactly two dimensions."""

    axes: _2Axes = (Axis(), Axis())


class CoordinateSystem3D(CoordinateSystem, on={"axes": _is3d}):
    """A coordinate system with exactly three dimensions."""

    axes: _3Axes = (Axis(), Axis(), Axis())


# ----------------------------------------------------------------------
#   PHYSICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class PhysicalCoordinateSystem(CoordinateSystem):
    """A coordinate system whose coordinates measure physical quantities.

    It has at least one axis, and every axis carries a unit, and a
    *meaningful* one: not `None`, which leaves the unit unspecified, and
    not [`SampleUnit`][], which says the coordinates count samples of an
    array. A system that declares itself physical therefore always has a
    conversion factor to another physical system of the same kind, and
    reversing one of its axes is a sign flip rather than the origin shift
    a sampled axis needs.

    This is a base to inherit deliberately rather than a dispatch target:
    a physical spatial system is selected as a spatial one, and the
    concrete systems that are physical by construction -- [`RASmm`][],
    [`LPSmm`][], [`RSAmm`][] -- compose it in. Their own constraint is
    what dispatch selects them on; this class checks every instance, so
    building one directly with an unspecified or sampled axis is refused
    too.

    A physical system is always closed. Its axes may not hold `...`:
    `...` is not an axis without a unit, but it stands for axes about
    which nothing is known -- their units included -- and a physical
    system vouches for the unit of every axis it has. Such a system is
    refused with its own message, which says to close it first.
    `PhysicalCoordinateSystem()`, whose axes default to `[...]`, states
    no axis at all, and is refused as a system without axes.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        name = type(self).__name__
        if all(axis is ... for axis in self.axes):
            # No axis at all, or only `[...]` (the default), which states
            # none.
            raise ValueError(
                f"{name} is a physical coordinate system, so it must have "
                f"axes, and every one of them must carry a physical unit. "
                f"It was given {list(self.axes)!r}."
            )
        if any(axis is ... for axis in self.axes):
            # `...` is not an axis without a unit: it stands for axes about
            # which nothing is known, the unit included. The class could
            # not then vouch for a conversion factor on every axis, so a
            # physical system lists every axis it has.
            raise ValueError(
                f"{name} is a physical coordinate system, so it lists "
                f"every one of its axes, each with a physical unit. Its "
                f"axes {list(self.axes)!r} hold `...`, which stands for "
                f"axes whose units are unknown: close the system first "
                f"(`expand`), or build a non-physical one."
            )
        for axis in self.axes:
            if is_physicalunit(getattr(axis, "unit", None)):
                continue
            raise ValueError(
                f"{name} is a physical coordinate system, so every one of "
                f"its axes must carry a unit that measures something. The "
                f"axis {axis.name or axis.type!r} carries {axis.unit!r}: "
                f"`None` leaves the unit unspecified, and `'sample'` says "
                f"the axis indexes an array."
            )


# ----------------------------------------------------------------------
#   ARRAY COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class ArrayCoordinateSystem(CoordinateSystem):
    """A coordinate system for a multidimensional array.

    Its coordinates count samples, so the axes it builds by default carry
    the sample unit (see [`SampleUnit`][]). By default, the array is
    assumed C-ordered: the first axis is the slowest changing in memory,
    and the last axis is the fastest changing.

    It is a base rather than a dispatch target: calling it with two or
    three axes builds the matching fixed-arity class, and a system of two
    or three sampled axes is selected as one of those from
    [`CoordinateSystem`][] too.
    """

    name: tx.Optional[str] = "array"


class CArrayCoordinateSystem(ArrayCoordinateSystem):
    """A coordinate system for a C-ordered multidimensional array."""

    name: tx.Optional[str] = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem):
    """A coordinate system for an F-ordered multidimensional array."""

    name: tx.Optional[str] = "farray"


def _dim(i: int) -> Axis:
    return Axis(f"dim{i}", unit=_SAMPLE)


# `ArrayCoordinateSystem` is not a dispatch target, so a class statement
# cannot say that `ArrayCoordinateSystem(axes=[a, b])` is two-dimensional
# without also claiming every two-dimensional system of sampled axes
# from `CoordinateSystem2D` -- which `on=` does too, so both are said.
@ArrayCoordinateSystem.register_polymorph(axes=_is2d)
class ArrayCoordinateSystem2D(
    CoordinateSystem2D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is2d, _is_array)},
):
    """A coordinate system for an array with two dimensions."""

    axes: _2Axes = (_dim(0), _dim(1))


@ArrayCoordinateSystem.register_polymorph(axes=_is3d)
class ArrayCoordinateSystem3D(
    CoordinateSystem3D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is3d, _is_array)},
):
    """A coordinate system for an array with three dimensions."""

    axes: _3Axes = (_dim(0), _dim(1), _dim(2))


# The memory order is not written on the axes, so nothing selects a C- or
# F-ordered class but its own ordered base: `on=None` keeps them out of
# the dispatch of every class above, and the decorator registers them
# with that one base.
@CArrayCoordinateSystem.register_polymorph(axes=_is2d)
class CArrayCoordinateSystem2D(
    CoordinateSystem2D, CArrayCoordinateSystem, on=None
):
    """A coordinate system for a C-ordered array with two dimensions."""

    axes: _2Axes = (_dim(0), _dim(1))


@CArrayCoordinateSystem.register_polymorph(axes=_is3d)
class CArrayCoordinateSystem3D(
    CoordinateSystem3D, CArrayCoordinateSystem, on=None
):
    """A coordinate system for a C-ordered array with three dimensions."""

    axes: _3Axes = (_dim(0), _dim(1), _dim(2))


@FArrayCoordinateSystem.register_polymorph(axes=_is2d)
class FArrayCoordinateSystem2D(
    CoordinateSystem2D, FArrayCoordinateSystem, on=None
):
    """A coordinate system for an F-ordered array with two dimensions."""

    axes: _2Axes = (_dim(0), _dim(1))


@FArrayCoordinateSystem.register_polymorph(axes=_is3d)
class FArrayCoordinateSystem3D(
    CoordinateSystem3D, FArrayCoordinateSystem, on=None
):
    """A coordinate system for an F-ordered array with three dimensions."""

    axes: _3Axes = (_dim(0), _dim(1), _dim(2))


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class SpatialCoordinateSystem(CoordinateSystem, on={"axes": _is_spatial}):
    """A coordinate system, whose axes have spatial meaning."""

    axes: AxisList[tx.Union[SpaceAxis, _Ellipsis]] = [...]


# A spatial system of sampled axes is both spatial and an array, and the
# spatial reading wins: the pixel and voxel systems below are spatial
# systems, and are reached through them.
class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: _2SpatialAxes = (SpaceAxis(), SpaceAxis())


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: _3SpatialAxes = (
        SpaceAxis(),
        SpaceAxis(),
        SpaceAxis(),
    )


def _space(name: str) -> SpaceAxis:
    return SpaceAxis(name=name, unit=_SAMPLE)


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for 2D pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: _2SpatialAxes = (_space("dim0"), _space("dim1"))


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: _3SpatialAxes = (
        _space("dim0"),
        _space("dim1"),
        _space("dim2"),
    )


@CArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
class CPixelCoordinateSystem(
    PixelCoordinateSystem, CArrayCoordinateSystem2D, on=None
):
    """A coordinate system for C-ordered 2D pixel grids."""

    name: tx.Optional[str] = "cpixel"
    axes: _2SpatialAxes = (_space("j"), _space("i"))


@FArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
class FPixelCoordinateSystem(
    PixelCoordinateSystem, FArrayCoordinateSystem2D, on=None
):
    """A coordinate system for F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    axes: _2SpatialAxes = (_space("i"), _space("j"))


@CArrayCoordinateSystem3D.register_polymorph(axes=_is_spatial)
class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D, on=None
):
    """A coordinate system for C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: _3SpatialAxes = (_space("k"), _space("j"), _space("i"))


@FArrayCoordinateSystem3D.register_polymorph(axes=_is_spatial)
class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D, on=None
):
    """A coordinate system for F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: _3SpatialAxes = (_space("i"), _space("j"), _space("k"))


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------
# An anatomical system fixes a direction per axis and says nothing about
# the metric: an array can be RAS-oriented and indexed in samples. Their
# default axes are instances of their own, built from the classes. They
# take precedence over the pixel and voxel systems, which say less about a
# system of oriented, sampled axes than the orientation does.


class RASCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RAS")}, priority=2
):
    """The RAS anatomical coordinate system.

    Coordinates increase toward the right, the anterior, and the
    superior directions. This coordinate system is used by NIfTI files,
    and by many other neuroimaging formats.
    """

    name: tx.Optional[str] = "RAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _axes.R(),
        _axes.A(),
        _axes.S(),
    )


class LPSCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("LPS")}, priority=2
):
    """The LPS anatomical coordinate system.

    Coordinates increase toward the left, the posterior, and the
    superior directions. This coordinate system is used by ITK, and
    therefore also by ANTs, 3D Slicer, and other ITK-based tools.
    """

    name: tx.Optional[str] = "LPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _axes.L(),
        _axes.P(),
        _axes.S(),
    )


class RSACoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RSA")}, priority=2
):
    """The RSA anatomical coordinate system.

    Coordinates increase toward the right, the superior, and the
    anterior directions. This coordinate system appears in some
    FreeSurfer LTA files.
    """

    name: tx.Optional[str] = "RSA"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisIS, _axes.AxisPA] = (
        _axes.R(),
        _axes.S(),
        _axes.A(),
    )


# ----------------------------------------------------------------------
#   PHYSICAL ANATOMICAL SPACES
# ----------------------------------------------------------------------
# These shorthands are the physical anatomical systems -- the millimetre
# spaces that nearly every file format means when it writes an anatomical
# affine -- and they inherit [`PhysicalCoordinateSystem`][], so an axis of
# theirs can never be left without a unit. Each is selected on its own
# orientation *and* a physical unit on every axis: the orientation is what
# its anatomical parent already checks, and checking it again keeps
# `PhysicalCoordinateSystem(axes=<LPS axes in mm>)` from reaching `RASmm`.


def _mm(axis: tx.Type[Axis]) -> Axis:
    return axis(unit="mm")


class RASmm(
    RASCoordinateSystem,
    PhysicalCoordinateSystem,
    on={"axes": _both(_is_anat("RAS"), _is_physical)},
):
    """[`RASCoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "RAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _mm(_axes.AxisLR),
        _mm(_axes.AxisPA),
        _mm(_axes.AxisIS),
    )


class LPSmm(
    LPSCoordinateSystem,
    PhysicalCoordinateSystem,
    on={"axes": _both(_is_anat("LPS"), _is_physical)},
):
    """[`LPSCoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "LPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _mm(_axes.AxisRL),
        _mm(_axes.AxisAP),
        _mm(_axes.AxisIS),
    )


class RSAmm(
    RSACoordinateSystem,
    PhysicalCoordinateSystem,
    on={"axes": _both(_is_anat("RSA"), _is_physical)},
):
    """[`RSACoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "RSA"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisIS, _axes.AxisPA] = (
        _mm(_axes.AxisLR),
        _mm(_axes.AxisIS),
        _mm(_axes.AxisPA),
    )


# ----------------------------------------------------------------------
#   ANATOMICAL VOXEL SPACES
# ----------------------------------------------------------------------
# An F-ordered grid lists its axes x, y, z; a C-ordered one lists them z,
# y, x. So an F-ordered RAS grid has axes that point R, A, S, and a
# C-ordered one has axes that point S, A, R -- which is what each is
# selected on, from its ordered voxel base. Like that base, they stay out
# of the dispatch of every other class (`on=None`): the C-ordered ones
# list their axes in an order their anatomical parent does not select.


def _sampled(axis: tx.Type[Axis], name: str) -> Axis:
    return axis(name=name, unit=_SAMPLE)


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RAS"))
class FRASCoordinateSystem(
    RASCoordinateSystem, FVoxelCoordinateSystem, on=None
):
    """Combines [`RASCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: tx.Optional[str] = "fRAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _sampled(_axes.AxisLR, "x"),
        _sampled(_axes.AxisPA, "y"),
        _sampled(_axes.AxisIS, "z"),
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("LPS"))
class FLPSCoordinateSystem(
    LPSCoordinateSystem, FVoxelCoordinateSystem, on=None
):
    """Combines [`LPSCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: tx.Optional[str] = "fLPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _sampled(_axes.AxisRL, "x"),
        _sampled(_axes.AxisAP, "y"),
        _sampled(_axes.AxisIS, "z"),
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RSA"))
class FRSACoordinateSystem(
    RSACoordinateSystem, FVoxelCoordinateSystem, on=None
):
    """Combines [`RSACoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: tx.Optional[str] = "fRSA"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisIS, _axes.AxisPA] = (
        _sampled(_axes.AxisLR, "x"),
        _sampled(_axes.AxisIS, "y"),
        _sampled(_axes.AxisPA, "z"),
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("SAR"))
class CRASCoordinateSystem(
    RASCoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """Combines [`RASCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: tx.Optional[str] = "cRAS"
    axes: tx.Tuple[_axes.AxisIS, _axes.AxisPA, _axes.AxisLR] = (
        _sampled(_axes.AxisIS, "z"),
        _sampled(_axes.AxisPA, "y"),
        _sampled(_axes.AxisLR, "x"),
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("SPL"))
class CLPSCoordinateSystem(
    LPSCoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """Combines [`LPSCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: tx.Optional[str] = "cLPS"
    axes: tx.Tuple[_axes.AxisIS, _axes.AxisAP, _axes.AxisRL] = (
        _sampled(_axes.AxisIS, "z"),
        _sampled(_axes.AxisAP, "y"),
        _sampled(_axes.AxisRL, "x"),
    )


@CVoxelCoordinateSystem.register_polymorph(axes=_is_anat("ASR"))
class CRSACoordinateSystem(
    RSACoordinateSystem, CVoxelCoordinateSystem, on=None
):
    """Combines [`RSACoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: tx.Optional[str] = "cRSA"
    axes: tx.Tuple[_axes.AxisPA, _axes.AxisIS, _axes.AxisLR] = (
        _sampled(_axes.AxisPA, "z"),
        _sampled(_axes.AxisIS, "y"),
        _sampled(_axes.AxisLR, "x"),
    )
