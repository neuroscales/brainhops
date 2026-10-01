"""Coordinate systems, from unitless arrays to anatomical spaces."""

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
]
# stdlib
import sys
from numbers import Integral

# externals
import typing_extensions as tx
from bagof.magic import fields, replace

# internals
from . import axes as _axes
from .axes import Axis, SpatialAxis
from .base import DataModelBase

_Ellipsis = type(Ellipsis)
# The type of `...`. Python 3.10 names it `types.EllipsisType`.

AXIS = tx.TypeVar("AXIS")
# The type of the items of an `AxisList`.

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis]
_3SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis, SpatialAxis]


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
        * `axes.index(TimeAxis())` finds the first time axis whose unit
          is the default second;
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
    one, item by item, to the type of axis the class declares. The
    system, or its axes, may also be `None`, which means the same as
    `[...]`. [`of`][brainhops.datamodel.systems.AxisList.of] reads the
    axes of any system, or of none, as an `AxisList`.

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
        [Axis(), Axis(), TimeAxis(name='t', unit='second')]
        >>> axes.restrict([-1, 0, 1])[1:]
        [Axis(name='x'), Axis()]
        ```
    """

    @classmethod
    def of(cls, system: tx.Optional["CoordinateSystem"]) -> tx.Self:
        """The axes of a coordinate system that may be missing.

        This is the one way to read the axes of a system whatever it is:

        * a missing system (`None`) and a system whose `axes` are `None`
          say nothing about their axes, and both read as `[...]`;
        * any other system reads as its axes, whether its class stores
          them as an `AxisList` or, for a fixed number of axes, as a
          tuple.

        The system is left as it is: its `axes` keep what was given.

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
            axes = None
        elif isinstance(system, CoordinateSystem):
            axes = system.axes
        else:
            raise TypeError(
                f"Expected a CoordinateSystem or None, not a "
                f"{type(system).__name__}."
            )
        return cls([...] if axes is None else axes)

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
            >>> x, t = SpatialAxis(name="x"), TimeAxis(name="t")
            >>> axes = AxisList([x, ..., t])
            >>> axes.index("t"), axes.index(SpatialAxis())
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
            >>> x, t = SpatialAxis(name="x"), TimeAxis()
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


class CoordinateSystem(DataModelBase, eq=False):
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
        * `axes=None` means the same as `axes=[...]`: nothing is known.
          It is stored as given, but every method, equality and
          [`compatible_with`][] treat the two spellings identically.

        A list or a tuple given as `axes` is stored as an [`AxisList`][].
        [`AxisList.of`][] reads the axes of any system, or of a missing
        one, as an [`AxisList`][], with `[...]` for `None`. An axis is
        read by its name as `AxisList.of(system)["x"]`, and found by
        [`index`][brainhops.datamodel.systems.AxisList.index].

        Classes with a fixed number of axes, such as
        [`CoordinateSystem3D`][], are always closed. They store their
        axes as a tuple, whose type fixes the number of axes and the
        class of each one, so they reject `...` and `axes=None`.

    !!! note "Equality"
        Two systems are equal when they are of the same class, have the
        same name, and have equal axes, where `axes=None` equals
        `axes=[...]`. A plain `CoordinateSystem` with no name and no known
        axis, which says nothing at all, also equals `None`, the missing
        endpoint of a transformation. [`compatible_with`][] is the looser
        question of whether two systems could describe the same space.
    """

    # `eq=False`: the hand-written `__eq__` below must also be the one of
    # every subclass. With the default `eq=True`, Magic writes a field-wise
    # `__eq__` into each subclass that does not define its own, which
    # would shadow this one and tell `axes=None` from `axes=[...]`.

    name: tx.Optional[str] = None
    axes: tx.Optional[AxisList[tx.Union[Axis, _Ellipsis]]] = None

    # --- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        if self.axes is not None and sum(a is ... for a in self.axes) > 1:
            raise ValueError(
                "The axes of a coordinate system hold at most one `...`, "
                "which stands for all the axes about which nothing is known."
            )

    # --- equality -----------------------------------------------------

    def __eq__(self, other: tx.Any) -> bool:
        if other is None:
            return _is_unknown(self)
        if type(other) is not type(self):
            return NotImplemented
        for field in fields(type(self)):
            if field.eq and field.name != "axes":
                if getattr(self, field.name) != getattr(other, field.name):
                    return False
        return AxisList.of(self) == AxisList.of(other)

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` when the system is open.

        A closed system has exactly `len(axes)` axes. An open system,
        whose axes hold `...` or are `None`, has an unknown number of
        axes, and its `ndim` is `None`. This is
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
        `Axis()` as needed to reach `ndim` axes. The class and the name
        are kept. Use it once the number of axes is known, for instance
        from the shape of the data.

        Parameters
        ----------
        ndim : int
            The number of axes.

        Returns
        -------
        CoordinateSystem
            A closed system of `ndim` axes, of the same class. A closed
            system is returned as itself.

        Raises
        ------
        ValueError
            If `ndim` is less than the number of explicit axes of an open
            system, or differs from the number of axes of a closed one.
        TypeError
            If `ndim` is not an integer.

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(name="x"), ...]).expand(3)
            CoordinateSystem(axes=[Axis(name='x'), Axis(), Axis()])
            >>> CoordinateSystem().expand(2)
            CoordinateSystem(axes=[Axis(), Axis()])
            ```
        """
        axes = AxisList.of(self)
        expanded = axes.expand(ndim)
        return self if not axes.is_open else replace(self, axes=expanded)

    def restrict(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """The system of the axes at some positions of this system.

        The axes are restricted by
        [`AxisList.restrict`][brainhops.datamodel.systems.AxisList.restrict]:
        a reference is a position in the space or a name, and a position
        of an open system that falls among the axes that `...` stands
        for gives an unknown `Axis()`. The axes are listed in the order
        of `refs`. The result is a closed, unnamed
        [`CoordinateSystem`][], whatever the class of this system,
        because it describes a different space.

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
            CoordinateSystem(axes=[Axis(name='z'), Axis(name='x')])
            >>> CoordinateSystem(axes=[x, ...]).restrict([0, 1])
            CoordinateSystem(axes=[Axis(name='x'), Axis()])
            ```
        """
        return CoordinateSystem(axes=AxisList.of(self).restrict(refs))

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """The system of a larger space in which this system's axes sit.

        This is the inverse of [`restrict`][]. The axes are embedded by
        [`AxisList.embed`][brainhops.datamodel.systems.AxisList.embed]:
        axis `j` of this system sits at `positions[j]` of the result,
        and every other position holds an unknown `Axis()`. The result is
        an unnamed [`CoordinateSystem`][], whatever the class of this
        system, because it describes a different space.

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
            An unnamed [`CoordinateSystem`][], closed when `ndim` is given
            and open otherwise.

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
            >>> CoordinateSystem(axes=[x]).embed([1], ndim=3)
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Axis()])
            ```
        """
        return CoordinateSystem(
            axes=AxisList.of(self).embed(positions, ndim=ndim)
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
            >>> x, t = SpatialAxis(name="x"), TimeAxis()
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


def _is_unknown(system: tx.Optional[CoordinateSystem]) -> bool:
    # Whether `system` says nothing at all: it is missing, or it is a plain
    # `CoordinateSystem` with no name whose axes are `None` or `[...]`.
    # Such a system equals `None`.
    if system is None:
        return True
    return (
        type(system) is CoordinateSystem
        and system.name is None
        and AxisList.of(system) == [...]
    )


class CoordinateSystem2D(CoordinateSystem):
    """A coordinate systems with exactly two dimensions."""

    axes: _2Axes = (Axis(), Axis())


class CoordinateSystem3D(CoordinateSystem):
    """A coordinate system with exactly three dimensions."""

    axes: _3Axes = (Axis(), Axis(), Axis())


# ----------------------------------------------------------------------
#   ARRAY COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class ArrayCoordinateSystem(CoordinateSystem):
    """A coordinate system for a unitless, multidimensional array.

    By default, the array is assumed C-ordered: the first axis is the
    slowest changing in memory, and the last axis is the fastest changing.
    """

    name: tx.Optional[str] = "array"


class CArrayCoordinateSystem(ArrayCoordinateSystem):
    """A coordinate system for a unitless, C-ordered multidimensional array."""

    name: tx.Optional[str] = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem):
    """A coordinate system for a unitless, F-ordered multidimensional array."""

    name: tx.Optional[str] = "farray"


class ArrayCoordinateSystem2D(CoordinateSystem2D, ArrayCoordinateSystem):
    """A coordinate system for a unitless array with two dimensions."""

    axes: _2Axes = (Axis("dim0"), Axis("dim1"))


class ArrayCoordinateSystem3D(CoordinateSystem3D, ArrayCoordinateSystem):
    """A coordinate system for a unitless array with three dimensions."""

    axes: _3Axes = (Axis("dim0"), Axis("dim1"), Axis("dim2"))


class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem):
    """A coordinate system for a unitless, C-ordered array with two
    dimensions."""


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem):
    """A coordinate system for a unitless, C-ordered array with three
    dimensions."""


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem):
    """A coordinate system for a unitless, F-ordered array with two
    dimensions."""


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem):
    """A coordinate system for a unitless, F-ordered array with three
    dimensions."""


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class SpatialCoordinateSystem(CoordinateSystem):
    """A coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[AxisList[tx.Union[SpatialAxis, _Ellipsis]]] = None


class SpatialCoordinateSystem2D(CoordinateSystem2D, SpatialCoordinateSystem):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: _2SpatialAxes = (SpatialAxis(), SpatialAxis())


class SpatialCoordinateSystem3D(CoordinateSystem3D, SpatialCoordinateSystem):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: _3SpatialAxes = (
        SpatialAxis(),
        SpatialAxis(),
        SpatialAxis(),
    )


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for (unitless) 2D pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: _2SpatialAxes = (
        SpatialAxis(name="dim0", unit=None),
        SpatialAxis(name="dim1", unit=None),
    )


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: _3SpatialAxes = (
        SpatialAxis(name="dim0", unit=None),
        SpatialAxis(name="dim1", unit=None),
        SpatialAxis(name="dim2", unit=None),
    )


class CPixelCoordinateSystem(PixelCoordinateSystem, CArrayCoordinateSystem2D):
    """A coordinate system for (unitless) C-ordered 2D pixel grids."""

    name: tx.Optional[str] = "cpixel"
    axes: _2SpatialAxes = (
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="i", unit=None),
    )


class FPixelCoordinateSystem(PixelCoordinateSystem, FArrayCoordinateSystem2D):
    """A coordinate system for (unitless) F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    axes: _2SpatialAxes = (
        SpatialAxis(name="i", unit=None),
        SpatialAxis(name="j", unit=None),
    )


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: _3SpatialAxes = (
        SpatialAxis(name="k", unit=None),
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="i", unit=None),
    )


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: _3SpatialAxes = (
        SpatialAxis(name="i", unit=None),
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="k", unit=None),
    )


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class RASCoordinateSystem(SpatialCoordinateSystem3D):
    """The RAS anatomical coordinate system.

    Coordinates increase toward the right, the anterior, and the
    superior directions. This coordinate system is used by NIfTI files,
    and by many other neuroimaging formats.
    """

    name: str = "RAS"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.PosteriorToAnteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (_axes.R, _axes.A, _axes.S)


class LPSCoordinateSystem(SpatialCoordinateSystem3D):
    """The LPS anatomical coordinate system.

    Coordinates increase toward the left, the posterior, and the
    superior directions. This coordinate system is used by ITK, and
    therefore also by ANTs, 3D Slicer, and other ITK-based tools.
    """

    name: str = "LPS"
    axes: tx.Tuple[
        _axes.RightToLeftAxis,
        _axes.AnteriorToPosteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (_axes.L, _axes.P, _axes.S)


class RSACoordinateSystem(SpatialCoordinateSystem3D):
    """The RSA anatomical coordinate system.

    Coordinates increase toward the right, the superior, and the
    anterior directions. This coordinate system appears in some
    FreeSurfer LTA files.
    """

    name: str = "RSA"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.InferiorToSuperiorAxis,
        _axes.PosteriorToAnteriorAxis,
    ] = (_axes.R, _axes.S, _axes.A)


class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RASCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: str = "fRAS"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.PosteriorToAnteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (
        _axes.LeftToRightAxis(name="x"),
        _axes.PosteriorToAnteriorAxis(name="y"),
        _axes.InferiorToSuperiorAxis(name="z"),
    )


class FLPSCoordinateSystem(LPSCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`LPSCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: str = "fLPS"
    axes: tx.Tuple[
        _axes.RightToLeftAxis,
        _axes.AnteriorToPosteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (
        _axes.RightToLeftAxis(name="x"),
        _axes.AnteriorToPosteriorAxis(name="y"),
        _axes.InferiorToSuperiorAxis(name="z"),
    )


class FRSACoordinateSystem(RSACoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RSACoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: str = "fRSA"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.InferiorToSuperiorAxis,
        _axes.PosteriorToAnteriorAxis,
    ] = (
        _axes.LeftToRightAxis(name="x"),
        _axes.InferiorToSuperiorAxis(name="y"),
        _axes.PosteriorToAnteriorAxis(name="z"),
    )


class CRASCoordinateSystem(RASCoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`RASCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: str = "cRAS"
    axes: tx.Tuple[
        _axes.InferiorToSuperiorAxis,
        _axes.PosteriorToAnteriorAxis,
        _axes.LeftToRightAxis,
    ] = (
        _axes.InferiorToSuperiorAxis(name="z"),
        _axes.PosteriorToAnteriorAxis(name="y"),
        _axes.LeftToRightAxis(name="x"),
    )


class CLPSCoordinateSystem(LPSCoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`LPSCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: str = "cLPS"
    axes: tx.Tuple[
        _axes.InferiorToSuperiorAxis,
        _axes.AnteriorToPosteriorAxis,
        _axes.RightToLeftAxis,
    ] = (
        _axes.InferiorToSuperiorAxis(name="z"),
        _axes.AnteriorToPosteriorAxis(name="y"),
        _axes.RightToLeftAxis(name="x"),
    )


class CRSACoordinateSystem(RSACoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`RSACoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: str = "cRSA"
    axes: tx.Tuple[
        _axes.PosteriorToAnteriorAxis,
        _axes.InferiorToSuperiorAxis,
        _axes.LeftToRightAxis,
    ] = (
        _axes.PosteriorToAnteriorAxis(name="z"),
        _axes.InferiorToSuperiorAxis(name="y"),
        _axes.LeftToRightAxis(name="x"),
    )
