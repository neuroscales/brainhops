__all__ = [
    "AxisQuery",
    "AxisSequence",
    "AxisSequenceLike",
    "AxisTuple",
    "AxisList",
]

import abc
import sys
from collections.abc import Mapping
from numbers import Integral

import typing_extensions as tx
from bagof.magic import fields

from .base import Axis, AxisLike

if tx.TYPE_CHECKING:
    from types import EllipsisType as _Ellipsis

else:
    # `types.EllipsisType` exists from Python 3.10.
    _Ellipsis: tx.TypeAlias = type(Ellipsis)


# Type of the items of an `AxisSequence`.
AXIS = tx.TypeVar("AXIS")

AxisQuery: tx.TypeAlias = tx.Union[
    Axis,
    str,
    tx.Type[Axis],
    tx.Mapping[str, tx.Any],
    tx.Callable[[Axis], bool],
]
"""A description of the axes to find in an [`AxisSequence`][].

A query is one of the following:

- a name, which stands for `Axis(name=...)`;
- an axis used as a template, which matches the axes that are instances
  of its class and have the same value for every field that it sets;
- an axis class, which matches its instances, so that `ComponentAxis`
  matches every point, vector, displacement and tensor axis;
- a dictionary of axis fields, which stands for `Axis(**fields)`, so that
  `{"type": "space"}` is the template `SpaceAxis()`;
- a function that receives an axis and returns whether it matches.

The `...` of a variadic sequence matches no query.
"""

# Types of the items of an `AxisTuple`, in order.
AXES = tx.TypeVarTuple("AXES")


class AxisSequence(tx.Sequence[AXIS]):
    """Axes of a coordinate system, some of which may be unknown.

    An axis sequence holds [`Axis`][] objects and at most one `...`,
    which stands for zero or more unknown axes. A sequence that holds
    `...` is variadic, because its number of axes is unknown; otherwise
    it is finite and lists every axis. The `...` is an entry but never
    counts as an axis. A sequence with more than one `...` describes no
    axes: the methods that read axes raise `ValueError`, and coordinate
    systems refuse it. For example, `[..., TimeAxis()]` says that the
    last axis is a time axis, and `[...]` says nothing.

    This class is the read-only base of [`AxisList`][], which is mutable,
    and [`AxisTuple`][], which is immutable. Slicing and the methods that
    build new sequences return the type of the original sequence. The type
    parameter is the type of the items, so an `AxisSequence[Axis]`, whose
    items are all axes, is finite.

    !!! note "Entries and axes"
        Length, iteration, equality and `axes[i]` work on entries,
        including `...`. [`ndim`][] counts axes, and [`at`][],
        [`expand`][], [`restrict`][], [`embed`][] and [`compatible_with`][]
        work on positions in the space, where `...` stands for as many axes
        as needed. Non-negative positions count from the first axis and
        negative positions from the last. In a finite sequence, entries and
        axes coincide. In `[x, ..., t]`, however, `axes[2]` is `t` while
        `axes.at(2)` is one of the axes that `...` stands for.

    !!! note "Finding an axis"
        [`index`][] finds the first axis that matches a query, and
        [`indices`][] finds every axis that matches it. A query is a name,
        a template axis, an axis class, a dictionary of axis fields or a
        function, as [`AxisQuery`][] describes. Thus
        `axes.index(TimeAxis())` finds the first time axis whatever its
        unit, and `axes.indices(SpaceAxis)` finds every spatial axis. Both
        methods return numbers that are entries and positions at once:
        an axis before `...` is counted from the start and an axis after
        `...` from the end, so that `axes[i]` and `axes.at(i)` give the
        same axis. The `...` matches nothing, and an unknown `Axis()`
        matches no query that sets a field.

    !!! note "Names"
        `axes["t"]` returns the explicit axis named `"t"`, `"t" in axes`
        tests for one, and [`names`][] lists the names. A name that several
        axes share is refused. There is no mapping interface, because axes
        may be unnamed or share names.

    !!! example
        ```pycon
        >>> x, t = Axis(name="x"), TimeAxis(name="t")
        >>> axes = AxisList([x, ..., t])
        >>> axes.ndim is None, axes.is_variadic
        (True, True)
        >>> axes.index("t"), axes.index(TimeAxis()), axes["t"] is t
        (-1, -1, True)
        >>> "x" in axes, axes.names
        (True, ('x', Ellipsis, 't'))
        >>> axes[2] is t, axes.at(2), axes.at(-1) is t
        (True, Axis(), True)
        >>> axes.expand(4)[1:]
        [Axis(), Axis(), TimeAxis(name='t')]
        >>> axes.restrict([-1, 0, 1])[1:]
        [Axis(name='x'), Axis()]
        ```
    """

    __slots__ = ()

    @abc.abstractmethod
    def _entry(self, key: tx.Any) -> tx.Any:
        # Read the underlying list or tuple storage.
        ...

    @tx.overload
    def __getitem__(self, key: tx.SupportsIndex) -> AXIS: ...

    @tx.overload
    def __getitem__(self, key: slice) -> tx.Self: ...

    @tx.overload
    def __getitem__(self, key: str) -> Axis: ...

    def __getitem__(self, key: tx.Any) -> tx.Any:
        """Return an entry, a slice, or the explicit axis with a name.

        Integers and slices index entries, as they do for a list, whereas
        [`at`][] reads the axis at a position in the space.

        !!! example
            ```pycon
            >>> x, t = Axis(name="x"), TimeAxis(name="t")
            >>> AxisList([x, ..., t])["t"] is t
            True
            >>> AxisList([x, ..., t])[2] is t
            True
            ```

        Raises
        ------
        KeyError
            If no axis has the name.
        ValueError
            If several axes have the name.
        """
        if isinstance(key, str):
            return self._entry(self._entry_named(key))
        if isinstance(key, slice):
            return type(self)(self._entry(key))
        return self._entry(key)

    def __contains__(self, item: object) -> bool:
        """Return whether an item is an entry, or a string names an axis."""
        if isinstance(item, str):
            return bool(self._entries_named(item))
        return any(entry is item or entry == item for entry in self)

    def index(
        self,
        query: AxisQuery,
        start: tx.SupportsIndex = 0,
        stop: tx.SupportsIndex = sys.maxsize,
    ) -> int:
        """Return the index of the first axis that matches a query.

        The test is looser than equality: unset fields of a template are
        not compared, as described in the class documentation. The index
        of an axis after `...` counts from the end, so that it is both
        an entry and a position in the space.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis(name="t", unit="s")
            >>> axes = AxisList([x, ..., t])
            >>> axes.index("t"), axes.index(SpaceAxis())
            (-1, 0)
            >>> axes.index(Axis(unit="second")), axes.index(Axis())
            (-1, 0)
            >>> axes.index({"type": "time"}), axes.index(TimeAxis)
            (-1, -1)
            >>> axes.index("y")
            Traceback (most recent call last):
              ...
            ValueError: Axis(name='y') is not in list
            ```

        Parameters
        ----------
        query : AxisQuery
            A name, a template axis, an axis class, a dictionary of axis
            fields or a function, as [`AxisQuery`][] describes.
        start : int, default=0
            Start of the search window among the entries, as for
            `list.index`.
        stop : int, default=sys.maxsize
            End of the search window among the entries, as for
            `list.index`.

        Returns
        -------
        int
            Index of the axis. It is non-negative for an axis before `...`
            or in a finite sequence, and negative for an axis after `...`.

        Raises
        ------
        ValueError
            If no axis matches.
        TypeError
            If the query is not one of the forms of [`AxisQuery`][].
        """
        matches, label = _matcher(query)
        entries = list(self)
        for i in range(*slice(start, stop).indices(len(entries))):
            if entries[i] is not ... and matches(entries[i]):
                return self._position_of_entry(i)
        raise ValueError(f"{label} is not in list")

    def indices(self, query: AxisQuery) -> tx.List[int]:
        """Return the indices of every axis that matches a query.

        The axes are found as by [`index`][], and their indices follow the
        order of the sequence. An index is both an entry and a position in
        the space: it counts from the start for an axis before `...`, or
        in a finite sequence, and from the end for an axis after `...`.
        The indices can therefore be passed to [`restrict`][] and
        [`at`][].

        !!! example
            ```pycon
            >>> x, y = SpaceAxis(name="x"), SpaceAxis(name="y")
            >>> t = TimeAxis(name="t")
            >>> AxisList([x, y, t]).indices(SpaceAxis)
            [0, 1]
            >>> AxisList([x, ..., y, t]).indices({"type": "space"})
            [0, -2]
            >>> AxisList([x, y, t]).indices(lambda axis: axis.name != "y")
            [0, 2]
            ```

        Parameters
        ----------
        query : AxisQuery
            A name, a template axis, an axis class, a dictionary of axis
            fields or a function, as [`AxisQuery`][] describes.

        Returns
        -------
        list of int
            The indices of the matching axes, which is empty when no axis
            matches.

        Raises
        ------
        TypeError
            If the query is not one of the forms of [`AxisQuery`][].
        """
        matches, _ = _matcher(query)
        return [
            self._position_of_entry(i)
            for i, entry in enumerate(self)
            if entry is not ... and matches(entry)
        ]

    @property
    def names(self) -> tx.Tuple[tx.Union[str, None, _Ellipsis], ...]:
        """Name of each entry, or `None` for an unnamed axis.

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
        """Number of axes, or `None` when the sequence is variadic.

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
    def is_variadic(self) -> bool:
        """
        Whether the sequence holds `...`, so its number of axes is unknown.

        `AxisList([...])` is variadic, and an empty list is finite.
        """
        return self._split()[1] is not None

    def expand(self, ndim: int) -> tx.Self:
        """Return a finite sequence of `ndim` axes.

        In a variadic sequence, `...` is replaced by as many unknown
        axes as needed. A finite sequence is returned as a copy.

        !!! example
            ```pycon
            >>> AxisList([Axis(name="x"), ...]).expand(3)
            [Axis(name='x'), Axis(), Axis()]
            >>> AxisList([...]).expand(2)
            [Axis(), Axis()]
            ```

        Raises
        ------
        ValueError
            If `ndim` is smaller than the number of explicit axes of a
            variadic sequence, or differs from the length of a finite
            sequence.
        TypeError
            If `ndim` is not an integer.
        """
        ndim = _as_int(ndim, "ndim")
        prefix, suffix = self._split()
        if suffix is None:
            if ndim != len(prefix):
                raise ValueError(
                    f"Cannot expand a finite list of {len(prefix)} axes "
                    f"to {ndim} axes."
                )
            return type(self)(prefix)
        explicit = len(prefix) + len(suffix)
        if ndim < explicit:
            raise ValueError(
                f"Cannot expand a variadic list with {explicit} explicit "
                f"axes to {ndim} axes."
            )
        fill = [Axis() for _ in range(ndim - explicit)]
        return type(self)(prefix + fill + suffix)

    def restrict(
        self,
        refs: tx.Union[AxisQuery, tx.Iterable[tx.Union[int, AxisQuery]]],
    ) -> tx.Self:
        """Return the axes that some references designate, in that order.

        This is how a subspace is described, for example the spatial axes
        of an image. A reference is a position in the space, the name of
        an explicit axis as in `axes[name]`, or any other query of
        [`AxisQuery`][], which designates every axis that it matches, in
        the order of the sequence. A single query may be given instead of
        a list. In a finite sequence of `n` axes, positions lie in
        `[-n, n)`. In a variadic sequence, every position is valid, and
        the positions that fall among the axes of `...` give unknown axes.

        !!! example
            ```pycon
            >>> x, y = SpaceAxis(name="x"), SpaceAxis(name="y")
            >>> t = TimeAxis(name="t")
            >>> AxisList([x, y, t]).restrict(["t", 0])
            [TimeAxis(name='t'), SpaceAxis(name='x')]
            >>> AxisList([x, y, t]).restrict(SpaceAxis)
            [SpaceAxis(name='x'), SpaceAxis(name='y')]
            >>> AxisList([x, ...]).restrict([0, 1])
            [SpaceAxis(name='x'), Axis()]
            ```

        Parameters
        ----------
        refs : AxisQuery or iterable of (int or AxisQuery)
            The positions, names and other queries that designate the axes
            to keep.

        Returns
        -------
        AxisSequence
            A new finite sequence with the designated axes.

        Raises
        ------
        IndexError
            If a position lies outside a finite sequence.
        KeyError
            If no explicit axis has a name.
        ValueError
            If a name is shared, or two references designate the same axis.
        TypeError
            If a reference is neither an integer nor a query.
        """
        return type(self)([self.at(p) for p in self.positions(refs)])

    def positions(
        self,
        refs: tx.Union[AxisQuery, tx.Iterable[tx.Union[int, AxisQuery]]],
    ) -> tx.List[int]:
        """Return the positions of the axes that references designate.

        The references are read as [`restrict`][] reads them, and the
        result lists the position of each designated axis, in the order
        of `refs`, instead of the axes themselves. A position in a
        finite sequence is made non-negative. A position in a variadic
        sequence is returned as it is given, and the position of an
        axis after `...` counts from the end, as in [`indices`][].

        !!! example
            ```pycon
            >>> x, y = SpaceAxis(name="x"), SpaceAxis(name="y")
            >>> t = TimeAxis(name="t")
            >>> AxisList([x, y, t]).positions(["t", -3])
            [2, 0]
            >>> AxisList([x, ..., t]).positions([SpaceAxis, "t", 1])
            [0, -1, 1]
            ```

        Parameters
        ----------
        refs : AxisQuery or iterable of (int or AxisQuery)
            The positions, names and other queries that designate the
            axes.

        Returns
        -------
        list of int
            The position of each designated axis.

        Raises
        ------
        IndexError
            If a position lies outside a finite sequence.
        KeyError
            If no explicit axis has a name.
        ValueError
            If a name is shared, or two references designate the same
            axis.
        TypeError
            If a reference is neither an integer nor a query.
        """
        if _is_query(refs) or isinstance(refs, str):
            refs = [refs]
        positions = []
        for ref in _as_list(refs, "refs"):
            if isinstance(ref, str):
                entry = self._entry_named(ref)
                positions.append(self._position_of_entry(entry))
            elif _is_query(ref):
                positions.extend(self.indices(ref))
            else:
                positions.append(self._position(ref))
        _check_unique(positions, "refs")
        return positions

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> tx.Self:
        """Return the axes of a larger space in which these axes sit.

        This method is the inverse of [`restrict`][]. Axis `j` of this
        sequence sits at `positions[j]` in the larger space, and the other
        axes of the larger space are unknown. A variadic sequence is first
        broadcast to `len(positions)` axes, as by [`expand`][]. Positions
        refer to a space that does not exist yet, so names are not accepted.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> AxisList([x]).embed([1])
            [Axis(), Axis(name='x'), Ellipsis]
            >>> AxisList([x]).embed([1], ndim=3)
            [Axis(), Axis(name='x'), Axis()]
            ```

        Parameters
        ----------
        positions : iterable of int
            Non-negative position of each axis.
        ndim : int, optional
            Number of axes of the larger space. When it is omitted, the
            result is open-ended and ends with `...`.

        Raises
        ------
        ValueError
            If a position is negative or repeated, if `ndim` does not
            exceed every position, or if the sequence cannot be
            restricted to `len(positions)` axes.
        TypeError
            If a position or `ndim` is not an integer.
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
        """Return whether two sequences could describe the same axes.

        The sequences are compatible when some choice of what each `...` stands
        for makes them match axis by axis, with each pair compatible in the
        sense of [`Axis.compatible_with`][].
        Unlike equality, an unknown axis matches any axis and `[...]` matches
        every sequence. The relation is symmetric but not transitive.

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis()
            >>> AxisList([x, ...]).compatible_with([x, Axis(), t])
            True
            >>> AxisList([..., t]).compatible_with([x])
            False
            ```

        Raises
        ------
        TypeError
            If `other` is not a list or a tuple.
        """
        if not isinstance(other, (list, tuple)):
            raise TypeError(
                f"A list of axes is compatible only with another list of "
                f"axes, not with {type(other).__name__}."
            )
        p1, s1 = self._split()
        p2, s2 = _split(other)
        if s1 is None and s2 is None:
            return len(p1) == len(p2) and _pairwise(p1, p2)
        if s1 is None:
            # Make the first sequence the variadic one.
            (p1, s1), (p2, s2) = (p2, s2), (p1, s1)
        assert s1 is not None
        if s2 is None:
            # The explicit axes of the opvariadicen sequence must fit at
            # the start and at the end of the finite one.
            n = len(p2)
            if len(p1) + len(s1) > n:
                return False
            return _pairwise(p1, p2[: len(p1)]) and _pairwise(
                s1, p2[n - len(s1) :]
            )
        # When both sequences are variadic, each `...` can stand for
        # enough axes that only the axes that both sequences state at
        # the start, or both state at the end, are compared.
        k = min(len(p1), len(p2))
        m = min(len(s1), len(s2))
        return _pairwise(p1[:k], p2[:k]) and _pairwise(
            s1[len(s1) - m :], s2[len(s2) - m :]
        )

    # --- private helpers ----------------------------------------------
    # Positions in the space count from the first axis when non-negative and
    # from the last axis when negative.

    def _split(self) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
        return _split(self)

    def _position(self, position: int) -> int:
        # In a finite sequence, a negative position is made non-negative.
        # In a variadic sequence, every position is valid and returned
        # unchanged.
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
        # An entry after `...` counts from the end, since its distance from the
        # start is unknown.
        entries = list(self)
        entry = range(len(entries))[entry]
        if entries[entry] is ...:
            raise ValueError("`...` stands for axes, and is not one.")
        after = ... in entries[:entry]
        return entry - len(entries) if after else entry

    def at(self, position: int) -> Axis:
        """Return the axis at a position in the space.

        Non-negative positions count from the first axis and negative
        positions from the last. In a finite sequence, `axes.at(i)`
        equals `axes[i]` and the position lies in `[-ndim, ndim)`.
        In a variadic sequence, every position is valid, and a position
        that falls among the axes of `...` gives a new unknown axis.

        !!! example
            ```pycon
            >>> x, t = Axis(name="x"), TimeAxis(name="t")
            >>> axes = AxisList([x, ..., t])
            >>> axes.at(0) is x, axes.at(-1) is t, axes.at(1)
            (True, True, Axis())
            >>> axes[2] is t, axes.at(2)
            (True, Axis())
            ```

        Raises
        ------
        IndexError
            If the position lies outside a finite sequence.
        TypeError
            If the position is not an integer.
        """
        position = self._position(position)
        prefix, suffix = self._split()
        if suffix is None or 0 <= position < len(prefix):
            return prefix[position]
        if -len(suffix) <= position < 0:
            return suffix[position]
        return Axis()

    def _entries_named(self, name: str) -> tx.List[int]:
        return [i for i, axis in enumerate(self) if _name(axis) == name]

    def _entry_named(self, name: str) -> int:
        entries = self._entries_named(name)
        if not entries:
            raise KeyError(f"No axis of the list is named {name!r}.")
        if len(entries) > 1:
            raise ValueError(
                f"Cannot read the axis named {name!r}: {len(entries)} axes "
                f"of the list have that name."
            )
        return entries[0]


class AxisTuple(tuple, AxisSequence, tx.Generic[tx.Unpack[AXES]]):
    """Immutable [`AxisSequence`][] for systems with a fixed number of axes.

    The type parameters fix the item type at each position, and thereby the
    number of items. A field typed `AxisTuple[SpaceAxis, SpaceAxis]`
    converts each item to the type of its position and refuses a wrong
    count, `None` and `...`. A bare `AxisTuple` holds any items.

    !!! example
        ```pycon
        >>> axes = RASCoordinateSystem().axes
        >>> type(axes).__name__, axes.ndim, axes.names[0]
        ('AxisTuple', 3, 'left-to-right')
        >>> axes["left-to-right"] is axes[0] is axes.at(-3)
        True
        ```
    """

    # `tuple` comes first among the bases so that it provides the storage.
    # The methods of `AxisSequence` that read items and names would then be
    # hidden by those of `tuple`, so they are bound explicitly.
    __getitem__ = AxisSequence.__getitem__
    __contains__ = AxisSequence.__contains__
    index = AxisSequence.index
    _entry = tuple.__getitem__


class AxisList(AxisSequence[AXIS], list):
    """Mutable [`AxisSequence`][] for systems with any number of axes.

    A list or tuple given to such a system is converted item by item to the
    axis type of the system. The sequence `[...]` says nothing about the
    axes. A coordinate system may instead store `None` for unknown axes,
    which [`get_axes`][brainhops.datamodel._sugar.get_axes] reads as
    `[...]`. An `AxisList[Axis]` is finite.

    !!! example
        ```pycon
        >>> axes = CoordinateSystem(axes=[Axis(name="x"), ...]).axes
        >>> type(axes).__name__, axes.is_variadic, axes["x"]
        ('AxisList', True, Axis(name='x'))
        >>> axes.append(TimeAxis(name="t"))
        >>> axes.at(-1)
        TimeAxis(name='t')
        ```
    """

    # `collections.abc.Sequence` sits between the bases in the MRO and
    # would otherwise supply its generic mixin methods and an abstract
    # `__len__`, so the methods of `list` are bound explicitly.
    __len__ = list.__len__
    __iter__ = list.__iter__
    __reversed__ = list.__reversed__
    count = list.count
    _entry = list.__getitem__


AXIS_LIKE = tx.Union[AXIS, AxisLike, tx.Literal[...]]

AxisSequenceLike: tx.TypeAlias = tx.Union[
    AxisSequence[AXIS], tx.Sequence[AXIS_LIKE]
]
"""
A value that is a [`AxisSequence`][] or that can be converted to one.
"""

# ---- private helpers -------------------------------------------------


def _split(
    entries: tx.Iterable[tx.Any],
) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
    """
    Return the explicit axes before and after `...`.
    The second list is `None` for a finite sequence.
    """
    entries = list(entries)
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


def _is_query(value: tx.Any) -> bool:
    # Return whether a value is a query other than a name. A string is left
    # out because a name is also a reference of its own in `restrict`.
    if isinstance(value, (Axis, Mapping)):
        return True
    if isinstance(value, type):
        return issubclass(value, Axis)
    return callable(value)


def _matcher(query: tx.Any) -> tx.Tuple[tx.Callable[[tx.Any], bool], str]:
    # Return a function that tests an axis against a query, and a label of
    # the query for error messages.
    if isinstance(query, str):
        query = Axis(name=query)
    elif isinstance(query, Mapping):
        query = Axis(**query)
    if isinstance(query, Axis):
        template = query
        return (lambda axis: _matches(axis, template)), repr(template)
    if isinstance(query, type) and issubclass(query, Axis):
        cls = query
        return (lambda axis: isinstance(axis, cls)), cls.__name__
    if callable(query) and not isinstance(query, type):
        return query, repr(query)
    raise TypeError(
        f"An axis is found by a name, an Axis, an Axis class, a dict of "
        f"axis fields or a function, not by a {type(query).__name__}."
    )


def _matches(candidate: tx.Any, query: Axis) -> bool:
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
