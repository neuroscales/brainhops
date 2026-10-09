import abc
import sys
from numbers import Integral

import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import ConvertTo, fields

from .axes import Axis

if tx.TYPE_CHECKING:
    from types import EllipsisType as _Ellipsis

else:
    # `types.EllipsisType` exists from Python 3.10.
    _Ellipsis: tx.TypeAlias = type(Ellipsis)


# Type of the items of an `AxisSequence`.
AXIS = tx.TypeVar("AXIS")

# Types of the items of an `AxisTuple`, in order.
AXES = tx.TypeVarTuple("AXES")


class AxisSequence(tx.Sequence[AXIS]):
    """Axes of a coordinate system, some of which may be unknown.

    An axis sequence holds [`Axis`][] objects and at most one `...`, which
    stands for zero or more unknown axes. A sequence that holds `...` is
    open, because its number of axes is unknown; otherwise it is closed and
    lists every axis. The `...` is an entry but never counts as an axis. A
    sequence with more than one `...` describes no axes: the methods that
    read axes raise `ValueError`, and coordinate systems refuse it. For
    example, `[..., TimeAxis()]` says that the last axis is a time axis,
    and `[...]` says nothing.

    This class is the read-only base of [`AxisList`][], which is mutable,
    and [`AxisTuple`][], which is immutable. Slicing and the methods that
    build new sequences return the type of the original sequence. The type
    parameter is the item type, so `AxisSequence[Axis]` is closed.

    !!! note "Entries and axes"
        Length, iteration, equality, `axes[i]` and [`index`][] work on
        entries, including `...`. [`ndim`][] counts axes, and [`at`][],
        [`expand`][], [`restrict`][], [`embed`][] and [`compatible_with`][]
        work on positions in the space, where `...` stands for as many axes
        as needed. Non-negative positions count from the first axis and
        negative positions from the last. In a closed sequence, entries and
        axes coincide. In `[x, ..., t]`, however, `axes[2]` is `t` while
        `axes.at(2)` is one of the axes that `...` stands for.

    !!! note "Finding an axis"
        [`index`][] finds the first entry that matches a query, given as an
        axis or as a name, which stands for `Axis(name=...)`. An entry
        matches when it is an instance of the class of the query and has
        the same value for every field that the query sets. Thus
        `axes.index(TimeAxis())` finds the first time axis whatever its
        unit, and `axes.index(Axis())` finds the first axis. The `...`
        matches nothing, and an unknown `Axis()` matches no query that sets
        a field.

    !!! note "Names"
        `axes["t"]` returns the explicit axis named `"t"`, `"t" in axes`
        tests for one, and [`names`][] lists the names. A name that several
        axes share is refused. There is no mapping interface, because axes
        may be unnamed or share names.

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

        Integers and slices index entries, as for a list; [`at`][] reads the
        axis at a position in the space.

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
        query: tx.Union[Axis, str],
        start: tx.SupportsIndex = 0,
        stop: tx.SupportsIndex = sys.maxsize,
    ) -> int:
        """Return the index of the first entry that matches an axis or a name.

        The test is looser than equality: unset fields of the query are not
        compared, as described in the class documentation.

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

        Parameters
        ----------
        query
            An axis, or the name of an axis.
        start, stop
            Search window, as for `list.index`.

        Returns
        -------
        int
            Index of the entry, which is a position in the space only when the
            sequence is closed.

        Raises
        ------
        ValueError
            If no entry matches.
        TypeError
            If the query is neither an axis nor a string.
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
        """Name of each entry, or `None` for an unnamed axis.

        !!! example
            ```pycon
            >>> AxisList([Axis(name="x"), Axis(), ...]).names
            ('x', None, Ellipsis)
            ```
        """
        return tuple(... if axis is ... else _name(axis) for axis in self)

    @property
    def ndim(self) -> tx.Optional[int]:
        """Number of axes, or `None` when the sequence is open.

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
        """Whether the sequence holds `...`, so its number of axes is unknown.

        `AxisList([...])` is open, and an empty list is closed.
        """
        return self._split()[1] is not None

    def expand(self, ndim: int) -> tx.Self:
        """Return a closed sequence of `ndim` axes.

        In an open sequence, `...` is replaced by as many unknown axes as
        needed. A closed sequence is returned as a copy.

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
            If `ndim` is smaller than the number of explicit axes of an open
            sequence, or differs from the length of a closed sequence.
        TypeError
            If `ndim` is not an integer.
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
        """Return the axes at some positions or with some names, in that order.

        A position is an integer in the space, and a name refers to an
        explicit axis as in `axes[name]`. In a closed sequence of `n` axes,
        positions lie in `[-n, n)`. In an open sequence, every position is
        valid, and the positions that fall among the axes of `...` give
        unknown axes.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> AxisList([x, y, z]).restrict(["z", 0])
            [Axis(name='z'), Axis(name='x')]
            >>> AxisList([x, ...]).restrict([0, 1])
            [Axis(name='x'), Axis()]
            ```

        Returns
        -------
        AxisSequence
            A new closed sequence with one axis per reference.

        Raises
        ------
        IndexError
            If a position lies outside a closed sequence.
        KeyError
            If no explicit axis has a name.
        ValueError
            If a name is shared, or two references designate the same axis.
        TypeError
            If a reference is neither an integer nor a string, or `refs` is a
            string.
        """
        positions = []
        for ref in _as_list(refs, "refs"):
            if isinstance(ref, str):
                entry = self._entry_named(ref)
                positions.append(self._position_of_entry(entry))
            else:
                positions.append(self._position(ref))
        _check_unique(positions, "refs")
        return type(self)([self.at(p) for p in positions])

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> tx.Self:
        """Return the axes of a larger space in which these axes sit.

        This is the inverse of [`restrict`][]: axis `j` sits at `positions[j]`,
        and the other axes are unknown. An open sequence is first closed to
        `len(positions)` axes, as by [`expand`][]. Positions refer to a space
        that does not exist yet, so names are not accepted.

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
        positions
            Non-negative position of each axis.
        ndim
            Number of axes of the larger space. When it is omitted, the result
            is open and ends with `...`.

        Raises
        ------
        ValueError
            If a position is negative or repeated, if `ndim` does not exceed
            every position, or if the sequence cannot be closed to
            `len(positions)` axes.
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
            # Make the first sequence the open one.
            (p1, s1), (p2, s2) = (p2, s2), (p1, s1)
        assert s1 is not None
        if s2 is None:
            # The explicit axes of the open sequence must fit at the start and
            # at the end of the closed one.
            n = len(p2)
            if len(p1) + len(s1) > n:
                return False
            return _pairwise(p1, p2[: len(p1)]) and _pairwise(
                s1, p2[n - len(s1) :]
            )
        # With enough axes behind each `...`, only the axes that both state at
        # the start, or both at the end, meet.
        k = min(len(p1), len(p2))
        m = min(len(s1), len(s2))
        return _pairwise(p1[:k], p2[:k]) and _pairwise(
            s1[len(s1) - m :], s2[len(s2) - m :]
        )

    # Positions in the space count from the first axis when non-negative and
    # from the last axis when negative.

    def _split(self) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
        return _split(self)

    def _position(self, position: int) -> int:
        # In a closed sequence, a negative position is made non-negative. In an
        # open sequence, every position is valid and returned unchanged.
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

        Non-negative positions count from the first axis and negative positions
        from the last. In a closed sequence, `axes.at(i)` equals `axes[i]` and
        the position lies in `[-ndim, ndim)`. In an open sequence, every
        position is valid, and a position that falls among the axes of `...`
        gives a new unknown axis.

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
            If the position lies outside a closed sequence.
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

    # `tuple` comes first for storage, so the item and name reading of
    # `AxisSequence` is bound explicitly.
    __getitem__ = AxisSequence.__getitem__
    __contains__ = AxisSequence.__contains__
    index = AxisSequence.index
    _entry = tuple.__getitem__


class AxisList(AxisSequence[AXIS], list):
    """Mutable [`AxisSequence`][] for systems with any number of axes.

    A list or tuple given to such a system is converted item by item to the
    axis type of the system. The default, `[...]`, says nothing, and
    `axes=None` reads as that default. `AxisList[Axis]` is closed.

    !!! example
        ```pycon
        >>> axes = CoordinateSystem(axes=[Axis(name="x"), ...]).axes
        >>> type(axes).__name__, axes.is_open, axes["x"]
        ('AxisList', True, Axis(name='x'))
        >>> axes.append(TimeAxis(name="t"))
        >>> axes.at(-1)
        TimeAxis(name='t')
        ```
    """

    # `collections.abc.Sequence` sits between the bases in the MRO and would
    # supply generic mixins and an abstract `__len__`.
    __len__ = list.__len__
    __iter__ = list.__iter__
    __reversed__ = list.__reversed__
    count = list.count
    _entry = list.__getitem__


class _NoneReadsAsDefault:
    """Field converter that reads `None` as the default of the field.

    Other values go to the converter that the type hint would have had.
    `axes=None` means the default, which is `[...]` for an open system and
    its own axes for a fixed system. Only the field knows its default, so
    each field gets its own converter, which `bind_axes_default` points at
    the field. The converter is not registered for `AxisSequence`, since a
    registered converter would be asked to convert itself.
    """

    def __init__(self, hint: tx.Any) -> None:
        self.hint = hint
        self.field: tx.Any = None
        self._convert: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None

    def __call__(self, value: tx.Any) -> tx.Any:
        if value is None and self.field is not None:
            factory = self.field.factory
            value = factory() if callable(factory) else self.field.default
        if self._convert is None:
            self._convert = Converter.get(self.hint)
        return self._convert(value)


class Axes:
    """`Axes[hint]` types a field whose `None` reads as its default."""

    def __class_getitem__(cls, hint: tx.Any) -> tx.Any:
        return tx.Annotated[hint, ConvertTo(_NoneReadsAsDefault(hint))]


def bind_axes_default(cls: type) -> None:
    """Point the converter of the `axes` field of `cls` at the field."""
    for field in fields(cls):
        if field.name == "axes" and isinstance(
            field.converter, _NoneReadsAsDefault
        ):
            field.converter.field = field


def _split(
    entries: tx.Iterable[tx.Any],
) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
    # Explicit axes before and after `...`; the second is `None` for a closed
    # sequence.
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
