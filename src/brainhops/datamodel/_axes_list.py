# stdlib
import abc
import sys
from numbers import Integral
from types import EllipsisType as _Ellipsis

# datamodel
import typing_extensions as tx
from bagof.converters import Converter
from bagof.magic import ConvertTo, fields

# datamodel
from .axes import Axis

# typing

AXIS = tx.TypeVar("AXIS")
# The type of the items of an `AxisSequence`, and of an `AxisList`.

AXES = tx.TypeVarTuple("AXES")
# The type of each item of an `AxisTuple`, in order.


class AxisSequence(tx.Sequence[AXIS]):
    """The axes of a coordinate system, which may leave some unknown.

    An `AxisSequence` is a sequence of [`Axis`][] that may hold one `...`
    ([`Ellipsis`][]), anywhere in it. `...` stands for *zero or more axes
    about which nothing is known*.

    * A sequence that holds `...` is *open*: its number of axes is
      unknown.
    * A sequence without it is *closed*: it lists every axis.
    * `...` is an entry of the sequence, but never counts as an axis.
    * A sequence that holds `...` more than once describes no axes:
      every method that reads the axes raises a `ValueError` on it. A
      coordinate system refuses such a sequence when it is built.

    `[..., TimeAxis()]` says that the last axis is time, and nothing
    about the others. `[Axis(name="x"), ...]` says that the first axis
    is `x`. `[...]` says nothing at all.

    This is the read-only base of two containers, which share all of its
    API, and whose methods that build a new sequence (a slice,
    [`expand`][], [`restrict`][], [`embed`][]) build one of their own
    type:

    * [`AxisList`][], a `list`, is mutable. A coordinate system whose
      number of axes is not fixed by its class stores its axes as one.
    * [`AxisTuple`][], a `tuple`, is immutable. A coordinate system
      with a fixed number of axes, such as an `RASCoordinateSystem`,
      stores its axes as one.

    !!! note "Entries and axes"
        `len()`, iteration, equality, `repr`, `[i]` (an integer or a
        slice) and [`index`][] are about the *entries* of the sequence,
        `...` included, as in the `list` or `tuple` it is.

        [`ndim`][] counts the *axes* the sequence describes, and
        [`at`][], [`expand`][], [`restrict`][], [`embed`][] and
        [`compatible_with`][] place them in the space, where `...` stands
        for as many axes as needed. A position in the space is counted
        from the first axis when it is non-negative, and from the last
        one when it is negative.

        In a closed sequence, the entries are the axes, in order. In an
        open one, they are not: in `[x, ..., t]`, entry 2 (`axes[2]`) is
        `t`, which is the last axis, and the axis at position 2
        (`axes.at(2)`) is one of the axes that `...` stands for, which
        has no entry. So `for i in range(len(axes)): axes[i]` walks the
        entries, not the axes.

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
        sequence, and changing an axis by its name would be a trap.

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

    The type parameter is the type of the items:
    `AxisSequence[Union[Axis, EllipsisType]]` may be open, and
    `AxisSequence[Axis]` is closed.
    """

    __slots__ = ()

    @abc.abstractmethod
    def _entry(self, key: tx.Any) -> tx.Any:
        # The builtin storage, read: `list.__getitem__` in an `AxisList`,
        # `tuple.__getitem__` in an `AxisTuple`.
        ...

    @tx.overload
    def __getitem__(self, key: tx.SupportsIndex) -> AXIS: ...

    @tx.overload
    def __getitem__(self, key: slice) -> tx.Self: ...

    @tx.overload
    def __getitem__(self, key: str) -> Axis: ...

    def __getitem__(self, key: tx.Any) -> tx.Any:
        """An entry (`int`), some entries (`slice`), or the axis with a
        name (`str`).

        An integer or a slice indexes the *entries* of the sequence, as
        in any `list` or `tuple`, and a slice gives a sequence of the same
        type. A name gives the one explicit axis that has it. The axis at
        a *position* in the space is [`at`][] that position.

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
            If no explicit axis has the name.
        ValueError
            If more than one explicit axis has the name.
        IndexError, TypeError
            As `list` indexing does.
        """
        if isinstance(key, str):
            return self._entry(self._entry_named(key))
        if isinstance(key, slice):
            return type(self)(self._entry(key))
        return self._entry(key)

    def __contains__(self, item: object) -> bool:
        """Whether an explicit axis has a name (`str`), or whether an
        entry equals `item` (anything else, as in any `list`)."""
        if isinstance(item, str):
            return bool(self._entries_named(item))
        return any(entry is item or entry == item for entry in self)

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

        !!! example
            ```pycon
            >>> AxisList([Axis(name="x"), ...]).expand(3)
            [Axis(name='x'), Axis(), Axis()]
            >>> AxisList([...]).expand(2)
            [Axis(), Axis()]
            ```

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

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> AxisList([x, y, z]).restrict(["z", 0])
            [Axis(name='z'), Axis(name='x')]
            >>> AxisList([x, ...]).restrict([0, 1])
            [Axis(name='x'), Axis()]
            ```

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
        """The axes of a larger space in which this list's axes sit.

        This is the inverse of [`restrict`][]: axis `j` of this list sits
        at `positions[j]` of the result, and every other position holds
        an unknown `Axis()`. An open list is first closed to
        `len(positions)` axes, as by [`expand`][].

        The positions are absolute positions in the larger space, which
        does not exist yet, so they cannot be names.

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

        !!! example
            ```pycon
            >>> x, t = SpaceAxis(name="x"), TimeAxis()
            >>> AxisList([x, ...]).compatible_with([x, Axis(), t])
            True
            >>> AxisList([..., t]).compatible_with([x])
            False
            ```

        Parameters
        ----------
        other : AxisSequence, or list or tuple of Axis
            The axes to compare with. A plain list or tuple is read as
            an axis sequence.

        Returns
        -------
        bool
            Whether the two lists could describe the same axes.

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
        return _split(self)

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

    def at(self, position: int) -> Axis:
        """The axis at a position in the space.

        Where `axes[i]` reads *entry* `i` of the sequence, `axes.at(i)`
        reads the axis at *position* `i` of the space it describes:
        counted from the first axis when `i` is non-negative, and from
        the last one when it is negative.

        * In a closed sequence, the two are the same, and a position lies
          in `[-ndim, ndim)`.
        * In an open sequence, every position is valid, because `...`
          stands for any number of axes. A non-negative position reads
          the explicit axes before `...`, and a negative one the explicit
          axes after it. Any other position falls among the axes that
          `...` stands for, and gives a new, unknown `Axis()`.

        !!! example
            ```pycon
            >>> x, t = Axis(name="x"), TimeAxis(name="t")
            >>> axes = AxisList([x, ..., t])
            >>> axes.at(0) is x, axes.at(-1) is t, axes.at(1)
            (True, True, Axis())
            >>> axes[2] is t, axes.at(2)
            (True, Axis())
            ```

        Parameters
        ----------
        position : int
            The position of the axis in the space.

        Returns
        -------
        Axis
            The explicit axis at that position, or a new `Axis()`.

        Raises
        ------
        IndexError
            If the sequence is closed, and the position lies outside it.
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


class AxisTuple(tuple, AxisSequence, tx.Generic[tx.Unpack[AXES]]):
    """An immutable [`AxisSequence`][brainhops.datamodel.systems.AxisSequence].

    It is a `tuple`, with all the API of an `AxisSequence`: indexing by
    name, [`at`][], [`expand`][] and so on. A slice, and every method that
    builds a new sequence, gives an `AxisTuple`.

    A coordinate system with a fixed number of axes, such as an
    `RASCoordinateSystem`, stores its axes as one, so they are closed.
    The type parameters are the type of each item, in order, and fix the
    number of items: `AxisTuple[SpaceAxis, SpaceAxis]` is two spatial
    axes. A field of that type converts what it is given item by item,
    each to the type of its position, and refuses a wrong number of
    items, `None`, or `...` (which is not an axis). A bare `AxisTuple`
    holds any number of items, `...` included.

    !!! example
        ```pycon
        >>> axes = RASCoordinateSystem().axes
        >>> type(axes).__name__, axes.ndim, axes.names[0]
        ('AxisTuple', 3, 'left-to-right')
        >>> axes["left-to-right"] is axes[0] is axes.at(-3)
        True
        ```
    """

    # The `tuple` comes first, for its storage, but its own reading of an
    # item or of a name is not the one this class means.
    __getitem__ = AxisSequence.__getitem__
    __contains__ = AxisSequence.__contains__
    index = AxisSequence.index
    _entry = tuple.__getitem__


class AxisList(AxisSequence[AXIS], list):
    """A mutable [`AxisSequence`][brainhops.datamodel.systems.AxisSequence].

    It is a `list`, with all the API of an `AxisSequence`: indexing by
    name, [`at`][], [`expand`][] and so on. A slice, and every method that
    builds a new sequence, gives an `AxisList`.

    A coordinate system whose number of axes is not fixed by its class
    stores its axes as one: a list or a tuple given to the system is
    converted to one, item by item, to the type of axis the class
    declares. Its default, `[...]`, says nothing about the axes, and
    `axes=None` reads as that default.

    The type parameter is the type of the items:
    `AxisList[Union[Axis, EllipsisType]]` may be open, and
    `AxisList[Axis]` is closed.

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

    # `AxisSequence` comes first, for its reading of an item or of a name.
    # The rest is the `list`'s: `collections.abc.Sequence`, between the
    # two in the method resolution order, would otherwise answer with its
    # generic mixins (and its abstract `__len__`).
    __len__ = list.__len__
    __iter__ = list.__iter__
    __reversed__ = list.__reversed__
    count = list.count
    _entry = list.__getitem__


# ---- converter -------------------------------------------------------


class _NoneReadsAsDefault:
    """A field's converter, with `None` read as the field's default.

    `None` is not a sequence of axes, and what a caller who writes
    `axes=None` means is "say nothing about them", which is what the
    default says: `[...]` for an open system, its own axes for a system
    whose number of axes is fixed. The field knows that default and the
    converter does not, so each `axes` field gets a converter of its own,
    which `bind_axes_default` points at the field. Everything else is
    handed to the converter the hint would have had.

    It is *not* registered for `AxisSequence`: it converts the one field
    it is attached to, and a converter registered for the sequence class
    would be asked to convert itself.
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
    """
    `Axes[hint]` types an `axes` field as `hint`, with `None` reading as
    the field's default (see `_NoneReadsAsDefault`).
    """

    def __class_getitem__(cls, hint: tx.Any) -> tx.Any:
        return tx.Annotated[hint, ConvertTo(_NoneReadsAsDefault(hint))]


def bind_axes_default(cls: type) -> None:
    """Point the converter of the `axes` field of `cls` at that field."""
    for field in fields(cls):
        if field.name == "axes" and isinstance(
            field.converter, _NoneReadsAsDefault
        ):
            field.converter.field = field


# ---- private helpers -------------------------------------------------


def _split(
    entries: tx.Iterable[tx.Any],
) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
    # The explicit axes before and after `...`. The second list is `None`
    # for a closed sequence, whose axes are then all in the first.
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
