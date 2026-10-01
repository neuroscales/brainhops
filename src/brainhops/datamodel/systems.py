"""Coordinate systems, from unitless arrays to anatomical spaces."""

__all__ = [
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

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis]
_3SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis, SpatialAxis]


class CoordinateSystem(DataModelBase, eq=False):
    """A coordinate system defines the meaning of coordinates in a space.

    It describes each axis in the system (name, unit and/or other properties),
    and can be named.

    !!! note "Open systems"
        The list of axes may hold at most one `...` (`Ellipsis`), anywhere
        in the list, that stands for *zero or more axes about which nothing
        is known*. A system with `...` is *open*: its number of axes is
        unknown. A system without it is *closed*.

        * `[..., TimeAxis()]` says that the last axis is time, and nothing
          about the others.
        * `[Axis(name="x"), ...]` says that the first axis is `x`.
        * `axes=None` means the same as `axes=[...]`: nothing is known.
          It is stored as given, but every method, equality and
          [`compatible`][] treat the two spellings identically.

        The `...` entry is never counted as an axis. Classes with a fixed
        number of axes, such as [`CoordinateSystem3D`][], are always
        closed: they reject `...` and `axes=None`.

    !!! note "Equality"
        Two systems are equal when they are of the same class, have the
        same name, and have equal axes, where `axes=None` equals
        `axes=[...]`. A plain `CoordinateSystem` with no name and no known
        axis, which says nothing at all, also equals `None`, the missing
        endpoint of a transformation. [`compatible`][] is the looser
        question of whether two systems could describe the same space.
    """

    name: tx.Optional[str] = None
    axes: tx.Optional[tx.List[tx.Union[Axis, _Ellipsis]]] = None

    # The number of axes every instance of the class has, or `None` when
    # it is not fixed. Classes with a fixed number of axes are closed.
    _FIXED_NDIM = None

    # --- validation ---------------------------------------------------

    def __pre_init__(self, arguments: tx.Any) -> None:
        # A fixed-dimension class is refused an open list before its
        # conversion to a fixed-length tuple, which would otherwise fail
        # with a message that does not say why.
        ndim = type(self)._FIXED_NDIM
        if ndim is None:
            return
        axes = arguments.get("axes")
        if axes is None or (
            isinstance(axes, (list, tuple)) and any(a is ... for a in axes)
        ):
            raise ValueError(
                f"A {type(self).__name__} has exactly {ndim} axes, so its "
                f"axes cannot be left open. Give {ndim} axes, not None or "
                f"a list that holds `...`."
            )

    def __post_init__(self) -> None:
        axes = self.axes
        if axes is not None and sum(a is ... for a in axes) > 1:
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
        return _axes_of(self) == _axes_of(other)

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` when the system is open.

        A closed system has exactly `len(axes)` axes. An open system,
        whose axes hold `...` or are `None`, has an unknown number of
        axes, and its `ndim` is `None`.

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
        return _ndim_of(self)

    # --- axis references ----------------------------------------------

    def position(self, ref: tx.Union[int, str]) -> int:
        """Resolve an axis reference to a position in this system.

        A reference is a position (`int`) or the name of an axis (`str`).

        * **A name** resolves when exactly one explicit axis (any axis of
          the list but `...`) has that name. In a closed system, and in
          the part of an open system before `...`, the result is the
          non-negative position of that axis. In the part after `...`,
          the distance from the start is unknown, so the result is the
          negative position of that axis, counted from the end.
        * **A position** is checked against the system and returned.
          In a closed system of `n` axes, it must lie in `[-n, n)`, and a
          negative position is returned as its non-negative equivalent.
          In an open system, every position is valid, because `...`
          stands for any number of axes. It is returned unchanged:
          non-negative positions count from the start and negative ones
          from the end. Nothing is ever clamped.

        The result is always a position that [`axis`][] reads back.

        Parameters
        ----------
        ref : int or str
            The position or the name of the axis.

        Returns
        -------
        int
            The position of the axis. It is non-negative, except for an
            axis of an open system that is counted from the end.

        Raises
        ------
        ValueError
            If no explicit axis has the name, or if more than one has it.
            A name that no explicit axis carries is not resolved, even if
            it might be one of the axes that `...` stands for.
        IndexError
            If the position lies outside a closed system.
        TypeError
            If `ref` is neither an integer nor a string.

        !!! example
            ```pycon
            >>> x, y, t = Axis(name="x"), Axis(name="y"), TimeAxis(name="t")
            >>> CoordinateSystem(axes=[x, y, t]).position("t")
            2
            >>> CoordinateSystem(axes=[x, y, t]).position(-1)
            2
            >>> CoordinateSystem(axes=[x, ..., t]).position("t")
            -1
            >>> CoordinateSystem(axes=[x, ..., t]).position(5)
            5
            ```
        """
        return _position_in(self, ref)

    def axis(self, ref: tx.Union[int, str]) -> Axis:
        """The axis at a position, or with a name, in this system.

        The reference is resolved by [`position`][], so it raises in the
        same cases. A non-negative position reads the part of the list
        before `...`, and a negative position the part after it. In an
        open system, a position that falls in neither part may lie among
        the axes that `...` stands for, about which nothing is known, so
        it reads as the unknown `Axis()`.

        Parameters
        ----------
        ref : int or str
            The position or the name of the axis.

        Returns
        -------
        Axis
            The axis, or a new unknown `Axis()` for a position of an open
            system that no explicit axis occupies.

        Raises
        ------
        ValueError, IndexError, TypeError
            As [`position`][] does.

        !!! example
            ```pycon
            >>> x, t = Axis(name="x"), TimeAxis(name="t")
            >>> system = CoordinateSystem(axes=[x, ..., t])
            >>> system.axis(0) is x, system.axis(-1) is t
            (True, True)
            >>> system.axis(1)
            Axis()
            ```
        """
        return _axis_in(self, ref)

    # --- operations ---------------------------------------------------

    def expand(self, ndim: int) -> tx.Self:
        """The closed system of `ndim` axes that this system describes.

        In an open system, `...` is replaced with as many unknown
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
            A closed system of `ndim` axes. A closed system is returned
            as itself.

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
        ndim = _as_int(ndim, "ndim")
        prefix, suffix = _split(self)
        if suffix is None:
            if ndim != len(prefix):
                raise ValueError(
                    f"Cannot expand a closed system of {len(prefix)} axes "
                    f"to {ndim} axes."
                )
            return self
        explicit = len(prefix) + len(suffix)
        if ndim < explicit:
            raise ValueError(
                f"Cannot expand an open system with {explicit} explicit "
                f"axes to {ndim} axes."
            )
        fill = [Axis() for _ in range(ndim - explicit)]
        return replace(self, axes=prefix + fill + suffix)

    def take(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """The system of the axes at some positions of this system.

        Each reference is resolved by [`axis`][], so a position of an
        open system that falls among the axes that `...` stands for gives
        an unknown `Axis()`. The axes are listed in the order of `refs`.
        The result is a closed, unnamed [`CoordinateSystem`][], because
        it describes a different space.

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
        ValueError
            If two references resolve to the same position, or as
            [`position`][] does.
        IndexError, TypeError
            As [`position`][] does, or if `refs` is a string rather than
            a list of references.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> CoordinateSystem(axes=[x, y, z]).take(["z", 0])
            CoordinateSystem(axes=[Axis(name='z'), Axis(name='x')])
            >>> CoordinateSystem(axes=[x, ...]).take([0, 1])
            CoordinateSystem(axes=[Axis(name='x'), Axis()])
            ```
        """
        refs = _as_list(refs, "refs")
        positions = [self.position(ref) for ref in refs]
        _check_unique(positions, "refs")
        return CoordinateSystem(axes=[self.axis(p) for p in positions])

    def place(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """The system of a larger space in which this system's axes sit.

        This is the inverse of [`take`][]: axis `j` of this system is
        placed at `positions[j]` of the result, and every other position
        holds an unknown `Axis()`. An open system is first closed to
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
            the last placed axis.

        Returns
        -------
        CoordinateSystem
            An unnamed [`CoordinateSystem`][], closed when `ndim` is given
            and open otherwise.

        Raises
        ------
        ValueError
            If a position is negative or repeated, if `ndim` does not
            exceed every position, or if this system cannot be closed to
            `len(positions)` axes.
        TypeError
            If a position or `ndim` is not an integer.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> CoordinateSystem(axes=[x]).place([1])
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Ellipsis])
            >>> CoordinateSystem(axes=[x]).place([1], ndim=3)
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Axis()])
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
        prefix, suffix = _split(self)
        count = len(prefix) + len(suffix or [])
        if (suffix is None and count != len(positions)) or (
            count > len(positions)
        ):
            raise ValueError(
                f"Cannot place a system of {count}"
                f"{'' if suffix is None else ' explicit'} axes at "
                f"{len(positions)} positions."
            )
        axes = _axes_of(self.expand(len(positions)))
        size = max(positions) + 1 if positions else 0
        if ndim is not None:
            ndim = _as_int(ndim, "ndim")
            if ndim < size:
                raise ValueError(
                    f"Cannot place an axis at position {size - 1} of a "
                    f"space of {ndim} axes."
                )
            size = ndim
        full: tx.List[tx.Any] = [Axis() for _ in range(size)]
        for axis, p in zip(axes, positions):
            full[p] = axis
        if ndim is None:
            full.append(...)
        return CoordinateSystem(axes=full)

    def compatible(self, other: tx.Optional["CoordinateSystem"]) -> bool:
        """Whether `self` and `other` could describe the same space.

        Two systems are compatible when some choice of the axes that each
        `...` stands for makes them match axis by axis, each pair being
        [`Axis.compatible`][brainhops.datamodel.axes.Axis.compatible].
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

        !!! example
            ```pycon
            >>> x, t = SpatialAxis(name="x"), TimeAxis()
            >>> CoordinateSystem(axes=[x, ...]).compatible(
            ...     CoordinateSystem(axes=[x, Axis(), t])
            ... )
            True
            >>> CoordinateSystem(axes=[..., t]).compatible(
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
        return _compatible(self, other)


def _axes_of(system: tx.Optional[CoordinateSystem]) -> tx.List[tx.Any]:
    # The one accessor of a system's axes, as a new list. A missing system
    # and a system whose axes are `None` are read as `[...]`, so that no
    # caller has to tell the spellings of "nothing is known" apart.
    axes = None if system is None else system.axes
    return [...] if axes is None else list(axes)


def _expand_of(
    system: tx.Optional[CoordinateSystem], ndim: int
) -> CoordinateSystem:
    # `CoordinateSystem.expand`, for a system that may be missing, which is
    # read as `CoordinateSystem()`.
    return (CoordinateSystem() if system is None else system).expand(ndim)


def _split(
    system: tx.Optional[CoordinateSystem],
) -> tx.Tuple[tx.List[Axis], tx.Optional[tx.List[Axis]]]:
    # The explicit axes before and after `...`. The second list is `None`
    # for a closed system, whose axes are then all in the first.
    axes = _axes_of(system)
    for i, axis in enumerate(axes):
        if axis is ...:
            return axes[:i], axes[i + 1 :]
    return axes, None


def _ndim_of(system: tx.Optional[CoordinateSystem]) -> tx.Optional[int]:
    # The number of axes of a closed system, `None` for an open or a
    # missing one.
    prefix, suffix = _split(system)
    return len(prefix) if suffix is None else None


def _is_unknown(system: tx.Optional[CoordinateSystem]) -> bool:
    # Whether `system` says nothing at all: it is missing, or it is a plain
    # `CoordinateSystem` with no name whose axes are `None` or `[...]`.
    # Such a system equals `None`.
    if system is None:
        return True
    return (
        type(system) is CoordinateSystem
        and system.name is None
        and _axes_of(system) == [...]
    )


def _position_in(
    system: tx.Optional[CoordinateSystem], ref: tx.Union[int, str]
) -> int:
    # `CoordinateSystem.position`, for a system that may be missing.
    prefix, suffix = _split(system)
    if isinstance(ref, str):
        found = [i for i, a in enumerate(prefix) if _name(a) == ref]
        if suffix is not None:
            n = len(suffix)
            found += [i - n for i, a in enumerate(suffix) if _name(a) == ref]
        if len(found) == 1:
            return found[0]
        if found:
            raise ValueError(
                f"Cannot resolve the axis name {ref!r}: {len(found)} axes "
                f"of the system carry it."
            )
        if not prefix and not suffix:
            raise ValueError(
                f"Cannot resolve the axis name {ref!r} without a system "
                f"that names it: nothing is known about the axes."
            )
        raise ValueError(
            f"Cannot resolve the axis name {ref!r}: no axis of the system "
            f"carries it."
        )
    if isinstance(ref, bool) or not isinstance(ref, Integral):
        raise TypeError(
            f"An axis is referred to by its position (int) or its name "
            f"(str), not by a {type(ref).__name__}."
        )
    ref = int(ref)
    if suffix is not None:
        return ref
    n = len(prefix)
    if not -n <= ref < n:
        raise IndexError(
            f"Axis position {ref} is out of range for a system of {n} axes."
        )
    return ref + n if ref < 0 else ref


def _axis_in(
    system: tx.Optional[CoordinateSystem], ref: tx.Union[int, str]
) -> Axis:
    # `CoordinateSystem.axis`, for a system that may be missing.
    position = _position_in(system, ref)
    prefix, suffix = _split(system)
    if suffix is None:
        return prefix[position]
    if 0 <= position < len(prefix):
        return prefix[position]
    if -len(suffix) <= position < 0:
        return suffix[position]
    return Axis()


def _compatible(
    first: tx.Optional[CoordinateSystem],
    second: tx.Optional[CoordinateSystem],
) -> bool:
    # `CoordinateSystem.compatible`, for systems that may be missing.
    p1, s1 = _split(first)
    p2, s2 = _split(second)
    if s1 is None and s2 is None:
        return len(p1) == len(p2) and _pairwise(p1, p2)
    if s1 is None:
        # Let the first system be the open one.
        (p1, s1), (p2, s2) = (p2, s2), (p1, s1)
    assert s1 is not None
    if s2 is None:
        # Open against closed: the explicit axes of the open system must
        # fit, and match the axes at the start and at the end.
        n = len(p2)
        if len(p1) + len(s1) > n:
            return False
        return _pairwise(p1, p2[: len(p1)]) and _pairwise(
            s1, p2[n - len(s1) :]
        )
    # Open against open: with enough axes in each `...`, only the axes
    # that both systems state at the start, or both at the end, meet.
    k = min(len(p1), len(p2))
    m = min(len(s1), len(s2))
    return _pairwise(p1[:k], p2[:k]) and _pairwise(
        s1[len(s1) - m :], s2[len(s2) - m :]
    )


def _pairwise(first: tx.List[Axis], second: tx.List[Axis]) -> bool:
    return all(a.compatible(b) for a, b in zip(first, second))


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


class CoordinateSystem2D(CoordinateSystem):
    """A coordinate systems with exactly two dimensions."""

    axes: _2Axes = (Axis(), Axis())
    _FIXED_NDIM = 2


class CoordinateSystem3D(CoordinateSystem):
    """A coordinate system with exactly three dimensions."""

    axes: _3Axes = (Axis(), Axis(), Axis())
    _FIXED_NDIM = 3


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

    axes: tx.Optional[tx.List[tx.Union[SpatialAxis, _Ellipsis]]] = None


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
