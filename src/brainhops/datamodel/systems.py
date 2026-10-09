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
| ... and in millimetres                        | `RASmm`                    |

and likewise for LPS and RSA. A class that inherits from two dispatch
targets -- `SpatialCoordinateSystem3D` from `CoordinateSystem3D` and
`SpatialCoordinateSystem` -- is selected on what both stand for, with no
constraint of its own.

The memory order of an array is not written on its axes, so the C- and
F-ordered variants are selected on `order` (`"C"`, `"F"`, or `None` when
it is not specified), together with the axes:

| `order=`, and the axes are...           | ...so the system is            |
| --------------------------------------- | ------------------------------ |
| `"C"` / `"F"`, anything                 | `C`/`FArrayCoordinateSystem`   |
| ... two or three                        | `C`/`FArrayCoordinateSystem2D/3D` |
| ... two, spatial                        | `C`/`FPixelCoordinateSystem`   |
| ... three, spatial                      | `C`/`FVoxelCoordinateSystem`   |
| `"F"`, oriented R, A, S                 | `FRASCoordinateSystem`         |
| `"C"`, oriented S, A, R (a C-ordered    | `CRASCoordinateSystem`         |
| grid lists its axes z, y, x)            |                                |

`order` is a field of every system, so every class can be called with it
and pass it on: `CoordinateSystem(axes=<RAS axes>, order="F")` builds an
`FRASCoordinateSystem`, and so do `ArrayCoordinateSystem`,
`FArrayCoordinateSystem` and `FVoxelCoordinateSystem` called the same
way. Only an array system has an order, so a system the axes and the
order do not make one of (`RASmm(order="F")`) refuses it.

Every row of the table says something about *all* the axes, so only a
closed system is dispatched. An open system -- one whose axes hold `...`
([`AxisList`][]) -- does not know all its axes, so it is built as the
class it was called as: `CoordinateSystem(axes=[x, ...])` is
not two-dimensional, and `CoordinateSystem(axes=[R(), A(), S(), ...])` is
not an `RASCoordinateSystem`. [`CoordinateSystem.expand`][] closes it,
and the closed system is dispatched like any other.
"""

__all__ = [
    "AxisSequence",
    "AxisTuple",
    "AxisList",
    "CoordinateSystem",
    "CoordinateSystem2D",
    "CoordinateSystem3D",
    "PhysicalCoordinateSystem",
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
    "RASmm",
    "LPSmm",
    "RSAmm",
]
# externals
import typing_extensions as tx
from bagof.magic import Factory, replace

# internals
from . import axes as _axes
from ._axes_list import (
    Axes,
    AxisList,
    AxisSequence,
    AxisTuple,
    bind_axes_default,
)
from .axes import Axis, SpaceAxis
from .base import DataModelBase
from .units import Unit, is_indexunit, is_physicalunit

if tx.TYPE_CHECKING:
    from types import EllipsisType as _Ellipsis

else:
    _Ellipsis: tx.TypeAlias = type(Ellipsis)
    # The type of `...`. Python 3.10 names it `types.EllipsisType`.

AXIS = tx.TypeVar("AXIS")
# The type of the items of an `AxisSequence`, and of an `AxisList`.

_INDEX = Unit("index")

_2Axes: tx.TypeAlias = AxisTuple[Axis, Axis]
_3Axes: tx.TypeAlias = AxisTuple[Axis, Axis, Axis]
_2SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis]
_3SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis, SpaceAxis]
_EllipsisOr: tx.TypeAlias = tx.Union[AXIS, _Ellipsis]


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


def _isnd(n: int) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    def isnd(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        closed = _closed(axes)
        return closed is not None and len(closed) == n

    isnd.__name__ = isnd.__qualname__ = f"_is{n}d"
    return isnd


_is2d = _isnd(2)
_is3d = _isnd(3)


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
_is_array = _all(lambda axis: is_indexunit(axis.unit), "_is_array")
_MM = Unit("mm")
_is_mm = _all(lambda axis: axis.unit == _MM, "_is_mm")


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
        * `[...]`, the default, says nothing at all. `axes=None` reads
          as not giving the axes, so it is `[...]` too, and is stored as
          `[...]`: `CoordinateSystem(axes=None) == CoordinateSystem()`.

        A list or a tuple given as `axes` is stored as an [`AxisList`][].
        An axis is read by its name as `system.axes["x"]`, at a position
        as `system.axes.at(i)`, and found by [`index`][AxisSequence.index].

        Classes with a fixed number of axes, such as
        [`CoordinateSystem3D`][], are always closed. They store their
        axes as an [`AxisTuple`][], whose type fixes the number of axes
        and the class of each one, so they reject `...`. It has the API
        of an [`AxisList`][], but is immutable. `axes=None` reads as not
        giving the axes there too, so it builds the class's default axes:
        `CoordinateSystem3D(axes=None) == CoordinateSystem3D()`.

        Calling a class builds the most specific system its axes
        describe (see the module), and only a closed system is
        dispatched: `CoordinateSystem(axes=[x, y])` is a
        [`CoordinateSystem2D`][], whose axes are an `AxisTuple`, but
        `CoordinateSystem(axes=[x, ...])` -- whose `...` may stand for
        no axis, or for many -- stays a `CoordinateSystem`. Closing an
        open system with [`expand`][] dispatches it again.

    !!! note "Equality"
        Equality is field by field: two systems are equal when they are
        of the same class, have the same name, and have equal axes. A
        system is never equal to `None`, not even a plain
        `CoordinateSystem()` that says nothing at all: as the endpoint of a
        transformation, `None` is no system (the transformation defers to
        its context for it), and a `CoordinateSystem()` is one, kept as
        given.
        [`compatible_with`][] is the looser question of whether two
        systems could describe the same space.
    """

    name: tx.Optional[str] = None
    """The name of the coordinate system."""

    axes: Axes[AxisList[_EllipsisOr[Axis]]] = [...]
    """The axes of the coordinate system, in order. `[...]`, the default,
    says nothing about them; `axes=None` reads as the default."""

    order: tx.Optional[tx.Literal["C", "F"]] = None
    """The memory order of the array the coordinates index: `"C"` (the
    last axis changes fastest), `"F"` (the first axis does), or `None`
    when it is not specified. Only an [`ArrayCoordinateSystem`][] indexes
    an array, so any other system refuses an order; the field is
    declared here so that every class can be called with it, and pass it
    on to the C- or F-ordered class it selects."""

    # --- construction -------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Every subclass declares its own `axes`, with its own default,
        # and its own converter to read `None` as that default: the
        # converter is pointed at the field here, where the field exists.
        bind_axes_default(cls)

    # --- validation ---------------------------------------------------

    def __post_init__(self) -> None:
        if self.axes is None:
            self.axes = [...]
        if sum(a is ... for a in self.axes) > 1:
            raise ValueError(
                "The axes of a coordinate system hold at most one `...`, "
                "which stands for all the axes about which nothing is known."
            )
        if self.order is not None and not isinstance(
            self, ArrayCoordinateSystem
        ):
            raise ValueError(
                f"A {type(self).__name__} indexes no array, so it has no "
                f"memory order: order={self.order!r} is for a system whose "
                f"coordinates index an array, an ArrayCoordinateSystem, "
                f"which the order selects whenever the axes allow one."
            )

    # --- properties ---------------------------------------------------

    @property
    def ndim(self) -> tx.Optional[int]:
        """The number of axes, or `None` when the system is open.

        A closed system has exactly `len(axes)` axes. An open system,
        whose axes hold `...`, has an unknown number of axes, and its
        `ndim` is `None`. This is [`AxisSequence.ndim`][] of its axes.

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
        return self.axes.ndim

    # --- operations ---------------------------------------------------

    def expand(self, ndim: int) -> tx.Self:
        """The closed system of `ndim` axes that this system describes.

        The axes are expanded by [`AxisSequence.expand`][]:
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

        !!! example
            ```pycon
            >>> CoordinateSystem(axes=[Axis(name="x"), ...]).expand(4)
            CoordinateSystem(axes=[Axis(name='x'), Axis(), Axis(), Axis()])
            >>> CoordinateSystem().expand(2)
            CoordinateSystem2D(axes=(Axis(), Axis()))
            ```

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
        """
        expanded = self.axes.expand(ndim)
        if not self.axes.is_open:
            return self
        # The class converts the closed axes item by item, and dispatches
        # on them once converted.
        return replace(self, axes=expanded)

    def restrict(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """The system of the axes at some positions of this system.

        The axes are restricted by [`AxisSequence.restrict`][]:
        a reference is a position in the space or a name, and a position
        of an open system that falls among the axes that `...` stands
        for gives an unknown `Axis()`. The axes are listed in the order
        of `refs`. The result describes a different space, so the class
        and the name of this system are not carried over: it is the
        closed system that `CoordinateSystem(axes=...)` builds from the
        restricted axes, which is a [`CoordinateSystem2D`][], an
        [`RASCoordinateSystem`][], ... when the axes select one.

        !!! example
            ```pycon
            >>> x, y, z = Axis(name="x"), Axis(name="y"), Axis(name="z")
            >>> CoordinateSystem(axes=[x, y, z]).restrict(["z", 0])
            CoordinateSystem2D(axes=(Axis(name='z'), Axis(name='x')))
            >>> CoordinateSystem(axes=[x, ...]).restrict([0])
            CoordinateSystem(axes=[Axis(name='x')])
            ```

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
            As [`AxisSequence.restrict`][] does.
        """
        return CoordinateSystem(axes=self.axes.restrict(refs))

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """The system of a larger space in which this system's axes sit.

        This is the inverse of [`restrict`][]. The axes are embedded by
        [`AxisSequence.embed`][]:
        axis `j` of this system sits at `positions[j]` of the result,
        and every other position holds an unknown `Axis()`. The result
        describes a different space, so the class and the name of this
        system are not carried over: it is the system that
        `CoordinateSystem(axes=...)` builds from the embedded axes -- a
        plain, open `CoordinateSystem` when `ndim` is not given, and the
        closed system the axes select when it is.

        !!! example
            ```pycon
            >>> x = Axis(name="x")
            >>> CoordinateSystem(axes=[x]).embed([1])
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Ellipsis])
            >>> CoordinateSystem(axes=[x]).embed([1], ndim=4)
            CoordinateSystem(axes=[Axis(), Axis(name='x'), Axis(), Axis()])
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
        CoordinateSystem
            A system that is closed when `ndim` is given, and open
            otherwise.

        Raises
        ------
        ValueError, TypeError
            As [`AxisSequence.embed`][] does.
        """
        return CoordinateSystem(axes=self.axes.embed(positions, ndim=ndim))

    def compatible_with(self, other: tx.Optional["CoordinateSystem"]) -> bool:
        """Whether `self` and `other` could describe the same space.

        Two systems are compatible when their axes are
        [`AxisSequence.compatible_with`][] each other: some choice of
        the axes that each `...` stands for makes them match axis by
        axis, each pair being [`Axis.compatible_with`][].
        Only the axes are compared, not the names of the systems. `None`
        is read as a system about which nothing is known, which is
        compatible with every system.

        For two closed systems, this asks for the same number of axes,
        pairwise compatible. Unlike `==`, an unknown `Axis()` matches
        any axis. The relation is symmetric, but not transitive.

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
        """
        if other is not None and not isinstance(other, CoordinateSystem):
            raise TypeError(
                f"A coordinate system is compatible only with another "
                f"CoordinateSystem or None, not with {type(other).__name__}."
            )
        other_axes = getattr(other, "axes", AxisList([...]))
        return self.axes.compatible_with(other_axes)


bind_axes_default(CoordinateSystem)


class CoordinateSystem2D(CoordinateSystem, on={"axes": _is2d}):
    """A coordinate systems with exactly two dimensions."""

    axes: Axes[_2Axes] = Factory()


class CoordinateSystem3D(CoordinateSystem, on={"axes": _is3d}):
    """A coordinate system with exactly three dimensions."""

    axes: Axes[_3Axes] = Factory()


# ----------------------------------------------------------------------
#   PHYSICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class PhysicalCoordinateSystem(CoordinateSystem):
    """A coordinate system whose coordinates measure physical quantities.

    Every axis it states is measured in a physical unit, or in a unit not
    yet specified (`None`): a millimetre or a second, never
    [`IndexUnit`][], which says the coordinates count the samples of an
    array. So reversing one of its axes is a sign flip, never the origin
    shift a sampled axis needs, and a conversion factor to another
    physical system of the same kind exists as soon as the units are all
    given.

    The unit of an axis is of the kind its axis measures: a spatial axis
    takes a unit of space and a time axis a unit of time. The type of the
    axis already enforces that -- `SpaceAxis(unit="s")` is refused -- so
    this class only refuses the index units.

    It may be open, and its units may be unspecified, since neither says
    anything non-physical: `...` stands for axes about which nothing is
    known, and `None` for a unit about which nothing is. A system with no
    axis at all, `[]`, has nothing to refuse either. No array, pixel or
    voxel system is a physical one: their axes count samples.

    This is a base to inherit deliberately rather than a dispatch target:
    a physical spatial system is selected as a spatial one. The concrete
    systems that are physical by construction -- [`RASmm`][],
    [`LPSmm`][], [`RSAmm`][] -- compose it in, and are stricter: they are
    in millimetres, on every axis. Their own constraint is what dispatch
    selects them on, so axes in RAS order in centimetres, or with no
    unit, build an `RASCoordinateSystem`, and built by name, `RASmm`
    refuses them.

    !!! example
        ```pycon
        >>> PhysicalCoordinateSystem(axes=[SpaceAxis(unit="mm"), ...]).ndim
        >>> PhysicalCoordinateSystem(axes=[R(), A(unit="cm")]).ndim
        2
        >>> mm = [R(unit="mm"), A(unit="mm"), S(unit="mm")]
        >>> type(PhysicalCoordinateSystem(axes=mm)).__name__
        'RASmm'
        >>> type(CoordinateSystem(axes=[R(), A(), S()])).__name__
        'RASCoordinateSystem'
        ```
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        name = type(self).__name__
        for axis in self.axes:
            if axis is ...:
                continue
            unit = getattr(axis, "unit", None)
            if unit is None or is_physicalunit(unit):
                continue
            what = (
                f"counts samples (its unit is {unit!r}), which says it "
                "indexes an array"
                if is_indexunit(unit)
                else f"carries {unit!r}, which measures nothing"
            )
            raise ValueError(
                f"{name} is a physical coordinate system, so none of its "
                f"axes counts samples: each is measured in a physical unit, "
                f"or in one not yet given (`None`). The axis "
                f"{axis.name or axis.type!r} {what}."
            )


# ----------------------------------------------------------------------
#   ARRAY COORDINATE SYSTEMS
# ----------------------------------------------------------------------

# --- factories --------------------------------------------------------


def _index_axis(i: int) -> Axis:
    return Axis(f"dim{i}", unit=_INDEX)


class _AxesFactory(Factory):
    _AxisFactory = tx.Callable[..., AXIS]

    def __init__(
        self, ndim: int, axis_factory: tx.Optional[_AxisFactory] = None
    ) -> None:
        self.ndim = ndim
        self.axis_factory = axis_factory or (lambda *a: Axis())
        super().__init__(self._factory)

    def _factory(self) -> tx.Tuple[AXIS, ...]:
        return tuple(map(self.axis_factory, range(self.ndim)))


class _ArrayAxesFactory(_AxesFactory):
    def __init__(self, ndim: int) -> None:
        super().__init__(ndim, _index_axis)


# --- API --------------------------------------------------------------


class ArrayCoordinateSystem(CoordinateSystem):
    """A coordinate system for a multidimensional array.

    Its coordinates count samples, so the axes it builds by default carry
    the index unit (see [`IndexUnit`][]). Its `order` is the memory
    order of the array, `None` when it is not specified: an
    `ArrayCoordinateSystem` says nothing about it.

    It is a base rather than a dispatch target: calling it with two or
    three axes builds the matching fixed-arity class, and a system of two
    or three sampled axes is selected as one of those from
    [`CoordinateSystem`][] too. Calling any class with `order="C"` or
    `order="F"` builds the C- or F-ordered class that the order and the
    axes select (see the module).
    """

    name: tx.Optional[str] = "array"


class CArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "C"}):
    """A coordinate system for a C-ordered multidimensional array.

    The first axis is the slowest changing in memory, and the last axis
    the fastest changing. It is what `order="C"` selects.
    """

    name: tx.Optional[str] = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "F"}):
    """A coordinate system for an F-ordered multidimensional array.

    The first axis is the fastest changing in memory, and the last axis
    the slowest changing. It is what `order="F"` selects.
    """

    name: tx.Optional[str] = "farray"


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

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


@ArrayCoordinateSystem.register_polymorph(axes=_is3d)
class ArrayCoordinateSystem3D(
    CoordinateSystem3D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is3d, _is_array)},
):
    """A coordinate system for an array with three dimensions."""

    axes: Axes[_3Axes] = _ArrayAxesFactory(3)


# The C- and F-ordered classes below inherit from two dispatch targets --
# a fixed-arity class, selected on its axes, and an ordered one, selected
# on `order` -- so bagof selects them on what both stand for, from every
# class above them: `CoordinateSystem(axes=<3 axes>, order="F")` is an
# `FArrayCoordinateSystem3D`.
class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem):
    """A coordinate system for a C-ordered array with two dimensions."""

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem):
    """A coordinate system for a C-ordered array with three dimensions."""

    axes: Axes[_3Axes] = _ArrayAxesFactory(3)


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem):
    """A coordinate system for an F-ordered array with two dimensions."""

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem):
    """A coordinate system for an F-ordered array with three dimensions."""

    axes: Axes[_3Axes] = _ArrayAxesFactory(3)


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


def _SpaceAxes(*names) -> tx.Tuple[SpaceAxis, ...]:
    return tuple(map(SpaceAxis, names))


def _SpaceArrayAxis(name: str) -> SpaceAxis:
    return SpaceAxis(name, unit=_INDEX)


def _SpaceArrayAxes(*names) -> tx.Tuple[SpaceAxis, ...]:
    return tuple(map(_SpaceArrayAxis, names))


def _SpaceAxesFactory(*names) -> Factory:
    return Factory(lambda: _SpaceAxes(*names))


def _SpaceArrayAxesFactory(*names) -> Factory:
    return Factory(lambda: _SpaceArrayAxes(*names))


class SpatialCoordinateSystem(CoordinateSystem, on={"axes": _is_spatial}):
    """A coordinate system, whose axes have spatial meaning."""

    axes: Axes[AxisList[_EllipsisOr[SpaceAxis]]] = [...]


# A spatial system of sampled axes is both spatial and an array, and the
# spatial reading wins: the pixel and voxel systems below are spatial
# systems, and are reached through them. The C- and F-ordered voxel
# systems inherit from two dispatch targets -- a 3D spatial system and an
# ordered 3D array -- and bagof selects them on what both stand for, from
# every class above. A C- or F-ordered pixel system is a pixel system, and
# a pixel system asks for axes that count samples, which an ordered one
# does not (the order already says the axes index an array, so a spatial
# axis whose unit is not given is enough): it stays out of the dispatch of
# every class (`on=None`), and is registered by hand with its ordered
# array base (on its spatial axes), and with the pixel and the 2D spatial
# systems (on its order), through which every class above reaches it.


class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: Axes[_2SpatialAxes] = _SpaceAxesFactory("dim0", "dim1")


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: Axes[_3SpatialAxes] = _SpaceAxesFactory("dim0", "dim1", "dim2")


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for 2D pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("dim0", "dim1")


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory(
        "dim0", "dim1", "dim2"
    )


@CArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
@PixelCoordinateSystem.register_polymorph(order="C")
@SpatialCoordinateSystem2D.register_polymorph(order="C")
class CPixelCoordinateSystem(
    PixelCoordinateSystem, CArrayCoordinateSystem2D, on=None
):
    """A coordinate system for C-ordered 2D pixel grids."""

    name: tx.Optional[str] = "cpixel"
    order: tx.Literal["C"] = "C"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("j", "i")


@FArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
@PixelCoordinateSystem.register_polymorph(order="F")
@SpatialCoordinateSystem2D.register_polymorph(order="F")
class FPixelCoordinateSystem(
    PixelCoordinateSystem, FArrayCoordinateSystem2D, on=None
):
    """A coordinate system for F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    order: tx.Literal["F"] = "F"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("i", "j")


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """A coordinate system for C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory("k", "j", "i")


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """A coordinate system for F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory("i", "j", "k")


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
    axes: Axes[AxisTuple[_axes.R, _axes.A, _axes.S]] = Factory()


class LPSCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("LPS")}, priority=2
):
    """The LPS anatomical coordinate system.

    Coordinates increase toward the left, the posterior, and the
    superior directions. This coordinate system is used by ITK, and
    therefore also by ANTs, 3D Slicer, and other ITK-based tools.
    """

    name: tx.Optional[str] = "LPS"
    axes: Axes[AxisTuple[_axes.L, _axes.P, _axes.S]] = Factory()


class RSACoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RSA")}, priority=2
):
    """The RSA anatomical coordinate system.

    Coordinates increase toward the right, the superior, and the
    anterior directions. This coordinate system appears in some
    FreeSurfer LTA files.
    """

    name: tx.Optional[str] = "RSA"
    axes: Axes[AxisTuple[_axes.R, _axes.S, _axes.A]] = Factory()


# ----------------------------------------------------------------------
#   PHYSICAL ANATOMICAL SPACES
# ----------------------------------------------------------------------
# These shorthands are the millimetre anatomical systems -- the spaces
# that nearly every file format means when it writes an anatomical affine.
# Each is in millimetres, and nothing else: every axis is measured in mm,
# not in another unit of length (which would need the data rescaled, not
# the system relabelled), and not in a unit left unspecified. Each is
# selected on its own orientation *and* the millimetre on every axis: the
# orientation is what its anatomical parent already checks, and checking
# it again keeps `PhysicalCoordinateSystem(axes=<LPS axes in mm>)` from
# reaching `RASmm`. RAS axes in another unit, or with no unit, build an
# `RASCoordinateSystem` -- there is no physical RAS system for another
# unit to select -- or, from `PhysicalCoordinateSystem`, stay one.


class MillimetricCoordinateSystem(
    SpatialCoordinateSystem, PhysicalCoordinateSystem
):
    """A physical coordinate system in millimetres, on every axis.

    Building one with an axis in another unit, or with none, is refused:
    the class says what the unit is.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for axis in self.axes:
            if axis is ... or axis.unit == _MM:
                continue
            name = type(self).__name__
            raise ValueError(
                f"{name} is in millimetres, so every one of its axes is "
                f"measured in mm. The axis {axis.name or axis.type!r} "
                f"carries {axis.unit!r}. Build an "
                f"{type(self).__mro__[1].__name__} (or a "
                f"PhysicalCoordinateSystem) for axes in another unit, or "
                f"with none."
            )


class RASmm(
    RASCoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("RAS"), _is_mm)},
):
    """[`RASCoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "RAS"
    axes: Axes[AxisTuple[_axes.R, _axes.A, _axes.S]] = Factory(
        lambda: (
            _axes.R("x", unit=_MM),
            _axes.A("y", unit=_MM),
            _axes.S("z", unit=_MM),
        )
    )


class LPSmm(
    LPSCoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("LPS"), _is_mm)},
):
    """[`LPSCoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "LPS"
    axes: Axes[AxisTuple[_axes.L, _axes.P, _axes.S]] = Factory(
        lambda: (
            _axes.L("x", unit=_MM),
            _axes.P("y", unit=_MM),
            _axes.S("z", unit=_MM),
        )
    )


class RSAmm(
    RSACoordinateSystem,
    MillimetricCoordinateSystem,
    on={"axes": _both(_is_anat("RSA"), _is_mm)},
):
    """[`RSACoordinateSystem`][] in millimetres."""

    name: tx.Optional[str] = "RSA"
    axes: Axes[AxisTuple[_axes.R, _axes.S, _axes.A]] = Factory(
        lambda: (
            _axes.R("x", unit=_MM),
            _axes.S("y", unit=_MM),
            _axes.A("z", unit=_MM),
        )
    )


# ----------------------------------------------------------------------
#   ANATOMICAL VOXEL SPACES
# ----------------------------------------------------------------------
# An F-ordered grid lists its axes x, y, z; a C-ordered one lists them z,
# y, x. So an F-ordered RAS grid has axes that point R, A, S, and a
# C-ordered one has axes that point S, A, R.
#
# An F-ordered one is what both its parents stand for -- its orientation
# and an F-ordered voxel grid -- so bagof selects it from every class
# above it, with no constraint of its own. A C-ordered one lists its axes
# in an order its anatomical parent does not select (S, A, R is not R, A,
# S), so it cannot stand for what that parent does: it stays out of the
# dispatch of every class (`on=None`), and is registered by hand with its
# C-ordered voxel base, on its own axis order. Every class that reaches
# that base reaches it too.


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RAS"))
class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RASCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: tx.Optional[str] = "fRAS"
    axes: Axes[AxisTuple[_axes.R, _axes.A, _axes.S]] = Factory(
        lambda: (
            _axes.R("x", unit=_INDEX),
            _axes.A("y", unit=_INDEX),
            _axes.S("z", unit=_INDEX),
        )
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("LPS"))
class FLPSCoordinateSystem(LPSCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`LPSCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: tx.Optional[str] = "fLPS"
    axes: Axes[AxisTuple[_axes.L, _axes.P, _axes.S]] = Factory(
        lambda: (
            _axes.L("x", unit=_INDEX),
            _axes.P("y", unit=_INDEX),
            _axes.S("z", unit=_INDEX),
        )
    )


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RSA"))
class FRSACoordinateSystem(RSACoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RSACoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: tx.Optional[str] = "fRSA"
    axes: Axes[AxisTuple[_axes.R, _axes.S, _axes.A]] = Factory(
        lambda: (
            _axes.R("x", unit=_INDEX),
            _axes.S("y", unit=_INDEX),
            _axes.A("z", unit=_INDEX),
        )
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
    order: tx.Literal["C"] = "C"
    axes: Axes[AxisTuple[_axes.S, _axes.A, _axes.R]] = Factory(
        lambda: (
            _axes.S("z", unit=_INDEX),
            _axes.A("y", unit=_INDEX),
            _axes.R("x", unit=_INDEX),
        )
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
    order: tx.Literal["C"] = "C"
    axes: Axes[AxisTuple[_axes.S, _axes.P, _axes.L]] = Factory(
        lambda: (
            _axes.S("z", unit=_INDEX),
            _axes.P("y", unit=_INDEX),
            _axes.L("x", unit=_INDEX),
        )
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
    order: tx.Literal["C"] = "C"
    axes: Axes[AxisTuple[_axes.A, _axes.S, _axes.R]] = Factory(
        lambda: (
            _axes.A("x", unit=_INDEX),
            _axes.S("y", unit=_INDEX),
            _axes.R("z", unit=_INDEX),
        )
    )
