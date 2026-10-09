"""Coordinate systems, from unitless arrays to anatomical spaces.

A coordinate system describes the axes of a space. Calling
[`CoordinateSystem`][] does not always build a plain `CoordinateSystem`: it
builds the most specific subclass that the given axes describe, and this
selection of a subclass is called dispatch below. Each class declares, in its
`on=` constraint, the axes that it stands for:

| The axes are...                               | ...so the system is        |
| --------------------------------------------- | -------------------------- |
| two or three of anything                      | `CoordinateSystem2D`/`3D`  |
| all spatial                                   | `SpatialCoordinateSystem*` |
| two or three, all measured in samples         | `ArrayCoordinateSystem*`   |
| spatial *and* measured in samples             | `Pixel`/`VoxelCoordinate…` |
| oriented right, anterior, superior (in order) | `RASCoordinateSystem`      |
| ... and in millimetres                        | `RASmm`                    |

The same rules apply to LPS and RSA systems. Some classes inherit from two
classes that dispatch can select. For example, `SpatialCoordinateSystem3D`
inherits from `CoordinateSystem3D` and from `SpatialCoordinateSystem`. Such a
class has no constraint of its own and is selected when the constraints of
both parents hold.

The memory order of an array cannot be read from its axes, so the C-ordered
and F-ordered variants are selected on the `order` field together with the
axes. The field holds `"C"`, `"F"`, or `None` when the order is unspecified:

| `order=`, and the axes are...           | ...so the system is            |
| --------------------------------------- | ------------------------------ |
| `"C"` / `"F"`, anything                 | `C`/`FArrayCoordinateSystem`   |
| ... two or three                        | `C`/`FArrayCoordinateSystem2D/3D` |
| ... two, spatial                        | `C`/`FPixelCoordinateSystem`   |
| ... three, spatial                      | `C`/`FVoxelCoordinateSystem`   |
| `"F"`, oriented R, A, S                 | `FRASCoordinateSystem`         |
| `"C"`, oriented S, A, R (a C-ordered    | `CRASCoordinateSystem`         |
| grid lists its axes z, y, x)            |                                |

Every class accepts `order` and passes it on, so `CoordinateSystem(axes=<RAS
axes>, order="F")` builds an `FRASCoordinateSystem`. Only array systems have an
order, so `RASmm(order="F")` raises a ValueError.

Every row of these tables states something about all the axes of a system, so
only closed systems are dispatched. An open system is a system whose
[`AxisList`][] holds `...`. Such a system does not know all its axes, so it is
built as the class it was called as: `CoordinateSystem(axes=[x, ...])` is not
two-dimensional. Once [`CoordinateSystem.expand`][] has closed the system, the
closed system is dispatched normally.
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
import typing_extensions as tx
from bagof.magic import Factory, replace

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
    # This is the type of `...`, which Python 3.10 names
    # `types.EllipsisType`.

AXIS = tx.TypeVar("AXIS")

_INDEX = Unit("index")

_2Axes: tx.TypeAlias = AxisTuple[Axis, Axis]
_3Axes: tx.TypeAlias = AxisTuple[Axis, Axis, Axis]
_2SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis]
_3SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis, SpaceAxis]
_EllipsisOr: tx.TypeAlias = tx.Union[AXIS, _Ellipsis]


# ----------------------------------------------------------------------
#   DISPATCH PREDICATES
# ----------------------------------------------------------------------
# Dispatch uses the predicates below to decide whether a class matches the
# axes. Each predicate states something about all the axes, so no predicate
# holds for an open system, whose `...` may stand for any number of axes.


def _closed(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Optional[tx.List[Axis]]:
    """Return the axes as a list, or `None` if they are missing or open."""
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
    """Build a predicate that holds when closed, non-empty axes pass `test`."""

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
    """Build a predicate that compares the axes with the orientation `code`.

    The letters are those of [`brainhops.datamodel.axes`][], so `"RAS"` means
    left-to-right, posterior-to-anterior and inferior-to-superior axes, in that
    order. Only orientations are compared, not names or units.
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
    """A coordinate system, which gives coordinates in a space their meaning.

    A coordinate system lists its axes in order and describes each of them
    with properties such as a name and a unit. The system itself can also
    have a name.

    !!! note "Open systems"
        The axes are an [`AxisList`][] that holds at most one `...`, standing
        for zero or more axes about which nothing is known. A system with `...`
        is open, since its number of axes is unknown, and a system without it
        is closed:

        * `[..., TimeAxis()]` says that the last axis is time, and nothing
          is known about the others.
        * `[Axis(name="x"), ...]` says that the first axis is x.
        * `[...]`, the default, says nothing about the axes. `axes=None` is
          read as if the axes were not given, so
          `CoordinateSystem(axes=None) == CoordinateSystem()`.

        Fixed-arity classes such as [`CoordinateSystem3D`][] are always closed.
        They store their axes as an immutable [`AxisTuple`][], whose type fixes
        the number and class of the axes.

        Only closed systems are dispatched: `CoordinateSystem(axes=[x, y])` is
        a [`CoordinateSystem2D`][], whereas `CoordinateSystem(axes=[x, ...])`
        stays a `CoordinateSystem` until [`expand`][] closes it.

    !!! note "Equality"
        Two systems are equal when they have the same class, the same name and
        equal axes. A system never equals `None`, not even
        `CoordinateSystem()`. As the input or output of a transformation,
        `None` means that no system is given and lets the context decide,
        whereas `CoordinateSystem()` is a system and is kept as given.
        [`compatible_with`][] asks the looser question of whether two systems
        could describe the same space.
    """

    name: tx.Optional[str] = None
    """The name of the system, if any."""

    axes: Axes[AxisList[_EllipsisOr[Axis]]] = [...]
    """The axes, in order. The default, `[...]`, says nothing about them."""

    order: tx.Optional[tx.Literal["C", "F"]] = None
    """The memory order of the indexed array, or `None` when unspecified.

    With `"C"` the last axis varies fastest, and with `"F"` the first one does.
    Only an [`ArrayCoordinateSystem`][] indexes an array, so any other system
    raises a ValueError when it is given an order.
    """

    # --- construction -------------------------------------------------

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass declares its own default for `axes`, and `axes=None`
        # must be read as that default.
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
        """The number of axes, or `None` for an open system.

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
        """Return the closed system of `ndim` axes that this system describes.

        The `...` is replaced with unknown axes of the type that the class
        declares, as [`AxisSequence.expand`][] does. The class is then called
        again with the new axes and the other fields unchanged, so the result
        may be a subclass. For example, a [`SpatialCoordinateSystem`][] closed
        to three axes is a [`SpatialCoordinateSystem3D`][]. A system that is
        already closed is returned unchanged.

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
            The closed system.

        Raises
        ------
        ValueError
            If `ndim` is smaller than the number of explicit axes of an open
            system, or differs from the number of axes of a closed system.
        TypeError
            If `ndim` is not an integer.
        """
        expanded = self.axes.expand(ndim)
        if not self.axes.is_open:
            return self
        return replace(self, axes=expanded)

    def restrict(
        self, refs: tx.Iterable[tx.Union[int, str]]
    ) -> "CoordinateSystem":
        """Return the system formed by a selection of the axes.

        Each reference selects an axis by position or by name, as in
        [`AxisSequence.restrict`][], and the selected axes follow the order of
        `refs`. A position covered by the `...` of an open system gives an
        unknown axis. The result describes another space,
        so neither the class nor the name is carried over: the result is
        whatever `CoordinateSystem(axes=...)` builds from the selected axes.

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
            A closed system with one axis per reference.

        Raises
        ------
        ValueError, IndexError, TypeError
            As raised by [`AxisSequence.restrict`][].
        """
        return CoordinateSystem(axes=self.axes.restrict(refs))

    def embed(
        self,
        positions: tx.Iterable[int],
        ndim: tx.Optional[int] = None,
    ) -> "CoordinateSystem":
        """Return the system of a larger space that contains this system's
        axes.

        The method is the inverse of [`restrict`][]. As in
        [`AxisSequence.embed`][], axis `j` of this system is placed at position
        `positions[j]`, and every other position holds an unknown axis. As with
        [`restrict`][], neither the class nor the name is carried over.

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
            The number of axes of the larger space. If omitted, the result is
            open and ends with `...` after the last embedded axis.

        Returns
        -------
        CoordinateSystem
            A closed system if `ndim` is given, and an open one otherwise.

        Raises
        ------
        ValueError, TypeError
            As raised by [`AxisSequence.embed`][].
        """
        return CoordinateSystem(axes=self.axes.embed(positions, ndim=ndim))

    def compatible_with(self, other: tx.Optional["CoordinateSystem"]) -> bool:
        """Return whether two systems could describe the same space.

        Only the axes are compared, with [`AxisSequence.compatible_with`][]:
        the systems are compatible when some choice of the axes that each `...`
        stands for makes them match pairwise under [`Axis.compatible_with`][].
        Unlike `==`, an unknown axis matches any axis, and `None` is compatible
        with every system. The relation is symmetric but not transitive.

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

        Raises
        ------
        TypeError
            If `other` is neither a coordinate system nor `None`.
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
    """A coordinate system with exactly two axes."""

    axes: Axes[_2Axes] = Factory()


class CoordinateSystem3D(CoordinateSystem, on={"axes": _is3d}):
    """A coordinate system with exactly three axes."""

    axes: Axes[_3Axes] = Factory()


# ----------------------------------------------------------------------
#   PHYSICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class PhysicalCoordinateSystem(CoordinateSystem):
    """A coordinate system whose coordinates measure physical quantities.

    Each explicitly listed axis has a physical unit, such as mm or s, or a
    unit that is not specified (`None`), but never an [`IndexUnit`][].
    Reversing an axis therefore flips the sign of its coordinates, whereas
    reversing a sampled axis shifts its origin. The axis types already check
    that each unit is of the right kind, so this class only rejects index
    units.

    The class has no constraint of its own, so dispatch never selects it from
    a parent class. Called directly with matching axes, it can still build
    one of its subclasses, such as [`RASmm`][], [`LPSmm`][] or [`RSAmm`][],
    which require millimetres on every axis.

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
    """A coordinate system that indexes a multidimensional array.

    The coordinates count samples, so the default axes of the two- and
    three-dimensional subclasses carry an [`IndexUnit`][]. Dispatch never
    selects the class itself from a parent class. When the class is called
    directly with two or three axes, however, it builds the matching subclass
    with a fixed number of axes, and when it is called with `order="C"` or
    `order="F"`, it builds the matching ordered subclass.
    """

    name: tx.Optional[str] = "array"


class CArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "C"}):
    """A C-ordered array system, whose last axis varies fastest in memory."""

    name: tx.Optional[str] = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "F"}):
    """An F-ordered array system, whose first axis varies fastest in memory."""

    name: tx.Optional[str] = "farray"


# ArrayCoordinateSystem has no constraint of its own, so its 2-D subclass is
# registered with it by hand: called with two axes of any kind,
# ArrayCoordinateSystem builds an ArrayCoordinateSystem2D. The `on=`
# constraint covers the other route, from CoordinateSystem2D, which requires
# axes that count samples.
@ArrayCoordinateSystem.register_polymorph(axes=_is2d)
class ArrayCoordinateSystem2D(
    CoordinateSystem2D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is2d, _is_array)},
):
    """An array coordinate system with two axes."""

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


@ArrayCoordinateSystem.register_polymorph(axes=_is3d)
class ArrayCoordinateSystem3D(
    CoordinateSystem3D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is3d, _is_array)},
):
    """An array coordinate system with three axes."""

    axes: Axes[_3Axes] = _ArrayAxesFactory(3)


# Each class below inherits from a class with a fixed number of axes and
# from an ordered class. It needs no constraint of its own, because it is
# selected when the constraints of both parents hold.
class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem):
    """A C-ordered array coordinate system with two axes."""

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem):
    """A C-ordered array coordinate system with three axes."""

    axes: Axes[_3Axes] = _ArrayAxesFactory(3)


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem):
    """An F-ordered array coordinate system with two axes."""

    axes: Axes[_2Axes] = _ArrayAxesFactory(2)


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem):
    """An F-ordered array coordinate system with three axes."""

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
    """A coordinate system whose axes are all spatial."""

    axes: Axes[AxisList[_EllipsisOr[SpaceAxis]]] = [...]


# A spatial system whose axes count samples is both spatial and an array,
# and the pixel and voxel systems are reached through the spatial systems.
# An ordered pixel system does not require axes that count samples, because
# the order already says that the axes index an array. It is therefore
# excluded from automatic dispatch (`on=None`) and registered by hand with
# its parent classes.


class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={}, priority=1
):
    """A spatial coordinate system with two axes."""

    axes: Axes[_2SpatialAxes] = _SpaceAxesFactory("dim0", "dim1")


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={}, priority=1
):
    """A spatial coordinate system with three axes."""

    axes: Axes[_3SpatialAxes] = _SpaceAxesFactory("dim0", "dim1", "dim2")


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for two-dimensional pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("dim0", "dim1")


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for three-dimensional voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory("dim0", "dim1", "dim2")


@CArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
@PixelCoordinateSystem.register_polymorph(order="C")
@SpatialCoordinateSystem2D.register_polymorph(order="C")
class CPixelCoordinateSystem(
    PixelCoordinateSystem, CArrayCoordinateSystem2D, on=None
):
    """A coordinate system for C-ordered pixel grids, with axes j and i."""

    name: tx.Optional[str] = "cpixel"
    order: tx.Literal["C"] = "C"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("j", "i")


@FArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
@PixelCoordinateSystem.register_polymorph(order="F")
@SpatialCoordinateSystem2D.register_polymorph(order="F")
class FPixelCoordinateSystem(
    PixelCoordinateSystem, FArrayCoordinateSystem2D, on=None
):
    """A coordinate system for F-ordered pixel grids, with axes i and j."""

    name: tx.Optional[str] = "fpixel"
    order: tx.Literal["F"] = "F"
    axes: Axes[_2SpatialAxes] = _SpaceArrayAxesFactory("i", "j")


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """A coordinate system for C-ordered voxel grids, with axes k, j and i."""

    name: tx.Optional[str] = "cvoxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory("k", "j", "i")


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """A coordinate system for F-ordered voxel grids, with axes i, j and k."""

    name: tx.Optional[str] = "fvoxel"
    axes: Axes[_3SpatialAxes] = _SpaceArrayAxesFactory("i", "j", "k")


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------
# An anatomical system fixes the direction of each axis but says nothing
# about the metric, so an array can be RAS-oriented and indexed in samples.
# Anatomical systems take precedence over pixel and voxel systems, because
# the orientation says more about oriented, sampled axes than a pixel or
# voxel system does.


class RASCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RAS")}, priority=2
):
    """The RAS anatomical system, used by NIfTI and many neuroimaging formats.

    Coordinates increase towards the right, anterior and superior directions.
    """

    name: tx.Optional[str] = "RAS"
    axes: Axes[AxisTuple[_axes.R, _axes.A, _axes.S]] = Factory()


class LPSCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("LPS")}, priority=2
):
    """The LPS anatomical system, used by ITK and hence by ANTs and 3D Slicer.

    Coordinates increase towards the left, posterior and superior directions.
    """

    name: tx.Optional[str] = "LPS"
    axes: Axes[AxisTuple[_axes.L, _axes.P, _axes.S]] = Factory()


class RSACoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RSA")}, priority=2
):
    """The RSA anatomical system, found in some FreeSurfer LTA files.

    Coordinates increase towards the right, superior and anterior directions.
    """

    name: tx.Optional[str] = "RSA"
    axes: Axes[AxisTuple[_axes.R, _axes.S, _axes.A]] = Factory()


# ----------------------------------------------------------------------
#   PHYSICAL ANATOMICAL SPACES
# ----------------------------------------------------------------------
# The classes below are the anatomical spaces in millimetres, which most
# file formats mean when they store an anatomical affine. Another unit of
# length is not accepted, because it would require rescaling the data rather
# than relabelling the system. Each class checks its orientation again,
# together with the unit, so that LPS axes in mm do not reach RASmm.


class MillimetricCoordinateSystem(
    SpatialCoordinateSystem, PhysicalCoordinateSystem
):
    """A physical coordinate system in millimetres on every axis.

    An axis in another unit, or without a unit, raises a ValueError.
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
    """A [`RASCoordinateSystem`][] in millimetres."""

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
    """An [`LPSCoordinateSystem`][] in millimetres."""

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
    """An [`RSACoordinateSystem`][] in millimetres."""

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
# An F-ordered grid lists its axes as x, y, z, whereas a C-ordered grid
# lists them as z, y, x. The axes of an F-ordered RAS grid therefore point
# R, A, S, and those of a C-ordered RAS grid point S, A, R. An F-ordered
# system is selected when the constraints of both its parents hold. A
# C-ordered system lists its axes in an order that its anatomical parent
# does not accept, so it is excluded from automatic dispatch (`on=None`)
# and registered by hand with its C-ordered voxel parent, on its own axis
# order.


@FVoxelCoordinateSystem.register_polymorph(axes=_is_anat("RAS"))
class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
    """An F-ordered voxel grid whose axes point in RAS order."""

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
    """An F-ordered voxel grid whose axes point in LPS order."""

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
    """An F-ordered voxel grid whose axes point in RSA order."""

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
    """A C-ordered RAS voxel grid, whose axes are listed S, A, R."""

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
    """A C-ordered LPS voxel grid, whose axes are listed S, P, L."""

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
    """A C-ordered RSA voxel grid, whose axes are listed A, S, R."""

    name: tx.Optional[str] = "cRSA"
    order: tx.Literal["C"] = "C"
    axes: Axes[AxisTuple[_axes.A, _axes.S, _axes.R]] = Factory(
        lambda: (
            _axes.A("x", unit=_INDEX),
            _axes.S("y", unit=_INDEX),
            _axes.R("z", unit=_INDEX),
        )
    )
