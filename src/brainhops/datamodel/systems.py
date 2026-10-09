"""Coordinate systems, from unitless arrays to anatomical spaces.

Calling [`CoordinateSystem`][] builds the most specific system that its axes
describe. Each class declares the axes it stands for through its `on=`
constraint:

| The axes are...                               | ...so the system is        |
| --------------------------------------------- | -------------------------- |
| two or three of anything                      | `CoordinateSystem2D`/`3D`  |
| all spatial                                   | `SpatialCoordinateSystem*` |
| two or three, all measured in samples         | `ArrayCoordinateSystem*`   |
| spatial *and* measured in samples             | `Pixel`/`VoxelCoordinate…` |
| oriented right, anterior, superior (in order) | `RASCoordinateSystem`      |
| ... and in millimetres                        | `RASmm`                    |

The same holds for LPS and RSA. A class that inherits from two dispatch
targets, such as `SpatialCoordinateSystem3D`, is selected on what both of them
stand for, without a constraint of its own.

The memory order of an array is not visible on its axes, so the C- and
F-ordered variants are selected on the `order` field (`"C"`, `"F"`, or `None`
when unspecified) together with the axes:

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
order, so `RASmm(order="F")` is refused.

Every row of these tables is a statement about all the axes, so only closed
systems are dispatched. An open system, whose [`AxisList`][] holds `...`, does
not know all its axes and is built as the class it was called as:
`CoordinateSystem(axes=[x, ...])` is not two-dimensional. Closing it with
[`CoordinateSystem.expand`][] dispatches it normally.
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
    # Python 3.10 names this type types.EllipsisType.

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
# Each predicate states something about all the axes, so none holds for an
# open system.


def _closed(
    axes: tx.Optional[tx.Sequence[tx.Any]],
) -> tx.Optional[tx.List[Axis]]:
    """Return the axes as a list if they list every axis, else `None`."""
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
    """Build a predicate: the axes are closed, non-empty and pass `test`."""

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
    """Build a predicate: the axes point as the letters of `code` say.

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

    A coordinate system describes each of its axes (name, unit and other
    properties) and can be named.

    !!! note "Open systems"
        The axes are an [`AxisList`][] that holds at most one `...`, standing
        for zero or more axes about which nothing is known. A system with `...`
        is open, since its number of axes is unknown, and a system without it
        is closed:

        * `[..., TimeAxis()]`: the last axis is time, and nothing is known
          of the others.
        * `[Axis(name="x"), ...]`: the first axis is x.
        * `[...]`, the default, says nothing. `axes=None` reads as not given,
          so `CoordinateSystem(axes=None) == CoordinateSystem()`.

        Fixed-arity classes such as [`CoordinateSystem3D`][] are always closed.
        They store their axes as an immutable [`AxisTuple`][], whose type fixes
        the number and class of the axes.

        Only closed systems are dispatched: `CoordinateSystem(axes=[x, y])` is
        a [`CoordinateSystem2D`][], whereas `CoordinateSystem(axes=[x, ...])`
        stays a `CoordinateSystem` until [`expand`][] closes it.

    !!! note "Equality"
        Two systems are equal when they have the same class, the same name and
        equal axes. A system never equals `None`, not even
        `CoordinateSystem()`: as an endpoint of a transformation, `None` is no
        system and defers to the context, whereas `CoordinateSystem()` is a
        system and is kept as given. [`compatible_with`][] asks the looser
        question of whether two systems could describe the same space.
    """

    name: tx.Optional[str] = None
    """The name of the system, if any."""

    axes: Axes[AxisList[_EllipsisOr[Axis]]] = [...]
    """The axes, in order. The default, `[...]`, says nothing about them."""

    order: tx.Optional[tx.Literal["C", "F"]] = None
    """The memory order of the indexed array, or `None` when unspecified.

    With `"C"` the last axis varies fastest, and with `"F"` the first one does.
    Only an [`ArrayCoordinateSystem`][] indexes an array, and other systems
    refuse an order with a ValueError.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass declares its own `axes` default, read for `None`.
        bind_axes_default(cls)

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

    def expand(self, ndim: int) -> tx.Self:
        """Return the closed system of `ndim` axes that this system describes.

        [`AxisSequence.expand`][] replaces `...` with unknown axes of the type
        the class declares, and the class is called again with the other fields
        kept, so the result may be a subclass: a [`SpatialCoordinateSystem`][]
        closed to three axes is a [`SpatialCoordinateSystem3D`][]. A closed
        system is returned as is.

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
        """Return the system of the axes at some positions.

        The axes are selected by [`AxisSequence.restrict`][], by position or
        name, in the order of `refs`. A position covered by the `...` of an
        open system gives an unknown axis. The result describes another space,
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

        This is the inverse of [`restrict`][]. With [`AxisSequence.embed`][],
        axis `j` sits at `positions[j]` and every other position holds an
        unknown axis. As with [`restrict`][], neither the class nor the name is
        carried over.

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

    Each stated axis has a physical unit, such as mm or s, or an unspecified
    one (`None`), but never an [`IndexUnit`][]. Reversing an axis is therefore
    a sign flip rather than the origin shift of a sampled axis. Axis types
    already enforce the kind of each unit, so this class only refuses index
    units.

    The class is a base, not a dispatch target. [`RASmm`][], [`LPSmm`][] and
    [`RSAmm`][] build on it and require mm on every axis.

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


class ArrayCoordinateSystem(CoordinateSystem):
    """A coordinate system that indexes a multidimensional array.

    The coordinates count samples, so the default axes of the two- and
    three-dimensional subclasses carry an [`IndexUnit`][]. The class is not a
    dispatch target itself: called with two or three axes, it builds the
    matching fixed-arity class, and called with `order="C"` or `"F"`, the
    matching ordered class.
    """

    name: tx.Optional[str] = "array"


class CArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "C"}):
    """A C-ordered array system, whose last axis varies fastest in memory."""

    name: tx.Optional[str] = "carray"


class FArrayCoordinateSystem(ArrayCoordinateSystem, on={"order": "F"}):
    """An F-ordered array system, whose first axis varies fastest in memory."""

    name: tx.Optional[str] = "farray"


# Not a dispatch target, so the 2-D case is registered by hand; `on=`
# covers the systems reached from CoordinateSystem2D.
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


# Two dispatch targets (arity and order), so these classes are selected
# on what both stand for, without a constraint of their own.
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


# Spatial systems of sampled axes are reached through the spatial ones.
# An ordered pixel system needs no sampled axes (the order already says
# that they index an array), so it stays out of dispatch and is
# registered by hand with its bases.


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
# An anatomical system fixes a direction per axis but no metric. It takes
# precedence over pixel and voxel systems, which say less about oriented
# sampled axes.


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
# Anatomical spaces in mm, as most formats mean by an anatomical affine.
# Another length unit would need the data rescaled, not the system
# relabelled. The orientation is checked again so that LPS axes in mm do
# not reach RASmm.


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
# An F-ordered grid lists its axes x, y, z and a C-ordered one z, y, x.
# The F-ordered systems are selected on both parents. The C-ordered ones
# list their axes in an order their anatomical parent does not select,
# so they stay out of dispatch and are registered on their own order.


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
