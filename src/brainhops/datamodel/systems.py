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
"""

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
    "PhysicalCoordinateSystem",
    "RASmm",
    "LPSmm",
    "RSAmm",
]
# externals
import typing_extensions as tx

# internals
from . import axes as _axes
from .axes import Axis, SpaceAxis
from .base import DataModelBase
from .units import is_physicalunit, is_sampleunit

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis]
_3SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis, SpaceAxis]

_SAMPLE = "sample"


# ----------------------------------------------------------------------
#   DISPATCH PREDICATES
# ----------------------------------------------------------------------


def _ndim(n: int) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        return axes is not None and len(axes) == n

    check.__name__ = check.__qualname__ = f"_is{n}d"
    return check


_is2d = _ndim(2)
_is3d = _ndim(3)


def _all(
    test: tx.Callable[[Axis], bool], name: str
) -> tx.Callable[[tx.Optional[tx.Sequence[Axis]]], bool]:
    """Whether there are axes and every one of them passes `test`."""

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        return bool(axes) and all(test(axis) for axis in axes)

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
        getattr(_axes, letter).orientation.value for letter in code
    )

    def check(axes: tx.Optional[tx.Sequence[Axis]]) -> bool:
        if axes is None or len(axes) != len(expected):
            return False
        return all(
            getattr(getattr(axis, "orientation", None), "value", None) == value
            for axis, value in zip(axes, expected)
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
    """

    name: tx.Optional[str] = None
    """The name of the coordinate system."""

    axes: tx.Optional[tx.List[Axis]] = None
    """The axes of the coordinate system, in order."""


class CoordinateSystem2D(CoordinateSystem, on={"axes": _is2d}):
    """A coordinate systems with exactly two dimensions."""

    axes: tx.Optional[_2Axes] = (Axis(), Axis())


class CoordinateSystem3D(CoordinateSystem, on={"axes": _is3d}):
    """A coordinate system with exactly three dimensions."""

    axes: tx.Optional[_3Axes] = (Axis(), Axis(), Axis())


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
    """

    def __post_init__(self) -> None:
        name = type(self).__name__
        if not self.axes:
            raise ValueError(
                f"{name} is a physical coordinate system, so it must have "
                f"axes, and every one of them must carry a physical unit. "
                f"It was given {self.axes!r}."
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

    axes: tx.Optional[_2Axes] = (_dim(0), _dim(1))


@ArrayCoordinateSystem.register_polymorph(axes=_is3d)
class ArrayCoordinateSystem3D(
    CoordinateSystem3D,
    ArrayCoordinateSystem,
    on={"axes": _both(_is3d, _is_array)},
):
    """A coordinate system for an array with three dimensions."""

    axes: tx.Optional[_3Axes] = (_dim(0), _dim(1), _dim(2))


# The memory order is not written on the axes, so nothing selects a C- or
# F-ordered class but its own ordered base: `on=None` keeps them out of
# the dispatch of every class above, and the decorator registers them
# with that one base.
@CArrayCoordinateSystem.register_polymorph(axes=_is2d)
class CArrayCoordinateSystem2D(
    CoordinateSystem2D, CArrayCoordinateSystem, on=None
):
    """A coordinate system for a C-ordered array with two dimensions."""

    axes: tx.Optional[_2Axes] = (_dim(0), _dim(1))


@CArrayCoordinateSystem.register_polymorph(axes=_is3d)
class CArrayCoordinateSystem3D(
    CoordinateSystem3D, CArrayCoordinateSystem, on=None
):
    """A coordinate system for a C-ordered array with three dimensions."""

    axes: tx.Optional[_3Axes] = (_dim(0), _dim(1), _dim(2))


@FArrayCoordinateSystem.register_polymorph(axes=_is2d)
class FArrayCoordinateSystem2D(
    CoordinateSystem2D, FArrayCoordinateSystem, on=None
):
    """A coordinate system for an F-ordered array with two dimensions."""

    axes: tx.Optional[_2Axes] = (_dim(0), _dim(1))


@FArrayCoordinateSystem.register_polymorph(axes=_is3d)
class FArrayCoordinateSystem3D(
    CoordinateSystem3D, FArrayCoordinateSystem, on=None
):
    """A coordinate system for an F-ordered array with three dimensions."""

    axes: tx.Optional[_3Axes] = (_dim(0), _dim(1), _dim(2))


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class SpatialCoordinateSystem(CoordinateSystem, on={"axes": _is_spatial}):
    """A coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[tx.List[SpaceAxis]] = None


# A spatial system of sampled axes is both spatial and an array, and the
# spatial reading wins: the pixel and voxel systems below are spatial
# systems, and are reached through them.
class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_2SpatialAxes] = (SpaceAxis(), SpaceAxis())


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={}, priority=1
):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_3SpatialAxes] = (
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
    axes: tx.Optional[_2SpatialAxes] = (_space("dim0"), _space("dim1"))


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: tx.Optional[_3SpatialAxes] = (
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
    axes: tx.Optional[_2SpatialAxes] = (_space("j"), _space("i"))


@FArrayCoordinateSystem2D.register_polymorph(axes=_is_spatial)
class FPixelCoordinateSystem(
    PixelCoordinateSystem, FArrayCoordinateSystem2D, on=None
):
    """A coordinate system for F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    axes: tx.Optional[_2SpatialAxes] = (_space("i"), _space("j"))


@CArrayCoordinateSystem3D.register_polymorph(axes=_is_spatial)
class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D, on=None
):
    """A coordinate system for C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: tx.Optional[_3SpatialAxes] = (_space("k"), _space("j"), _space("i"))


@FArrayCoordinateSystem3D.register_polymorph(axes=_is_spatial)
class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D, on=None
):
    """A coordinate system for F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: tx.Optional[_3SpatialAxes] = (_space("i"), _space("j"), _space("k"))


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------
# An anatomical system fixes a direction per axis and says nothing about
# the metric: an array can be RAS-oriented and indexed in samples. Their
# default axes are built here rather than taken from the module-level
# `R`, `A`, `S`, ...: those are mutable, and a system must never hand them
# out. They
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
        _axes.AxisLR(),
        _axes.AxisPA(),
        _axes.AxisIS(),
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
        _axes.AxisRL(),
        _axes.AxisAP(),
        _axes.AxisIS(),
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
        _axes.AxisLR(),
        _axes.AxisIS(),
        _axes.AxisPA(),
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
