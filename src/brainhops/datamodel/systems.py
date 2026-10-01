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
from .units import is_physicalunit

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis]
_3SpatialAxes = tx.Tuple[SpaceAxis, SpaceAxis, SpaceAxis]


def _is2d(axes: tx.Optional[tx.List[Axis]]) -> bool:
    return axes is not None and len(axes) == 2


def _is3d(axes: tx.Optional[tx.List[Axis]]) -> bool:
    return axes is not None and len(axes) == 3


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


def _is_physical(axes: tx.Optional[tx.List[Axis]]) -> bool:
    """Whether every axis measures a physical quantity."""
    if not axes:
        return False
    return all(is_physicalunit(getattr(axis, "unit", None)) for axis in axes)


class PhysicalCoordinateSystem(CoordinateSystem):
    """A coordinate system whose coordinates measure physical quantities.

    Every axis carries a unit, and a *meaningful* one: not `None`, which
    leaves the unit unspecified, and not [`SampleUnit`][], which says the
    coordinates count samples of an array. A system that declares itself
    physical therefore always has a conversion factor to another physical
    system of the same kind, and reversing one of its axes is a sign flip
    rather than the origin shift a sampled axis needs.

    Like [`ArrayCoordinateSystem`][], this is a base to inherit
    deliberately rather than a dispatch target. Physical-ness is
    orthogonal to the arity and the axis types the `on=` predicates select
    on, so making it a target would need one class per combination; the
    concrete systems that are physical by construction -- [`RASmm`][],
    [`LPSmm`][], [`RSAmm`][] -- compose it in.
    """

    def __post_init__(self) -> None:
        # `Validate(_is_physical)` would be the natural spelling, but in
        # bagof 0.2.1 such a field stores the predicate's *result* rather
        # than the value it checked, so the check is written out here.
        for axis in self.axes or ():
            if is_physicalunit(getattr(axis, "unit", None)):
                continue
            raise ValueError(
                f"{type(self).__name__} is a physical coordinate system, so "
                f"every one of its axes must carry a unit that measures "
                f"something. The axis {axis.name or axis.type!r} carries "
                f"{axis.unit!r}: `None` leaves the unit unspecified, and "
                f"`'sample'` says the axis indexes an array."
            )


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

    axes: tx.Optional[_2Axes] = (Axis("dim0"), Axis("dim1"))


ArrayCoordinateSystem.register_polymorph(
    ArrayCoordinateSystem2D, on={"axes": _is2d}
)


class ArrayCoordinateSystem3D(CoordinateSystem3D, ArrayCoordinateSystem):
    """A coordinate system for a unitless array with three dimensions."""

    axes: tx.Optional[_3Axes] = (Axis("dim0"), Axis("dim1"), Axis("dim2"))


ArrayCoordinateSystem.register_polymorph(
    ArrayCoordinateSystem3D, on={"axes": _is3d}
)


class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem):
    """A coordinate system for a unitless, C-ordered array with two
    dimensions."""


CArrayCoordinateSystem.register_polymorph(
    CArrayCoordinateSystem2D, on={"axes": _is2d}
)


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem):
    """A coordinate system for a unitless, C-ordered array with three
    dimensions."""


CArrayCoordinateSystem.register_polymorph(
    CArrayCoordinateSystem3D, on={"axes": _is3d}
)


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem):
    """A coordinate system for a unitless, F-ordered array with two
    dimensions."""


FArrayCoordinateSystem.register_polymorph(
    FArrayCoordinateSystem2D, on={"axes": _is2d}
)


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem):
    """A coordinate system for a unitless, F-ordered array with three
    dimensions."""


FArrayCoordinateSystem.register_polymorph(
    FArrayCoordinateSystem3D, on={"axes": _is3d}
)


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


def _is_spatial(axes: tx.Optional[tx.List[Axis]]) -> bool:
    return axes is not None and all(axis.type == "space" for axis in axes)


def _is_spatial_2d(axes: tx.Optional[tx.List[Axis]]) -> bool:
    return _is_spatial(axes) and len(axes) == 2


def _is_spatial_3d(axes: tx.Optional[tx.List[Axis]]) -> bool:
    return _is_spatial(axes) and len(axes) == 3


class SpatialCoordinateSystem(
    CoordinateSystem, on={"axes": _is_spatial}, priority=1
):
    """A coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[tx.List[SpaceAxis]] = None


class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={"axes": _is_spatial_2d}
):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_2SpatialAxes] = (SpaceAxis(), SpaceAxis())


SpatialCoordinateSystem.register_polymorph(
    SpatialCoordinateSystem2D, on={"axes": _is_spatial_2d}
)


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={"axes": _is_spatial_3d}
):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_3SpatialAxes] = (
        SpaceAxis(),
        SpaceAxis(),
        SpaceAxis(),
    )


SpatialCoordinateSystem.register_polymorph(
    SpatialCoordinateSystem3D, on={"axes": _is_spatial_3d}
)


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for (unitless) 2D pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpaceAxis(name="dim0"),
        SpaceAxis(name="dim1"),
    )


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpaceAxis(name="dim0"),
        SpaceAxis(name="dim1"),
        SpaceAxis(name="dim2"),
    )


class CPixelCoordinateSystem(PixelCoordinateSystem, CArrayCoordinateSystem2D):
    """A coordinate system for (unitless) C-ordered 2D pixel grids."""

    name: tx.Optional[str] = "cpixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpaceAxis(name="j"),
        SpaceAxis(name="i"),
    )


class FPixelCoordinateSystem(PixelCoordinateSystem, FArrayCoordinateSystem2D):
    """A coordinate system for (unitless) F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpaceAxis(name="i"),
        SpaceAxis(name="j"),
    )


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpaceAxis(name="k"),
        SpaceAxis(name="j"),
        SpaceAxis(name="i"),
    )


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpaceAxis(name="i"),
        SpaceAxis(name="j"),
        SpaceAxis(name="k"),
    )


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


def _is_anat(orientation: tx.Optional[str]) -> bool:

    def check(axes: tx.Optional[tx.List[Axis]]) -> bool:
        if axes is None:
            return False
        if len(axes) != len(orientation):
            return False
        for axis, ref in zip(axes, orientation):
            orient = getattr(axis.orientation, "value", None)
            ref = getattr(_axes, ref.value.upper()).orientation.value
            if orient != ref:
                return False
        return True

    return check


class RASCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RAS")}
):
    """The RAS anatomical coordinate system.

    Coordinates increase toward the right, the anterior, and the
    superior directions. This coordinate system is used by NIfTI files,
    and by many other neuroimaging formats.
    """

    name: str = "RAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _axes.R,
        _axes.A,
        _axes.S,
    )


class LPSCoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("LPS")}
):
    """The LPS anatomical coordinate system.

    Coordinates increase toward the left, the posterior, and the
    superior directions. This coordinate system is used by ITK, and
    therefore also by ANTs, 3D Slicer, and other ITK-based tools.
    """

    name: str = "LPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _axes.L,
        _axes.P,
        _axes.S,
    )


class RSACoordinateSystem(
    SpatialCoordinateSystem3D, on={"axes": _is_anat("RSA")}
):
    """The RSA anatomical coordinate system.

    Coordinates increase toward the right, the superior, and the
    anterior directions. This coordinate system appears in some
    FreeSurfer LTA files.
    """

    name: str = "RSA"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisIS, _axes.AxisPA] = (
        _axes.LR,
        _axes.IS,
        _axes.PA,
    )


# ----------------------------------------------------------------------
#   PHYSICAL ANATOMICAL SPACES
# ----------------------------------------------------------------------
# The anatomical systems above fix a direction per axis and say nothing
# about the metric, because the two are independent: an array can be
# RAS-oriented and indexed in samples. These shorthands are the physical
# ones -- the millimetre spaces that nearly every file format means when it
# writes an anatomical affine -- and they inherit
# [`PhysicalCoordinateSystem`][], so an axis of theirs can never be left
# without a unit.


class RASmm(RASCoordinateSystem, PhysicalCoordinateSystem):
    """[`RASCoordinateSystem`][] in millimetres."""

    name: str = "RAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _axes.AxisLR(unit="mm"),
        _axes.AxisPA(unit="mm"),
        _axes.AxisIS(unit="mm"),
    )


RASCoordinateSystem.register_polymorph(RASmm, on={"axes": _is_physical})


class LPSmm(LPSCoordinateSystem, PhysicalCoordinateSystem):
    """[`LPSCoordinateSystem`][] in millimetres."""

    name: str = "LPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _axes.AxisRL(unit="mm"),
        _axes.AxisAP(unit="mm"),
        _axes.AxisIS(unit="mm"),
    )


LPSCoordinateSystem.register_polymorph(LPSmm, on={"axes": _is_physical})


class RSAmm(RSACoordinateSystem, PhysicalCoordinateSystem):
    """[`RSACoordinateSystem`][] in millimetres."""

    name: str = "RSA"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisIS, _axes.AxisPA] = (
        _axes.AxisLR(unit="mm"),
        _axes.AxisIS(unit="mm"),
        _axes.AxisPA(unit="mm"),
    )


RSACoordinateSystem.register_polymorph(RSAmm, on={"axes": _is_physical})


class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RASCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: str = "fRAS"
    axes: tx.Tuple[_axes.AxisLR, _axes.AxisPA, _axes.AxisIS] = (
        _axes.AxisLR(name="x"),
        _axes.AxisPA(name="y"),
        _axes.AxisIS(name="z"),
    )


FVoxelCoordinateSystem.register_polymorph(
    FRASCoordinateSystem, on={"axes": _is_anat("RAS")}
)


class FLPSCoordinateSystem(LPSCoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`LPSCoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: str = "fLPS"
    axes: tx.Tuple[_axes.AxisRL, _axes.AxisAP, _axes.AxisIS] = (
        _axes.AxisRL(name="x"),
        _axes.AxisAP(name="y"),
        _axes.AxisIS(name="z"),
    )


FVoxelCoordinateSystem.register_polymorph(
    FLPSCoordinateSystem, on={"axes": _is_anat("LPS")}
)


class FRSACoordinateSystem(RSACoordinateSystem, FVoxelCoordinateSystem):
    """Combines [`RSACoordinateSystem`][] with [`FVoxelCoordinateSystem`][].

    This coordinate system describes an F-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: str = "fRSA"
    axes: tx.Tuple[
        _axes.AxisLR,
        _axes.AxisIS,
        _axes.AxisPA,
    ] = (
        _axes.AxisLR(name="x"),
        _axes.AxisIS(name="y"),
        _axes.AxisPA(name="z"),
    )


FVoxelCoordinateSystem.register_polymorph(
    FRSACoordinateSystem, on={"axes": _is_anat("RSA")}
)


class CRASCoordinateSystem(RASCoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`RASCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RAS order.
    """

    name: str = "cRAS"
    axes: tx.Tuple[_axes.AxisIS, _axes.AxisPA, _axes.AxisLR] = (
        _axes.AxisIS(name="z"),
        _axes.AxisPA(name="y"),
        _axes.AxisLR(name="x"),
    )


CVoxelCoordinateSystem.register_polymorph(
    CRASCoordinateSystem, on={"axes": _is_anat("RAS")}
)


class CLPSCoordinateSystem(LPSCoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`LPSCoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in LPS order.
    """

    name: str = "cLPS"
    axes: tx.Tuple[_axes.AxisIS, _axes.AxisAP, _axes.AxisRL] = (
        _axes.AxisIS(name="z"),
        _axes.AxisAP(name="y"),
        _axes.AxisRL(name="x"),
    )


CVoxelCoordinateSystem.register_polymorph(
    CLPSCoordinateSystem, on={"axes": _is_anat("LPS")}
)


class CRSACoordinateSystem(RSACoordinateSystem, CVoxelCoordinateSystem):
    """Combines [`RSACoordinateSystem`][] with [`CVoxelCoordinateSystem`][].

    This coordinate system describes a C-ordered voxel grid whose axes
    already point in RSA order.
    """

    name: str = "cRSA"
    axes: tx.Tuple[_axes.AxisPA, _axes.AxisIS, _axes.AxisLR] = (
        _axes.AxisPA(name="z"),
        _axes.AxisIS(name="y"),
        _axes.AxisLR(name="x"),
    )


CVoxelCoordinateSystem.register_polymorph(
    CRSACoordinateSystem, on={"axes": _is_anat("RSA")}
)
