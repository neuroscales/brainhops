__all__ = [
    "CoordinateSystem",
    "SpatialCoordinateSystem",
    "SpatialCoordinateSystem3D",
    "RASCoordinateSystem",
    "LPSCoordinateSystem",
]
# externals
import typing_extensions as tx

# internals
from .base import DataModelBase
from .axes import Axis, SpatialAxis, same_axis_type
from . import axes as _axes
from .axes import Axis, SpatialAxis
from .base import DataModelBase

_2Axes = tx.Tuple[Axis, Axis]
_3Axes = tx.Tuple[Axis, Axis, Axis]
_2SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis]
_3SpatialAxes = tx.Tuple[SpatialAxis, SpatialAxis, SpatialAxis]


class CoordinateSystem(DataModelBase):
    """A coordinate system defines the meaning of coordinates in a space.

    It describes each axis in the system (name, unit and/or other properties),
    and can be named.
    """

    name: tx.Optional[str] = None
    axes: tx.Optional[tx.List[Axis]] = None


class CoordinateSystem2D(CoordinateSystem):
    """A coordinate systems with exactly two dimensions."""

    axes: tx.Optional[_2Axes] = (Axis(), Axis())


class CoordinateSystem3D(CoordinateSystem):
    """A coordinate system with exactly three dimensions."""

    axes: tx.Optional[_3Axes] = (Axis(), Axis(), Axis())


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


class ArrayCoordinateSystem3D(CoordinateSystem3D, ArrayCoordinateSystem):
    """A coordinate system for a unitless array with three dimensions."""

    axes: tx.Optional[_3Axes] = (Axis("dim0"), Axis("dim1"), Axis("dim2"))


class CArrayCoordinateSystem2D(CoordinateSystem2D, CArrayCoordinateSystem): ...


class CArrayCoordinateSystem3D(CoordinateSystem3D, CArrayCoordinateSystem): ...


class FArrayCoordinateSystem2D(CoordinateSystem2D, FArrayCoordinateSystem): ...


class FArrayCoordinateSystem3D(CoordinateSystem3D, FArrayCoordinateSystem): ...


# ----------------------------------------------------------------------
#   SPATIAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class SpatialCoordinateSystem(CoordinateSystem):
    """A coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[tx.List[SpatialAxis]] = None


class SpatialCoordinateSystem2D(CoordinateSystem2D, SpatialCoordinateSystem):
    """A 2D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_2SpatialAxes] = (SpatialAxis(), SpatialAxis())


class SpatialCoordinateSystem3D(CoordinateSystem3D, SpatialCoordinateSystem):
    """A 3D coordinate system, whose axes have spatial meaning."""

    axes: tx.Optional[_3SpatialAxes] = (
        SpatialAxis(),
        SpatialAxis(),
        SpatialAxis(),
    )

    axes: tx.Optional[_3SpatialAxes] = (
        SpatialAxis(),
        SpatialAxis(),
        SpatialAxis(),
    )


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """A coordinate system for (unitless) 2D pixel grids."""

    name: tx.Optional[str] = "pixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpatialAxis(name="dim0", unit=None),
        SpatialAxis(name="dim1", unit=None),
    )


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) 3D voxel grids."""

    name: tx.Optional[str] = "voxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpatialAxis(name="dim0", unit=None),
        SpatialAxis(name="dim1", unit=None),
        SpatialAxis(name="dim2", unit=None),
    )


class CPixelCoordinateSystem(PixelCoordinateSystem, CArrayCoordinateSystem2D):
    """A coordinate system for (unitless) C-ordered 2D pixel grids."""

    name: tx.Optional[str] = "cpixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="i", unit=None),
    )


class FPixelCoordinateSystem(PixelCoordinateSystem, FArrayCoordinateSystem2D):
    """A coordinate system for (unitless) F-ordered 2D pixel grids."""

    name: tx.Optional[str] = "fpixel"
    axes: tx.Optional[_2SpatialAxes] = (
        SpatialAxis(name="i", unit=None),
        SpatialAxis(name="j", unit=None),
    )


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) C-ordered 3D voxel grids."""

    name: tx.Optional[str] = "cvoxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpatialAxis(name="k", unit=None),
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="i", unit=None),
    )


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """A coordinate system for (unitless) F-ordered 3D voxel grids."""

    name: tx.Optional[str] = "fvoxel"
    axes: tx.Optional[_3SpatialAxes] = (
        SpatialAxis(name="i", unit=None),
        SpatialAxis(name="j", unit=None),
        SpatialAxis(name="k", unit=None),
    )


# ----------------------------------------------------------------------
#   ANATOMICAL COORDINATE SYSTEMS
# ----------------------------------------------------------------------


class RASCoordinateSystem(SpatialCoordinateSystem3D):
    # Used in NIfTI files, and many others.
    name: str = "RAS"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.PosteriorToAnteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (_axes.R, _axes.A, _axes.S)


class LPSCoordinateSystem(SpatialCoordinateSystem3D):
    # Used in ITK (and therefore also ANTs, Slicer, etc.)
    name: str = "LPS"
    axes: tx.Tuple[
        _axes.RightToLeftAxis,
        _axes.AnteriorToPosteriorAxis,
        _axes.InferiorToSuperiorAxis,
    ] = (_axes.L, _axes.P, _axes.S)


class RSACoordinateSystem(SpatialCoordinateSystem3D):
    # Used in some (rare) Freesurfer LTAs.
    name: str = "RSA"
    axes: tx.Tuple[
        _axes.LeftToRightAxis,
        _axes.InferiorToSuperiorAxis,
        _axes.PosteriorToAnteriorAxis,
    ] = (_axes.R, _axes.S, _axes.A)


class FRASCoordinateSystem(RASCoordinateSystem, FVoxelCoordinateSystem):
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


def get_missing(c1: CoordinateSystem, c2: CoordinateSystem):
    """
    Find all axes in c1 that don't match an axis in c2

    Parameters
    ----------
    c1: CoordinateSystem
        The coordinate system containing axes that we would like to see if are missing from c2
    c2: CoordinateSystem
        The coordinate system that we want to see if there are any missing axes in

    Returns
    -------
    list[Axis]
        all axes in c1 that have no match in c2


    """

    if c1 is None or c1.axes is None or c2 is None or c2.axes is None:
        return []
    missing = []
    for i in range(len(c1.axes)):
        found = False
        for j in range(len(c2.axes)):
            if same_axis_type(c1.axes[i], c2.axes[j]):
                found = True
        if not found:
            missing.append(c1.axes[i])
    return missing
