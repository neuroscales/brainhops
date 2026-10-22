__all__ = [
    "SpatialCoordinateSystem",
    "SpatialCoordinateSystem2D",
    "SpatialCoordinateSystem3D",
    "PixelCoordinateSystem",
    "VoxelCoordinateSystem",
    "CPixelCoordinateSystem",
    "CVoxelCoordinateSystem",
    "FPixelCoordinateSystem",
    "FVoxelCoordinateSystem",
    "MillimetricCoordinateSystem",
]
import typing_extensions as tx
from bagof.magic import Inherit

from .._sugar import get_axes
from ..axes import AxisTuple, SpaceAxis
from ..units import Unit
from .array_finite import (
    ArrayCoordinateSystem2D,
    ArrayCoordinateSystem3D,
    CArrayCoordinateSystem2D,
    CArrayCoordinateSystem3D,
    FArrayCoordinateSystem2D,
    FArrayCoordinateSystem3D,
)
from .base import CoordinateSystem, _VariadicAxes
from .dispatch import is_spatial
from .factories import SpaceArrayAxesFactory, SpaceAxesFactory
from .finite import CoordinateSystem2D, CoordinateSystem3D
from .physical import PhysicalCoordinateSystem

# typing
_2SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis]
_3SpatialAxes: tx.TypeAlias = AxisTuple[SpaceAxis, SpaceAxis, SpaceAxis]


class SpatialCoordinateSystem(CoordinateSystem, on={"axes": is_spatial}):
    """A coordinate system whose axes are all spatial.

    Parameters
    ----------
    axes : sequence[SpaceAxis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known.
    name : str, optional
        The name of the system, if any.
    """

    axes: Inherit[tx.Optional[_VariadicAxes[SpaceAxis]]] = None


# A spatial system whose axes count samples is both spatial and an array,
# and the pixel and voxel systems are reached through the spatial systems.
# An ordered pixel system does not require axes that count samples, because
# the order already says that the axes index an array. It is therefore
# excluded from automatic dispatch (`on=None`) and registered by hand with
# its parent classes.

_DIMS3D = ("dim0", "dim1", "dim2")
_DIMS2D = _DIMS3D[:2]
_VOXDIMS = ("i", "j", "k")
_PIXDIMS = _VOXDIMS[:2]


class SpatialCoordinateSystem2D(
    CoordinateSystem2D, SpatialCoordinateSystem, on={}, priority=1
):
    """
    A spatial coordinate system with two axes.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis], default=(SpaceAxis("dim0"), SpaceAxis("dim1"))
        The two axes, in order.
    name : str, optional
        The name of the system, if any.
    """  # noqa: E501

    axes: Inherit[_2SpatialAxes] = SpaceAxesFactory(*_DIMS2D)


class SpatialCoordinateSystem3D(
    CoordinateSystem3D, SpatialCoordinateSystem, on={}, priority=1
):
    """
    A spatial coordinate system with three axes.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis, SpaceAxis], default=(SpaceAxis("dim0"), SpaceAxis("dim1"), SpaceAxis("dim2"))
        The three axes, in order.
    name : str, optional
        The name of the system, if any.
    """  # noqa: E501

    axes: Inherit[_3SpatialAxes] = SpaceAxesFactory(*_DIMS3D)


class PixelCoordinateSystem(
    SpatialCoordinateSystem2D, ArrayCoordinateSystem2D
):
    """
    A coordinate system for two-dimensional pixel grids.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis], default=(SpaceAxis("dim0"), SpaceAxis("dim1"))
        The two axes, in order.
    name : str, default="pixel"
        The name of the system, if any.
    order : literal["C", "F"], optional
        The memory order of the indexed array, or `None` when unspecified.
    """  # noqa: E501

    name: ... = "pixel"
    axes: Inherit[_2SpatialAxes] = SpaceArrayAxesFactory(*_DIMS2D)


class VoxelCoordinateSystem(
    SpatialCoordinateSystem3D, ArrayCoordinateSystem3D
):
    """
    A coordinate system for three-dimensional voxel grids.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis, SpaceAxis], default=(SpaceAxis("dim0"), SpaceAxis("dim1"), SpaceAxis("dim2"))
        The two axes, in order.
    name : str, default="voxel"
        The name of the system, if any.
    order : literal["C", "F"], optional
        The memory order of the indexed array, or `None` when unspecified.
    """  # noqa: E501

    name: ... = "voxel"
    axes: Inherit[_3SpatialAxes] = SpaceArrayAxesFactory(*_DIMS3D)


@CArrayCoordinateSystem2D.register_polymorph(axes=is_spatial)
@PixelCoordinateSystem.register_polymorph(order="C")
class CPixelCoordinateSystem(
    PixelCoordinateSystem, CArrayCoordinateSystem2D, on=None
):
    """
    A coordinate system for C-ordered pixel grids, with axes j and i.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis], default=(SpaceAxis("j"), SpaceAxis("i"))
        The two axes, in order.
    name : str, default="cpixel"
        The name of the system, if any.
    order : literal["C"], default="C"
        The pinned memory order.
    """  # noqa: E501

    name: ... = "cpixel"
    order: Inherit[tx.Literal["C"]] = "C"
    axes: Inherit[_2SpatialAxes] = SpaceArrayAxesFactory(*_PIXDIMS[::-1])


@FArrayCoordinateSystem2D.register_polymorph(axes=is_spatial)
@PixelCoordinateSystem.register_polymorph(order="F")
class FPixelCoordinateSystem(
    PixelCoordinateSystem, FArrayCoordinateSystem2D, on=None
):
    """
    A coordinate system for F-ordered pixel grids, with axes i and j.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis], default=(SpaceAxis("i"), SpaceAxis("j"))
        The two axes, in order.
    name : str, default="fpixel"
        The name of the system, if any.
    order : literal["F"], default="F"
        The pinned memory order.
    """  # noqa: E501

    name: ... = "fpixel"
    order: Inherit[tx.Literal["F"]] = "F"
    axes: Inherit[_2SpatialAxes] = SpaceArrayAxesFactory(*_PIXDIMS)


class CVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, CArrayCoordinateSystem3D
):
    """
    A coordinate system for C-ordered voxel grids, with axes k, j and i.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis, SpaceAxis], default=(SpaceAxis("k"), SpaceAxis("j"), SpaceAxis("i"))
        The two axes, in order.
    name : str, default="cvoxel"
        The name of the system, if any.
    order : literal["C"], default="C"
        The pinned memory order.
    """  # noqa: E501

    name: ... = "cvoxel"
    axes: Inherit[_3SpatialAxes] = SpaceArrayAxesFactory(*_VOXDIMS[::-1])


class FVoxelCoordinateSystem(
    SpatialCoordinateSystem3D, FArrayCoordinateSystem3D
):
    """
    A coordinate system for F-ordered voxel grids, with axes i, j and k.

    Parameters
    ----------
    axes : tuple[SpaceAxis, SpaceAxis, SpaceAxis], default=(SpaceAxis("i"), SpaceAxis("j"), SpaceAxis("k"))
        The three axes, in order.
    name : str, default="fvoxel"
        The name of the system, if any.
    order : literal["F"], default="F"
        The pinned memory order.
    """  # noqa: E501

    name: ... = "fvoxel"
    axes: Inherit[_3SpatialAxes] = SpaceArrayAxesFactory(*_VOXDIMS)


class MillimetricCoordinateSystem(
    SpatialCoordinateSystem, PhysicalCoordinateSystem
):
    """A physical coordinate system in millimetres on every axis.

    An axis in another unit, or without a unit, raises a ValueError.

    Parameters
    ----------
    axes : sequence[SpaceAxis | {...}], optional
        The axes, in order, where `...` stands for zero or more axes
        about which nothing is known. All axes must have physical units.
    name : str, optional
        The name of the system, if any.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        for axis in get_axes(self):
            if axis is ... or axis.unit == Unit("mm"):
                continue
            name = type(self).__name__
            raise ValueError(
                f"{name} is in millimetres, so every one of its axes is "
                f"measured in mm. The axis {axis.name or axis.type!r} "
                f"carries {axis.unit!r}. Build an "
                f"{type(self).__mro__[1].__name__} (or a "
                f"PhysicalCoordinateSystem) for axes in another unit, "
                f"or with none."
            )
