__all__ = [
    "LTACoordinateSystem",
    "LTAVoxelSystem",
    "LTAScaledSystem",
    "LTAPhysicalSystem"
]

# externals
import typing_extensions as _tx

# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import orientation as _orientation
from brainhops.datamodel import systems as _systems

from ._matrix_utils import _get_orient

# local
from ._struct import LTAStruct

# type hints
_3SpatialAxes = _tx.Tuple[
    _axes.SpatialAxis,
    _axes.SpatialAxis,
    _axes.SpatialAxis,
]


def _make_axes(
    names: _tx.Tuple[str, str, str],
    unit: _tx.Optional[str] = None,
    orientation: _tx.Optional[str] = None
) -> _3SpatialAxes:
    if orientation:
        orientation = tuple(getattr(_orientation, o) for o in orientation)
    else:
        orientation = (None,) * 3
    return tuple(
        _axes.SpatialAxis(name=name, unit=unit, orientation=orient)
        for name, orient in zip(names, orientation)
    )


class LTACoordinateSystem(
    _systems.SpatialCoordinateSystem3D,
    reverse=False,  # We want `struct` to be the last field.
):
    """Base class for coordinate systems specific to LTA files.

    Concrete subclasses
    -------------------
    LTAVoxelSystem
        Voxel space (unitless) of a volume.
        Coordinate (0,0,0) is the center of the first (corner) voxel.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    LTAScaledSystem
        Scaled voxel space (in mm) of a volume.
        Coordinate (0,0,0) is the center of the first (corner) voxel.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    LTAPhysicalSystem
        Physical space of a volume (source or destination).
        Coordinate (0,0,0) is the center of volume.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    """
    ...


class LTAVoxelSystem(LTACoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Voxel space (unscaled) of a volume (source or destination)."""

    name: str = "voxel"
    axes:  _3SpatialAxes = _make_axes(("i", "j", "k"))
    struct: _tx.Optional[LTAStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LTAStruct.VolumeInfo,
        names: _tx.Tuple[str, str, str] = ("i", "j", "k")
    ) -> _tx.Self:
        return cls(
            name=struct.filename or struct.NAME,
            axes=_make_axes(names, orientation=_get_orient(struct)),
            struct=struct
        )


class LTAScaledSystem(LTACoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Voxel space (scaled) of a volume (source or destination)."""

    name: str = "scaled"
    axes: _3SpatialAxes = _make_axes(("x", "y", "z"), unit="mm")
    struct: _tx.Optional[LTAStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LTAStruct.VolumeInfo,
        names: _tx.Tuple[str, str, str] = ("x", "y", "z")
    ) -> _tx.Self:
        return cls(
            name=struct.filename or struct.NAME,
            units="mm",
            axes=_make_axes(names, orientation=_get_orient(struct)),
            struct=struct
        )


class LTAPhysicalSystem(LTACoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Physical space of a volume (source or destination).

    This is the scaled voxel space, with an additional shift such that
    the origin is at the center of the volume rather than the corner.
    """

    name: str = "physical"
    axes: _3SpatialAxes = _make_axes(("x", "y", "z"), unit="mm")
    struct: _tx.Optional[LTAStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LTAStruct.VolumeInfo,
        names: _tx.Tuple[str, str, str] = ("x", "y", "z")
    ) -> _tx.Self:
        return cls(
            name=struct.filename or struct.NAME,
            units="mm",
            axes=_make_axes(names, orientation=_get_orient(struct)),
            struct=struct
        )
