__all__ = [
    "LtaCoordinateSystem",
    "LtaVoxelSystem",
    "LtaScaledSystem",
    "LtaPhysicalSystem",
]

# externals
import typing_extensions as tx

# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import orientation as _orientation
from brainhops.datamodel import systems as _systems

# local
from ._matrix_utils import _get_orient
from ._struct import LtaStruct

# type hints
_3SpatialAxes = tx.Tuple[
    _axes.SpaceAxis,
    _axes.SpaceAxis,
    _axes.SpaceAxis,
]


def _make_axes(
    names: tx.Tuple[str, str, str],
    unit: tx.Optional[str] = None,
    orientation: tx.Optional[str] = None,
) -> _3SpatialAxes:
    if orientation:
        orientation = tuple(getattr(_orientation, o) for o in orientation)
    else:
        orientation = (None,) * 3
    return tuple(
        _axes.SpaceAxis(name=name, unit=unit, orientation=orient)
        for name, orient in zip(names, orientation)
    )


def _name(struct: LtaStruct.VolumeInfo) -> tx.Dict[str, str]:
    """The name a volume gives its system, as keyword arguments.

    The file name of the volume, or else the role of the volume in the
    transform (`"src"` or `"dst"`). A volume with neither -- a bare
    `VolumeInfo` with no file name -- gives nothing, and the system keeps
    the name of its class.
    """
    name = struct.filename or getattr(struct, "NAME", None)
    return {"name": name} if name else {}


# A voxel space counts samples; the scaled and physical spaces are in
# millimetres.
_INDEX = "index"
_MM = "mm"


class LtaCoordinateSystem(
    _systems.SpatialCoordinateSystem3D,
    reverse=False,  # We want `struct` to be the last field.
):
    """Base class for coordinate systems specific to LTA files.

    Concrete subclasses
    -------------------
    LtaVoxelSystem
        Voxel space (unitless) of a volume.
        Coordinate (0,0,0) is the center of the first (corner) voxel.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    LtaScaledSystem
        Scaled voxel space (in mm) of a volume.
        Coordinate (0,0,0) is the center of the first (corner) voxel.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    LtaPhysicalSystem
        Physical space of a volume (source or destination).
        Coordinate (0,0,0) is the center of volume.
        Axes correspond to the F-ordered dimensions of the volume,
        where the first axis is the fastest changing in memory.
    """

    ...


class LtaVoxelSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Voxel space (unscaled) of a volume (source or destination)."""

    name: tx.Optional[str] = "voxel"
    axes: _3SpatialAxes = _make_axes(("i", "j", "k"), unit=_INDEX)
    struct: tx.Optional[LtaStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LtaStruct.VolumeInfo,
        names: tx.Tuple[str, str, str] = ("i", "j", "k"),
    ) -> tx.Self:
        """Build the voxel system of the volume described by `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(
                names, unit=_INDEX, orientation=_get_orient(struct)
            ),
            struct=struct,
        )


class LtaScaledSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Voxel space (scaled) of a volume (source or destination)."""

    name: tx.Optional[str] = "scaled"
    axes: _3SpatialAxes = _make_axes(("x", "y", "z"), unit=_MM)
    struct: tx.Optional[LtaStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LtaStruct.VolumeInfo,
        names: tx.Tuple[str, str, str] = ("x", "y", "z"),
    ) -> tx.Self:
        """Build the scaled voxel system of the volume described by
        `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(names, unit=_MM, orientation=_get_orient(struct)),
            struct=struct,
        )


class LtaPhysicalSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Physical space of a volume (source or destination).

    This is the scaled voxel space, with an additional shift such that
    the origin is at the center of the volume rather than the corner.
    """

    name: tx.Optional[str] = "physical"
    axes: _3SpatialAxes = _make_axes(("x", "y", "z"), unit=_MM)
    struct: tx.Optional[LtaStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LtaStruct.VolumeInfo,
        names: tx.Tuple[str, str, str] = ("x", "y", "z"),
    ) -> tx.Self:
        """Build the physical system of the volume described by `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(names, unit=_MM, orientation=_get_orient(struct)),
            struct=struct,
        )
