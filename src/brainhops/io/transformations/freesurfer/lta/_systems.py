__all__ = [
    "LtaCoordinateSystem",
    "LtaVoxelSystem",
    "LtaScaledSystem",
    "LtaPhysicalSystem",
]

import typing_extensions as tx

from brainhops.datamodel import axes as _axes
from brainhops.datamodel import orientations as _orientation
from brainhops.datamodel import systems as _systems

from ._matrix_utils import _get_orient
from ._struct import LtaStruct

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
    """Return the name that a volume gives its system, as keyword arguments.

    The name is the file name of the volume, or else its role (`"src"` or
    `"dst"`). A bare `VolumeInfo` with neither gives nothing, and the system
    keeps the default name of its class.
    """
    name = struct.filename or getattr(struct, "NAME", None)
    return {"name": name} if name else {}


# Voxel coordinates count samples, whereas scaled and physical coordinates
# are in millimetres.
_INDEX = "index"
_MM = "mm"


class LtaCoordinateSystem(
    _systems.SpatialCoordinateSystem3D,
    reverse=False,  # `struct` must be the last field
):
    """Base class of the coordinate systems specific to LTA files.

    There are three concrete systems, which all have axes that follow the
    dimensions of the volume in Fortran order (the first axis is the fastest
    in memory). [`LtaVoxelSystem`][] is the unitless voxel space, whose origin
    is the center of the first corner voxel. [`LtaScaledSystem`][] is the
    voxel space scaled to millimetres, with the same origin.
    [`LtaPhysicalSystem`][] is the physical space of the source or
    destination volume, whose origin is the center of the volume.
    """

    ...


class LtaVoxelSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Unscaled voxel space of a source or destination volume."""

    name: tx.Optional[str] = "voxel"
    axes: _3SpatialAxes = _make_axes(("i", "j", "k"), unit=_INDEX)
    struct: tx.Optional[LtaStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LtaStruct.VolumeInfo,
        names: tx.Tuple[str, str, str] = ("i", "j", "k"),
    ) -> tx.Self:
        """Return the voxel system of the volume described by `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(
                names, unit=_INDEX, orientation=_get_orient(struct)
            ),
            struct=struct,
        )


class LtaScaledSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Voxel space of a volume, scaled to millimetres."""

    name: tx.Optional[str] = "scaled"
    axes: _3SpatialAxes = _make_axes(("x", "y", "z"), unit=_MM)
    struct: tx.Optional[LtaStruct.VolumeInfo] = None

    @classmethod
    def from_struct(
        cls,
        struct: LtaStruct.VolumeInfo,
        names: tx.Tuple[str, str, str] = ("x", "y", "z"),
    ) -> tx.Self:
        """Return the scaled system of the volume described by `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(names, unit=_MM, orientation=_get_orient(struct)),
            struct=struct,
        )


class LtaPhysicalSystem(LtaCoordinateSystem, _systems.FVoxelCoordinateSystem):
    """Physical space of a volume.

    The physical space is the scaled voxel space, shifted so that its origin
    lies at the center of the volume rather than at its corner.
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
        """Return the physical system of the volume described by `struct`."""
        return cls(
            **_name(struct),
            axes=_make_axes(names, unit=_MM, orientation=_get_orient(struct)),
            struct=struct,
        )
