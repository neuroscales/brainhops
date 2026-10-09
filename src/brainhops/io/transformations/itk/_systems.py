from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems

# The oriented axes of the data model leave the unit unspecified.
_MM = "mm"


def _lps_axes(ndim: int) -> tuple:
    """LPS millimetre axes, followed by extra spatial axes."""
    lps = (_axes.L, _axes.P, _axes.S)
    axes = tuple(axis(unit=_MM) for axis in lps[:ndim])
    extra = max(0, ndim - len(lps))
    return axes + tuple(_axes.SpaceAxis(unit=_MM) for _ in range(extra))


def _make_system(ndim: int) -> _systems.SpatialCoordinateSystem:
    """The ITK spatial coordinate system of a given dimensionality."""
    if ndim == 2:
        return _systems.SpatialCoordinateSystem2D(axes=_lps_axes(2))
    elif ndim == 3:
        # The space of `LPSToVoxel` and `VoxelToLPS`, shared with warp blocks.
        return _systems.LPSmm()
    else:
        return _systems.SpatialCoordinateSystem(axes=_lps_axes(ndim))
