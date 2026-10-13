from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems

# ITK works in LPS millimetres, but the oriented axes of the data model leave
# their unit unspecified, so the unit is given here.
_MM = "mm"


def _lps_axes(ndim: int) -> tuple:
    """Return LPS millimetre axes, followed by plain spatial axes if needed.

    The first three dimensions use the L, P and S axes, and any further
    dimension uses an unnamed spatial axis in millimetres.
    """
    lps = (_axes.L, _axes.P, _axes.S)
    axes = tuple(axis(unit=_MM) for axis in lps[:ndim])
    extra = max(0, ndim - len(lps))
    return axes + tuple(_axes.SpaceAxis(unit=_MM) for _ in range(extra))


def _make_system(ndim: int) -> _systems.SpatialCoordinateSystem:
    """Return the ITK spatial coordinate system with `ndim` dimensions."""
    if ndim == 2:
        return _systems.SpatialCoordinateSystem2D(axes=_lps_axes(2))
    elif ndim == 3:
        # The named LPS system is also the space of `LPSToVoxel` and
        # `VoxelToLPS`, so affine blocks and warp blocks refer to one space.
        return _systems.LPSmm()
    else:
        return _systems.SpatialCoordinateSystem(axes=_lps_axes(ndim))
