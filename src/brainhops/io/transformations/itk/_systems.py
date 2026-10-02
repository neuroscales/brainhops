# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems

# ITK places everything in LPS millimetres. The oriented axes of
# `brainhops.datamodel.axes` leave their unit unspecified, so it is given
# here.
_MM = "mm"


def _lps_axes(ndim: int) -> tuple:
    """The first `ndim` axes of LPS millimetres, and spatial ones after."""
    lps = (_axes.L, _axes.P, _axes.S)
    axes = tuple(axis(unit=_MM) for axis in lps[:ndim])
    extra = max(0, ndim - len(lps))
    return axes + tuple(_axes.SpaceAxis(unit=_MM) for _ in range(extra))


def _make_system(ndim: int) -> _systems.SpatialCoordinateSystem:
    """
    Create a spatial coordinate system with the specified number of dimensions.
    """
    if ndim == 2:
        return _systems.SpatialCoordinateSystem2D(axes=_lps_axes(2))
    elif ndim == 3:
        # ITK works in LPS, and the named system is the one that the
        # format-independent affines (`LPSToVoxel`, `VoxelToLPS`) use, so
        # an affine block and a warp block name the same space.
        return _systems.LPSmm()
    else:
        return _systems.SpatialCoordinateSystem(axes=_lps_axes(ndim))
