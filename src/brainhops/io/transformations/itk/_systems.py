# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel import systems as _systems


def _make_system(ndim: int) -> _systems.SpatialCoordinateSystem:
    """
    Create a spatial coordinate system with the specified number of dimensions.
    """
    if ndim == 2:
        return _systems.SpatialCoordinateSystem2D(axes=(_axes.L, _axes.P))
    elif ndim == 3:
        return _systems.SpatialCoordinateSystem3D(axes=(_axes.L,
                                                        _axes.P,
                                                        _axes.S))
    else:
        return _systems.SpatialCoordinateSystem(
            axes=(_axes.L, _axes.P, _axes.S)[:ndim] +
            (_axes.Axis(),) * max(0, ndim - 3))
