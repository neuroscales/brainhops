from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
