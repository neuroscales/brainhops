"""Format-independent fields of world coordinates."""

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
