"""Format-independent fields of world coordinates."""

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class RasCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.RASmm()


class LpsCoordinatesField(_xforms.CoordinatesField):
    """Field of LPS coordinates."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.LPSmm()
