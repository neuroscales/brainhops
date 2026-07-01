from brainhops.datamodel import systems as _systems, transformations as _xforms


class RASCoordinatesField(_xforms.CoordinatesField):
    """Field of RAS coordinates."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class RASDisplacementField(_xforms.DisplacementField):
    """Field of RAS displacements."""

    input: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
