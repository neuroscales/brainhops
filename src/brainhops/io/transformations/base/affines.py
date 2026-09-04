from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel space to RAS space."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class RASToVoxel(_xforms.Affine):
    """Affine transformation from RAS space to voxel space."""

    input: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()


class VoxelToLPS(_xforms.Affine):
    """Affine transformation from voxel space to LPS space."""

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()


class LPSToVoxel(_xforms.Affine):
    """Affine transformation from LPS space to voxel space."""

    input: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()
    output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
