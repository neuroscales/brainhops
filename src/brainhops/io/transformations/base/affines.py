"""Format-independent affine transformations between the standard voxel,
RAS and LPS coordinate systems."""

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class VoxelToRAS(_xforms.Affine):
    """Affine transformation from voxel space to RAS space."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class RASToVoxel(_xforms.Affine):
    """Affine transformation from RAS space to voxel space."""

    _input: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()

    # The two halves of the pair map the same spaces in opposite
    # directions, so each one's inverse is the other. Declaring it here,
    # on the half defined second, pairs them both ways.
    _reverseof: tx.ClassVar[type] = VoxelToRAS


class VoxelToLPS(_xforms.Affine):
    """Affine transformation from voxel space to LPS space."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()


class LPSToVoxel(_xforms.Affine):
    """Affine transformation from LPS space to voxel space."""

    _input: _systems.CoordinateSystem = _systems.LPSCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()

    _reverseof: tx.ClassVar[type] = VoxelToLPS
