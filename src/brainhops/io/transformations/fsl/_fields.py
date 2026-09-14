# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

from ._systems import FSLCoordinateSystem


class ScaledMMCoordinatesField(_xforms.CoordinatesField):
    """A field of FSL scaled-mm coordinates defined on a voxel grid.

    Each voxel of the grid holds the scaled-mm coordinate, in some image,
    that the voxel maps to. A FNIRT deformation field is read as one of
    these, defined on the reference voxel grid and holding moving-image
    scaled-mm coordinates.
    """

    input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    output: _systems.CoordinateSystem = FSLCoordinateSystem()
