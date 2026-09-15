# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class RASToWarpField(_xforms.Affine):
    """Affine from reference world (RAS) space to a FNIRT warp grid.

    The warp grid is the grid on which a FNIRT warp field is stored. For a
    dense deformation field this is the reference voxel grid. For a
    coefficient field this is the coarse B-spline knot grid. Coordinates
    in the warp grid are the positions at which the warp's spline basis is
    evaluated.
    """

    input: _systems.CoordinateSystem = _systems.RASCoordinateSystem()


class WarpFieldToRAS(_xforms.Affine):
    """Affine from a FNIRT warp grid to moving world (RAS) space.

    This affine carries the warped position, expressed on the warp grid,
    into the moving image world coordinates. It folds in the scaled-mm
    geometry of the moving image and the initial FLIRT affine.
    """

    output: _systems.CoordinateSystem = _systems.RASCoordinateSystem()
