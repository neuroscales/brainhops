from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms


class RASToWarpField(_xforms.Affine):
    """Affine transformation from reference world (RAS) to an FNIRT warp grid.

    The warp grid is the grid on which the warp field is stored: the voxel
    grid of the reference image for a dense deformation field, or the coarse
    grid of B-spline knots for a coefficient field. Coordinates on the warp
    grid are the positions at which the spline basis of the warp is
    evaluated.
    """

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()


class WarpFieldToRAS(_xforms.Affine):
    """Affine transformation from an FNIRT warp grid to moving world (RAS).

    The affine carries a warped position on the warp grid into the world
    coordinates of the moving image. It folds in the scaled-mm geometry of
    the moving image and the initial FLIRT affine.
    """

    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()
