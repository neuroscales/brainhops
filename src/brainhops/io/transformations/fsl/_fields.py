import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base.affines import RAS, check_endpoint


class RASToWarpField(_xforms.Affine):
    """Affine transformation from the reference RAS space to a warp grid.

    The warp grid is the grid on which an FNIRT warp field is stored. It is
    the voxel grid of the reference image for a dense deformation field, and
    the coarse grid of B-spline knots for a coefficient field. Coordinates on
    the warp grid are the positions at which the spline basis of the warp is
    evaluated.
    """

    _input: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        check_endpoint(self, "input", RAS)


class WarpFieldToRAS(_xforms.Affine):
    """Affine transformation from an FNIRT warp grid to the moving RAS space.

    The affine carries a warped position on the warp grid into the world
    coordinates of the moving image. It includes both the scaled-mm geometry
    of the moving image and the initial FLIRT affine.
    """

    _output: KwOnly[_systems.CoordinateSystem] = _systems.RASmm()

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        check_endpoint(self, "output", RAS)
