import typing_extensions as tx

from brainhops.datamodel import axes as _axes
from brainhops.datamodel.systems import SpatialCoordinateSystem3D


class FslCoordinateSystem(SpatialCoordinateSystem3D):
    """FSL "scaled millimetre" coordinate system of one image.

    Coordinates are voxel indices multiplied by the voxel sizes, with the
    x-axis flipped when the voxel-to-world affine has a positive determinant.
    Since both depend on the image, every image has its own FSL system.
    """

    name: str = "fsl"
    axes: tx.Tuple[_axes.SpaceAxis, _axes.SpaceAxis, _axes.SpaceAxis] = (
        _axes.SpaceAxis(name="x", unit="mm"),
        _axes.SpaceAxis(name="y", unit="mm"),
        _axes.SpaceAxis(name="z", unit="mm"),
    )
