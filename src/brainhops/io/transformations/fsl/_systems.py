# externals
import typing_extensions as tx

# internals
from brainhops.datamodel import axes as _axes
from brainhops.datamodel.systems import SpatialCoordinateSystem3D


class FSLCoordinateSystem(SpatialCoordinateSystem3D):
    """The FSL "scaled-mm" coordinate system of an image.

    Coordinates are voxel indices scaled by the pixel sizes, with the
    x-axis flipped when the voxel-to-world affine has a positive
    determinant. FLIRT and FNIRT express their transformations in this
    coordinate system. Each image has its own scaled-mm system, because
    the scaling and the flip depend on that image's pixel sizes and
    shape.
    """

    name: str = "fsl"
    axes: tx.Tuple[_axes.SpatialAxis, _axes.SpatialAxis, _axes.SpatialAxis] = (
        _axes.SpatialAxis(name="x", unit="mm"),
        _axes.SpatialAxis(name="y", unit="mm"),
        _axes.SpatialAxis(name="z", unit="mm"),
    )
