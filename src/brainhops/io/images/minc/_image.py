import numpy as np
import typing_extensions as tx
from bagof.magic import replace

from brainhops._core.dependencies import HAS_H5PY
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import (
    SpaceUnit,
    TimeUnit,
    Unit,
    is_physicalunit,
)
from brainhops.io.base._base import register_format
from brainhops.io.common.minc import MincDimension, MincParser
from brainhops.io.images.base import ImageFormat

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_WORLD = "world"
"""Name of the MINC world (RAS) space, the preferred world space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_UNIT_TYPES = {"space": SpaceUnit, "time": TimeUnit}


class MincImage(MincParser, ImageFormat, SingleScaleImage):
    """An image stored in a MINC file, of version 1 or 2.

    This class answers to the format hint `"minc"` and reads both
    versions. It hands the file to [`Minc1Image`][] or [`Minc2Image`][],
    which are the classes registered as formats. The voxels are read with
    nibabel and scaled to real values. They are in Fortran order, so a
    file with dimensions `zspace, yspace, xspace` gives data indexed
    `(x, y, z)`.

    The transformations are, in order, a [`Scaling`][] to `"physical"` by
    the absolute step of each dimension, in its unit (millimeters by
    default for spatial dimensions), and the preferred [`Affine`][] to the
    RAS space `"world"`, which moves along the direction cosines of each
    spatial dimension by its signed step from `start`. The affine is
    recorded only for volumes with three spatial dimensions. MINC files
    cannot be written.

    !!! note "Why the bases are in this order"
        As for [`NiftiImage`][brainhops.io.images.nifti.NiftiImage],
        [`SingleScaleImage`][] comes last because its `data` field has no
        default, and [`MincParser`][] comes first so that its lazy
        properties take precedence.
    """

    @property
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the dimensions.

        Transformations that were set explicitly are returned as they are.
        """
        if getattr(self, "_transformations", None):
            return self._transformations
        if not self.dimensions:
            return list(getattr(self, "_transformations", None) or [])
        return _minc_to_transformations(self)

    @transformations.setter
    def transformations(self, value: tx.List[Transformation]) -> None:
        self._transformations = value


@register_format
class Minc1Image(MincImage):
    """An image read from a MINC1 (NetCDF classic) file.

    The file may be gzipped. Besides `"minc"`, this class answers to the
    hints `"minc1"` and `"minc.1"`. See [`MincImage`][].
    """

    HINTS = ("minc1", "1")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc", ".mnc.gz")
    VERSION: tx.ClassVar[int] = 1


class Minc2Image(MincImage):
    """An image read from a MINC2 (HDF5) file, which needs `h5py`.

    Besides `"minc"`, this class answers to the hints `"minc2"` and
    `"minc.2"`. See [`MincImage`][].
    """

    HINTS = ("minc2", "2")
    VERSION: tx.ClassVar[int] = 2


MincImage.VARIANTS = (Minc1Image, Minc2Image)

# Without h5py, MINC2 is not registered, and a request for it by hint
# reports which package to install instead (see the package `__init__`).
if HAS_H5PY:
    register_format(Minc2Image)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _unit(dimension: MincDimension, axis_type: tx.Optional[str]) -> tx.Any:
    """Return the unit of the step of a dimension.

    The unit is the `units` of the dimension if it is a physical unit of
    the axis type, and otherwise millimeters for a spatial dimension and
    `None` for any other.
    """
    kind = _UNIT_TYPES.get(axis_type)
    if kind is None:
        return None
    unit = None
    if dimension.units:
        try:
            unit = Unit(dimension.units)
        except ValueError:
            unit = None
        if unit is not None and (
            not isinstance(unit, kind) or not is_physicalunit(unit)
        ):
            unit = None
    if unit is None and axis_type == "space":
        unit = "mm"
    return unit


def _minc_to_transformations(image: MincParser) -> tx.List[Transformation]:
    """Return the voxel-to-physical and voxel-to-world transformations."""
    voxel_space = image.system
    axes = list(voxel_space.axes)
    dimensions = list(reversed(image.dimensions))

    phys_axes = [
        replace(axis, unit=_unit(dim, axis.type))
        for axis, dim in zip(axes, dimensions)
    ]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    scale = [abs(dim.spacing) for dim in dimensions]
    xforms = [Scaling(input=voxel_space, output=phys_space, scale=scale)]

    vox2world = image.vox2world
    if vox2world is None:
        return xforms
    keep = [i for i, dim in enumerate(dimensions) if dim.is_spatial]
    unit = phys_axes[keep[0]].unit
    world_axes = [
        Axis(
            name,
            "space",
            unit=unit,
            orientation=Orientation(type="anatomical", value=value),
        )
        for name, value in _RAS_ORIENTATION.items()
    ]
    world_space = CoordinateSystem(name=_WORLD, axes=world_axes)
    # Non-spatial voxel axes, such as time, have zero columns in the
    # matrix, so they do not move a world point.
    matrix = np.zeros((3, len(axes) + 1))
    matrix[:, keep] = vox2world[:3, :3]
    matrix[:, -1] = vox2world[:3, 3]
    xforms.append(Affine(input=voxel_space, output=world_space, matrix=matrix))
    return xforms
