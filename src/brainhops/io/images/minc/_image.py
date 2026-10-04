# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import replace

# internals
from brainhops._core.dependencies import HAS_H5PY
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientation import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import (
    SpaceUnit,
    TimeUnit,
    Unit,
    is_physicalunit,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.minc import MincDimension, MincParser
from brainhops.io.images.base import FileBasedImage

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_WORLD = "world"
"""Name of the MINC world (RAS) space, the preferred one."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_UNIT_TYPES = {"space": SpaceUnit, "time": TimeUnit}


class MincImage(MincParser, FileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a MINC file (MINC1 or MINC2).

    This class reads both versions and hands a file to
    [`Minc1Image`][brainhops.io.images.minc.Minc1Image] or
    [`Minc2Image`][brainhops.io.images.minc.Minc2Image], which are the
    classes registered for dispatch. It answers to the hint `"minc"`.

    The voxels are read with `nibabel`, scaled to real values (MINC's
    `image-min`/`image-max`, per slice or global), and presented in F
    order: the dimensions are reversed from the order the file lists
    them in, so that a `zspace, yspace, xspace` file reads as
    `(x, y, z)`. The voxel axes are named after the MINC dimensions
    (`xspace` -> `x`, `yspace` -> `y`, `zspace` -> `z`, `time` -> `t`),
    whatever their order in the file.

    The transformations are, in order:

    1. a [`Scaling`][brainhops.datamodel.transformations.Scaling] from
       voxels to the scaled voxel space `"physical"`: the absolute `step`
       of each dimension, in its `units` (millimetres by default for the
       spatial ones);
    2. the voxel-to-world affine, whose output is the RAS space named
       `"world"`: each spatial dimension runs along its direction
       cosines, scaled by its (signed) `step`, from its `start`. It is
       the preferred transformation. It is only recorded for a volume
       with the three spatial dimensions.

    MINC cannot be written: `nibabel` only reads it.

    !!! note "Why the bases are in this order"
        As for [`NiftiImage`][brainhops.io.images.nifti.NiftiImage]:
        `SingleScaleImage.data` has no default, so it comes last, and
        `MincParser` leads so that its lazy `data`/`system` properties
        win.
    """

    @property
    def transformations(self) -> tx.List[Transformation]:
        """The voxel-to-physical and voxel-to-world transformations
        recorded by the dimensions, decoded on first access unless set
        explicitly."""
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
    """
    An image that is encoded by a MINC1 file: a NetCDF classic file,
    possibly gzipped (`.mnc.gz`).

    It answers to the hints `"minc1"` and `"minc.1"`. See
    [`MincImage`][brainhops.io.images.minc.MincImage].
    """

    HINTS = ("minc1", "1")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc", ".mnc.gz")
    VERSION: tx.ClassVar[int] = 1


class Minc2Image(MincImage):
    """
    An image that is encoded by a MINC2 file: an HDF5 file with a
    `/minc-2.0` group. Reading it needs `h5py`.

    It answers to the hints `"minc2"` and `"minc.2"`. See
    [`MincImage`][brainhops.io.images.minc.MincImage].
    """

    HINTS = ("minc2", "2")
    VERSION: tx.ClassVar[int] = 2


MincImage.VARIANTS = (Minc1Image, Minc2Image)

# MINC2 is HDF5: without h5py, it is not registered, and asking for it by
# hint says what to install (see the package's `__init__`).
if HAS_H5PY:
    register_format(Minc2Image)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _unit(dimension: MincDimension, axis_type: tx.Optional[str]) -> tx.Any:
    """The unit of a dimension's `step`: its `units` when they are a
    unit of the axis' type, else millimetres for a spatial dimension and
    none for any other."""
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
    """Convert the dimensions of a MINC image to its list of
    transformations: voxel-to-physical scaling and voxel-to-world."""
    voxel_space = image.system
    axes = list(voxel_space.axes)
    dimensions = list(reversed(image.dimensions))

    # >> Physical space: the same axes, in the dimensions' units.
    phys_axes = [
        replace(axis, unit=_unit(dim, axis.type))
        for axis, dim in zip(axes, dimensions)
    ]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    scale = [abs(dim.spacing) for dim in dimensions]
    xforms = [Scaling(input=voxel_space, output=phys_space, scale=scale)]

    # >> World space: RAS, with the unit of the spatial dimensions.
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
    # The matrix has a column per voxel axis: the non-spatial ones (such
    # as time) do not move a point in the world.
    matrix = np.zeros((3, len(axes) + 1))
    matrix[:, keep] = vox2world[:3, :3]
    matrix[:, -1] = vox2world[:3, 3]
    xforms.append(Affine(input=voxel_space, output=world_space, matrix=matrix))
    return xforms
