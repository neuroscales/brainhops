"""The coordinate systems of a BigDataViewer dataset and the matrices of
its registrations and resolution levels.

The conventions are described in [`brainhops.io.common.bdv`][]. Voxel
coordinates are in the order x, y, z, with integers at the centers of
the voxels, and the global space has no unit.
"""

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.units import SpaceUnit, Unit, is_physicalunit

# this format
from ._raw import BdvRaw

_AXES = ("x", "y", "z")
"""The names of the axes of a view, in the order of ImgLib2."""

VOXEL = "voxel"
"""The name of the voxel system of a view."""

PHYSICAL = "physical"
"""The name of the system of a view scaled by its voxel size."""

GLOBAL = "global"
"""The name of the global system of a dataset, as BigDataViewer names it."""


def voxel_system() -> CoordinateSystem:
    """Return the voxel system of a view, with axes x, y and z."""
    axes = [Axis(name, "space", unit="index") for name in _AXES]
    return CoordinateSystem(name=VOXEL, axes=axes, order="F")


def global_system() -> CoordinateSystem:
    """Return the global system of a dataset, whose axes have no unit."""
    axes = [Axis(name, "space") for name in _AXES]
    return CoordinateSystem(name=GLOBAL, axes=axes)


def physical_system(unit: tx.Optional[str]) -> CoordinateSystem:
    """Return the system of a view scaled by its voxel size.

    The axes have the unit of the voxel size when it is a unit of
    length (see [`space_unit`][]), and no unit otherwise.
    """
    unit = space_unit(unit)
    axes = [Axis(name, "space", unit=unit) for name in _AXES]
    return CoordinateSystem(name=PHYSICAL, axes=axes)


def space_unit(text: tx.Optional[str]) -> tx.Optional[SpaceUnit]:
    """Return the unit of length that the unit of a voxel size names.

    Fiji writes the unit of the calibration of the image, such as `µm`,
    `um`, `micron` or `nm`, and `pixel` for an image without calibration.
    A unit that is not a physical unit of length, such as `pixel`, gives
    `None`. So does `px`, which the unit parser would otherwise read as
    the CSS pixel of 1/96 inch, while Fiji means an uncalibrated pixel.
    """
    if not text:
        return None
    try:
        unit = Unit(text)
    except ValueError:
        return None
    if not isinstance(unit, SpaceUnit) or not is_physicalunit(unit):
        return None
    if str(unit) == "css_pixel":
        return None
    return unit


def affine_matrix(values: tx.Sequence[float]) -> np.ndarray:
    """Return the `(4, 4)` matrix of the twelve values of a transform.

    The values are the first three rows of the matrix, row after row.
    """
    matrix = np.eye(4)
    matrix[:3] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return matrix


def registration_matrix(registration: BdvRaw.Registration) -> np.ndarray:
    """Return the `(4, 4)` matrix from the voxels of a view to the global
    space.

    The matrix is the product of the matrices of the transforms in the
    order of the file, as `ViewRegistration.updateModel` concatenates
    them, so the last transform is applied first.

    Raises
    ------
    ValueError
        If a transform is generic, since its values cannot be read.
    """
    matrix = np.eye(4)
    for item in registration.transforms:
        if item.affine is None:
            raise ValueError(
                f"The view of time point {registration.timepoint} and setup "
                f"{registration.setup} has a generic transform, which "
                f"brainhops cannot read."
            )
        matrix = matrix @ affine_matrix(item.affine)
    return matrix


def level_matrix(resolution: tx.Sequence[float]) -> np.ndarray:
    """Return the `(4, 4)` matrix from the voxels of a level to the voxels
    of the full resolution.

    A level downsampled by `f` along an axis places its voxel `i` at
    `f * i + (f - 1) / 2`, the center of the block of voxels that it
    averages (`MipmapTransforms.getMipmapTransformDefault`).
    """
    factors = np.asarray(resolution, dtype=np.float64)
    matrix = np.diag(np.append(factors, 1.0))
    matrix[:3, 3] = (factors - 1.0) / 2.0
    return matrix
