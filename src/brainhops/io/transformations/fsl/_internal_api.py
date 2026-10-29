"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.fsl` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "best_affine",
    "FslAffineFormat",
    "FslTransformationFormat",
    "ImageGeometry",
    "pop_aliased",
    "RASToWarpField",
    "spatial_shape",
    "stored_repr",
    "WarpFieldToRAS",
]

from ._affines import ImageGeometry, best_affine, spatial_shape
from ._fields import RASToWarpField, WarpFieldToRAS
from ._formats import FslAffineFormat, FslTransformationFormat
from ._keywords import pop_aliased
from ._repr import stored_repr
