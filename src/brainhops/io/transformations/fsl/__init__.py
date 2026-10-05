"""Readers for FSL transformation formats.

FSL expresses its transformations in *scaled-mm* coordinates: voxel
indices scaled by pixel size, with the x-axis flipped when the
voxel-to-world affine has a positive determinant. FLIRT stores a linear
transformation as a `.mat` matrix, and FNIRT stores a non-linear
transformation as a NIfTI warp field or coefficient field.
"""

__all__ = [
    "FslCoordinateSystem",
    "flirt",
    "fnirt",
    "FlirtTransform",
    "FnirtWarpField",
]

from . import flirt, fnirt
from ._systems import FslCoordinateSystem
from .flirt import FlirtTransform
from .fnirt import FnirtWarpField
