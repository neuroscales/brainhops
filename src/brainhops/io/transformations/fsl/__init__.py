"""Readers of FSL transformation formats.

FSL expresses transformations in scaled-mm coordinates, which are voxel
indices scaled by the pixel size, with x flipped when the voxel-to-world
affine has a positive determinant. FLIRT stores a linear transformation
as a `.mat` matrix, and FNIRT stores a non-linear transformation as a
NIfTI warp field or coefficient field.
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
