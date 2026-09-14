"""FNIRT non-linear transformations stored in NIfTI files.

FNIRT writes its non-linear registration as either a deformation (warp)
field or a coefficient field, distinguished by the NIfTI intent code.
Both express displacements in FSL scaled-mm coordinates.
"""

__all__ = [
    "FNIRTTransformation",
    "FNIRTWarpField",
    "FNIRTDeformationField",
    "FNIRTCoefficientField",
]

from ._base import FNIRTTransformation, FNIRTWarpField
from ._coeff import FNIRTCoefficientField
from ._warp import FNIRTDeformationField
