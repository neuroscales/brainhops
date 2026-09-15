"""FNIRT non-linear transformations stored in NIfTI files.

FNIRT writes its non-linear registration as either a deformation (warp)
field or a coefficient field, distinguished by the NIfTI intent code. One
order-parameterized reader handles both. Both express displacements in FSL
scaled-mm coordinates.
"""

__all__ = ["FNIRTWarpField"]

from ._base import FNIRTWarpField
