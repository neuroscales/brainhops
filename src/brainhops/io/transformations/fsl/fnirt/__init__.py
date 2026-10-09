"""FSL FNIRT non-linear transforms.

FNIRT stores a deformation field or a spline coefficient field in a NIfTI file,
told apart by the intent code. One reader handles both.
"""

__all__ = ["FnirtWarpField"]

from ._base import FnirtWarpField
