"""FSL FNIRT non-linear transforms.

FNIRT stores either a deformation field or a spline coefficient field in a
NIfTI file, and the intent code of the file tells the two apart. A single
reader handles both kinds of field.
"""

__all__ = ["FnirtWarpField"]

from ._base import FnirtWarpField
