"""FSL FNIRT non-linear transforms.

FNIRT stores either a deformation field or a spline coefficient field in a
NIfTI file, and the intent code of the file tells the two apart. A single
reader handles both kinds of field.
"""

__all__ = ["FnirtWarpField"]

from ._base import FnirtWarpField

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses rather than relabels.
from . import _converters  # noqa: E402, F401  isort: skip
