"""FSL FNIRT non-linear transforms.

FNIRT stores either a deformation field or a spline coefficient field in a
NIfTI file, and the intent code of the file tells the two apart. A single
reader handles both kinds of field.
"""

__all__ = ["FnirtWarpField"]

from ._base import FnirtWarpField

# Importing the private converters registers them with the data model,
# so that a conversion into one of these formats, such as
# `t.to(Format)`, fails with a reason instead of building a wrong
# object.
from . import _converters  # noqa: E402, F401  isort: skip
