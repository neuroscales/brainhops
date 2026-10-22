"""FSL FNIRT non-linear transforms.

FNIRT stores either a deformation field (`--fout`) or a spline
coefficient field (`--cout`) in a NIfTI file, and the intent code of the
file tells the two apart. A single reader handles both kinds of field,
and it writes each back as the kind that it read.
"""

__all__ = ["FnirtWarpField"]

from ._base import FnirtWarpField

# Importing the private converters registers them with the data model,
# so that a transformation converts into a FNIRT warp when it is given
# the reference and moving images.
from . import _converters  # noqa: E402, F401  isort: skip
