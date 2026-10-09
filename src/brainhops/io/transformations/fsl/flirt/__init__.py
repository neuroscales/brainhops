"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine that maps the scaled millimetre
coordinates of the moving image to the scaled millimetre coordinates of the
reference image.
"""

__all__ = ["FlirtMatrixReader", "FlirtTransform"]

from ._parser import FlirtMatrixReader
from ._xform import FlirtTransform

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses rather than relabels.
from . import _converters  # noqa: E402, F401  isort: skip
