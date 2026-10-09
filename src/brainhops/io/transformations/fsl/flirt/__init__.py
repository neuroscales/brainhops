"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine that maps the scaled millimetre
coordinates of the moving image to the scaled millimetre coordinates of the
reference image.
"""

__all__ = ["FlirtMatrixReader", "FlirtTransform"]

from ._parser import FlirtMatrixReader
from ._xform import FlirtTransform

# Importing the private converters registers them with the data model,
# so that a conversion into one of these formats, such as
# `t.to(Format)`, fails with a reason instead of building a wrong
# object.
from . import _converters  # noqa: E402, F401  isort: skip
