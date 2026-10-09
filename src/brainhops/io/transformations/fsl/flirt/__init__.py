"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine that maps the scaled millimetre
coordinates of the moving image to the scaled millimetre coordinates of the
reference image.
"""

__all__ = ["FlirtMatrixReader", "FlirtTransform"]

from ._parser import FlirtMatrixReader
from ._xform import FlirtTransform
