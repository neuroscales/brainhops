"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine from the scaled millimetres of the
moving image to those of the reference.
"""

__all__ = ["FlirtMatrixParser", "FlirtTransform"]

from ._parser import FlirtMatrixParser
from ._xform import FlirtTransform
