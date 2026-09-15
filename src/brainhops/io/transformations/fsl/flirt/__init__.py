"""FLIRT linear transformation matrices (`.mat`).

A FLIRT `.mat` file holds a `(4, 4)` affine that maps moving-image
scaled-mm coordinates to reference-image scaled-mm coordinates.
"""

__all__ = ["FLIRTMatrixParser", "FLIRTTransform"]

from ._parser import FLIRTMatrixParser
from ._xform import FLIRTTransform
