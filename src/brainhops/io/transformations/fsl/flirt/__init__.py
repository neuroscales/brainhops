"""FLIRT linear transformation matrices (`.mat`).

A FLIRT `.mat` file holds a `(4, 4)` affine that maps moving-image
scaled-mm coordinates to reference-image scaled-mm coordinates.
"""

__all__ = ["FlirtMatrixParser", "FlirtMetadata", "FlirtTransform"]

from ._metadata import FlirtMetadata
from ._parser import FlirtMatrixParser
from ._xform import FlirtTransform
