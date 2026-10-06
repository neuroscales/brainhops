"""FLIRT linear transformation matrices (`.mat`).

A FLIRT `.mat` file holds a `(4, 4)` affine that maps moving-image
scaled-mm coordinates to reference-image scaled-mm coordinates.

It stores no metadata. Its `metadata`, a [`FlirtMetadata`][], holds the
paths of the moving and reference images the reader was given, when they
were read from files (`moving`, `fixed`), in memory only.
"""

__all__ = ["FlirtMatrixParser", "FlirtMetadata", "FlirtTransform"]

from ._metadata import FlirtMetadata
from ._parser import FlirtMatrixParser
from ._xform import FlirtTransform
