"""Readers and writers for SPM transformations.

SPM writes its deformation fields as NIfTI files of RAS coordinates,
whose names start with `y_` or `iy_`. They are read and written by
[`SpmCoordinatesField`][], which holds the header of its file as
[`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata].
"""

__all__ = ["SpmCoordinatesField"]

from ._fields import SpmCoordinatesField

# Importing the private converters registers them with the data model,
# so that `t.to(SpmCoordinatesField)` converts a transformation.
from . import _converters  # noqa: E402, F401  isort: skip
