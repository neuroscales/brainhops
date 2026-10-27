"""Readers and writers for SPM transformations.

SPM writes its deformation fields as NIfTI files of RAS coordinates,
whose names start with `y_` or `iy_`. They are read and written by
[`SpmCoordinateField`][], which holds the header of its file as
[`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata].
"""

__all__ = ["SpmCoordinateField"]

from ._fields import SpmCoordinateField

# Importing the private converters registers them with the data model,
# so that `t.to(SpmCoordinateField)` converts a transformation.
from . import _converters  # noqa: E402, F401  isort: skip


# -- deprecated names ----------------------------------------------------
# The old spellings of the coordinate fields (#547) still import, with a
# DeprecationWarning, for one release.
from brainhops._core.deprecation import (  # noqa: E402
    renamed_attributes as _renamed_attributes,
)

__getattr__ = _renamed_attributes(
    __name__,
    {
        "SpmCoordinatesField": "SpmCoordinateField",
    },
)
