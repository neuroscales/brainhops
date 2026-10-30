"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.nifti` from this module, and not from
the modules that define them. This module only re-exports these names.
No module of the package imports it, so it cannot create a circular
import inside the package. The rules are explained in
`spec/design/internal-api.md`.
"""

__all__ = [
    "consolidate_nifti",
    "coordinates_intent",
    "field_to_disk",
    "field_to_model",
    "intent_overrides",
    "NiftiImageBackedTransformation",
    "RAS",
]

from ._base import (
    NiftiImageBackedTransformation,
    consolidate_nifti,
    coordinates_intent,
    intent_overrides,
)
from ._converters import RAS
from ._views import field_to_disk, field_to_model
