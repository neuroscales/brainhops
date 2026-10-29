"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.nifti` from this module, and not from
the modules that define them. This module only re-exports these names.
No module of the package imports it, so it cannot create a circular
import inside the package. The rules are explained in
`spec/design/internal-api.md`.
"""

__all__ = [
    "check_single_field",
    "consolidated_metadata",
    "field_to_disk",
    "field_to_model",
    "RAS",
]

from ._base import consolidated_metadata
from ._converters import RAS
from ._views import check_single_field, field_to_disk, field_to_model
