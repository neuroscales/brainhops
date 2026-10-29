"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.metadata` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "consolidated",
    "decode_object",
    "encode_changes",
    "encode_extra",
    "lost_fields",
    "metadata_options",
    "metadata_to_write",
    "overruled_fields",
    "requested_fields",
    "unsupported_values",
]

from ._base import (
    consolidated,
    lost_fields,
    metadata_options,
    metadata_to_write,
    overruled_fields,
    requested_fields,
    unsupported_values,
)
from ._json import decode_object, encode_changes, encode_extra
