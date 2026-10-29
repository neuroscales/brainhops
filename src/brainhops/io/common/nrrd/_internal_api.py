"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.nrrd` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "axis_roles",
    "DATA_EXTENSIONS",
    "DECLINE_MESSAGE",
    "decode",
    "dtype_to_nrrd",
    "encode_data",
    "ENCODINGS",
    "format_float",
    "format_strings",
    "format_vectors",
    "image_order",
    "NrrdProxy",
    "OPTION_OWNER",
    "read_nrrd",
    "SPACE_NAMES",
    "SPACES",
]

from ._codecs import (
    dtype_to_nrrd,
    format_float,
    format_strings,
    format_vectors,
)
from ._constants import DATA_EXTENSIONS, ENCODINGS, SPACE_NAMES, SPACES
from ._data import NrrdProxy, decode, encode_data, read_nrrd
from ._layout import axis_roles, image_order
from ._raw import DECLINE_MESSAGE, OPTION_OWNER
