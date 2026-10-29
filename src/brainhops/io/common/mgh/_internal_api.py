"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.mgh` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "compressed_name",
    "encode",
    "encode_mgh",
    "MGH_AXES",
    "MghProxy",
    "MRI_PARAMS",
    "read_mgh",
    "reads_mgh",
]

from ._constants import MGH_AXES, MRI_PARAMS
from ._data import MghProxy, reads_mgh
from ._metadata import encode
from ._raw import compressed_name, encode_mgh, read_mgh
