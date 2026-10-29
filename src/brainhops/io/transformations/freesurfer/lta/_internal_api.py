"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.freesurfer.lta` from this module, and
not from the modules that define them. This module only re-exports these
names. No module of the package imports it, so it cannot create a
circular import inside the package. The rules are explained in
`spec/design/internal-api.md`.
"""

__all__ = [
    "check_tkr",
    "get_tkr2tkr",
    "lta_written",
    "volume_geometry",
    "volume_info",
]

from ._converters import check_tkr
from ._matrix_utils import get_tkr2tkr, volume_geometry, volume_info
from ._xforms import lta_written
