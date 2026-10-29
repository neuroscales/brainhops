"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.hdf5` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "read_string",
]

from ._parsers import read_string
