"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.bdv` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "global_system",
    "level_count",
    "level_matrix",
    "physical_system",
    "read_dataset",
    "registration_matrix",
    "voxel_system",
]

from ._geometry import (
    global_system,
    level_matrix,
    physical_system,
    registration_matrix,
    voxel_system,
)
from ._hdf5 import level_count, read_dataset
