"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.zarr` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "absolute_location",
    "as_node",
    "opened_node",
    "reads_inside",
    "reads_store",
    "store_chunks",
    "store_location",
    "write_array",
    "write_into",
    "ZarrNodeReaderWriter",
    "ZarrProxy",
]

from ._data import (
    ZarrProxy,
    reads_inside,
    reads_store,
    store_chunks,
    store_location,
    write_array,
    write_into,
)
from ._parsers import (
    ZarrNodeReaderWriter,
    absolute_location,
    as_node,
    opened_node,
)
