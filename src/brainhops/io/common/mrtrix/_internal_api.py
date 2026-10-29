"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.mrtrix` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "DECLINE_MESSAGE",
    "dtype_to_mrtrix",
    "file_entry",
    "keeps_storage",
    "merged_keyval",
    "mrtrix_bytes",
    "MrtrixProxy",
    "OPTION_OWNER",
    "placed_record",
    "read_mrtrix",
    "reads_from",
    "split_voxel_to_scanner",
    "stored_voxels",
    "stores_as_recorded",
    "voxel_to_ras",
    "whole_file",
    "write_mrtrix",
    "writer_options",
    "writer_strides",
    "written_files",
]

from ._codecs import dtype_to_mrtrix
from ._data import MrtrixProxy, read_mrtrix, whole_file
from ._files import (
    file_entry,
    keeps_storage,
    merged_keyval,
    mrtrix_bytes,
    placed_record,
    reads_from,
    stored_voxels,
    stores_as_recorded,
    write_mrtrix,
    writer_options,
    writer_strides,
    written_files,
)
from ._geometry import split_voxel_to_scanner, voxel_to_ras
from ._raw import DECLINE_MESSAGE, OPTION_OWNER
