"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common` from this module, and not from the modules that
define them. This module only re-exports these names. No module of the
package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "arrange_voxel_to_ras",
    "Arrangement",
    "ArrayContainerError",
    "ArrayReader",
    "AxisLayout",
    "CsvArrayReader",
    "declared_axes",
    "embed_affine",
    "held_images",
    "held_transformations",
    "is_numeric_array",
    "Mat73ArrayReader",
    "MatArrayReader",
    "MatLegacyArrayReader",
    "NpyArrayReader",
    "NpzArrayReader",
    "own_level",
    "own_levels",
    "ras_conversion",
    "RAS_FROM_ORIENTATION",
    "read_mat73",
    "read_mat_legacy",
    "read_npy",
    "read_npz",
    "read_text_array",
    "read_text_rows",
    "reduce_to_affine",
    "same_transformations",
    "select_array",
    "TsvArrayReader",
    "TxtArrayReader",
]

from ._arrays import (
    ArrayContainerError,
    ArrayReader,
    CsvArrayReader,
    Mat73ArrayReader,
    MatArrayReader,
    MatLegacyArrayReader,
    NpyArrayReader,
    NpzArrayReader,
    TsvArrayReader,
    TxtArrayReader,
    is_numeric_array,
    read_mat73,
    read_mat_legacy,
    read_npy,
    read_npz,
    read_text_array,
    read_text_rows,
    select_array,
)
from ._geometry import (
    RAS_FROM_ORIENTATION,
    Arrangement,
    AxisLayout,
    arrange_voxel_to_ras,
    declared_axes,
    embed_affine,
    held_transformations,
    ras_conversion,
    reduce_to_affine,
    same_transformations,
)
from ._pyramids import held_images, own_level, own_levels
