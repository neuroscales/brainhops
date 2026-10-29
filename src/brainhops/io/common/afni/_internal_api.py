"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.afni` from this module, and not from the modules
that define them. This module only re-exports these names. No module of
the package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "afni_cardinal_matrix",
    "afni_dataset_files",
    "afni_geometry_from_matrix",
    "afni_view",
    "AFNI_VIEWS",
    "afni_voxel_to_dicom",
    "afni_world",
    "afni_written_files",
    "brick_code",
    "brick_dtype",
    "brik_proxy",
    "BrikProxy",
    "declined",
    "decode_bricks",
    "DICOM_TO_RAS",
    "encode_bricks",
    "format_attributes",
    "looks_like_head",
    "no_options",
    "read_afni",
    "reads_brik",
    "sniffed_raw",
    "sniffed_raw_at",
    "stores_brik",
    "whole_brik",
    "write_brik",
]

from ._constants import AFNI_VIEWS, DICOM_TO_RAS
from ._data import (
    BrikProxy,
    brick_code,
    brick_dtype,
    brik_proxy,
    decode_bricks,
    encode_bricks,
    read_afni,
    reads_brik,
    stores_brik,
    whole_brik,
    write_brik,
)
from ._files import afni_dataset_files, afni_written_files
from ._geometry import (
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_view,
    afni_voxel_to_dicom,
    afni_world,
)
from ._raw import (
    declined,
    format_attributes,
    looks_like_head,
    no_options,
    sniffed_raw,
    sniffed_raw_at,
)
