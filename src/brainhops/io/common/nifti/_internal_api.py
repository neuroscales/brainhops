"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.common.nifti` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "apply_like",
    "apply_overrides",
    "best_affine",
    "embed_nifti_affine",
    "has_nifti_magic",
    "header_of",
    "image_with_geometry",
    "keep_given_fields",
    "NIFTI_FIELD_INTENTS",
    "nifti_intent",
    "NIFTI_INTENT_DISPVECT",
    "nifti_intent_name",
    "NIFTI_INTENT_NAME_ITK_COORDINATES",
    "NIFTI_INTENT_NAME_ITK_DISPLACEMENT",
    "NIFTI_INTENT_NAME_MAPPING",
    "NIFTI_INTENT_NAME_NIFTYREG",
    "NIFTI_INTENT_NONE",
    "NIFTI_INTENT_VECTOR",
    "NIFTI_AXES",
    "nifti_shape",
    "nifti_to_axes",
    "nifti_to_unit",
    "NIFTI_XCODES",
    "NiftiObject",
    "NiftiProxy",
    "record_image",
    "scale_spatial",
    "sniffed_header",
    "split_options",
    "storage_override",
    "stored_parts",
    "unit_scale",
    "voxel_to_ras",
    "written_nifti",
]

from ._constants import (
    NIFTI_AXES,
    NIFTI_FIELD_INTENTS,
    NIFTI_INTENT_DISPVECT,
    NIFTI_INTENT_NAME_ITK_COORDINATES,
    NIFTI_INTENT_NAME_ITK_DISPLACEMENT,
    NIFTI_INTENT_NAME_MAPPING,
    NIFTI_INTENT_NAME_NIFTYREG,
    NIFTI_INTENT_NONE,
    NIFTI_INTENT_VECTOR,
    NIFTI_XCODES,
)
from ._data import NiftiProxy
from ._files import has_nifti_magic
from ._geometry import (
    embed_nifti_affine,
    image_with_geometry,
    scale_spatial,
    unit_scale,
    voxel_to_ras,
)
from ._header import (
    NiftiObject,
    apply_like,
    apply_overrides,
    best_affine,
    header_of,
    keep_given_fields,
    nifti_intent,
    nifti_intent_name,
    nifti_shape,
    nifti_to_axes,
    record_image,
    split_options,
    storage_override,
    stored_parts,
    written_nifti,
)
from ._raw import sniffed_header
from ._units import nifti_to_unit
