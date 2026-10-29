"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops._core` from this module, and not from the modules that
define them. This module only re-exports these names. No module of the
package imports it, so it cannot create a circular import inside the
package. The rules are explained in `spec/design/internal-api.md`.
"""

__all__ = [
    "affine_expm",
    "affine_inv",
    "affine_logm",
    "affine_sqrtm",
    "alias",
    "apply_error_policy",
    "array_backend",
    "ArrayLike",
    "ArrayProtocol",
    "ArrayProxy",
    "axis_scales",
    "BackendLike",
    "cached_value",
    "CartesianIndex",
    "cast_output",
    "Clip",
    "clip_mask",
    "coeff2value_field",
    "compression_suffix",
    "copy_array",
    "creation_options",
    "da",
    "decompress",
    "DeviceLike",
    "differs",
    "DTypeLike",
    "enum_name",
    "error_policy",
    "expm",
    "external_stacklevel",
    "field_default",
    "FileOrContentLike",
    "float32_repr",
    "has_abczarr_driver",
    "has_dask",
    "HAS_H5PY",
    "HAS_NIBABEL",
    "HAS_OPENSLIDE",
    "HAS_PILLOW",
    "HAS_TENSORSTORE",
    "HAS_TIFFFILE",
    "IntEnum",
    "Internal",
    "inv",
    "InvalidatorInAttribute",
    "is_dask",
    "is_instance_or_subclass",
    "is_lazy",
    "is_same_file",
    "is_seekable",
    "lazyproperty",
    "logm",
    "matmul",
    "may_share_memory",
    "ndimage_backend",
    "normalize_clip",
    "npmatrix",
    "npvector",
    "open_compressed",
    "open_compressing",
    "open_decompressed",
    "openslide",
    "partial",
    "Path",
    "peekable_lines",
    "PLACEHOLDER",
    "policy_action",
    "preserve_position",
    "pull",
    "pull_axes",
    "pull_field",
    "renamed_attributes",
    "replace",
    "replace_directory",
    "replace_file",
    "require_positive",
    "resolve_backend",
    "sampling_dtype",
    "ScaledProxy",
    "Shape",
    "SharedOptions",
    "shortest_decimal",
    "singleton",
    "smart_replace",
    "smartproperty",
    "spline_matrix",
    "sqrtm",
    "StoredProxy",
    "stores",
    "StrEnum",
    "StridedProxy",
    "tifffile",
    "to_backend",
    "to_backend_of",
    "to_compact",
    "to_homogeneous",
    "to_numpy",
    "to_promoted_backend",
    "ts",
    "UNSET",
    "value2coeff_field",
]

from .affines import (
    axis_scales,
    matmul,
    to_compact,
    to_homogeneous,
)
from .affines import (
    expm as affine_expm,
)
from .affines import (
    inv as affine_inv,
)
from .affines import (
    logm as affine_logm,
)
from .affines import (
    sqrtm as affine_sqrtm,
)
from .backends import (
    BackendLike,
    DeviceLike,
    array_backend,
    copy_array,
    creation_options,
    has_dask,
    is_dask,
    is_lazy,
    may_share_memory,
    ndimage_backend,
    resolve_backend,
    to_backend,
    to_backend_of,
    to_numpy,
    to_promoted_backend,
)
from .compare import differs
from .compat import PLACEHOLDER, StrEnum, partial
from .dependencies import (
    HAS_H5PY,
    HAS_NIBABEL,
    HAS_OPENSLIDE,
    HAS_PILLOW,
    HAS_TENSORSTORE,
    HAS_TIFFFILE,
    da,
    has_abczarr_driver,
    openslide,
    tifffile,
    ts,
)
from .deprecation import external_stacklevel, renamed_attributes
from .enum import IntEnum, enum_name
from .errors import UNSET, apply_error_policy, error_policy, policy_action
from .linalg import expm, inv, logm, require_positive, sqrtm
from .magic import SharedOptions, field_default, replace, smart_replace, stores
from .numeric import float32_repr, shortest_decimal
from .path import (
    FileOrContentLike,
    Path,
    is_same_file,
    replace_directory,
    replace_file,
)
from .peek import peekable_lines
from .properties import (
    InvalidatorInAttribute,
    alias,
    cached_value,
    lazyproperty,
    smartproperty,
)
from .proxy import ArrayProxy, ScaledProxy, StoredProxy
from .singleton import singleton
from .streams import (
    compression_suffix,
    decompress,
    is_seekable,
    open_compressed,
    open_compressing,
    open_decompressed,
    preserve_position,
)
from .strided import StridedProxy
from .typing import (
    ArrayLike,
    ArrayProtocol,
    CartesianIndex,
    Clip,
    DTypeLike,
    Internal,
    Shape,
    is_instance_or_subclass,
    npmatrix,
    npvector,
)

# isort: split
# The module `bsplines` imports `brainhops.datamodel.enums`, which
# imports names of this module, so it is imported after the other
# modules.
from .bsplines import (
    cast_output,
    clip_mask,
    coeff2value_field,
    normalize_clip,
    pull,
    pull_axes,
    pull_field,
    sampling_dtype,
    spline_matrix,
    value2coeff_field,
)
