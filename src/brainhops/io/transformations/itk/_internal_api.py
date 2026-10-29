"""Internal names of this package that other packages import.

The modules of other packages of `brainhops` import the internal names
of `brainhops.io.transformations.itk` from this module, and not from the
modules that define them. This module only re-exports these names. No
module of the package imports it, so it cannot create a circular import
inside the package. The rules are explained in `spec/design/internal-
api.md`.
"""

__all__ = [
    "block_of",
    "check_composite",
    "check_position",
    "check_systems",
    "decode_chain",
    "encode_chain",
    "euler_to_matrix",
    "is_composite",
    "ItkBSplineStruct",
    "make_system",
    "read_only",
    "same_chain",
    "TRANSFORM_TYPE_RE",
    "type_fields",
    "type_name",
]

from ._chains import (
    block_of,
    check_composite,
    check_position,
    check_systems,
    decode_chain,
    encode_chain,
    is_composite,
    read_only,
    same_chain,
    type_name,
)
from ._common import (
    TRANSFORM_TYPE_RE,
    ItkBSplineStruct,
    euler_to_matrix,
    type_fields,
)
from ._systems import make_system
