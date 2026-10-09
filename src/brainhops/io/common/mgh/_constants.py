"""The constants of the MGH/MGZ file layout."""

# stdlib
import struct

# internals
from brainhops.datamodel.axes import Axis

MGH_HEADER_SIZE = 284
"""Size in bytes of the fixed MGH header; the voxels start right after."""

MGH_FOOTER_SIZE = 20
"""Size in bytes of the MRI-parameter footer (five big-endian floats)."""

# The leading fields of the header, enough to recognise a file:
# version, 4 dims, type, dof (all int32), goodRASFlag (int16).
_PREFIX = struct.Struct(">7ih")

# Voxel types an MGH file can hold, by code: UCHAR, INT, FLOAT, SHORT.
_MGH_TYPES = {0: 1, 1: 4, 3: 4, 4: 2}

# The four axes of an MGH volume, in storage (F) order.
_MGH_AXES = [
    Axis("x", "space"),
    Axis("y", "space"),
    Axis("z", "space"),
    Axis("t", "time"),
]

_MRI_PARAMS = ("tr", "flip_angle", "te", "ti", "fov")
"""The footer fields, as `nibabel` names them."""
