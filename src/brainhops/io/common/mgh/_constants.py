"""The constants of the MGH/MGZ file layout."""

# stdlib
import struct

# internals
from brainhops.datamodel.axes import Axis

MGH_HEADER_SIZE = 284
"""The size in bytes of the fixed MGH header, after which the voxels start."""

MGH_FOOTER_SIZE = 20
"""The size in bytes of the footer, five big-endian floats."""

# The leading header fields, enough to recognise a file: the version, the
# four dimensions, the type and dof (int32), and goodRASFlag (int16).
_PREFIX = struct.Struct(">7ih")

# MGH voxel type codes: UCHAR, INT, FLOAT, SHORT.
_MGH_TYPES = {0: 1, 1: 4, 3: 4, 4: 2}

# The four MGH axes, in storage (Fortran) order.
_MGH_AXES = [
    Axis("x", "space"),
    Axis("y", "space"),
    Axis("z", "space"),
    Axis("t", "time"),
]

_MRI_PARAMS = ("tr", "flip_angle", "te", "ti", "fov")
"""The footer fields, as nibabel names them."""
