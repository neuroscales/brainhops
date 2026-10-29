"""The constants of the MINC1 and MINC2 containers."""

NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02")
"""The magic numbers of NetCDF classic files (MINC1)."""

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
"""The signature of HDF5 files (MINC2)."""

MINC2_ROOT = "minc-2.0"
"""The root group of a MINC2 file."""

SNIFF_SIZE = 1 << 16
"""
The number of leading bytes of a MINC1 file searched for dimension
names.
"""

SPATIAL_DIMENSIONS = ("xspace", "yspace", "zspace")
"""The spatial MINC dimensions, in world-axis order."""

DEFAULT_COSINES = {
    "xspace": (1.0, 0.0, 0.0),
    "yspace": (0.0, 1.0, 0.0),
    "zspace": (0.0, 0.0, 1.0),
}

# Other dimensions keep their own name and have no axis type.
AXES = {
    "xspace": ("x", "space"),
    "yspace": ("y", "space"),
    "zspace": ("z", "space"),
    "time": ("t", "time"),
}

OPTION_OWNER = "MINC reader"
"""What refuses unknown options, as the error message names it."""

DECLINE_MESSAGE = "Content is not a MINC file of this version."
"""The message of the error raised when a sniffer declines a stream."""
