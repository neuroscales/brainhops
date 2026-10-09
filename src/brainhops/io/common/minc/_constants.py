"""The constants of the MINC1 and MINC2 containers."""

_NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02")
"""The magic numbers of NetCDF classic files (MINC1)."""

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
"""The signature at the start of an HDF5 file (MINC2)."""

_MINC2_ROOT = "minc-2.0"
"""The root group of a MINC2 file."""

_SNIFF_SIZE = 1 << 16
"""How many leading bytes of a MINC1 file are searched for its
dimension names (the NetCDF header lists them first)."""

SPATIAL_DIMENSIONS = ("xspace", "yspace", "zspace")
"""The names of the spatial MINC dimensions, in world-axis order."""

_DEFAULT_COSINES = {
    "xspace": (1.0, 0.0, 0.0),
    "yspace": (0.0, 1.0, 0.0),
    "zspace": (0.0, 0.0, 1.0),
}

# The axes that the usual MINC dimensions stand for. Any other dimension
# keeps its own name and has no type.
_AXES = {
    "xspace": ("x", "space"),
    "yspace": ("y", "space"),
    "zspace": ("z", "space"),
    "time": ("t", "time"),
}
