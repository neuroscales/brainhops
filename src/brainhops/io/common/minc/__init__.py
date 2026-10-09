"""
Reading of MINC files.

MINC, the volume format of the MINC toolkit, comes in two containers
that share the `.mnc` extension. A MINC1 file is a NetCDF classic file
whose `image` variable holds the voxels. A MINC2 file is an HDF5 file
whose voxels are the dataset `/minc-2.0/image/0/image`. In both, each
dimension is described by a variable of its own name, and the
dimensions are listed slowest first, in any order.

A dimension has a `start`, a `step`, which may be negative, and
`direction_cosines`. World coordinates, in RAS millimetres by default,
are

```
world = sum_d (start_d + index_d * step_d) * direction_cosines_d
```

over the spatial dimensions `xspace`, `yspace` and `zspace`. Integer
voxels are scaled to real values through `image-min` and `image-max`.
Files are read with nibabel (and h5py for MINC2), which cannot write
MINC.

!!! warning "Limitations, inherited from nibabel"
    Irregularly spaced dimensions are not supported, nor are dimensions
    without a describing variable (such as `vector_dimension`), nor
    volumes without `image-min` and `image-max`.
"""

__all__ = [
    "MincParser",
    "MincDimension",
]

from ._dimensions import MincDimension
from ._parsers import MincParser
