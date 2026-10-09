"""
The shared MINC-reading machinery.

MINC is the volume format of the Montreal Neurological Institute (MNI)
and of the MINC toolkit. It comes in two containers, both with the
extension `.mnc`:

- **MINC1** is a NetCDF classic file (magic `CDF\\x01`, or `CDF\\x02`
  for the 64-bit offset variant). The voxels are the variable `image`,
  whose dimensions name the axes, and each spatial dimension is
  described by a variable of the same name.
- **MINC2** is an HDF5 file with a root group `/minc-2.0`. The voxels are
  the dataset `/minc-2.0/image/0/image`, whose attribute `dimorder`
  names the axes, and each dimension is described by a dataset under
  `/minc-2.0/dimensions`.

In both, the dimensions are listed slowest first (C order), in any order:
`zspace, yspace, xspace` is common (transverse slices), but sagittal or
coronal orders are just as valid. A dimension is described by its
`start`, its `step` (which may be negative), its `direction_cosines`
and its `units`. The *world* coordinates of a voxel are

```
world = sum_d (start_d + index_d * step_d) * direction_cosines_d
```

over the spatial dimensions `xspace`, `yspace` and `zspace`, in a world
space whose axes run left-to-right, posterior-to-anterior and
inferior-to-superior (RAS), in millimetres by default. A dimension
without direction cosines runs along its own world axis.

Integer voxels are scaled to real values through `image-min` and
`image-max`, which map the valid range of the type to real values,
possibly with one scaling per slice.

The headers and the voxels are read with `nibabel` (`nibabel.minc1`,
`nibabel.minc2`); MINC2 needs `h5py`, imported only when a MINC2 file
is read. `nibabel` cannot write MINC, so neither can `brainhops`.

!!! warning "Limitations, inherited from `nibabel`"
    Dimensions with irregular spacing (`spacing = "irregular"`) and
    dimensions that have no describing variable (such as MINC's
    `vector_dimension`, used for RGB volumes and displacement grids)
    are not supported, nor is a volume without `image-min` and
    `image-max` (which the MINC library always writes).
"""

__all__ = [
    "MincParser",
    "MincDimension",
]

from ._dimensions import MincDimension
from ._parsers import MincParser
