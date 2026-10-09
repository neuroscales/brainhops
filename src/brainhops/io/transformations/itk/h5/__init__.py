"""ITK transforms stored in HDF5 (`.h5`) files.

HDF5 files hold the same transform types as text files, but they are the
usual container for warps, which text files rarely store. The two warp
blocks are encoded as follows.

* `BSplineTransform`
  This transform deforms space with a sparse regular grid of control
  points, which represents a free-form distortion.

  - `Parameters`:
    Cubic B-spline coefficients, stored as one scalar coefficient image
    per spatial axis, back to back.

    * Total count = (Number of Grid Control Points) x Spatial Dimension.

  - `FixedParameters`:
    The geometry of the grid, laid out as follows:

    * `[0-2]`  Grid Size (number of control points along each axis)
    * `[3-5]`  Grid Origin (physical starting coordinates)
    * `[6-8]`  Grid Spacing (physical distance between control points)
    * `[9-17]` Grid Direction Matrix (orientation of the grid, 3 x 3 row-major)

* `DisplacementFieldTransform`
  This transform is a dense deformation map, in which every voxel of the
  image has its own displacement vector.

  - `Parameters`:
    The displacement vectors of all voxels.

    * Total count = (Total Image Voxels) x Spatial Dimension.

  - `FixedParameters`:
    The geometry of the voxel grid, laid out as for `BSplineTransform`.

## Composite transforms

A composite file, such as the ANTs `<prefix>Composite.h5`, starts with a
`CompositeTransform` block without parameters, followed by the blocks of its
transform queue from front to back. ITK applies the queue from back to front,
so the file `[Composite, T0, T1]` maps `x` to `T0(T1(x))`. A brainhops
[`Sequence`][brainhops.datamodel.transformations.Sequence] lists
transformations in application order, so the reader returns `[T1, T0]`.

When a file holds several blocks without a composite header, the blocks are
separate transforms, and the reader loads only one of them. By default the
reader loads the first one and warns that there are others, as SimpleITK's
`ReadTransform` does. Another transform can be selected with `position=`, as
in `H5Transform.from_file(path, position=1)`.

## File layout

The root holds the provenance strings `/ITKVersion`, `/HDFVersion`,
`/OSName` and `/OSVersion`. The blocks are sub-groups of `/TransformGroup`
named by their position from zero, and ITK orders them numerically (`10`
comes after `9`, not after `1`). Every block group contains:

```text
/TransformGroup
  └── /0
       ├── TransformType             (Dataset: String attribute/value)
       ├── TransformParameters       (Dataset: 1D Floating-point array)
       └── TransformFixedParameters  (Dataset: 1D Floating-point array)
```

`TransformType` is the C++ name of the ITK class with its precision and
dimensions, such as `"AffineTransform_double_3_3"`. The two parameter
arrays are 1-D float or double datasets. Older ITK versions misspelled
them `TranformParameters` and `TranformFixedParameters`, and both
spellings are read.
"""

__all__ = ["H5Transform", "H5Header", "H5TransformParser", "DelayedH5Array"]

from ._parser import DelayedH5Array, H5Header, H5TransformParser
from ._xform import H5Transform
