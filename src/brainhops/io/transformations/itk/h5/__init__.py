"""
ITK binary transformations are saved in H5 format and support a variety
of (chained) transformations.

They support the same transformation types as the text-based TFM format,
although TFM files are rarely used to store displacement fields, which are
usually stored in H5 files.

The supported displacement fields have the following encoding:

* `BSplineTransform`
  Deforms space using a sparse regular grid of control points to
  represent free-form distortions.

  - `Parameters`:
    Array of multi-dimensional deformation vector displacements (x, y, z)
    for every single control point in the grid.

    * Total count = (Number of Grid Control Points) x Spatial Dimension.

  - `FixedParameters`:
    Information describing the spatial bounds of the grid:

    * `[0-2]`  Grid Size (number of control points along each axis)
    * `[3-5]`  Grid Origin (physical starting coordinates)
    * `[6-8]`  Grid Spacing (physical distance between control points)
    * `[9-17]` Grid Direction Matrix (orientation of the grid, 3 x 3 row-major)

 * `DisplacementFieldTransform`
    A dense deformation map where every individual pixel/voxel in the
    image gets its own explicit displacement vector.

    - `Parameters`:
      Multi-dimensional displacement values for every voxel.

      * Total count = (Total Image Voxels) x Spatial Dimension.

    - `FixedParameters`:
      Empty. The transform topology matches the physical coordinate space
      of the primary image metadata instead.

Furthermore, the H5 format supports nested chained transforms. A group of
chained transforms is encoded by a group with class name `CompositeTransform`,
whose `Parameters` and `FixedParameters` are empty, and whose children
are the individual transform blocks.

## Approximate specification

### 1. File Level Attributes

At the root directory level (`/`), the format embeds global provenance
metadata to validate software compatibility. These are saved as standard
HDF5 Attributes or standalone datasets:

* `/ITKVersion`: String attribute tracking the compiling library version
  (e.g., "5.3.0").
* `/HDFVersion`: String attribute detailing the underlying dataset
  library version.
* `/OSName`: String attribute indicating the operating system name.
* `/OSVersion`: String attribute indicating the operating system version.

### 2. The `/TransformGroup` Directory Hierarchy

All transformation records are wrapped inside a top-level group path
named exactly `/TransformGroup`.

If a file contains a chain of transforms or a `CompositeTransform`, they
are stored inside numerical sub-groups corresponding sequentially to
their execution stack order:
* `/TransformGroup/0`
* `/TransformGroup/1`
* `/TransformGroup/2`

### 3. Internal Group Structure

Every numerical sub-group (e.g., `/TransformGroup/0`) must contain the
following specific objects:

```text
/TransformGroup
  └── /0
       ├── TransformType             (Dataset: String attribute/value)
       ├── TransformParameters       (Dataset: 1D Floating-point array)
       └── TransformFixedParameters  (Dataset: 1D Floating-point array)
```

1. `TransformType`

    * Data Type: String (Variable length or fixed character array).
    * Specification: Holds the exact C++ Run-Time Type Information (RTTI)
      name of the ITK class.
    * Example Value: "AffineTransform_double_3_3" or "BSplineTransform_double_3_3".

2. `TransformParameters`

    * Data Type: 1D Array of HDF5 Floats
      (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE).
    * Specification: Stores the optimisable/variable coefficients of the
      transform.
    * Note on Bug-Compatibility: To maintain backward compatibility with
    a legacy spelling error in older ITK versions, modern ITK parsers
    fallback to search for `TranformParameters` (missing the "s" in "Trans")
    if `TransformParameters` is missing.

3. `TransformFixedParameters`

    * Data Type: 1D Array of HDF5 Floats
      (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE).
    * Specification: Stores structural constants (such as centers of
      rotation or deformation grid bounds).
    * Note on Bug-Compatibility: The parser will fall back to look for
      the legacy misspelled string `TranformFixedParameters` if the
      properly spelled version cannot be indexed.
"""
__all__ = [
    "H5Transform",
    "H5Header",
    "H5TransformParser",
    "DelayedH5Array"
]

from ._xform import H5Transform
from ._parser import H5Header, H5TransformParser, DelayedH5Array
