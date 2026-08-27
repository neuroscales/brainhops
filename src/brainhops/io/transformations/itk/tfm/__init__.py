# ruff: disable[E501]
"""
ITK "TFM" transformations are saved in a text format and support a
variety of (chained) transformations.

!!! example "2D rotation encoded by Euler angles"
    ```text
    # Insight Transform File V1.0
    # Number of parameters = 3
    # Transform 0:
    # Class name = Euler2DTransform
    # Parameters = 0 10 20
    Transform: Euler2DTransform_double_2_2
    Parameters: 0 10 20
    FixedParameters: 50 50
    ```

!!! example "Chained transformations"
    ```text
    # Insight Transform File V1.0
    # Transform 0
    Transform: TranslationTransform_double_3_3
    Parameters: 10.5 -5.0 20.0
    FixedParameters:

    # Transform 1
    Transform: Euler3DTransform_double_3_3
    Parameters: 0.1 0.0 -0.2 0.0 0.0 0.0
    FixedParameters: 128.0 128.0 64.0
    ```

## Approximate specification

### 1. The Header Line

The very first non-blank line of the file must be a strict match for the
format version header:

```text
# Insight Transform File V1.0
```

If this header is missing or altered, the ITK parser will immediately
reject the file.

### 2. The Transform Block

Every transform in the file is parsed sequentially as an object. A block
contains exactly three required, case-sensitive tags:

* `Transform: {ClassName}_{Precision}_{InputDim}_{OutputDim}`
    - Specifies the RTTI (Run-Time Type Information) class name.
    - Precision must be double or float.
    - Dimensions specify the spatial manipulation (e.g., _3_3 for 3D-to-3D).
* `Parameters: {Space-separated floating-point numbers}`
    - The variable, optimizable values.
    - If a transform type does not have variable parameters (like an
      identity block), this line must still exist but can be left empty.
* `FixedParameters: {Space-separated floating-point numbers}`
    - Parameter constants that do not change during registration
      optimization (typically the center of rotation coordinates).
    - If none exist, this tag must still be explicitly typed out and left blank.

### 3. Comments and Whitespace

Any line starting with a `#` is treated as a comment and skipped by the
parser.

Empty lines between blocks are ignored.

### Implicit Geometrical Specifications

Beyond text formatting, the data inside the file must adhere to ITK's
structural physics guidelines:

* Coordinate System: The numerical values inside a .tfm file are
  strictly calculated using the LPS (Left-Posterior-Superior) coordinate
  system. If you export a transform from software that defaults to RAS
  (Right-Anterior-Superior), like 3D Slicer, the values are automatically
  matrix-converted to LPS before saving to the .tfm file.

* Array Ordering: Multi-dimensional matrices (such as the rotation
  elements in an AffineTransform) are written out in row-major order
  (linearized row by row).

### Transformation types

The text-based .tfm standard is intended only for linear, rigid, or
affine transformations.

| Transform Class Name   | Variable Parameters (Optimisable)                 | Length | FixedParameters |                          | Description / Note |
| -----------------------|---------------------------------------------------|--------|-----------------|--------------------------|--------------------|
| IdentityTransform      | None                                              | 0      | None            |                          | Maps input coordinates completely unaltered.
| TranslationTransform   | [t_x, t_y, ...]                                   | D      | None            |                          | Standard shifts along spatial axes (e.g., 2 or 3 parameters).
| ScaleTransform         | [s_x, s_y, ...]                                   | D      | [c_x, c_y, ...] | Center of scaling        | Anisotropic scaling along spatial axes.
| Euler2DTransform       | [angle, t_x, t_y]                                 | 3      | [c_x, c_y]      | Center of rotation       | Rigid 2D transform (1 rotation parameter in radians, 2 translations).
| Euler3DTransform       | [angle_x, angle_y, angle_z, t_x, t_y, t_z]        | 6      | [c_x, c_y, c_z] | Center of rotation       | Rigid 3D transform (3 Euler rotation angles in radians, 3 translations).
| VersorTransform        | [v_x, v_y, v_z]                                   | 3      | [c_x, c_y, c_z] | Center of rotation       | Pure 3D rotation defined using a unit quaternion vector (versor).
| VersorRigid3DTransform | [v_x, v_y, v_z, t_x, t_y, t_z]                    | 6      | [c_x, c_y, c_z] | Center of rotation       | Standard 3D rigid transform. Uses versors for cleaner rotation optimization.
| Similarity2DTransform  | [scale, angle, t_x, t_y]                          | 4      | [c_x, c_y]      | Center of rotation/scale | Rigid 2D transformation plus uniform scaling factor.
| Similarity3DTransform  | [v_x, v_y, v_z, t_x, t_y, t_z, scale]             | 7      | [c_x, c_y, c_z] | Center of rotation/scale | Rigid 3D transformation plus uniform scaling factor.
| AffineTransform        | [Matrix elements (row-major), Translation vector] | D² + D | [c_x, c_y, ...] | Center of rotation       | Fully unbounded linear mapping (Translation, Rotation, Shearing, and Scale). Example (3D): 9 matrix values + 3 translations = 12 parameters.
"""
# ruff: enable[E501]
__all__ = [
    "TFMTransform",
    "TFMTransformParser",
]

from ._parser import TFMTransformParser
from ._xform import TFMTransform
