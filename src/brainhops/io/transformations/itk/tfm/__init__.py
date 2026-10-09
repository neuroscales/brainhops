# ruff: disable[E501]
"""
Transformations stored in ITK text `.tfm` files.

A `.tfm` file is a text file that holds one or more transformations, each
written as a block of tagged lines. A file can hold several independent
transformations or a single composite transformation built from several blocks.

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

!!! example "Composite transformation"
    ```text
    # Insight Transform File V1.0
    # Transform 0
    Transform: CompositeTransform_double_3_3
    # Transform 1
    Transform: TranslationTransform_double_3_3
    Parameters: 10.5 -5.0 20.0
    FixedParameters:

    # Transform 2
    Transform: Euler3DTransform_double_3_3
    Parameters: 0.1 0.0 -0.2 0.0 0.0 0.0
    FixedParameters: 128.0 128.0 64.0
    ```

    ITK applies the blocks of a composite transformation from last to first, so
    this file rotates a point and then translates it. The reader therefore
    lists the blocks in application order, as `[Euler3D, Translation]`.

## Composite transformations

ITK writes a composite transformation as a `CompositeTransform` header block,
which has no parameters, followed by the blocks of its queue from front to
back. Its `TransformPoint` method applies the queue from back to front, so the
file `[Composite, T0, T1]` maps a point `x` to `T0(T1(x))`. A brainhops
[`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
transformations in application order, so the reader reverses the queue and
returns `[T1, T0]`.

A file with several blocks but no composite header holds separate
transformations that are not composed. By default, the reader loads the first
one, as SimpleITK's `ReadTransform` does, and warns that the file holds
several. Another one is selected by its position, as in
`TfmTransform.from_file(path, position=1)`. A composite file holds a single
transformation, at position 0. A `CompositeTransform` block that is not the
first block of the file is refused, since ITK never writes one.

## File format

1. The first non-blank line must be exactly `# Insight Transform File V1.0`,
   otherwise ITK rejects the file.
2. Each block is made of three case-sensitive tagged lines:
    - `Transform: {ClassName}_{Precision}_{InputDim}_{OutputDim}` names the
      class of the transformation, its precision (`double` or `float`) and its
      input and output dimensions, such as `_3_3`.
    - `Parameters:` lists the optimizable parameters as space-separated
      numbers.
    - `FixedParameters:` lists the constant parameters, which are typically the
      center of rotation.

    In a transformation block, both parameter lines are present, and a line is
    left empty when there are no values. A `CompositeTransform` header block
    has neither line. The reader accepts any block in which either line is
    missing and reads the missing values as empty.
3. Lines that start with `#` are comments, and blank lines between blocks are
   ignored.

All values are expressed in LPS coordinates, and matrices are stored in
row-major order. Software that works in RAS coordinates, such as 3D Slicer,
converts its transformations to LPS when it saves them.

## Transformation classes

The `.tfm` format is intended for linear transformations, such as rigid and
affine ones. The classes below are the most common.

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
    "TfmTransform",
    "TfmTransformParser",
]

from ._parser import TfmTransformParser
from ._xform import TfmTransform
