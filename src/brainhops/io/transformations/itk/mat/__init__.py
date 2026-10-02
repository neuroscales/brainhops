"""
ITK can save linear transformations in a binary MATLAB file
(`itk::MatlabTransformIO`). This is the format ANTs uses for every linear
transform it writes: `antsRegistration` saves `<prefix>0GenericAffine.mat`,
and also `Rigid.mat`, `Affine.mat`, `Similarity.mat`, `Translation.mat`
and `DerivedInitialMovingTranslation.mat`.

The file holds the same transform blocks as the text-based TFM format --
a transform class, its parameters and its fixed parameters -- in a
MATLAB v4 container instead of text.

!!! example "A 3-D affine, as `scipy.io.loadmat` sees it"
    ```python
    {
        "AffineTransform_double_3_3": array([[1.1], [0.1], ..., [3.0]]),
        "fixed": array([[4.0], [5.0], [6.0]]),
    }
    ```

!!! note "Not to be confused with FSL FLIRT"
    FSL FLIRT also saves its affines as `.mat` files, but as plain text.
    The two are told apart by content, not by name: this reader claims
    only files that start with a MATLAB v4 header naming an ITK transform.

## Approximate specification

### 1. Variables

A MATLAB v4 file is a flat sequence of variables, each a header, a name
and the values. ITK writes every block of the transform as two variables,
in this order:

1. The parameters, a column vector named after the transform class,
   `{ClassName}_{Precision}_{InputDim}_{OutputDim}` -- for example
   `AffineTransform_double_3_3`. Older ANTs releases name affines
   `MatrixOffsetTransformBase_double_3_3`, which has the same parameters.
2. The fixed parameters, a column vector named `fixed` -- typically the
   center of rotation.

A file with several blocks repeats the pair, so the same name may appear
more than once. A chain written from a `CompositeTransform` starts with a
pair for the composite itself, which only points to the blocks after it.

### 2. Variable header

Each variable starts with five 32-bit integers:

| Field    | Meaning                                                   |
| -------- | --------------------------------------------------------- |
| `type`   | `M*1000 + O*100 + P*10 + T` (see below)                   |
| `mrows`  | number of rows (the length of the vector)                 |
| `ncols`  | number of columns (`1`)                                   |
| `imagf`  | `1` if the values are complex (`0`)                       |
| `namlen` | length of the name, including its terminating `NUL`       |

followed by the `NUL`-terminated name and the `mrows * ncols` values in
column-major order. The digits of `type` are:

* `M`: byte order, `0` for little-endian and `1` for big-endian. ITK
  (through `vnl_matlab_write`) writes in the native order of the machine,
  and the header integers are in that same order.
* `O`: reserved, always `0`.
* `P`: precision, `0` for `double` and `1` for `float`.
* `T`: matrix type, `0` for a full numeric matrix.

### Implicit Geometrical Specifications

As in every ITK format, the parameters map points of the fixed space to
points of the moving space, in LPS world coordinates, and a matrix is
stored row-major.
"""

__all__ = ["MATTransform", "MATTransformParser"]

from ._parser import MATTransformParser
from ._xform import MATTransform
