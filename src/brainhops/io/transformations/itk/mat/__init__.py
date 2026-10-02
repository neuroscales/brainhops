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

ITK's reader takes the variables in pairs, and the second of each pair is
the fixed parameters whatever its name; both must be column vectors. A
file with several blocks repeats the pair, so the same name may appear
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
* `O`: `0`. MATLAB reserves this digit; VNL sets it to `1` for a matrix
  it writes row by row, which reads the same for a vector.
* `P`: precision, `0` for `double` and `1` for `float`. The parameters of
  a `float` transform are `float`, but fixed parameters are always
  `double`.
* `T`: matrix type, `0` for a full numeric matrix.

The layout is that of `itk::MatlabTransformIOTemplate::Read` and `Write`
(`Modules/IO/TransformMatlab/src/itkMatlabTransformIO.cxx`), which go
through VNL's `vnl_matlab_write` and `vnl_matlab_readhdr`
(`vnl/vnl_matlab_write.cxx`, `vnl/vnl_matlab_read.cxx`).

### Implicit Geometrical Specifications

As in every ITK format, the parameters map points of the fixed space to
points of the moving space, in LPS world coordinates, and a matrix is
stored row-major. For an `AffineTransform` with matrix `A`, translation
`t` (the last `D` parameters) and center `c` (the fixed parameters), a
point `x` of the fixed space maps to the point

```text
y = A (x - c) + c + t
```

of the moving space: the transform *pulls* the moving image onto the
fixed grid. That is the chain an
[`ITKAffineBase`][brainhops.io.transformations.itk.ITKAffineBase] block
holds -- `recenter`, `linear`, `uncenter`, `translation`.

## Writing

[`MatTransform`][brainhops.io.transformations.itk.mat.MatTransform]
writes a file as `itk::MatlabTransformIO` does: the parameters, then
the fixed parameters, as two column vectors, little-endian and `double`
by default (`save(..., byteorder=">", precision="float")` changes
either). `scipy.io.loadmat` reads what it writes.

- A block read from an ITK file is written back unchanged -- its
  class, its parameters and its center -- so a file read and saved again
  is the same file, byte for byte.
- Any other affine (or a transformation that converts to one, such as a
  `Translation`) is written as an `AffineTransform` whose center is the
  origin: `fixed` is zero, and the translation is the last column of the
  matrix. A brainhops affine has no center, and with `c = 0` ITK reads
  back exactly `y = A x + t`. Its endpoints must be ITK's space (LPS
  millimetres) or unspecified; an affine between RAS spaces is refused
  rather than silently reinterpreted.

```python
from brainhops.datamodel.transformations import Affine
from brainhops.io.transformations.itk.mat import MatTransform

MatTransform([Affine(matrix)]).save("out0GenericAffine.mat")
```

Only one block is written, as ANTs writes one transform per `.mat` file.
A chain is refused: compose it first (`.compute()`).

## ANTs conventions

ANTs reads and writes its transforms through ITK, so the conventions
above are those of ANTs: LPS millimetres, and fixed to moving.

- **Warps.** `<prefix><n>Warp.nii.gz` and `<prefix><n>InverseWarp.nii.gz`
  are ITK NIfTI displacement fields, read by
  [`brainhops.io.transformations.itk.nifti`][].
- **Transform lists.** `antsApplyTransforms -t T1 -t T2 ... -t Tn`
  describes the image transform as a stack, "the last one listed is
  applied first" -- to the *moving image*. To the *points* of the fixed
  space, which is how the transforms are evaluated, they apply in the
  order listed: `T1` first. A brainhops
  [`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
  transformations in the order they are applied to points, so it is the
  ANTs list in the same order:

    - `-t out1Warp.nii.gz -t out0GenericAffine.mat` is
      `Sequence([warp, affine])`;
    - `-t [out0GenericAffine.mat,1] -t out1InverseWarp.nii.gz` is
      `Sequence([~affine, inverse_warp])`,

    with `warp = io.load("out1Warp.nii.gz", hint="ants")`,
    `inverse_warp = io.load("out1InverseWarp.nii.gz", hint="ants")` and
    `affine = io.load("out0GenericAffine.mat")` (a warp needs the hint:
    its header alone does not say it holds LPS vectors). The first maps
    the fixed space to the moving space (it resamples the moving image
    onto the fixed grid), the second maps the moving space back to the
    fixed one.
- **Inversion.** `[file.mat,1]` (`useInverse`) inverts a linear
  transform: it is `io.load("file.mat").inverse()` (or `~`). ANTs does not
  invert a warp this way; it writes the inverse warp to its own file.
- **Composite files.** Inside one ITK file holding a `CompositeTransform`
  (`<prefix>Composite.h5`), ITK lists the blocks the other way round: it
  applies the last block of the file first. That is why the writer does
  not write chains.
"""

__all__ = ["MatTransform", "MatTransformParser"]

from ._parser import MatTransformParser
from ._xform import MatTransform
