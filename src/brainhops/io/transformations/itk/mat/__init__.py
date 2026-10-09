"""ITK linear transforms stored in binary MATLAB (`.mat`) files.

This is the format of `itk::MatlabTransformIO`, which ANTs uses for all of
its linear transforms (`<prefix>0GenericAffine.mat`, and likewise the
Rigid, Affine, Similarity, Translation and DerivedInitialMovingTranslation
`.mat` files). A file holds the same blocks as an ITK text file, each made
of a class, parameters and fixed parameters, in a MATLAB v4 container.

!!! example "A 3-D affine, as `scipy.io.loadmat` sees it"
    ```python
    {
        "AffineTransform_double_3_3": array([[1.1], [0.1], ..., [3.0]]),
        "fixed": array([[4.0], [5.0], [6.0]]),
    }
    ```

!!! note "Not to be confused with FSL FLIRT"
    FLIRT also saves `.mat` files, but as plain text. The two are told
    apart by their content: this reader only claims files that start with
    a MATLAB v4 header naming an ITK transform.

## Approximate specification

### Variables

A file is a flat sequence of variables, each made of a header, a name and
values. Every block is a pair of column vectors: the parameters, named
`{ClassName}_{Precision}_{InputDim}_{OutputDim}` (for example
`AffineTransform_double_3_3`, or `MatrixOffsetTransformBase_double_3_3` in
older ANTs releases), followed by the fixed parameters, named `fixed`. ITK
reads the variables in pairs and takes the second one of each pair as the
fixed parameters, whatever its name. A file with several blocks repeats
the pair, so names may repeat.

### Variable header

The header consists of five 32-bit integers:

| Field    | Meaning                                                   |
| -------- | --------------------------------------------------------- |
| `type`   | `M*1000 + O*100 + P*10 + T` (see below)                   |
| `mrows`  | number of rows (the length of the vector)                 |
| `ncols`  | number of columns (`1`)                                   |
| `imagf`  | `1` if the values are complex (`0`)                       |
| `namlen` | length of the name, including its terminating `NUL`       |

The NUL-terminated name and `mrows * ncols` values follow. The digits of
`type` are:

- M, the byte order: 0 for little-endian and 1 for big-endian. ITK writes
  the native order of the machine, for the header and the values alike.
- O, which MATLAB reserves as 0. VNL sets it to 1 for row-by-row
  matrices, which for a vector is the same thing.
- P, the precision: 0 for double and 1 for float. The fixed parameters
  are always double.
- T, the matrix type: 0 for a full numeric matrix.

The layout follows `itk::MatlabTransformIOTemplate::Read` and `Write`,
which rely on VNL's `vnl_matlab_write` and `vnl_matlab_readhdr`.

### Geometry

As in every ITK format, the parameters map points of the fixed space to
points of the moving space, in LPS world coordinates, and matrices are
stored row-major. An `AffineTransform` with matrix A, translation t (the
last D parameters) and center c (the fixed parameters) maps a fixed-space
point x to

```text
y = A (x - c) + c + t
```

in moving space, so the transform pulls the moving image onto the fixed
grid. This is the chain held by
[`ItkAffineBase`][brainhops.io.transformations.itk.ItkAffineBase].

## Writing

[`MatTransform`][brainhops.io.transformations.itk.mat.MatTransform] writes
a single block, as ANTs does, in little-endian double precision by
default. A block read from an ITK file is written back unchanged, center
included. Any other affine is written as an `AffineTransform` centered on
the origin, which gives exactly `y = A x + t`. The endpoints of such an
affine must be the ITK space or left unspecified, and an affine between RAS
spaces is refused rather than reinterpreted. A chain of several
transformations must first be composed into one, for example with
`.compute()`.

```python
from brainhops.datamodel.transformations import Affine
from brainhops.io.transformations.itk.mat import MatTransform

MatTransform([Affine(matrix)]).save("out0GenericAffine.mat")
```

## ANTs transform lists

`antsApplyTransforms -t T1 -t T2 ... -t Tn` applies the last transform
first to the moving image, which means that points of the fixed space go
through the list in order, starting with T1. A brainhops
[`Sequence`][brainhops.datamodel.transformations.Sequence] lists its
transformations in point-application order, so it matches the ANTs list:

- `-t out1Warp.nii.gz -t out0GenericAffine.mat` is
  `Sequence([warp, affine])`, which maps fixed to moving space;
- `-t [out0GenericAffine.mat,1] -t out1InverseWarp.nii.gz` is
  `Sequence([~affine, inverse_warp])`, which maps moving back to fixed
  space.

Here `affine = io.load("out0GenericAffine.mat")` and
`warp = io.load("out1Warp.nii.gz", hint="ants")`. The warps need the hint
because their header does not say that they hold LPS vectors, and they are
read by [`brainhops.io.transformations.itk.nifti`][]. The `[file.mat,1]`
syntax inverts a linear transform, which corresponds to
`io.load("file.mat").inverse()` or to the `~` operator. For warps, ANTs
writes the inverse to a separate file instead.

A composite file (`<prefix>Composite.h5`) stores its blocks in the
opposite order: the file `[Composite, T0, T1]` maps `x` to `T0(T1(x))`.
Every ITK reader here lists composite blocks in application order, so the
file reads as `Sequence([T1, T0])`, and `-t <prefix>Composite.h5` equals
`-t T1 -t T0`. Blocks without a composite header are separate transforms.
The first one is read by default, with a warning, and another one can be
selected with `position=`, as in `MatTransform.from_file(path, position=1)`.
"""

__all__ = ["MatTransform", "MatTransformReaderWriter"]

from ._parser import MatTransformReaderWriter
from ._xform import MatTransform
