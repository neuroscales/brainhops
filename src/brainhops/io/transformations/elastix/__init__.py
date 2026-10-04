# ruff: disable[E501]
"""
elastix / transformix transform parameter files.

[elastix](https://github.com/SuperElastix/elastix) writes the result of
a registration as a *transform parameter file*,
`TransformParameters.<n>.txt`, which transformix (and SimpleElastix,
ITK-Elastix, ...) reads to resample the moving image onto the fixed
grid. It is a plain-text map from parameter names to values.

| Syntax             | Class                         | Extension | Hints                           | Writes |
| ------------------ | ----------------------------- | --------- | ------------------------------- | ------ |
| classic text       | [`ElastixParameterTransform`][] | `.txt`  | `elastix`, `transformix`, `elastix.params` | yes |
| TOML (elastix 5.2) | [`ElastixTomlTransform`][]    | `.toml`   | `elastix`, `transformix`, `elastix.toml`   | yes |

!!! example "A rigid transform"
    ```text
    (Transform "EulerTransform")
    (NumberOfParameters 6)
    (TransformParameters 0.1 -0.2 0.3 1.5 -2 3.25)
    (InitialTransformParameterFileName "NoInitialTransform")
    (HowToCombineTransforms "Compose")
    (FixedImageDimension 3)
    (MovingImageDimension 3)
    (Size 256 256 128)
    (Index 0 0 0)
    (Spacing 1 1 2)
    (Origin -127.5 -127.5 -127)
    (Direction 1 0 0 0 1 0 0 0 1)
    (UseDirectionCosines "true")
    (CenterOfRotationPoint 4.5 -2 7.25)
    (ComputeZYX "false")
    ```

## Syntax

From `Common/ParameterFileParser/itkParameterFileParser.cxx`:

- **text**: one `(Name value value ...)` per line. A value is a number
  or a double-quoted string (no escapes); `//` starts a comment; tabs
  are blanks; a name may not repeat. elastix writes booleans as quoted
  words (`"true"`).
- **TOML**: one `Name = value` or `Name = [value, ...]` per line, `#`
  comments. Only the one-line subset that elastix writes is parsed here
  (Python has no TOML parser before 3.11).

The parsed map is kept, in file order, in
[`parameter_map`][brainhops.io.transformations.elastix.ElastixTransform.parameter_map].
A file is claimed only if it names a `Transform` *and* carries its
parameters: elastix's registration parameter files share the syntax and
are not transforms.

## Conventions

- **Space.** elastix's transforms are ITK transforms acting on ITK's
  physical space: LPS millimetres.
- **Direction.** A transform maps *fixed*-image points to *moving*-image
  points -- the pull direction in which transformix resamples. The
  transformation read here maps the same way: its input is the fixed
  image's LPS space, its output the moving image's.
- **Blocks.** Each file's transform is decoded into the ITK block of
  [`brainhops.io.transformations.itk`][] that carries the same
  parameters (`y = M (x - c) + c + t` about the center `c`):

| elastix `Transform`          | Parameters                                     | Block                     |
| ---------------------------- | ---------------------------------------------- | ------------------------- |
| `TranslationTransform`       | `t` (D)                                        | `TranslationTransform`    |
| `EulerTransform` (2-D)       | `angle, t` (3)                                 | `Euler2DTransform`        |
| `EulerTransform` (3-D)       | `ax, ay, az, t` (6), `ComputeZYX`              | `Euler3DTransform`        |
| `SimilarityTransform` (2-D)  | `scale, angle, t` (4)                          | `Similarity2DTransform`   |
| `SimilarityTransform` (3-D)  | `versor (3), t, scale` (7)                     | `Similarity3DTransform`   |
| `AffineTransform`            | `M` row-major, `t` (D² + D)                    | `AffineTransform`         |
| `AffineLogTransform`         | `log M` row-major, `t` (D² + D)                | `AffineTransform`, `M = expm(log M)` |
| `AffineDTITransform`         | angles, shears, scales, `t` (7 / 12)           | `AffineTransform`         |
| `BSplineTransform`, `RecursiveBSplineTransform` | coefficients (D x grid)     | `BSplineTransform`, of degree `BSplineTransformSplineOrder` |

  The center is `CenterOfRotationPoint` (world coordinates). Maps that
  carry ITK's own `ITKTransformParameters` / `ITKTransformFixedParameters`
  are read too.
- **Fixed image.** `Size`, `Index`, `Spacing`, `Origin` and `Direction`
  describe the grid transformix resamples onto. It is exposed as
  [`fixed_geometry`][brainhops.io.transformations.elastix.ElastixTransform.fixed_geometry]
  (a [`Geometry`][brainhops.datamodel.geometry.Geometry] whose
  transformation maps voxels to LPS), since a transformation has no slot
  for its output domain.
- **Direction matrices are column-major.** elastix writes `Direction`
  and `GridDirection` column by column (`Conversion::ToVectorOfStrings`
  on an `itk::Matrix`), the transpose of ITK's row-major fixed
  parameters.
- **B-spline grid.** The coefficients are `D` images of world-space
  displacements, back to back, x fastest, over the region that starts at
  `GridIndex`: the first coefficient sits at
  `GridOrigin + GridDirection @ diag(GridSpacing) @ GridIndex`.

## Chains

A file may name an *initial transform*, in
`InitialTransformParameterFileName` (or the deprecated
`InitialTransformParametersFileName`), that applies before its own; that
file may name another, and so on. Each is read, and the chain is a
[`Sequence`][brainhops.datamodel.transformations.Sequence] of their
blocks, initial transforms first (elastix computes `T1(T0(x))` --
`AdvancedCombinationTransform::TransformPointUseComposition`). The
initial transform is also kept, as an `ElastixTransform`, in `initial`.

A relative name is looked for as elastix looks for it: from the working
directory, then from the directory of the file that names it. elastix
writes absolute paths, which break when an output folder is moved, so a
name that is found nowhere is then looked for by its base name next to
the file. `initial=False` reads a file's own transform alone;
`initial=<file name>` or `initial=<transformation>` supplies the initial
transform.

## Writing

A file read and not modified is written back as read (the files of its
initial transforms are not rewritten). Otherwise the chain must reduce to
one transform: a block that elastix has (translation, Euler, similarity,
affine, B-spline) keeps its class and center; anything else that reduces
to an affine is written as an `AffineTransform` centered on the origin.
The fixed-image geometry and the other parameters of the map are kept.

## Not supported

- `HowToCombineTransforms "Add"`: elastix then computes
  `T1(x) + T0(x) - x`, a sum that a chain of transformations cannot
  express. Such a file is refused unless read with `initial=False`.
- Other transforms: `DeformationFieldTransform` (a field stored in a
  separate image), `SplineKernelTransform`, `WeightedCombinationTransform`,
  `MultiBSplineTransformWithNormal`, `BSplineTransformWithDiffusion`,
  the stack transforms (`*StackTransform`), `ExternalTransform`, and
  cyclic B-splines (`UseCyclicTransform "true"`).
- `CenterOfRotation`, the voxel-index center written by elastix < 3.402,
  which current elastix no longer reads either.
- `UseDirectionCosines "false"`: elastix then ignored both images'
  directions, so the transform maps direction-less spaces that cannot be
  recovered without the moving image. Such a file is read as if it
  mapped LPS, with a warning.

!!! note "B-splines outside their valid region"
    elastix evaluates a B-spline only where the whole support of the
    spline lies inside the control-point grid, and returns the input point
    unchanged elsewhere (`InsideValidRegion`). Here, as for ITK's own
    B-splines, coefficients outside the grid are zero, so the two agree
    inside the valid region and differ in the outer band of the grid.

## Verified and assumed

Verified against transformix (ITK-Elastix 0.25.4): every
transform above in 2-D and 3-D where elastix has it, both Euler angle
orders, B-splines of degrees 1, 2 and 3 with a non-zero `GridIndex` and an
oblique `GridDirection`, non-symmetric fixed-image directions, two- and
three-link chains (including the deprecated key), and the TOML syntax --
see `tests/data/elastix/generate_elastix_fixtures.py`. The source
references are elastix's `elxTransformBase.hxx` (reading, initial
transforms, combination), `itkAdvancedEuler3DTransform.hxx`,
`itkAdvancedSimilarity{2,3}DTransform.hxx`,
`itkAdvancedBSplineDeformableTransformBase.hxx`,
`itkAffineLogTransform.hxx`, `itkAffineDTI{2,3}DTransform.hxx` and
`elxAdvancedBSplineTransform.hxx`.

Assumed: 4-D (and higher) affine and B-spline maps behave as their 2-D
and 3-D counterparts (not tested against transformix).
"""

# ruff: enable[E501]

__all__ = [
    "ElastixParameterTransform",
    "ElastixTomlTransform",
    "ElastixTransform",
]

from ._xform import (
    ElastixParameterTransform,
    ElastixTomlTransform,
    ElastixTransform,
)
