# ruff: disable[E501]
"""elastix transform parameter files.

A transform parameter file, such as `TransformParameters.0.txt`, is a
plain-text map from parameter names to values. It is the result of an
[elastix](https://github.com/SuperElastix/elastix) registration, and
transformix, SimpleElastix and ITK-Elastix read it to resample the moving
image onto the fixed grid.

Two syntaxes are supported:

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

The classic syntax (`itkParameterFileParser.cxx`) has one
`(Name value ...)` entry per line, where a value is a number or a
double-quoted string without escapes. `//` starts a comment, tabs count as
blanks, names may not repeat, and booleans are quoted words such as
`"true"`. The TOML syntax has one `Name = value` or `Name = [value, ...]`
entry per line, and `#` starts a comment. Only the one-line subset of TOML
that elastix writes is parsed, because Python has no TOML parser before
3.11.

The file is read in one pass. Its parameter map, in file order, its
syntax and its text are held as an [`ElastixRaw`][] record by
[`ElastixMetadata`][], the metadata of the file, and a transformation
holds that metadata, so the map of a transformation is
`xform.metadata.raw.parameters`. As in elastix, a file whose name ends in
`.toml` is read as TOML, and any other file in the classic syntax
(`ParameterFileParser::ReadParameterFile` in
`Common/ParameterFileParser/itkParameterFileParser.cxx`, elastix
3d044f8). Registration parameter files share the syntax, so a file is
recognised as a transform only if it names a `Transform` and gives its
parameters.

## Conventions

Points live in ITK physical space, which is LPS in millimetres. The
transform maps points of the fixed image to points of the moving image,
which is the pull direction that transformix uses to resample. The
transformation that is read therefore has the fixed LPS space as input and
the moving LPS space as output.

Each elastix transform is decoded into the block of
[`brainhops.io.transformations.itk`][] that has the same parameters, where
a block represents one ITK transform as a file stores it. The linear
blocks map a point `x` to `y = M (x - c) + c + t`, where `c` is the
center:

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

The center is `CenterOfRotationPoint`, in world coordinates. Maps that use
ITK's own `ITKTransformParameters` and `ITKTransformFixedParameters` are read
as well. The entries `Size`, `Index`, `Spacing`, `Origin` and `Direction`
describe the fixed grid onto which transformix resamples. A transformation has
no place for an output domain, so this grid is exposed separately as
[`fixed_geometry`][brainhops.io.transformations.elastix.ElastixTransform.fixed_geometry],
a [`Geometry`][brainhops.datamodel.geometry.Geometry] whose transformation maps
voxels to LPS.

`Direction` and `GridDirection` are written column by column
(`Conversion::ToVectorOfStrings` on an `itk::Matrix`), which is the
transpose of ITK's row-major fixed parameters. B-spline coefficients form
`D` images of world-space displacements, which are stored one after the
other with x varying fastest. The images cover a region that starts at
`GridIndex`, so the first coefficient lies at
`GridOrigin + GridDirection @ diag(GridSpacing) @ GridIndex`.

## Chains

A file may name an initial transform, applied before its own, in
`InitialTransformParameterFileName` (or the deprecated
`InitialTransformParametersFileName`), and that file may name another. The
chain is read as a
[`Sequence`][brainhops.datamodel.transformations.Sequence] of blocks with
the initial transform first, since elastix computes `T1(T0(x))`
(`AdvancedCombinationTransform::TransformPointUseComposition`).

The files of the chain are read when the file that names them is read,
and the record of each one is held in the record of the file that names
it, in [`ElastixRaw.initial`][brainhops.io.transformations.elastix.ElastixRaw.initial].
The name is kept as the file stores it. The initial transform is also
decoded, as an `ElastixTransform` that holds the record of its file, in
[`initial`][brainhops.io.transformations.elastix.ElastixTransform.initial].

A relative name is looked up as elastix does, in the working directory and
then next to the naming file (`TransformBase::ReadFromFile` in
`Core/ComponentBaseClasses/elxTransformBase.hxx`). elastix writes absolute
paths, which break when the output folder moves, so a name that is not
found is retried by its base name next to the file. With `initial=False`,
only the file's own transform is read. With `initial=<file name>`, that
file is read in place of the file that the map names, and with
`initial=<transformation>`, the transformation is the initial transform.
These options select what is read and decoded, and they do not change the
name that the file stores.

## Writing

A transformation whose chain was not assigned is written as its record.
The text of the record is written back as it was read, line endings and
comments included, while it still parses to the parameter map. When the
transformation is written to a path, each file of its chain whose name is
relative is written where the new file names it, which is beside it, so
that the copy of a chain in another directory reads the same chain. A file
that already holds the same bytes is not written again, and a file that
holds other bytes is never replaced: the writer then refuses. A name that
is absolute is left as it is, and nothing is written for a chain that was
not read, as with `initial=False`. A file object receives the file alone.

A transformation whose chain was assigned is consolidated into a new
record, which holds a single transform and no initial transform, so the
chain must reduce to one transform. A block that elastix supports
(translation, Euler, similarity, affine or B-spline) keeps its class and
center, and anything else that reduces to an affine is written as an
`AffineTransform` centered on the origin. The fixed-image geometry and the
other parameters of the old record are kept. Assigned `input` and `output`
systems must be LPS, which is what the file maps.

## Not supported

- `HowToCombineTransforms "Add"`, since `T1(x) + T0(x) - x` is not a
  chain. Such a file is refused unless `initial=False`.
- `DeformationFieldTransform` (a separate image), `SplineKernelTransform`, `WeightedCombinationTransform`,
  `MultiBSplineTransformWithNormal`, `BSplineTransformWithDiffusion`, the
  `*StackTransform` family, `ExternalTransform`, and cyclic B-splines
  (`UseCyclicTransform "true"`).
- `CenterOfRotation`, the voxel-index center of elastix before 3.402,
  which current elastix no longer reads.
- `UseDirectionCosines "false"`, under which elastix ignores directions.
  The true spaces cannot be recovered without the moving image, so such a
  file is read as if it were LPS, with a warning.

!!! note "B-splines outside their valid region"
    elastix evaluates a B-spline only where its whole support lies inside
    the control-point grid, and returns the input point unchanged
    elsewhere (`InsideValidRegion`). Here, as in ITK, coefficients outside
    the grid are zero, so the two agree inside the valid region and differ
    in the band around it.

## Verified and assumed

Results match transformix (ITK-Elastix 0.25.4) for every transform above,
in 2-D and 3-D where elastix has it: both Euler orders, B-splines of
degrees 1 to 3 with a non-zero `GridIndex` and an oblique `GridDirection`,
non-symmetric fixed directions, chains of two and three links (including
the deprecated key), and the TOML syntax. The fixtures are generated by
`tests/data/elastix/generate_elastix_fixtures.py`. Affines and B-splines
in four or more dimensions are assumed, without tests, to behave as in 2-D
and 3-D.

Every fixture is written back byte for byte, and so is a file with
Windows line endings and comments. Writing the files of a chain beside a
copy, and refusing to replace another file, are choices of brainhops, not
conventions of elastix. It is not verified that elastix reads an initial
file through the same parser, and so chooses its syntax from its extension
in the same way, although `TransformBase::ReadFromFile` reads it as a
parameter file.
"""

# ruff: enable[E501]

__all__ = [
    "ElastixMetadata",
    "ElastixParameterTransform",
    "ElastixRaw",
    "ElastixTomlTransform",
    "ElastixTransform",
]

from ._metadata import ElastixMetadata
from ._raw import ElastixRaw
from ._xform import (
    ElastixParameterTransform,
    ElastixTomlTransform,
    ElastixTransform,
)

# Importing the private converters registers them with the data model,
# so that an affine or a field converts into one of these formats, as in
# `t.to(Format)`.
from . import _converters  # noqa: E402, F401  isort: skip
