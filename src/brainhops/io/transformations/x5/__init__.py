"""Reader and writer for BIDS X5 transformation files.

X5 is the HDF5 format of BIDS extension proposal BEP014, which is still a
draft. This package follows its two implementations: nitransforms 25.0 and
later (`Version = 1`, the current layout) and fslpy 3.x
(`Version = "0.1.0"`, an earlier layout, which is read and written back in
the current layout).

!!! warning "A draft format"
    Anything that neither implementation defines is left unimplemented
    rather than guessed. The section "Not supported" lists these cases.

## Layout

```text
/                          attrs: Format = "X5", Version = 1
/TransformGroup/0          attrs: Type, SubType, Representation,
                                  Metadata (JSON), ArrayLength
    Transform              the parameters: a 4x4 matrix, or a field
    DimensionKinds         what each axis of Transform holds,
                           e.g. ["space", "space", "space", "vector"]
    Inverse                optional precomputed inverse
    Jacobian               optional cached Jacobian determinant
    AdditionalParameters   optional, depends on SubType
    Domain/                REQUIRED for nonlinear, RECOMMENDED for linear
        Grid               1 if the samples lie on a regular grid
        Size               the number of samples per dimension
        Mapping            voxel-to-world affine of the grid
        attrs: Coordinates e.g. "cartesian"
/TransformGroup/1 ...
/TransformChain/0          optional: a string "0/1/2" of node indices
```

ITK `.h5` files also have a `TransformGroup`, but no root `Format`
attribute, so an X5 file named `.h5` is read by this reader and not by
[`brainhops.io.transformations.itk.h5`][]. Arrays are read in the shape
that h5py returns, such as `(X, Y, Z, 3)`. When `DimensionKinds` places the
vector axis elsewhere, that axis is moved last.

## Direction

X5 does not name its spaces. Each transform maps the RAS millimetre world
of an image A to that of an image B, and it is read as a transformation from
`RASmm` to `RASmm`. Which
image A is depends on the writer. In nitransforms, A is the reference (fixed)
image, on whose grid fields are sampled. In fslpy, A is the source image of
a linear transform, but the reference image of a nonlinear one, as in FNIRT.
The caller must therefore know which image to resample from.

## Fields

Nonlinear transforms with the `SubType` `densefield`, or with no `SubType`,
are dense fields:

| `Representation`                      | Transformation             |
| ------------------------------------- | -------------------------- |
| `"displacements"` (fslpy `relative`)  | [`X5DisplacementField`][]  |
| `"deformations"` (fslpy `absolute`)   | [`X5CoordinatesField`][]   |

`"coordinates"` and `"absolute"` are accepted as aliases of
`"deformations"`. A displacement field is read as a chain of three
transformations, in the same way as a NIfTI `DISPVECT` field (see
[`brainhops.io.transformations.nifti`][]): a map from RAS to the voxels of
the `Domain`, the displacements in voxel units, and a map from the voxels
back to RAS. The field is interpolated linearly and takes its nearest value
outside the grid. nitransforms instead interpolates with cubic splines and
applies no displacement outside the grid, so the results of the two
implementations differ away from the grid nodes.

## B-splines

A nonlinear transform with the `SubType` `bspline` and the
`Representation` `coefficients`, as nitransforms writes a
`BSplineFieldTransform`, is read as an [`X5BSplineField`][]. Its `Transform`
holds cubic B-spline coefficients of a RAS millimetre displacement, with
shape `(X, Y, Z, 3)`, and its `AdditionalParameters` hold the voxel-to-RAS
affine of the knot grid, which places knot `k` at voxel `k`. Its `Domain`,
the reference grid that nitransforms uses in `to_field`, does not affect the
transform. The degree is not stored, so only cubic splines are read.

The B-spline is read as the same chain as a dense displacement field, with a
field of cubic coefficients and a zero boundary. The knot grid follows the
same convention as the ITK `BSplineTransform` (see
[`brainhops.io.transformations.itk`][]), except that ITK places its grid in
LPS rather than RAS coordinates.
On write, coefficients of another degree or boundary condition are refitted,
and the knot grid serves as the `Domain` that nitransforms requires.

## Chains

nitransforms stores a chain as a string such as `"0/1/2"` in
`/TransformChain/<n>`, and the nodes of this chain are applied as
`f2(f1(f0(x)))`. This is also the order of a brainhops `Sequence`, so the
chain is read as `Sequence([t0, t1, t2])`.
[`X5Transform.selection`][] describes which nodes are read.

## Metadata

The datamodel holds no metadata, so the reader keeps each node's JSON
`Metadata`, `Domain`, `Inverse`, `Jacobian` and other attributes as read, in
[`X5Transform.nodes`][] and [`X5Transform.header`][], and writes them back.
A transformation built from scratch is written without metadata.

## Not supported

The following nodes raise an error when interpreted, although they are
still read and written back unchanged:

- the `Type` `composite`, whose storage the draft does not specify and which
  no implementation writes;
- an `ArrayLength` greater than 1, which denotes a stack of one affine per
  volume (the nitransforms `LinearTransformsMapping`) and which the
  datamodel cannot hold;
- domains that are not regular, 3-D and cartesian, such as surfaces.

On write, a transformation is refused when it does not map `RASmm` to
`RASmm`, or when it is neither an affine, a dense field nor a B-spline.
"""

__all__ = [
    "X5BSplineField",
    "X5CoordinatesField",
    "X5DisplacementField",
    "X5Domain",
    "X5Header",
    "X5Node",
    "X5Transform",
    "X5TransformReaderWriter",
]

from ._blocks import X5BSplineField, X5CoordinatesField, X5DisplacementField
from ._struct import X5Domain, X5Header, X5Node
from ._xform import X5Transform, X5TransformReaderWriter

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses rather than relabels.
from . import _converters  # noqa: E402, F401  isort: skip
