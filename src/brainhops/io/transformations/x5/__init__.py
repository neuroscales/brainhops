"""Reader and writer for BIDS X5 transformation files.

X5 is the HDF5 format of BIDS extension proposal BEP014, which is still a
draft. This package follows its two implementations: nitransforms 25.0 and
later (`Version = 1`, the current layout) and fslpy 3.x
(`Version = "0.1.0"`, an earlier layout, which is read and written back in
the current layout).

!!! warning "A draft format"
    What neither implementation defines is left unimplemented rather than
    guessed; see "Not supported".

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
[`brainhops.io.transformations.itk.h5`][]. Arrays are read as h5py returns
them, such as `(X, Y, Z, 3)`; a vector axis that `DimensionKinds` places
elsewhere is moved last.

## Direction

X5 has no named spaces: each transform maps the RAS millimetre world of an
image A to that of an image B, and is read from `RASmm` to `RASmm`. Which
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
`"deformations"`. A displacement field is read as a chain from RAS to the
`Domain` voxels, displacements in voxel units, and voxels back to RAS, as a
NIfTI `DISPVECT` field is (see [`brainhops.io.transformations.nifti`][]).
It is interpolated linearly, with the nearest value outside the grid.
nitransforms uses cubic splines and no displacement outside the grid, so
results differ away from the grid nodes.

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
convention of the ITK `BSplineTransform` (see
[`brainhops.io.transformations.itk`][]), which places it in LPS instead.
On write, coefficients of another degree or boundary condition are refitted,
and the knot grid serves as the `Domain` that nitransforms requires.

## Chains

nitransforms stores a chain as a string such as `"0/1/2"` in
`/TransformChain/<n>`, applied as `f2(f1(f0(x)))`. This is the order of a
brainhops `Sequence`, so the chain is read as `Sequence([t0, t1, t2])`.
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
- an `ArrayLength` greater than 1, a stack of one affine per volume (the
  nitransforms `LinearTransformsMapping`), which the datamodel cannot hold;
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
    "X5TransformParser",
]

from ._blocks import X5BSplineField, X5CoordinatesField, X5DisplacementField
from ._struct import X5Domain, X5Header, X5Node
from ._xform import X5Transform, X5TransformParser
