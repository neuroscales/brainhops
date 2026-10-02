"""
Reader and writer for BIDS X5 (`.x5`) transformation files.

X5 is the HDF5-based transformation format drafted by the BIDS
extension proposal BEP014 ("Transforms"). The draft is not final; this
module follows what its two implementations read and write, for
interoperability with them:

- **nitransforms** >= 25.0 (`nitransforms/io/x5.py`, and the `to_x5` /
  `from_x5` functions of `linear.py`, `nonlinear.py` and `manip.py`),
  which writes the current layout, `Version = 1`;
- **fslpy** 3.x (`fsl/transform/x5.py`), which writes an earlier
  layout, `Version = "0.1.0"`. It is read, and written back in the
  current layout.

!!! warning "A draft format"
    Everything below that is not in nitransforms or fslpy is left
    unimplemented rather than guessed: see "Not supported".

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

Unlike ITK's `.h5`, which also keeps its transforms under a
`TransformGroup`, an X5 file has no `ITKVersion`, and says what it is in
its root `Format` attribute, which is what tells the two apart. An X5
file named `.h5` is therefore read by this reader, not by
[`brainhops.io.transformations.itk.h5`][].

Arrays are read as `h5py` returns them: a field written by nitransforms
or fslpy from a `numpy` array of shape `(X, Y, Z, 3)` is read with that
shape, with `DimensionKinds` `("space", "space", "space", "vector")`.
When `DimensionKinds` puts the `"vector"` axis elsewhere, it is moved
last.

## Direction

There are no named source and target spaces in X5. Every transform
maps **points** of one world space, A, to points of another, B, in RAS
millimetres -- so each one is read as a brainhops transformation whose
`input` and `output` are both `RASmm`, and which maps an input point to
an output point.

- nitransforms: the affine "maps coordinates from *reference* space
  into *moving* space" (`linear.py`, `Affine.__init__`), and a field
  is sampled on the reference grid (`Domain`), mapping each reference
  point to `x + u(x)` (`nonlinear.py`, `DenseFieldTransform.map`). A
  is the *reference* (fixed) space: the transform "pulls" the moving
  image onto the reference grid.
- fslpy: "X5 files enable a transformation from the world coordinate
  system of image A to the world coordinate system of image B"
  (`x5.py`, module docstring). For a **linear** file, A is the
  *source* and B the *reference* (`writeLinearX5(fname, xform, src,
  ref)` writes `src` to `/A`), so the matrix maps source points to
  reference points. For a **nonlinear** file, A is the *reference*
  (`writeNonLinearX5` writes `field.ref` to `/A`), as with FNIRT.

Both agree on "A to B", which is all that brainhops represents.
*Which image* A is differs between fslpy's linear files and every other
file. Resampling an image with a transformation read here is a matter
of the caller knowing which of the two images it was registered from.

## Fields

A `nonlinear` node with `SubType = "densefield"` (or none) is read as:

| `Representation`                      | Transformation             |
| ------------------------------------- | -------------------------- |
| `"displacements"` (fslpy `relative`)  | [`X5DisplacementField`][]  |
| `"deformations"` (fslpy `absolute`)   | [`X5CoordinatesField`][]   |

nitransforms writes `"deformations"` for a field of absolute
coordinates; `"coordinates"` and `"absolute"` are read as the same. A
field of displacements holds, at each voxel of the `Domain` grid, the
RAS displacement of the point at its centre, and is read as the chain
RAS to voxel, displacements in voxel units, voxel to RAS -- the same
chain as a NIfTI `DISPVECT` field
([`brainhops.io.transformations.nifti`][]), built by the same shared
helpers. It is interpolated linearly, and extended with its nearest
value outside its grid. nitransforms interpolates it with cubic splines
and treats points outside the grid as not displaced, so the two can
differ off the grid nodes.

## B-splines

A `nonlinear` node with `SubType = "bspline"` and `Representation =
"coefficients"` is nitransforms' `BSplineFieldTransform`
(`nonlinear.py`, `to_x5` / `from_x5`), and is read as an
[`X5BSplineField`][]:

- `Transform` holds the B-spline coefficients of a displacement in RAS
  millimetres, one 3-vector per knot, `(X, Y, Z, 3)` (`DimensionKinds`
  `("space", "space", "space", "vector")`);
- `AdditionalParameters` is the voxel-to-RAS affine of the grid of
  knots: knot `k` sits at voxel `k` of that grid;
- `Domain` is the grid of the reference image, on which nitransforms'
  `to_field` samples the field. The transform itself does not depend
  on it.

nitransforms maps a RAS point `x` to `x + sum_k c_k B3(i(x) - k)`
(`nonlinear.py`, `_map_xyz`), where `i(x)` are the coordinates of `x`
in the grid of knots, and `B3` is the tensor product of centred cubic
B-splines (`interp/bspline.py`, `_cubic_bspline`); only knots that
exist contribute, so coefficients beyond the grid are zero. The degree
is not stored: nitransforms evaluates cubics only. This is the chain
RAS to knot voxel, a `DisplacementField` of coefficients (`coeff=True`,
degree 3, zero boundary) rotated into knot units, knot voxel to RAS --
the same chain as a dense field of displacements, and the same
knot-grid convention as an ITK `BSplineTransform`
([`brainhops.io.transformations.itk`][]), whose fixed parameters place
its coefficient grid in LPS.

Any such chain whose input and output are `RASmm` is written as a
`bspline` node, with the knot grid as its `Domain` when it was not read
from a file (nitransforms requires a `Domain`, and has no other grid to
give it).

## Chains

A chain is stored by nitransforms (`manip.py`,
`TransformChain.to_filename`) as a string dataset
`/TransformChain/<n>` listing the indices of its nodes, `"0/1/2"`.
`TransformChain.map` applies them in that order, `f2(f1(f0(x)))`, which
is the order of a brainhops `Sequence`, so the chain `"0/1/2"` is read
as `Sequence([t0, t1, t2])`. Which nodes are read is described in
[`X5Transform.selection`][brainhops.io.transformations.x5.X5Transform.selection].

## Metadata

The datamodel holds no metadata. The JSON `Metadata` of every node --
and its `Domain`, `Inverse`, `Jacobian` and other attributes -- is kept,
as read, on the reader (`X5Transform.nodes`, `X5Transform.header`), and
written back. A transformation built from scratch is written with no
metadata.

## Not supported

These raise an error when the transformation is read (its raw nodes are
still read, and written back unchanged):

- `Type = "composite"`: the draft lists it, but does not say how its
  parts are stored, and neither nitransforms nor fslpy writes one;
- `ArrayLength > 1`: a stack of affines, one per volume of a series
  (nitransforms' `LinearTransformsMapping`), which the datamodel has no
  transformation for;
- domains that are not regular 3-D cartesian grids (surfaces).

These are refused when written: a transformation that does not map
`RASmm` to `RASmm`, or that is neither an affine, a dense field, nor a
cubic B-spline with a zero boundary.
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
