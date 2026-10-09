# FSL transformation IO: the coordinate chain

This document states, with explicit matrices, what the FSL readers build
from a FLIRT `.mat` file and from a FNIRT warp, and where the pixel sizes
and the conditional x-flip of the reference and moving images enter. The
chains were checked numerically against fslpy (`fsl.transform.flirt`,
`fnirt` and `nonlinear`, and `fsl.data.image`), for images stored both
neurologically (`det > 0`) and radiologically (`det < 0`).

## 0. Notation and the brainhops storage direction

`Transformation(input=I, output=O)` maps coordinates from `I` to `O`:
`t(x_I) = x_O`. As the `Transformation` docstring explains, an image that
is deformed from space A to space B is stored as the coordinate map with
`input=B` and `output=A`. In a registration of a moving image onto a
reference image, the image travels from the moving space to the reference
space, so the stored coordinate map runs from the reference to the moving
image. The ITK readers follow the same convention, since ITK natively maps
the fixed (reference) space to the moving space and needs no inversion.
Every FSL transformation is therefore built to map:

```
    reference world (RAS)  ->  moving world (RAS).
```

In what follows, `ref` is the reference image and `mov` is the moving
(source) image.

## 1. The FSL scaled-mm affine of an image

FSL works in scaled-mm coordinates: voxel indices multiplied by the pixel
size, with the x axis flipped when the image is stored neurologically.
For a voxel-to-world affine `V2W`, a shape `(nx, ny, nz)` and positive
pixel sizes `(px, py, pz)`, fslpy (`Nifti.generateAffines`) computes:

```
    V2F = diag(px, py, pz, 1)                     # voxel -> scaled-mm
    if det(V2W[:3,:3]) > 0:            # "neurological" storage
        x_off = (nx - 1) * px
        FLIP  = [[-1, 0, 0, x_off],
                 [ 0, 1, 0,     0],
                 [ 0, 0, 1,     0],
                 [ 0, 0, 0,     1]]
        V2F = FLIP @ V2F
```

In short, `V2F = FLIP? @ diag(pixdim)`. Three details were checked against
`img.getAffine('voxel', 'fsl')`:

- The pixel sizes are magnitudes (`abs(header.get_zooms()[:3])`), and the
  sign of the raw `qfac` in `pixdim[0]` is not used.
- The flip is decided by `det(V2W) > 0`, where `V2W` is the best affine
  (the sform if it is set, and the qform otherwise), not by the `qfac`.
- The flip offset `(nx - 1) * px` uses the image's own shape, so the
  scaled-mm affine depends on both the pixel sizes and the size of the x
  axis.

The other per-image affines follow:

```
    F2V = inv(V2F)                 # scaled-mm -> voxel
    F2R = V2W @ F2V                # scaled-mm -> world (RAS)
    R2F = V2F @ inv(V2W)           # world (RAS) -> scaled-mm
    R2V = inv(V2W)                 # RAS -> voxel
    V2R = V2W                      # voxel -> RAS
```

## 2. FLIRT `.mat`

A FLIRT file holds a 4x4 affine `M`. As the fslpy documentation states,
`M` maps the FSL coordinates of the source image to the FSL coordinates of
the reference image, so the raw matrix is `M : moving scaled-mm ->
reference scaled-mm`. brainhops needs the opposite direction and uses
`inv(M)`. The reader (`FlirtTransform`) returns an affine that maps
reference RAS to moving RAS through the following chain:

```
    ref RAS --R2V(ref)--> ref voxel --V2F(ref)--> ref scaled-mm
            --inv(M)--> mov scaled-mm --F2V(mov)--> mov voxel
            --V2R(mov)--> mov RAS
```

As a matrix product, read from right to left:

```
    A = V2R(mov) @ F2V(mov) @ inv(M) @ V2F(ref) @ R2V(ref)
      = F2R(mov) @ inv(M) @ R2F(ref)
```

The pixel sizes and x-flip of the reference image enter through
`V2F(ref)`, and those of the moving image through `F2V(mov)`. The
voxel-to-RAS affines are the sforms of the two images, used as `R2V(ref)`
and `V2R(mov)`. The chain was verified to satisfy
`A == inv(fromFlirt(M, mov, ref, 'world', 'world'))` for random matrices
`M` and for images of mismatched handedness.

A `.mat` file records no geometry, so both images must be supplied, either
as keyword arguments to `load` or `from_file` (`reference=`, `moving=`) or
by setting them on the object.

## 3. FNIRT deformation field (intent 2006)

As fslpy reads it (`fnirt._readFnirtDeformationField`), a FNIRT
deformation field is defined on the reference grid and maps the reference
to the source, both in scaled-mm (`srcSpace='fsl'`, `refSpace='fsl'`).
Each reference voxel holds either the absolute scaled-mm coordinate of the
corresponding source point, or a relative displacement `d`, with
`source_fsl = d + ref_fsl(voxel)` and `ref_fsl(voxel) = V2F(ref) @ [voxel, 1]`.

The file does not record which of the two it holds, so the reader infers
it, and the caller can override the inference. The map from reference RAS
to moving RAS is:

```
    ref RAS --R2V(ref)--> ref voxel
            --[voxel -> source scaled-mm, from the field]-->
            mov scaled-mm --F2V(mov)--> mov voxel --V2R(mov)--> mov RAS
```

The reader (`FnirtWarpField`) converts an absolute field to a relative one
(`d = abs - V2F(ref) @ voxel`) and returns an `ImmutableSequence` of three
transformations: reference RAS to warp-grid voxels, a `DisplacementField`
expressed in warp-grid units, and warp-grid voxels to moving RAS. The
scaled-mm affines and the change of units are absorbed into the two outer
affines and into the field values, so the sequence equals the chain above.

The pixel sizes and x-flip of the reference image enter only through the
conversion between absolute and relative coordinates and through the
detection heuristic. The reference sform still provides `R2V(ref)`. The
pixel sizes and x-flip of the moving image enter through `F2V(mov)`.

### Relative versus absolute detection

The reader reproduces the fslpy heuristic exactly:

```
    C   = V2F(ref) applied to every reference voxel index   # ref scaled-mm grid
    std_abs = data.std over spatial axes, summed over the 3 components
    std_rel = (data - C).std over spatial axes, summed
    type = 'absolute' if std_abs > std_rel else 'relative'
```

Absolute coordinates grow with position and have a large spread across the
volume, whereas relative displacements are small offsets. The caller can
bypass the heuristic with `deformation_type="absolute"` or
`deformation_type="relative"`. The detection was verified to match fslpy,
and both readings reproduce `fromFnirt(field, 'world', 'world')` at every
voxel.

The moving image must be supplied (`moving=`, or its aliases `mov=` and
`src=`). The reference image (`reference=` or `ref=`) defaults to the
header and shape of the warp file itself, since the warp lives on the
reference grid, and it can be overridden.

## 4. FNIRT coefficient fields (intents 2007 and 2009)

A coefficient field stores the coefficients of a cubic (intent 2007) or
quadratic (intent 2009) B-spline basis on a coarse knot grid overlaid on
the reference image. The knot spacing is stored in the pixel sizes, the
reference pixel sizes in `intent_p1` to `intent_p3`, and the initial FLIRT
affine (`srcToRefMat`) in the sform.

`FnirtWarpField` reads both kinds with the same chain as a deformation
field. The spline coefficients are kept on the knot grid with the degree
that the intent names, the knot spacing is rescaled when the current
reference has different pixel sizes, and the inverse of the initial affine
is applied on top of the spline displacement. The basis is evaluated only
when the sequence is computed, so the field is never expanded onto the
reference grid when it is read. A coefficient field carries the geometry
of neither image, so both the reference and the moving image are required.

A discrete-cosine-transform coefficient field (intent 2008) is recognised
but not supported: using it raises `NotImplementedError`, and it should be
converted to a deformation field with FSL's `fnirtfileutils` first.
