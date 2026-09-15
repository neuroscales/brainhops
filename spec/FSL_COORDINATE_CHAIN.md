# FSL transformation IO — coordinate chain (review aid)

This note states, with matrices, exactly what the FSL readers build, and
where the pixdim scaling and the conditional x-flip enter for the
reference and the moving image. It is a review aid for the Fable review
and is **not** shipped in `src/`.

All conventions below were checked against **fslpy** (`fsl.transform.flirt`,
`fsl.transform.fnirt`, `fsl.transform.nonlinear`, `fsl.data.image`) by
running it, not from memory. The brainhops chains were verified
numerically against fslpy for both a "neurological" (det > 0) and a
"radiological" (det < 0) image (see report).

## 0. Notation and the brainhops storage direction

A `Transformation(input=I, output=O)` maps **coordinates** from `I` to
`O`: `t(x_I) = x_O`. Per the `Transformation` docstring, an image that is
*deformed* from space A to space B is stored as the coordinate map
`input=B, output=A`. For a registration of a *moving* image onto a
*reference*, the image travels moving → reference, so the stored
coordinate map runs **reference → moving**. This matches the existing ITK
reader, which stores ITK's native fixed(=reference) → moving direction
without inverting. So every FSL transform below is built to map

    reference world (RAS)  ->  moving world (RAS).

## 1. FSL scaled-mm ("FSL coordinates") affine — per image

FSL works internally in *scaled-mm* (a.k.a. *FSL*) coordinates: voxel
indices scaled by pixel size, with the x-axis flipped when the image is
stored "neurologically". fslpy (`Nifti.generateAffines`) builds, for an
image with voxel→world affine `V2W`, spatial shape `(nx, ny, nz)` and
positive pixdims `(px, py, pz)`:

    V2F = diag(px, py, pz, 1)                     # voxel -> scaled-mm
    if det(V2W[:3,:3]) > 0:            # "neurological" storage
        x_off = (nx - 1) * px
        FLIP  = [[-1, 0, 0, x_off],
                 [ 0, 1, 0,     0],
                 [ 0, 0, 1,     0],
                 [ 0, 0, 0,     1]]
        V2F = FLIP @ V2F

So `V2F = FLIP? @ diag(pixdim)`.  Key points, both **checked against
`img.getAffine('voxel','fsl')`**:

- **pixdim** are magnitudes (`abs(header.get_zooms()[:3])`); the raw
  `pixdim[0]` qfac sign is not used here.
- The flip is decided by `det(V2W) > 0`, where `V2W` is the *best* affine
  (sform if set, else qform) — **not** by the pixdim qfac.
- The flip offset is `(nx - 1) * px`, using the reference/own **shape**.
  So the scaled-mm affine of an image depends on both its pixdim and its
  x-shape.

Derived affines used below (per image):

    F2V = inv(V2F)                 # scaled-mm -> voxel
    F2R = V2W @ F2V                # scaled-mm -> world (RAS)
    R2F = V2F @ inv(V2W)           # world (RAS) -> scaled-mm
    R2V = inv(V2W)                 # RAS -> voxel
    V2R = V2W                      # voxel -> RAS

`ref` denotes the reference image, `mov` the moving/source image.

## 2. FLIRT `.mat`

A FLIRT `.mat` is a 4×4 affine `M`. **Per fslpy's own module docstring:**
"FLIRT matrices ... encode a transformation from **source** image FSL
coordinates to **reference** image FSL coordinates." That is, the *raw
matrix* is

    M :  moving scaled-mm  ->  reference scaled-mm.

brainhops needs the opposite direction (reference → moving, §0), so the
core uses `inv(M)`. The reader returns a `Sequence` mapping reference RAS
→ moving RAS, built as the fully explicit chain:

    ref RAS --R2V(ref)--> ref voxel --V2F(ref)--> ref scaled-mm
            --inv(M)--> mov scaled-mm --F2V(mov)--> mov voxel
            --V2R(mov)--> mov RAS

As a matrix product (right-to-left):

    A = V2R(mov) @ F2V(mov) @ inv(M) @ V2F(ref) @ R2V(ref)
      = F2R(mov) @ inv(M) @ R2F(ref)

- **Reference** pixdim + x-flip enter in `V2F(ref)` (step 2).
- **Moving** pixdim + x-flip enter in `F2V(mov)` (step 4).
- Reference voxel↔RAS is the reference sform (`R2V(ref)`); moving
  voxel↔RAS is the moving sform (`V2R(mov)`).

**Verified:** `A == inv(fromFlirt(M, mov, ref, 'world', 'world'))` for
random `M` and mismatched-handedness ref/mov images.

> Convention flag for review: the issue text says the `.mat` "maps
> reference-scaled-mm → moving-scaled-mm". The *raw* fslpy matrix is the
> other way (moving → reference); the reference → moving direction is what
> brainhops stores, and is obtained by inverting `M`. Both readings agree
> on the final stored transform; they differ only on which direction the
> raw file is described as. The implementation follows fslpy for the raw
> direction and inverts to reach the brainhops direction.

`src`/`moving` and `ref`/`reference` images must be supplied (the `.mat`
carries no geometry). They are accepted as `load`/`from_*` keyword
arguments or set on the object.

## 3. FNIRT warp / deformation field (intent 2006)

Per fslpy (`fnirt._readFnirtDeformationField`), a FNIRT deformation field
is defined **on the reference grid**, with `srcSpace='fsl'`,
`refSpace='fsl'`; it maps reference → source, both in scaled-mm. Each
reference voxel holds either:

- **absolute**: the source scaled-mm coordinate directly, or
- **relative**: a displacement `d` with `source_fsl = d + ref_fsl(voxel)`,
  where `ref_fsl(voxel) = V2F(ref) @ [voxel, 1]` is the reference
  scaled-mm coordinate of that voxel.

FNIRT files do not record which; the type is inferred (fslpy
`detectDeformationType`), and can be overridden by the caller. The
reader stores the field as **absolute source scaled-mm coordinates** and
returns a `Sequence` mapping reference RAS → moving RAS:

    ref RAS --R2V(ref)--> ref voxel
            --[CoordinatesField: voxel -> abs source scaled-mm]-->
            mov scaled-mm --F2V(mov)--> mov voxel --V2R(mov)--> mov RAS

- **Reference** pixdim + x-flip enter **only** in the relative→absolute
  reconstruction `abs = d + V2F(ref) @ voxel` (and in the detection
  heuristic). For an *absolute* field the reference scaled-mm is not used
  at all; the reference sform still supplies `R2V(ref)`.
- **Moving** pixdim + x-flip enter in `F2V(mov)`.

The field is a `CoordinatesField` (grid → coordinate), exactly like the
existing SPM `y_` reader, but the stored coordinates are source scaled-mm
rather than RAS, so the moving scaled-mm→voxel→RAS tail converts them.

### Relative vs absolute detection

fslpy's heuristic, reproduced exactly:

    C   = V2F(ref) applied to every reference voxel index   # ref scaled-mm grid
    std_abs = data.std over spatial axes, summed over the 3 components
    std_rel = (data - C).std over spatial axes, summed
    type = 'absolute' if std_abs > std_rel else 'relative'

Rationale: absolute coordinates grow with position (large spatial std);
relative displacements are small offsets (small std). The caller may pass
`deformation_type='absolute'|'relative'` to bypass the heuristic.
**Verified:** detection returns 'absolute'/'relative' matching fslpy, and
both the absolute and relative chains reproduce
`fromFnirt(field,'world','world')` at every voxel.

`src`/`moving` must be supplied. `ref`/`reference` defaults to the warp
file's own header + shape (the FNIRT convention: the warp field lives on
the reference grid), and may be overridden.

## 4. FNIRT coefficient field (intent 2007 cubic / 2009 quadratic) — deferred

A coefficient field stores B-spline coefficients on a coarse knot grid
(knot spacing in pixdim), the reference pixdims in `intent_p1..3`, and the
initial FLIRT `srcToRefMat` in the sform. Turning it into a displacement
requires evaluating the FNIRT quadratic/cubic B-spline basis on the knot
grid and folding in the initial affine (fslpy
`coefficientFieldToDeformationField`). That spline evaluation is a
separate numerical component that cannot be verified from the scaled-mm
conventions alone, so this reader **recognizes** coefficient fields and
extracts their parameters, but raises an informative error when the
transformation chain is requested. This is the one deferred item flagged
for review (see report).
