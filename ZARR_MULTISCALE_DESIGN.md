# OME-Zarr multiscale fields — design note

Review aid for issue #52. This file is a review artifact and is not part
of the shipped package (it lives at the repository root, not under
`src/`). It records the class hierarchy, the exact sandwich matrices, the
level-selection policy, the byte-for-byte re-emit structure, and the
refused cases.

## Summary of the change

- Adds `MultiscaleCoordinatesField` and `MultiscaleDisplacementField` to
  `src/brainhops/datamodel/transformations.py`. Each subclasses its
  single-scale field type, so it is type-transparent to the compose
  engine and needs no new composers or dispatch entries.
- Adds the option-(b) sandwich builder, the voxel-to-voxel
  normalizations, the level-selection helpers, and the refusal checks to
  the same module.
- Wires level selection into `SingleScaleImage.reslice` in
  `src/brainhops/datamodel/images.py`.
- Adds `OmeZarrField`, a `Sequence`-subclass reader, in
  `src/brainhops/io/transformations/zarr/`. It holds the raw arrays and
  the raw OME metadata, and builds the sandwich lazily.
- `LayeredTransformation` does not exist on `main` (it is only on the
  draft branch `origin/james/zarr`), so #52's "remove it" is satisfied by
  not porting it. None of its bugs are carried over.

## Class hierarchy

```
Transformation
├── CoordinatesField
│   ├── CartesianField
│   └── MultiscaleCoordinatesField        (new)
├── DisplacementField
│   └── MultiscaleDisplacementField       (new)
└── Sequence
    ├── Geometry
    ├── SPMCoordinatesField               (existing, the pattern mirrored)
    └── OmeZarrField                       (new, io/transformations/zarr)
```

`MultiscaleCoordinatesField` **is** a `CoordinatesField`, and
`MultiscaleDisplacementField` **is** a `DisplacementField`. The compose
engine dispatches on `type(x)` through a class-distance search, and reads
the fields it needs (`.field`, `.order`, `.bound`, `.coeff`). The active
level's array is served through the `.field` property, so a multiscale
field composes exactly as its active single-scale level would, and
composition collapses the pyramid to that level. This is verified by
`test_composes_like_its_active_single_scale_level`.

### Multiscale field state

- `levels`: the array of each resolution level, finest first, shape
  `(*grid, ndim)`.
- `level_transforms`: the transformation `L_i` that maps level `i`'s grid
  to the finest grid, finest first. `L_0` is the identity. Each `L_i` is
  a scaling and a shift. Stored, not inferred, because pyramid builders
  differ in sub-pixel alignment.
- `level`: the active level index, default `0` (finest).
- `.field`: property returning `levels[level]` (or an explicit override
  set through the setter once the field has been collapsed).

The same-type converters for these two classes drop the derived `.field`
on rebuild (mirroring `CartesianField`), so `at_level` and any `.to(...)`
rebuild keep the pyramid rather than freezing one level's array onto
`field`.

## The (b) sandwich

`place_ome_field(field, placement)` returns

```
Sequence([placement.inverse(), field, placement])
```

which reads right-to-left as `placement @ field @ placement.inverse()`.
This is structurally identical to how `SPMCoordinatesField` is a
`Sequence` of a world-to-voxel affine and a field. `placement` is
`xform_0`, the finest level's voxel-to-world transformation (the OME
`coordinateTransformations` of the finest dataset).

Let `xform_0` have homogeneous form with linear part `L` (an `ndim ×
ndim` matrix) and translation `t`:

```
xform_0(v) = L v + t
inverse(xform_0)(w) = L⁻¹ (w − t)
```

The field inside the sandwich is stored voxel-to-voxel. The values read
from OME are in world units, so they are normalized:

- **Displacement** (`normalize_ome_displacement`):
  `D = disp · L⁻ᵀ`, i.e. `D[..., :] = disp[..., :] @ inv(L).T`.
  Only the linear part is used, because a displacement is a free vector.
  This is the `D = disp · inv(L)` of the issue.

- **Coordinate** (`normalize_ome_coordinates`):
  `C = inverse(xform_0)(coords)`, i.e.
  `C[..., :] = coords[..., :] @ inv(L).T − inv(L).T @ t`, computed as
  `coords @ M⁻¹[:, :-1].T + M⁻¹[:, -1]` where `M⁻¹` is the inverse affine
  matrix. A coordinate is an absolute point, so the full inverse affine
  is used, translation included.

Both normalizations are the identity when `xform_0` is a unit isotropic
scaling (`L = I`, `t = 0`), so a field on a 1 mm isotropic grid reads the
same whether or not the normalization is applied. **This is the trap the
tests guard against**: every normalization test uses an anisotropic
(`diag(2, 1/2)`) and rotated placement, so `inv(L)` is neither a scalar
nor diagonal and a wrong normalization is caught.

### Why voxel-to-voxel

With `D` (or `C`) voxel-to-voxel and the sandwich `xform_0 @ field @
inverse(xform_0)`, an incoming world coordinate `w` is first mapped to
the field's voxel grid by `inverse(xform_0)`, the field is applied in
voxel coordinates, and the result is mapped back to world by `xform_0`.
The whole sandwich is world-to-world, matching the OME field's own
world-to-world semantics, while the interpolation happens on the grid the
array is actually sampled on.

## Level selection

`multiscale_level_for(field, target)` returns the level index whose
downsampling factor best matches `target`, measured in log-scale (so the
levels below and above the target are weighed evenly; ties go to the
finer level). `target` is a per-axis factor relative to the finest grid,
a scalar, or a transformation whose linear scale supplies the factor. A
level's factor is the per-axis Euclidean norm of the columns of the
linear part of `L_i`, which is rotation-invariant.

`resolve_multiscale_level(transformation, target)` is the reslice hook.
It recognizes two shapes and otherwise returns the transformation
unchanged (so ordinary reslices are untouched, and the finest level stays
active by default):

1. A sandwich `Sequence([~xform_0, field, xform_0])` produced by
   `place_ome_field`. The target world resolution
   `scale(target)` is divided by the finest world resolution
   `scale(xform_0)` to get the relative factor, the matched level is
   chosen, and the sandwich is rebuilt at that level with the placement
   adjusted to `xform_0 @ L_level`.
2. A bare multiscale field, matched directly against `scale(target)`.

`SingleScaleImage.reslice` calls `resolve_multiscale_level(self.transformation,
geometry.transformation)` before composing, so reslicing onto a coarse
grid samples a coarse level. This is verified by
`test_resolve_multiscale_level_selects_inside_a_sandwich` and the
`reslice` regression is preserved (a transformation without a multiscale
field is returned identically, and the target resolution is not even
inspected in that case).

**Fork flagged for review** — see the last section. The bare-field branch
of `resolve_multiscale_level` treats `scale(target)` as a factor relative
to the finest grid, which is exact only when the finest field grid is
unit-isotropic. Inside a sandwich the comparison is against `xform_0` and
is exact. The policy is deliberately conservative (default to finest, and
never resample finer than requested by more than the nearest log-step),
but the precise grid→level rule is the one policy knob that #52 leaves to
implementation.

## Byte-for-byte re-emit

`OmeZarrField` stores `raw_levels` (the arrays as read, in world units),
`placement`, `level_transforms`, `axes`, and `ome_metadata` (the metadata
object exactly as read). The sandwich is derived on demand from these by
the `.field` and `.transformations` properties; the raw inputs are never
rewritten. `to_ome_metadata()` returns the identical `ome_metadata`
object, so a read that is not modified re-emits the OME metadata
unchanged. This is verified by
`test_untouched_read_re_emits_metadata_unchanged` (identity, not just
equality).

The normalization and the placement live entirely in the derived
sandwich, so they never touch the stored arrays or metadata. A consumer
that only reads and writes back therefore round-trips the bytes; a
consumer that composes or reslices works against the derived, normalized
sandwich.

## Refused cases

- **Mixed axes** (`check_ome_axes`): a field whose axes mix the
  `displacement` type and the `coordinate` type in one field is refused
  with `OmePlacementError`. Axes are classified by `axis.type`; all
  `displacement` is a displacement field, none is a coordinate field, and
  any-but-not-all is refused. The draft branch silently converted the
  displacement channels to coordinates instead; #52 refuses.
- **Non-linear-placed displacement**
  (`check_ome_displacement_placement`): a displacement field whose
  placement resolves to a coordinate or displacement field (rather than
  an affine) is refused with `OmePlacementError`, because a displacement
  is normalized by a single linear part that a non-linear placement does
  not have.

## Adopted vs rewritten from PR #22 (`origin/james/zarr`, "not yet tested")

- **Not adopted**: `LayeredTransformation` and `OmeZarrTransformation`
  (built on it). Their bugs — the shared class attribute `active_layer`,
  the `input`/`output` properties shadowing struct fields, `inverse()`
  dropping the spaces, and in-place mutation of the zarr-backed array —
  are designed out by the multiscale-field model.
- **Not adopted**: the `io/transformations/common/` refactor of the NIfTI
  IO. It diverged from `main`, which since landed a more complete
  `io/transformations/nifti/` (PRs #44, #53). Reusing `common/` would
  regress `main`.
- **Rewritten**: the displacement/coordinate axis classification. The
  draft's `_displacement_axis_mask` idea is kept, but "some axes are
  displacement" now refuses rather than silently converting, per #52.
- **Deferred (see fork)**: `io/base/omezarr.py` (the `OmeZarrParser`
  built on `abczarr`) and the OME-Zarr image reader.

## Forks for the Fable reviewer

1. **`abczarr` as a brainhops dependency + the file-reading path.** The
   `OmeZarrField` reader is complete and tested from in-memory arrays and
   metadata, but its OME-Zarr *file* parsing (turning an `abczarr` node
   and its `coordinateTransformations` into `raw_levels`, `placement`,
   `level_transforms`, and `axes`) is not implemented here. It requires
   adopting `abczarr` as a brainhops dependency and reconciling the
   draft's `OmeZarrParser`/`io/base/omezarr.py` against `main`'s
   `io/base` registry. `abczarr` is not currently a brainhops
   dependency; taking it on is an architectural decision #52 does not
   settle. Recommendation: land the datamodel and the in-memory reader
   now; scope the `abczarr` integration and the format-registry wiring as
   a follow-up, with an OME-Zarr round-trip fixture to lock the
   byte-for-byte guarantee end to end.
2. **The bare-field level→grid policy** (above): exact inside a sandwich,
   approximate for a bare field without a placement. Recommendation:
   accept the conservative log-nearest policy, and require a placement
   (the sandwich form) for exact selection.
