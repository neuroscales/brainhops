"""
Readers and writers for AFNI transformations: affine matrices
(`.aff12.1D`, `.1D`) and nonlinear warps (`3dQwarp`'s `_WARP` datasets,
as AFNI datasets or NIfTI files).

| Class               | Files                       | Hint                |
| ------------------- | --------------------------- | ------------------- |
| [`AfniAffine`][]    | `x.aff12.1D`, `x.1D`        | `"afni.aff12"`      |
| [`AfniBrikWarp`][]  | `x_WARP+tlrc.HEAD`, `.BRIK` | `"afni.warp.brik"`  |
| [`AfniNiftiWarp`][] | `x_WARP.nii[.gz]`           | `"afni.warp.nifti"` |

All of them answer to `hint="afni"`, and the two warps to
`hint="afni.warp"`. Only numpy is needed, but for NIfTI warps (nibabel).

```python
import brainhops.io as io

affine = io.load("anat_al.aff12.1D")       # an AfniAffine, LPS -> LPS
motion = io.load("epi_vr.aff12.1D", volume=12)   # one of many rows
warp = io.load("anat_WARP+tlrc.HEAD")      # an AfniBrikWarp, RAS -> RAS
io.save(warp, "anat_WARP.nii.gz")          # an AfniNiftiWarp
```

Conventions
-----------
Everything AFNI computes is in **DICOM order**: `x` increases to the
left, `y` to the back, `z` up -- LPS millimetres, which AFNI calls `RAI`
(see [`brainhops.io.base.afni`][]).

**Affine matrices.** A matrix file holds the 12 numbers of a `(3, 4)`
matrix `M`, row by row. `3dAllineate -1Dmatrix_save` writes "the
coordinate transformation from base to source DICOM coordinates":
`x_source = M @ x_base`, a "pull" from the base's grid, which is the
direction the data model resamples with -- so `M` is read as it is, from
`LPSmm` to `LPSmm`. A file may hold one row per volume of the source
(`3dvolreg`, `3dAllineate` on a time series); `3dQwarp -allineate`,
`align_epi_anat.py` and `cat_matvec -ONELINE` write the same layout,
and `cat_matvec` writes a single matrix as 3 lines of 4 numbers by
default (and with `0 0 0 1` as a fourth line with `-4x4`). AFNI also
reads 9 numbers as a matrix without shift.

**Warps.** A warp dataset has three sub-bricks, `x_delta`, `y_delta`,
`z_delta`: the DICOM displacement at each point of the base's grid, in
millimetres, to the corresponding point of the source --
`x_source = x + u(x)`, again a "pull". The components are in DICOM order
whatever the orientation of the grid. A warp is read here as a field of
RAS displacements (`x` and `y` negated, an exact change of frame), so
that it shares its chain with the other RAS fields. AFNI writes a warp
to NIfTI as a `(X, Y, Z, 1, 3)` array with no intent code and its
displacements unchanged (in DICOM, not RAS), and an AFNI header
extension holding the sub-brick labels.

**Obliquity.** AFNI programs compute on each dataset's *cardinal* grid
-- the closest grid whose axes run along the DICOM axes -- and ignore
the obliquity of its true geometry (`IJK_TO_DICOM_REAL`, or the NIfTI
sform/qform). Matrices and warps therefore map cardinal coordinates.
[`AfniAffine`][] takes the `base` and `source` images to map their true
coordinates instead, and [`cardinal_to_real`][] gives the correction to
compose with a warp. For datasets that are not oblique -- templates, and
the usual bases of `3dQwarp` -- the two are the same.

What was verified, and how
--------------------------
Checked in the AFNI sources (github.com/afni/afni, `src/`):

- `3dAllineate.c`: the help of `-1Dmatrix_save` (base to source DICOM,
  `Xin = M Xout`, one row per sub-brick, `.aff12.1D` suffix), the
  comment it writes (`# 3dAllineate matrices (DICOM-to-DICOM,
  row-by-row):`), `-1Dmatrix_apply` reading a 3x4 array as one matrix,
  and the cardinal matrices it computes on (`DSET_CMAT(dset,
  use_realaxes)` with `use_realaxes = 0` by default).
- `cat_matvec.c`: the layouts it reads (3 lines of 4, rows of 12 in a
  `.aff12.1D` file, 9 numbers) and writes (`-ONELINE`, `-4x4`).
- `3dQwarp.c`: the help of `-prefix` and the sections "CLARIFICATION
  about the very confusing forward and inverse warp issue" and "STORAGE
  of 3D warps in AFNI" (displacements in DICOM mm from the base grid to
  the source, `xd, yd, zd` in DICOM order whatever the grid's order, no
  header code marking a warp, linear extrapolation outside the grid).
- `mri_nwarp.c`: `IW3D_to_dataset` (the labels `x_delta`, `y_delta`,
  `z_delta`; displacements as `cmat @ index displacement`) and
  `IW3D_from_dataset` (`cmat = dset->daxes->ijk_to_dicom`, the cardinal
  matrix).
- `thd_niftiwrite.c`: a dataset of several sub-bricks that is not a
  time series is written with `dim[0] = 5` and its sub-bricks in
  `dim[5]`, no intent code, and the AFNI extension
  (`nifti_set_afni_extension`, code 4, the NIML of every attribute).
- `thd_niftiread.c`: the qform is preferred over the sform, and the
  cardinal grid of a NIfTI file keeps the matrix's translation as its
  origin (while `THD_daxes_from_mat44`, used for AFNI's own headers,
  projects it on each axis).
- `thd_nimlatr.c`, `niml/niml_elemio.c`: how attributes are written
  into the NIML extension (`<AFNI_atr atr_name="BRICK_LABS" ...>`, NUL
  characters written as `~`).
- NiTransforms (`nitransforms/io/afni.py`) agrees: it reads the matrix
  as LPS-to-LPS and flips it into RAS, corrects for the obliquity of the
  reference and moving images the same way, and negates the first two
  components of an AFNI NIfTI warp to make them RAS.

Assumed, not checked against AFNI itself (no AFNI binaries here):

- the exact text AFNI's NIML parser accepts for the minimal extension
  written with a NIfTI warp (written as `NI_write_element` formats one);
- that no AFNI tool writes warps without the `x_delta` labels: such a
  dataset is read as an image unless asked for with a hint.

Not supported
-------------
- `3dAllineate -1Dparam_save` files (`.param.1D`): parameters, not
  matrices.
- The command-line warp syntax of `3dNwarpApply` and `3dNwarpCat`
  (`INV(...)`, `SQRT(...)`, a list of warps and matrices, `IDENTITY`,
  `MATRIX(...)`): it describes how to combine files, not a file. Read
  each file, and compose or invert the transformations.
- Index warps (`3dQwarp -inwarp`), which have no use outside debugging.
- AFNI's linear extrapolation of a warp outside its grid (the data model
  has no such boundary condition: the nearest value is used).
- A file of several matrices as one transformation: the data model has
  no affine that varies along time, so one row is read at a time.
"""

__all__ = [
    "AfniAffine",
    "AfniBrikWarp",
    "AfniWarp",
    "cardinal_to_real",
]

from brainhops.io.base._dispatch import register_missing_format

from ._affine import AfniAffine
from ._utils import cardinal_to_real
from ._warp import AfniBrikWarp, AfniWarp

try:
    from ._nifti import AfniNiftiWarp

    __all__ += ["AfniNiftiWarp"]
except ImportError:  # nibabel is optional
    register_missing_format(["afni.warp.nifti"], "nibabel", "nibabel")
