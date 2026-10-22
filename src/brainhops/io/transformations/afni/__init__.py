"""
Readers and writers for AFNI transformations: affine matrices
(`.aff12.1D`, `.1D`) and nonlinear warps (the `_WARP` datasets of
`3dQwarp`, as AFNI datasets or NIfTI files).

| Class               | Files                       | Hint                |
| ------------------- | --------------------------- | ------------------- |
| [`AfniAffine`][]    | `x.aff12.1D`, `x.1D`        | `"afni.aff12"`      |
| [`AfniBrikWarp`][]  | `x_WARP+tlrc.HEAD`, `.BRIK` | `"afni.warp.brik"`  |
| [`AfniNiftiWarp`][] | `x_WARP.nii[.gz]`           | `"afni.warp.nifti"` |

All of them answer to `hint="afni"`, and the two warps to
`hint="afni.warp"`, through their family [`AfniWarpFormat`][]. Only NumPy is
needed, except for the NIfTI warps, which need nibabel.

A matrix file is short and is read in one pass, so its whole content is
held as an [`AfniAffineRaw`][] record by its [`AfniAffineMetadata`][], as
an LTA file is. A BRIK warp holds the header of its dataset as
[`AfniMetadata`][], whose record is an [`AfniRaw`][], and its sub-bricks,
read lazily, in `rawdata`, as an AFNI image does. A NIfTI warp holds the
header and the array of its file as every NIfTI transformation does. The
records and `rawdata` describe the files as they were read. Matrices,
sub-bricks and chains that are assigned are pending until
`consolidate()` encodes them into a copy, or until the object is
written, which writes such a copy and leaves the object as it was. An
object that is read and written again without changes keeps the bytes
of its files.

```python
import brainhops.io as io

affine = io.load("anat_al.aff12.1D")       # an AfniAffine, LPS -> LPS
motion = io.load("epi_vr.aff12.1D", volume=12)   # one of many rows
warp = io.load("anat_WARP+tlrc.HEAD")      # an AfniBrikWarp, RAS -> RAS
nifti = io.transformations.afni.AfniNiftiWarp.from_any(warp)
nifti.save("anat_WARP.nii.gz")             # an AfniNiftiWarp
```

## Conventions

Everything that AFNI computes is in the DICOM order: `x` increases to
the left, `y` to the back and `z` up, which gives LPS millimetres, and
which AFNI calls `RAI` (see [`brainhops.io.common.afni`][]).

A matrix file holds the 12 numbers of a `(3, 4)` matrix `M`, row by
row. `3dAllineate -1Dmatrix_save` writes "the coordinate transformation
from base to source DICOM coordinates", so `x_source = M @ x_base`. This
is a "pull" from the grid of the base, which is the direction in which
the data model resamples, so `M` is read as it is, from `LPSmm` to
`LPSmm`. A file may hold one row per volume of the source (`3dvolreg`,
or `3dAllineate` on a time series), and `3dQwarp -allineate`,
`align_epi_anat.py` and `cat_matvec -ONELINE` write the same layout.
`cat_matvec` writes a single matrix as 3 lines of 4 numbers by default,
and with `0 0 0 1` as a fourth line with `-4x4`. AFNI also reads 9
numbers as a matrix without a shift.

A warp dataset has three sub-bricks, `x_delta`, `y_delta` and `z_delta`,
which hold the DICOM displacement in millimetres at each point of the
grid of the base, to the corresponding point of the source, so that
`x_source = x + u(x)`. This is again a "pull". The components are in the
DICOM order whatever the orientation of the grid. A warp is read here as
a field of RAS displacements, with `x` and `y` negated, which is an exact
change of frame, so that it shares its chain with the other fields of
RAS displacements. AFNI writes a warp to NIfTI as a `(X, Y, Z, 1, 3)`
array without an intent code and with its displacements unchanged, in
DICOM and not in RAS, and with an AFNI header extension that holds the
labels of the sub-bricks.

AFNI programs compute on the *cardinal* grid of each dataset, which is
the closest grid whose axes run along the DICOM axes, and they ignore
the obliquity of its true geometry (`IJK_TO_DICOM_REAL`, or the sform
and qform of a NIfTI file). Matrices and warps therefore map cardinal
coordinates. [`AfniAffine`][] takes the `base` and `source` images to
map their true coordinates instead, and [`cardinal_to_real`][] gives the
correction to compose with a warp. For datasets that are not oblique,
such as templates and the usual bases of `3dQwarp`, the two are the
same.

## What was verified, and how

The following was checked in the AFNI sources (github.com/afni/afni,
`src/`).

- `3dAllineate.c`: the help of `-1Dmatrix_save` (base to source DICOM,
  `Xin = M Xout`, one row per sub-brick, `.aff12.1D` suffix), the
  comment that it writes (`# 3dAllineate matrices (DICOM-to-DICOM,
  row-by-row):`), `-1Dmatrix_apply` reading a 3x4 array as one matrix,
  and the cardinal matrices that it computes on (`DSET_CMAT(dset,
  use_realaxes)` with `use_realaxes = 0` by default).
- `cat_matvec.c`: the layouts that it reads (3 lines of 4, rows of 12 in
  a `.aff12.1D` file, 9 numbers) and writes (`-ONELINE`, `-4x4`).
- `3dQwarp.c`: the help of `-prefix` and the sections "CLARIFICATION
  about the very confusing forward and inverse warp issue" and "STORAGE
  of 3D warps in AFNI" (displacements in DICOM millimetres from the grid
  of the base to the source, `xd, yd, zd` in the DICOM order whatever
  the order of the grid, no header code that marks a warp, linear
  extrapolation outside the grid).
- `mri_nwarp.c`: `IW3D_to_dataset` (the labels `x_delta`, `y_delta` and
  `z_delta`, and displacements as `cmat @ index displacement`) and
  `IW3D_from_dataset` (`cmat = dset->daxes->ijk_to_dicom`, the cardinal
  matrix).
- `thd_niftiwrite.c`: a dataset of several sub-bricks that is not a
  time series is written with `dim[0] = 5` and its sub-bricks in
  `dim[5]`, without an intent code, and with the AFNI extension
  (`nifti_set_afni_extension`, code 4, the NIML of every attribute).
- `thd_niftiread.c`: the qform is preferred over the sform, and the
  cardinal grid of a NIfTI file keeps the translation of the matrix as
  its origin, while `THD_daxes_from_mat44`, which reads the headers of
  AFNI itself, projects it on each axis.
- `thd_nimlatr.c` and `niml/niml_elemio.c`: how attributes are written
  into the NIML extension (`<AFNI_atr atr_name="BRICK_LABS" ...>`, with
  NUL characters written as `~`).
- NiTransforms (`nitransforms/io/afni.py`) agrees. It reads the matrix
  as LPS to LPS and flips it into RAS, corrects for the obliquity of the
  reference and moving images in the same way, and negates the first two
  components of an AFNI NIfTI warp to make them RAS.

The following is assumed, and was not checked against AFNI itself,
since no AFNI binaries were available.

- The exact text that the NIML parser of AFNI accepts for the minimal
  extension that is written with a NIfTI warp, which is written as
  `NI_write_element` formats one.
- That no AFNI tool writes warps without the `x_delta` labels. Such a
  dataset is read as an image unless a hint asks for a warp.

## Not supported

- The `.param.1D` files of `3dAllineate -1Dparam_save`, which hold
  parameters, not matrices.
- The command-line warp syntax of `3dNwarpApply` and `3dNwarpCat`
  (`INV(...)`, `SQRT(...)`, a list of warps and matrices, `IDENTITY`,
  `MATRIX(...)`), which describes how to combine files and not a file.
  Each file is read, and the transformations are composed or inverted.
- Index warps (`3dQwarp -inwarp`), which have no use outside debugging.
- The linear extrapolation of a warp outside its grid by AFNI. The data
  model has no such boundary condition, so the nearest value is used.
- A file of several matrices as one transformation. The data model has
  no affine that varies along time, so one row is read at a time.
"""

__all__ = [
    "AfniAffine",
    "AfniAffineMetadata",
    "AfniAffineRaw",
    "AfniBrikWarp",
    "AfniMetadata",
    "AfniRaw",
    "AfniWarpFormat",
    "cardinal_to_real",
]

from brainhops.io.base._dispatch import register_missing_format
from brainhops.io.common.afni import AfniMetadata, AfniRaw

from ._affine import AfniAffine
from ._formats import AfniWarpFormat
from ._geometry import cardinal_to_real
from ._metadata import AfniAffineMetadata
from ._raw import AfniAffineRaw
from ._warp import AfniBrikWarp

try:
    from ._nifti import AfniNiftiWarp

    __all__ += ["AfniNiftiWarp"]
except ImportError:  # nibabel is optional
    register_missing_format(["afni.warp.nifti"], "nibabel", "nibabel")

# Importing the private converters registers them with the data model, so
# that a conversion into one of these formats, such as `t.to(Format)`,
# holds the very map or fails with a reason.
from . import _converters  # noqa: E402, F401  isort: skip
