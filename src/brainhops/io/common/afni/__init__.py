"""
The shared AFNI-reading and AFNI-writing machinery behind every AFNI
image and transformation format.

An AFNI dataset is a pair of files that share a *prefix* and a *view*:

- `prefix+view.HEAD`, a text header made of *attributes*;
- `prefix+view.BRIK`, the raw voxel values, optionally compressed
  (`.BRIK.gz`, `.BRIK.bz2`).

The view is one of `orig` (the original, scanner-based coordinates),
`acpc` (AC-PC aligned) and `tlrc` (Talairach, or any other template).

The conventions below were checked against the AFNI sources
(`README.attributes`, `3ddata.h`, `mrilib.h`, `thd_atr.c`,
`thd_writeatr.c`, `thd_dsetdblk.c`, `thd_initdblk.c`, `thd_editdaxes.c`,
`thd_matdaxes.c`, `thd_dsetatr.c`, `thd_niftiread.c`,
`thd_niftiwrite.c`, `thd_compress.h`).

**Attributes.** Each attribute is three lines -- `type = <kind>-attribute`,
`name = <NAME>`, `count = <n>` -- followed by its `n` values. The kind is
`integer`, `float` or `string`. The numbers are separated by any white
space. A string is `n` characters that follow a single quote `'`, in
which `~` stands for a NUL character (AFNI writes a `~` of the string
itself as `*`). A string attribute ends with a NUL, and may hold several
NUL-separated sub-strings (the sub-brick labels of `BRICK_LABS`). An
attribute with a count of zero is skipped, as AFNI does. A header whose
first kilobyte holds `<AFNI_` is AFNI's alternative NIML (XML) header,
which is not read here.

**Shape.** `DATASET_DIMENSIONS` holds the number of voxels along the
three axes, `nx, ny, nz`, and `DATASET_RANK[1]` the number of
*sub-bricks* (`nvals`): the volumes of a time series, the statistics of
a "bucket", or the components of a warp. The BRIK holds the sub-bricks
one after the other, each with `x` fastest: the data are a
Fortran-ordered `(nx, ny, nz, nvals)` array.

**Types and scaling.** `BRICK_TYPES` holds one code per sub-brick
(`mrilib.h`): 0 = `uint8`, 1 = `int16`, 2 = `int32`, 3 = `float32`,
4 = `float64`, 5 = `complex64` (6 = RGB and 7 = RGBA are not read). It
is `int16` when absent, and a list shorter than `nvals` repeats its last
code. `BYTEORDER_STRING` is `LSB_FIRST` or `MSB_FIRST`, and the native
order when absent. `BRICK_FLOAT_FACS` holds one factor per sub-brick: a
stored value `v` means `v * f`, and a factor of zero means no scaling.

**Orientation.** `ORIENT_SPECIFIC` holds one code per voxel axis, saying
which way it runs: 0 = right-to-left, 1 = left-to-right,
2 = posterior-to-anterior, 3 = anterior-to-posterior,
4 = inferior-to-superior, 5 = superior-to-inferior.

**World coordinates.** AFNI's world is "DICOM order": `x` increases to
the left, `y` to the back and `z` up -- LPS millimetres, which AFNI
also calls RAI (`R`, `A` and `I` are negative). `ORIGIN[i]` is the DICOM
coordinate of the centre of voxel 0 along the DICOM axis voxel axis `i`
runs along, and `DELTA[i]` is the signed voxel size along it, so that
the centre of voxel `n` is at `ORIGIN[i] + n * DELTA[i]`: `DELTA` is
negative for an axis that runs towards `R`, `A` or `I`. The voxel to
DICOM matrix is therefore a signed permutation (`THD_daxes_to_mat44`):
row `ORIENT_SPECIFIC[i] // 2`, column `i`, holds `DELTA[i]`, and the
same row of the last column holds `ORIGIN[i]`. This is the *cardinal*
matrix, which AFNI computes from `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA`
whenever it reads a header (the `IJK_TO_DICOM` attribute it writes is
the same matrix, and is ignored on reading).

**Obliquity.** `IJK_TO_DICOM_REAL`, when present, is the true
(possibly oblique) voxel-to-DICOM matrix, as 12 values (three rows of
four). AFNI programs compute on the cardinal grid, ignoring obliquity,
but AFNI's own NIfTI export (`3dAFNItoNIFTI`) stores the real matrix
(with `x` and `y` negated into RAS) as the sform. When a matrix is turned
back into `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA` (`THD_daxes_from_mat44`),
each axis gets the closest orientation, its voxel size is the length of
its column (signed by the orientation), and its origin is the projection
of the translation on the unit column (signed by the orientation).

**View.** `SCENE_DATA[0]` is the view: 0 = `orig`, 1 = `acpc`,
2 = `tlrc`. A NIfTI file read by AFNI goes to `tlrc` when its form names
a template (Talairach, MNI, other template), and to `orig` otherwise.

**Time.** A time series has `TAXIS_NUMS` (`[ntt, nslices, units]`, with
units 77001 = ms, 77002 = s, 77003 = Hz) and `TAXIS_FLOATS`
(`[origin, TR, duration, z origin, z step]`).

This package holds what an image reader and a transformation reader (an
AFNI warp, for instance) share: the header, the geometry, the decoding
of the voxel data and their encoding. Every AFNI format derives from
[`AfniFormat`][brainhops.io.common.afni.AfniFormat], so that the `"afni"`
hint selects them all.
"""

__all__ = [
    "AfniFormat",
    "AfniParser",
    "AfniHeader",
]

from ._header import AfniHeader
from ._parsers import AfniFormat, AfniParser
