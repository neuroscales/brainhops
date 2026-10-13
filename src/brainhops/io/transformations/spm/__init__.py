"""Readers and writers for SPM transformations.

SPM12 writes its deformation fields as NIfTI files of RAS coordinates,
whose names start with `y_` or `iy_`. They are read and written by
[`SpmCoordinatesField`][], which holds the header of its file as
[`NiftiMetadata`][brainhops.io.common.nifti.NiftiMetadata].

The older spatial normalisation of SPM2 to SPM8, which the `OldNorm`
toolbox of SPM12 keeps, saves its parameters in a MATLAB file whose name
ends in `_sn.mat`. Such a file is read by [`SpmSnTransformation`][],
which holds the whole file as an [`SpmSnRaw`][] record in its
[`SpmSnMetadata`][], and it cannot be written.

```python
from brainhops import io
from brainhops.io.transformations.spm import SpmCoordinatesField

sn = io.load("subject_sn.mat")              # or hint="sn"
y = sn.to(SpmCoordinatesField)              # the same map, as y_*.nii
io.save(y, "y_subject.nii")
```

## Specification of `*_sn.mat` files

The conventions below are taken from the SPM sources at these commits of
the mirrors of the SPM releases on GitHub:

- `spm/spm2` at d44663e0aaed24a5c3073aee74b257a9a06a3820:
  `spm_normalise.m` and `toolbox/Deformations/spm_sn2def.m`;
- `spm/spm5` at ff5e2ee3ae2ce117ba9dab1178f24f076b8e5ca4:
  `spm_normalise.m`, `spm_prep2sn.m` and `spm_write_sn.m`;
- `spm/spm8` at 29d67c03ecd49bac7daca905c734aacc15dadc78:
  `spm_normalise.m` and `spm_write_sn.m`;
- `spm/spm12` at 03ac9473cad402407b6472228377c0167fdc54b8:
  `toolbox/OldNorm/spm_normalise.m` (r5745),
  `toolbox/OldNorm/spm_write_sn.m` (r4878), the function `get_sn2def`
  of `spm_deformations.m` (r7700), `spm_dctmtx.m` (r6416),
  `@nifti/private/decode_qform0.m` and `spm_defaults.m`.

The evaluation of the warp is the same in all four releases.

### Variables

`spm_normalise` saves five variables with MATLAB's `save`, in the v6
format since SPM5, and `spm_prep2sn` saves the same five in
`*_seg_sn.mat` and `*_seg_inv_sn.mat`.

| Variable | Content                                                      |
| -------- | ------------------------------------------------------------ |
| `Affine` | 4x4 matrix from template voxels to source voxels             |
| `Tr`     | `Kx x Ky x Kz x 3` coefficients of the warp, or `[]`         |
| `VG`     | `spm_vol` structures of the templates; SPM uses `VG(1)`      |
| `VF`     | `spm_vol` structure of the source image                      |
| `flags`  | options of the estimation, such as `cutoff` and `reg`        |

An `spm_vol` structure has a `mat`, the 4x4 voxel-to-world matrix of the
image, and a `dim`, its number of voxels along each axis. In SPM2, which
reads Analyze headers, `dim` has a fourth element, the data type.

### Coordinates

SPM numbers voxels from 1, so the `mat` of an image maps voxel `(1, 1,
1)` to the centre of its first voxel. SPM reads a NIfTI header in this
convention, as `mat = M * [eye(4,3) [-1 -1 -1 1]']`, so the NIfTI
affine of an image is `mat` followed by adding 1 to each index (see
[`SPM_TO_NIFTI`][brainhops.io.transformations.spm._sn_raw.SPM_TO_NIFTI]).
`Affine` is also written for voxels numbered from 1, since
`spm_normalise` computes it as `inv(VG(1).mat \\ M * VF(1).mat)`.

### Deformation

`get_sn2def` in SPM12 and `spm_write_defs` in SPM2 evaluate the map on
the voxels `x` of the template, numbered from 1, as

```text
y(x) = VF.mat * Affine * (x + warp(x))
warp_c(x) = sum over a, b, d of Tr(a, b, d, c) Bx(x1, a) By(x2, b) Bz(x3, d)
```

where `Bx = spm_dctmtx(VG(1).dim(1), Kx, x1 - 1)` holds the first `Kx`
functions of the orthonormal basis of the type-II discrete cosine
transform, sampled at `x1 - 1`, and likewise along the other axes. The
warp is a displacement in template voxels, and `y(x)` is in the
millimetres of the source image. The map therefore goes from the
template to the source, and `spm_write_sn` samples the source image at
`y(x)` to write the normalised image. An empty `Tr` means an affine
normalisation.

[`SpmSnTransformation`][] holds the same map. Its field holds `x - 1 +
warp(x)` at each voxel `x - 1` of the template, numbered from 0, and the
affine after it is `VF(1).mat * Affine * [eye(4,3) [1 1 1 1]']`.

### What is not read

- The `*_sn3d.mat` files of SPM99 are MATLAB v4 files with the
  variables `mgc`, `Affine`, `Dims`, `Transform`, `MF` and `MG`
  (`spm_sn3d.m` of `spm/spm99` at
  1ad6395ecf7b9edad8f923484ead9f6f4498a10b). Their warp is evaluated in
  the same way, but they are not read.
- A file saved in the MATLAB v7.3 format, which SPM12 writes only when
  `defaults.mat.format` is changed, is refused.
- The fields `dt`, `pinfo`, `n` and `private` of the `spm_vol`
  structures, and the fields of `flags` that are neither text nor
  numbers, are not kept.

### Writing

A normalisation file is not written. `scipy.io.savemat` could write the
five variables, but not as MATLAB writes them: the `private` field of
an `spm_vol` structure holds a MATLAB `nifti` object, which the record
does not keep, and MATLAB's header text and padding differ. A file that
was read could therefore never be written back byte for byte. Nothing
fits the coefficients of a warp to another transformation either. The
map is written as a deformation field instead, with
`t.to(SpmCoordinatesField)`, whose values at the voxels of the template
are those that `get_sn2def` computes with its default bounding box and
voxel size (`NaN` in `config/spm_cfg_deformations.m`), which are those
of the template.
"""

__all__ = [
    "SpmCoordinatesField",
    "SpmSnMetadata",
    "SpmSnRaw",
    "SpmSnTransformation",
    "SpmVolume",
]

from ._fields import SpmCoordinatesField
from ._sn import SpmSnTransformation
from ._sn_metadata import SpmSnMetadata
from ._sn_raw import SpmSnRaw, SpmVolume

# Importing the private converters registers them with the data model,
# so that `t.to(SpmCoordinatesField)` converts a transformation.
from . import _converters  # noqa: E402, F401  isort: skip
