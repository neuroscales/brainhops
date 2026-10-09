"""FreeSurfer non-linear morphs (`.m3z`), such as `talairach.m3z`.

`recon-all` registers each subject to its Gaussian classifier atlas with
`mri_ca_register` and saves the result as `mri/transforms/talairach.m3z`.
This "GCA morph" (GCAM) is a regular grid of nodes on the atlas, and each
node records the position to which it maps in the subject image, which is
called the source image below.

```python
from brainhops import io

morph = io.load("talairach.m3z")              # or hint="m3z"
morph.save("copy.m3z")                         # the same file
```

## Specification

The layout follows `__m3zRead` and `__m3zWrite` in FreeSurfer's
`utils/gcamorph.cpp`, and the MATLAB reader `mris_read_m3z.m` and surfa
agree with it. A `.m3z` file is gzipped and a `.m3d` file is not. All
values are big-endian on every host, because FreeSurfer's `znzwriteInt`
and related functions (`utils/fio.cpp`) swap bytes on little-endian hosts.

### 1. Header

| Type    | Field     | Meaning                                    |
| ------- | --------- | ------------------------------------------ |
| float32 | `version` | always `1.0`                               |
| int32   | `width`   | number of nodes along x                    |
| int32   | `height`  | number of nodes along y                    |
| int32   | `depth`   | number of nodes along z                    |
| int32   | `spacing` | distance between nodes, in atlas voxels    |
| float32 | `exp_k`   | exponent of the registration's area term   |

### 2. Nodes

The header is followed by `width * height * depth` node records of 36
bytes each. The records are ordered with x varying slowest and z fastest,
so a C-ordered array of shape `(width, height, depth)` is indexed by
`[x, y, z]`.

| Type       | Fields                 | Meaning                         |
| ---------- | ---------------------- | ------------------------------- |
| float32 x3 | `origx, origy, origz`  | original (linear) position      |
| float32 x3 | `x, y, z`              | position                        |
| int32 x3   | `xn, yn, zn`           | the GCA node it is attached to  |

A node whose six coordinates are all zero is invalid
(`GCAM_POSITION_INVALID`), and FreeSurfer does not sample through it.

### 3. Tags

The nodes are followed by tags, until the end of the file or a zero tag.
Each tag is an int32 and, unlike an MGH tag, is not followed by the length
of its content.

| Tag                        | Content                                   |
| -------------------------- | ----------------------------------------- |
| `TAG_GCAMORPH_GEOM` (10)   | the source geometry, then the atlas's     |
| `TAG_GCAMORPH_TYPE` (11)   | int32: `GCAM_VOX` (2) or `GCAM_RAS` (1)   |
| `TAG_GCAMORPH_LABELS` (12) | int32 per node, in node order: its label  |
| `TAG_MGH_XFORM` (31)       | a linear transform, as text (see below)   |

A geometry, as `VOL_GEOM::write` writes it, consists of 4 int32 (valid,
width, height, depth), 15 float32 (voxel sizes, direction cosines
`x_r ... z_s` and centre `c_r, c_a, c_s`) and a 512-byte NUL-padded file
name. It is the FreeSurfer volume geometry that is described in
[`brainhops.io.common.freesurfer`][]. Without this tag, both volumes have
the default geometry of `initVolGeom`, which is 256³ voxels of 1 mm in
LIA orientation, centred on the origin.

`TAG_MGH_XFORM` is followed by an inner tag of its own, an int64 length
(1600) and a 1600-byte text buffer. The inner tag is `0`, or
`TAG_AUTO_ALIGN` (33) before FreeSurfer 7.2. The buffer holds a keyword
(`Matrix` or `AutoAlign`) followed by a 4x4 matrix, row by row. FreeSurfer
uses the determinant of this matrix to normalise node areas, but the
matrix plays no part in applying the morph. FreeSurfer writes the tags in
the order of the table, and it writes the matrix tag only if the morph has
a matrix. An unknown tag ends reading, and whatever follows it is kept
verbatim.

## Geometry

The command `mri_vol2vol --m3z`, through `GCAMmorphToAtlas` and
`GCAMsampleMorph`, resamples the source image on the atlas grid. For each
atlas voxel `v`, it interpolates the node positions trilinearly at
`v / spacing` and samples the source image at the resulting position,
which is expressed in source voxels. The morph therefore pulls the source
onto the atlas, so it maps atlas coordinates to source coordinates:

```text
atlas RAS --(node vox2ras)^-1--> node voxels --positions--> source voxels
          --(source vox2ras)--> source RAS
```

- With `GCAM_VOX`, which is the default and what `mri_ca_register`
  writes, the positions are 0-based voxel coordinates of the source
  image. With `GCAM_RAS`, they are scanner RAS coordinates of the source
  image (`GCAMvoxToRas`), and the last step of the chain is dropped.
- The node grid is the atlas grid subsampled by `spacing`. Node `n` is
  atlas voxel `n * spacing`, so the vox2ras of the node grid is that of
  the atlas multiplied by `diag(spacing, spacing, spacing, 1)`.
- Both vox2ras matrices map to scanner RAS, as `mri_info --vox2ras`
  reports it, and not to tkr RAS. They are built from the stored
  geometries, as `VGgetVoxelToRasXform` builds them.
- FreeSurfer samples nothing outside the node grid. The `nearest`
  boundary condition of the field matches the clamping that FreeSurfer
  applies in the last half-cell, and it extends the field beyond the
  grid.

The attribute `M3zMorph.struct`, an [`M3zStruct`][], keeps the content of
the file:

```python
morph.struct.spacing                  # distance between nodes
morph.struct.atlas_geometry.vox2ras   # atlas voxel-to-RAS
morph.struct.image_geometry.vox2ras   # source voxel-to-RAS
morph.struct.xform.matrix             # the linear transform, if any
```

## Writing

A morph that has not been modified is written back byte for byte, apart
from differences in the gzip compression. A morph whose chain was
assigned, either to change its field or to build a morph from scratch, is
written from the chain, as described in [`M3zMorph.to_struct`][]. The atlas
geometry is then rebuilt from the first affine of the chain, and the
source geometry from the last affine when the positions are voxel
coordinates. Whatever the chain does not describe, such as the original
positions and the labels, is kept from the struct when the node grid is
unchanged.

!!! note "Out of scope"
    FreeSurfer 8 can save a morph as an MGH or NIfTI "warpfield"
    (`mri_warp_convert`), which these classes do not read. Inverse morphs
    (`talairach.m3z.inv.{x,y,z}.mgz`) are plain MGH images.
"""

__all__ = [
    "GCAM_RAS",
    "GCAM_VOX",
    "M3zFormat",
    "M3zGeometry",
    "M3zMorph",
    "M3zReaderWriter",
    "M3zStruct",
    "M3zXform",
]

from ._struct import GCAM_RAS, GCAM_VOX, M3zGeometry, M3zStruct, M3zXform
from ._xform import M3zFormat, M3zMorph, M3zReaderWriter

# The converters into these formats register themselves when this module
# is imported, here, so that `t.to(Format)` refuses with a reason.
from . import _converters  # noqa: E402, F401  isort: skip
