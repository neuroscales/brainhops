"""FreeSurfer non-linear morphs (`.m3z`), such as `talairach.m3z`.

`recon-all` registers each subject to its Gaussian classifier atlas with
`mri_ca_register` and saves `mri/transforms/talairach.m3z`. This "GCA
morph" (GCAM) is a regular grid of nodes on the atlas, each recording
where it lands in the subject (source) image.

```python
from brainhops import io

morph = io.load("talairach.m3z")              # or hint="m3z"
morph.save("copy.m3z")                         # the same file
```

## Specification

The layout follows `__m3zRead` and `__m3zWrite` in FreeSurfer's
`utils/gcamorph.cpp`; `mris_read_m3z.m` and surfa agree. A `.m3z` file is
gzipped and a `.m3d` file is not. All values are big-endian on every
host, because FreeSurfer's `znzwriteInt` and related functions
(`utils/fio.cpp`) swap bytes on little-endian hosts.

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

`width * height * depth` records of 36 bytes follow, with x slowest and
z fastest, so a C-ordered array of shape `(width, height, depth)` is
indexed by `[x, y, z]`.

| Type       | Fields                 | Meaning                         |
| ---------- | ---------------------- | ------------------------------- |
| float32 x3 | `origx, origy, origz`  | original (linear) position      |
| float32 x3 | `x, y, z`              | position                        |
| int32 x3   | `xn, yn, zn`           | the GCA node it is attached to  |

A node whose six coordinates are all zero is invalid
(`GCAM_POSITION_INVALID`): FreeSurfer does not sample through it.

### 3. Tags

Tags follow until the end of the file or a zero tag. Each tag is an
int32 and, unlike an MGH tag, has no length.

| Tag                        | Content                                   |
| -------------------------- | ----------------------------------------- |
| `TAG_GCAMORPH_GEOM` (10)   | the source geometry, then the atlas's     |
| `TAG_GCAMORPH_TYPE` (11)   | int32: `GCAM_VOX` (2) or `GCAM_RAS` (1)   |
| `TAG_GCAMORPH_LABELS` (12) | int32 per node, in node order: its label  |
| `TAG_MGH_XFORM` (31)       | a linear transform, as text (see below)   |

A geometry (`VOL_GEOM::write`) is 4 int32 (valid, width, height, depth),
15 float32 (voxel sizes, direction cosines `x_r ... z_s` and centre
`c_r, c_a, c_s`) and a 512-byte NUL-padded file name: the FreeSurfer
volume geometry of [`brainhops.io.common.freesurfer`][]. Without this
tag, both volumes have the default geometry (`initVolGeom`): 256³ voxels
of 1 mm, LIA, centred on the origin.

`TAG_MGH_XFORM` is followed by a tag of its own (`0`, or `TAG_AUTO_ALIGN`
(33) before FreeSurfer 7.2), an int64 length (1600) and a 1600-byte text
buffer holding a keyword (`Matrix` or `AutoAlign`) and a 4x4 matrix, row
by row. FreeSurfer uses its determinant to normalise node areas; it plays
no part in applying the morph. FreeSurfer writes the tags in this order,
the matrix only if it has one. An unknown tag ends reading, and whatever
follows is kept verbatim.

## Geometry

`mri_vol2vol --m3z` (`GCAMmorphToAtlas`, `GCAMsampleMorph`) resamples the
source on the atlas grid: for each atlas voxel `v`, it interpolates the
node positions trilinearly at `v / spacing` and samples the source there,
in source voxels. The morph pulls the source onto the atlas, so it maps
atlas coordinates to source coordinates:

```text
atlas RAS --(node vox2ras)^-1--> node voxels --positions--> source voxels
          --(source vox2ras)--> source RAS
```

- For `GCAM_VOX`, the default and what `mri_ca_register` writes,
  positions are 0-based source voxel coordinates. For `GCAM_RAS` they are
  source scanner RAS (`GCAMvoxToRas`), and the last step is dropped.
- The node grid is the atlas grid subsampled by `spacing`: node `n` is
  atlas voxel `n * spacing`, so its vox2ras is that of the atlas times
  `diag(spacing, spacing, spacing, 1)`.
- Both vox2ras are scanner RAS (`mri_info --vox2ras`), not tkr RAS,
  built from the stored geometries (`VGgetVoxelToRasXform`).
- FreeSurfer samples nothing outside the node grid. The field's `nearest`
  boundary matches FreeSurfer's clamping in the last half-cell and
  extends the field beyond.

`M3zMorph.struct`, an [`M3zStruct`][], keeps the content of the file:

```python
morph.struct.spacing                  # distance between nodes
morph.struct.atlas_geometry.vox2ras   # atlas voxel-to-RAS
morph.struct.image_geometry.vox2ras   # source voxel-to-RAS
morph.struct.xform.matrix             # the linear transform, if any
```

## Writing

An untouched morph is written back byte for byte, up to gzip. A morph
whose chain was assigned, by changing its field or building it from
scratch, is written from the chain (see [`M3zMorph.to_struct`][]): its
geometries are rebuilt from its affines, and what the chain does not
describe (original positions, labels and so on) is kept from the struct
when the node grid is unchanged.

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
    "M3zParser",
    "M3zStruct",
    "M3zXform",
]

from ._struct import GCAM_RAS, GCAM_VOX, M3zGeometry, M3zStruct, M3zXform
from ._xform import M3zFormat, M3zMorph, M3zParser
