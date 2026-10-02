"""
FreeSurfer non-linear morphs (`.m3z`), such as `talairach.m3z`.

`recon-all` registers every subject to its Gaussian classifier atlas
with `mri_ca_register`, and saves the result as
`mri/transforms/talairach.m3z`: a "GCA morph" (`GCA_MORPH`, or GCAM), a
regular grid of nodes placed on the atlas, each of which records where
it lands in the subject (source) image.

```python
from brainhops import io

morph = io.load("talairach.m3z")              # or hint="m3z"
morph.save("copy.m3z")                         # the same file
```

## Specification

The layout is that of `__m3zRead` and `__m3zWrite` in FreeSurfer's
`utils/gcamorph.cpp`; MATLAB's `mris_read_m3z.m` and `surfa`
(`surfa/io/fsio.py`, `surfa/io/utils.py`) agree with it. A `.m3z` file
is gzipped, a `.m3d` file is not (FreeSurfer gzips a morph whose name
contains `.m3z`). Every value is big-endian, whatever the machine that
wrote it: FreeSurfer writes and reads through `znzwriteInt`,
`znzwriteFloat`, `znzreadInt`, ... (`utils/fio.cpp`), which byte-swap
on little-endian hosts.

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

`width * height * depth` records of 36 bytes, x slowest and z fastest
(`for x: for y: for z:`), so that a C-ordered array of shape
`(width, height, depth)` indexes them by `[x, y, z]`:

| Type       | Fields                 | Meaning                         |
| ---------- | ---------------------- | ------------------------------- |
| float32 x3 | `origx, origy, origz`  | original (linear) position      |
| float32 x3 | `x, y, z`              | position                        |
| int32 x3   | `xn, yn, zn`           | the GCA node it is attached to  |

A node whose six coordinates are all zero is invalid
(`GCAM_POSITION_INVALID`): FreeSurfer does not sample through it.

### 3. Tags

Tags follow until the end of the file (or a zero tag). Each is an int32,
and, unlike the tags of an MGH file, those of a morph have no length:

| Tag                        | Content                                   |
| -------------------------- | ----------------------------------------- |
| `TAG_GCAMORPH_GEOM` (10)   | the source geometry, then the atlas's     |
| `TAG_GCAMORPH_TYPE` (11)   | int32: `GCAM_VOX` (2) or `GCAM_RAS` (1)   |
| `TAG_GCAMORPH_LABELS` (12) | int32 per node, in node order: its label  |
| `TAG_MGH_XFORM` (31)       | a linear transform, as text (see below)   |

A geometry (`VOL_GEOM::write`) is four int32 -- `valid, width, height,
depth` -- then fifteen float32 -- `xsize, ysize, zsize`, the direction
cosines `x_r, x_a, x_s, y_r, y_a, y_s, z_r, z_a, z_s` and the centre
`c_r, c_a, c_s` -- then a 512-byte, `NUL`-padded file name: the
FreeSurfer volume geometry of
[`brainhops.io.base.freesurfer`][brainhops.io.base.freesurfer]. A file
without this tag has FreeSurfer's default geometry for both volumes
(`initVolGeom`: 256^3 voxels of 1 mm, LIA, centred on the origin).

`TAG_MGH_XFORM` is followed by a tag of its own (`0`; `TAG_AUTO_ALIGN`,
33, before FreeSurfer 7.2), an int64 length (1600) and a 1600-byte text
buffer: a keyword (`Matrix`; `AutoAlign`) and the 16 values of a 4x4
matrix, row by row. FreeSurfer uses its determinant to normalise the
node areas; it plays no part in applying the morph.

FreeSurfer writes the tags in this order, all of them except the matrix,
which it writes only if it has one. A tag it does not know ends the
reading; what follows is kept verbatim.

## Geometry

`mri_vol2vol --m3z` (through `GCAMmorphToAtlas` and `GCAMsampleMorph`)
resamples the source image on the atlas grid: for each atlas voxel
`v`, it interpolates the positions of the nodes trilinearly at node
coordinate `v / spacing`, and samples the source image at that position,
in source voxels. A morph therefore *pulls* the source onto the atlas,
and, in brainhops' convention, maps atlas coordinates to source
coordinates:

```text
atlas RAS --(node vox2ras)^-1--> node voxels --positions--> source voxels
          --(source vox2ras)--> source RAS
```

- The positions are source voxel coordinates (0-based, integers at voxel
  centres) when the type is `GCAM_VOX`, FreeSurfer's default and what
  `mri_ca_register` writes. When the type is `GCAM_RAS`, they are
  source scanner RAS (`GCAMvoxToRas`, through the source geometry), and
  the last step is dropped.
- The node grid is the atlas grid subsampled by `spacing`: node `n` is
  atlas voxel `n * spacing`, so its voxel-to-RAS matrix is the atlas's
  times `diag(spacing, spacing, spacing, 1)`.
- Both voxel-to-RAS matrices are scanner RAS (`mri_info --vox2ras`), not
  tkr RAS, built from the stored geometries as FreeSurfer does
  (`VGgetVoxelToRasXform`).
- FreeSurfer samples nothing outside the node grid. The field's boundary
  condition is `nearest`, which matches inside the last half-cell where
  FreeSurfer clamps, and extends the field beyond it.

`M3zMorph.struct`, an
[`M3zStruct`][brainhops.io.transformations.freesurfer.m3z.M3zStruct],
keeps the content of the file, whose members can be queried:

```python
morph.struct.spacing                  # distance between nodes
morph.struct.atlas_geometry.vox2ras   # atlas voxel-to-RAS
morph.struct.image_geometry.vox2ras   # source voxel-to-RAS
morph.struct.xform.matrix             # the linear transform, if any
```

## Writing

A morph read from a file and left untouched is written back byte for
byte (up to gzip). A morph whose chain was assigned -- a field changed,
or one built from scratch -- is written from the chain (see
[`M3zMorph.to_struct`][brainhops.io.transformations.freesurfer.m3z.M3zMorph.to_struct]):
the geometries are rebuilt from its affines, and what it does not say
(original positions, labels, ...) is kept from the struct when the node
grid is unchanged.

!!! note "Out of scope"
    FreeSurfer 8 can also save a morph as an MGH or NIfTI "warpfield"
    (`mri_warp_convert`), which these classes do not read. Inverse
    morphs (`talairach.m3z.inv.{x,y,z}.mgz`) are plain MGH images.
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
