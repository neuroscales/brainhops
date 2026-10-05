"""
Readers and writers for images stored as AFNI datasets.

[`AfniImage`][brainhops.io.images.afni.AfniImage] reads and writes AFNI's
native datasets, with no dependency beyond numpy: a text header,
`prefix+view.HEAD`, and the voxel values, `prefix+view.BRIK`, which may
be compressed (`.BRIK.gz`, `.BRIK.bz2`). Either file, or the dataset's
name without extension, can be given.

The header format, and every convention checked against the AFNI
sources, is described in [`brainhops.io.base.afni`][brainhops.io.base.afni],
which the image reader shares with the AFNI transformation formats.

```python
import brainhops.io as io

image = io.load("epi+orig.HEAD")       # an AfniImage
image.data                             # [x, y, z, sub-brick], scaled
image.transformation                   # voxel -> "orig", LPS mm (Affine)
image.header["HISTORY_NOTE"]           # any attribute of the header
image.header.labels                    # sub-brick labels (BRICK_LABS)
image.save("copy+orig.BRIK.gz")        # .HEAD + gzipped .BRIK
io.save(image, "epi.nii.gz")           # or any other image format
```

**Data.** The sub-bricks are stored one after the other, `x` fastest:
the array is indexed `[x, y, z]`, or `[x, y, z, sub-brick]` when there
are several sub-bricks, in F order. The fourth axis is `t` (of type
time) for a time series -- a dataset with a `TAXIS_NUMS` attribute --
and `brick` otherwise. The data of an uncompressed local BRIK stay
memory-mapped until they are indexed; a compressed BRIK, or one whose
sub-bricks have different types, is read into memory. The scaling
factor of each sub-brick (`BRICK_FLOAT_FACS`, zero meaning none) is
applied when `data` is first accessed; `dataobj` holds the stored
values.

**Coordinate systems.** AFNI's world is "DICOM order": LPS millimetres
(`x` to the left, `y` to the back, `z` up), which AFNI calls RAI. The
world spaces are named after the dataset's *view*: `orig`, `acpc` or
`tlrc`. The transformations are, in order:

1. `voxel` -> `physical`: a `Scaling` by the voxel sizes (`|DELTA|`),
   and by the repetition time of a time series;
2. `voxel` -> `<view>-cardinal`: the cardinal grid AFNI programs compute
   on, from `ORIENT_SPECIFIC`, `ORIGIN` and `DELTA`;
3. `voxel` -> `<view>`: the true, possibly oblique, geometry of
   `IJK_TO_DICOM_REAL` (the cardinal grid when the header has none). It
   is the preferred transformation, and the matrix AFNI itself exports
   to NIfTI (`3dAFNItoNIFTI`) and `nibabel` reads.

The two affines are the same unless the dataset is oblique.

**Writing.** The preferred transformation is converted to
voxel-to-DICOM (an RAS world is flipped) and written as
`IJK_TO_DICOM_REAL`; its closest cardinal grid becomes
`ORIENT_SPECIFIC`, `ORIGIN`, `DELTA` and `IJK_TO_DICOM`, as AFNI computes
it when it reads a NIfTI file. A 3D array is one sub-brick, and a 4D
array has one sub-brick per volume. Writer options set the `view`
(default: from the file name, `out+tlrc.HEAD`, else from the name of the
world space, else the view read, else `orig`), the stored `datatype`
(default: the data's own type, or the closest AFNI has) and extra or
removed `attributes`. The attributes read from the source header
(`HISTORY_NOTE`, `BRICK_LABS`, ...) are written back, except those that
no longer describe the data. The data are written unscaled, in
little-endian order, and a `.BRIK.gz` or `.BRIK.bz2` name compresses
them.

!!! note "One class for every view"
    The view (`+orig`, `+acpc`, `+tlrc`) is a property of the dataset --
    the world space its coordinates are in, recorded in `SCENE_DATA` --
    not a different file format: the three views are read and written
    the same way, so a single class reads them all, and names its world
    space after the view.
"""

__all__ = ["AfniImage"]

from ._image import AfniImage
