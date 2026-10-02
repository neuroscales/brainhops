"""
Readers and writers for images stored in MRtrix files.

[`MRtrixImage`][brainhops.io.images.mrtrix.MRtrixImage] reads and writes
the three variants of the MRtrix image format, with no dependency beyond
numpy:

| Extension  | Content                                                     |
| ---------- | ----------------------------------------------------------- |
| `.mif`     | text header and voxel data in one file                      |
| `.mif.gz`  | the same, gzip-compressed                                   |
| `.mih`     | text header only; the data are in the file it names (`.dat`)|

The header format, and every convention checked against the MRtrix3
sources, is described in
[`brainhops.io.base.mrtrix`][brainhops.io.base.mrtrix], which the image
reader shares with the MRtrix transformation formats.

```python
import brainhops.io as io

image = io.load("dwi.mif")        # an MRtrixImage
image.data                        # [x, y, z, volume], a memory-mapped view
image.transformation              # voxel -> scanner RAS+ mm (Affine)
image.header.keyval["dw_scheme"]  # free-form keys, one line per row
image.save("dwi.mih")             # header + dwi.dat
io.save(image, "dwi.nii.gz")      # or any other image format
```

**Data.** The array is indexed in the order of the header's axes
(`x, y, z, ...`), in F order, whatever `layout` the values are stored
in. A negative axis (`-0`) is stored in decreasing order of its index,
and a permuted layout (`+1,+2,+3,+0`, volume-contiguous, as MRtrix stores
DWI data) changes which axis is fastest in the file. Both are undone by
an array *view*, so the data of an uncompressed local `.mif` or `.mih`
stay memory-mapped until they are indexed. A `.mif.gz`, a stream and
`Bit` data are read into memory. The intensity `scaling` of the header,
if any, is applied when `data` is first accessed; `dataobj` holds the
stored values.

**Coordinate systems.** The transformations are, in order:

1. `voxel` -> `physical`: a `Scaling` by the voxel sizes (`vox`); a
   non-finite size, common on the volume axis, is a scale of 1;
2. `voxel` -> `scanner`: the `Affine` to scanner RAS+ millimetres.

The voxel axes are `x`, `y`, `z` (spatial), then `dim3`, `dim4`, ...,
which MRtrix gives no meaning of their own. The `scanner` space has the
three RAS+ axes. MRtrix's `transform` maps voxel coordinates
*multiplied by the voxel sizes*, so the voxel-to-scanner matrix is
`transform @ diag(vox)`. A header without a `transform` is given
MRtrix's default, which centres the field of view on the origin
(`-0.5 * (size - 1) * vox`): it is not the identity.

**Writing.** The preferred transformation is converted to
voxel-to-RAS+ (an LPS world is flipped) and split into unit direction
cosines and voxel sizes, as MRtrix stores them. A writer option may set
the `layout` (default: the layout the image was read with, else
`+0,+1,+2,...`), the `datatype` (default: the data's own type, with an
explicit byte order), an intensity `scaling`, and extra or removed
header keys (`keyval`). The keys read from the source header
(`dw_scheme`, `command_history`, ...) are written back.
"""

__all__ = ["MRtrixImage"]

from ._image import MRtrixImage
