"""MRtrix images.

[`MrtrixImage`][brainhops.io.images.mrtrix.MrtrixImage] reads and writes
the three variants of the MRtrix image format with numpy alone.

| Extension  | Content                                                     |
| ---------- | ----------------------------------------------------------- |
| `.mif`     | text header and voxel data in one file                      |
| `.mif.gz`  | the same, gzip-compressed                                   |
| `.mih`     | text header only; the data are in the file it names (`.dat`)|

The header format and its conventions are described in
[`brainhops.io.common.mrtrix`][brainhops.io.common.mrtrix], which is
shared with the MRtrix transformation formats.

```python
import brainhops.io as io

image = io.load("dwi.mif")        # an MrtrixImage
image.data                        # [x, y, z, volume], a memory-mapped view
image.transformation              # voxel -> scanner RAS+ mm (Affine)
image.header.keyval["dw_scheme"]  # free-form keys, one line per row
image.save("dwi.mih")             # header + dwi.dat
io.save(image, "dwi.nii.gz")      # or any other image format
```

The data are indexed `[x, y, z, ...]` in Fortran order, whatever `layout`
the file uses, because negative and permuted layouts are undone by a view.
The intensity `scaling` is applied to `data`, while `dataobj` holds the
stored values.

An image carries a [`Scaling`][brainhops.datamodel.transformations.Scaling]
from voxels to `"physical"` by the voxel sizes `vox`, and the preferred
[`Affine`][brainhops.datamodel.transformations.Affine] to scanner RAS+ in
millimeters. The MRtrix `transform` applies to voxel coordinates
multiplied by the voxel sizes, so the voxel-to-scanner matrix is
`transform @ diag(vox)`. Without a `transform`, the MRtrix default centres
the field of view on the origin. The writer accepts the `layout`,
`datatype`, `scaling` and `keyval` options, and writes back the keys of
the source header.
"""

__all__ = ["MrtrixImage"]

from ._image import MrtrixImage
