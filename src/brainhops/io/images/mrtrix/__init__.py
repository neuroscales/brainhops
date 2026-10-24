"""MRtrix images.

[`MrtrixImage`][brainhops.io.images.mrtrix.MrtrixImage] reads and writes
the three variants of the MRtrix image format with numpy alone.

| Extension  | Content                                                     |
| ---------- | ----------------------------------------------------------- |
| `.mif`     | text header and voxel data in one file                      |
| `.mif.gz`  | the same, gzip-compressed                                   |
| `.mih`     | text header only; the data are in the file it names (`.dat`)|

The header format and its conventions are described in
[`brainhops.io.common.mrtrix`][brainhops.io.common.mrtrix].

```python
import brainhops.io as io

image = io.load("dwi.mif")                # an MrtrixImage
image.data                                # [x, y, z, volume]
image.transformation                      # voxel -> scanner RAS+ mm (Affine)
image.metadata.raw.keyval["dw_scheme"]    # free-form keys, one line per row
image.save("dwi.mih")                     # header + dwi.dat
io.save(image, "dwi.nii.gz")              # or any other image format
```

The data are indexed `[x, y, z, ...]` in Fortran order, whatever `layout`
the file uses, because negative and permuted layouts are undone by a view.
After a read, `rawdata` is a lazy array of the voxels, which reads them
only when `data` is first accessed and applies the intensity `scaling`
to them. Local data that is not compressed is memory-mapped.

An image carries a [`Scaling`][brainhops.datamodel.transformations.Scaling]
from voxels to `"physical"` by the voxel sizes `vox`, and the preferred
[`Affine`][brainhops.datamodel.transformations.Affine] to scanner RAS+ in
millimeters. The MRtrix `transform` applies to voxel coordinates
multiplied by the voxel sizes, so the voxel-to-scanner matrix is
`transform @ diag(vox)`. Without a `transform`, the MRtrix default centres
the field of view on the origin. The writer accepts the `layout`,
`datatype`, `scaling` and `keyval` options, and writes back the keys of
the source header. An image that is read and written again without
changes keeps its bytes.

Some images hold the components of vectors or tensors along their
fourth axis, such as the diffusion tensors of `dwi2tensor`, in the order
`D11`, `D22`, `D33`, `D12`, `D13`, `D23`, and the warps of `warpconvert`.
MRtrix expresses these components in scanner RAS millimetres. The header
has no key that says what the volumes are, so the reader makes the
fourth axis a tensor, vector, displacement or coordinate axis only when
the command that wrote the file, or the key `brainhops_field_type` that
brainhops writes, says so, and records the scanner as the frame of the
values (see
[`MrtrixImage.frames`][brainhops.io.images.mrtrix.MrtrixImage.frames]).
The writer re-expresses the values in the scanner and stores the
coefficients of a tensor in the order of `dwi2tensor`.
"""

__all__ = ["MrtrixImage", "MrtrixMetadata", "MrtrixRaw"]

from brainhops.io.common.mrtrix import MrtrixMetadata, MrtrixRaw

from ._image import MrtrixImage
