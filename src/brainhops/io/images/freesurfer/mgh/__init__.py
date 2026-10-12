"""FreeSurfer MGH and MGZ images.

An `.mgh` file holds a 3D or 4D volume with its geometry, and an `.mgz`
file holds the same bytes compressed with gzip. Both are read and written
through `nibabel` (`pip install brainhops[mgh]`) and are recognized from
their content, whatever the file name.

```python
import brainhops.io as io

image = io.images.load("orig.mgz")
image.data.shape              # (x, y, z) or (x, y, z, frames), F order
image.transformation          # voxel -> scanner RAS (preferred)
image.transformations[1]      # voxel -> tkr (surface) RAS
record = image.metadata.raw   # the MghRaw record of the file
record.vox2ras, record.vox2tkr  # the same, as (4, 4) arrays
record.mri_params             # {"tr": ..., "flip_angle": ..., ...}
image.save("copy.mgz")        # gzipped because of the name
```

An image maps its voxels to the `"physical"` space by the voxel size (and
the TR of the frames), to the tkr RAS space `"tkr"` in which surfaces
live, and to the preferred scanner RAS space `"scanner"`. An
[`MghImage`][] holds the record of its file as [`MghMetadata`][], whose
record is an [`MghRaw`][], and the voxels as stored in the file. The two
classes of the record are defined in
[`brainhops.io.common.mgh`][brainhops.io.common.mgh] and exported here as
well. The header and the file layout are described in
[`brainhops.io.common.freesurfer`][brainhops.io.common.freesurfer] and
[`brainhops.io.common.mgh`][brainhops.io.common.mgh].

Like FreeSurfer, the reader ignores the stored geometry when
`goodRASFlag` is not positive. The MRI parameters and trailing tags of the
footer are written back, but they are lost when the image is converted to
another format.
"""

__all__ = ["MghImage", "MghMetadata", "MghRaw"]

from brainhops.io.common.mgh import MghMetadata, MghRaw

from ._image import MghImage
