"""
Readers and writers for images stored in FreeSurfer's MGH / MGZ format.

An `.mgh` file holds a 3D or 4D volume and its geometry; an `.mgz` file
is the same bytes, gzipped. Both are read and written through `nibabel`
(the `nibabel` extra, also available as `brainhops[mgh]`), and are
recognised from their content -- a big-endian version number of 1 at the
start of the (decompressed) stream -- whatever their name.

```python
import brainhops.io as io

image = io.images.load("orig.mgz")
image.data.shape              # (x, y, z) or (x, y, z, frames), F order
image.transformation          # voxel -> scanner RAS (preferred)
image.transformations[1]      # voxel -> tkr (surface) RAS
image.vox2ras, image.vox2tkr  # the same, as (4, 4) arrays
image.mri_params              # {"tr": ..., "flip_angle": ..., ...}
image.save("copy.mgz")        # gzipped because of the name
```

**Coordinate systems.** The voxel axes are `(x, y, z[, t])`, x fastest
on disk. The transformations are a scaling to the `"physical"` scaled
voxel space (voxel size in mm, and TR in ms for the frames when the
footer records one), the voxel-to-tkr RAS affine (`"tkr"`, FreeSurfer's
`Torig`, the space surfaces live in) and the voxel-to-scanner RAS affine
(`"scanner"`, FreeSurfer's `Norig`), which is preferred. See
[`brainhops.io.common.freesurfer`][] for how both derive from the header,
and [`brainhops.io.common.mgh`][] for the file layout.

**`goodRASFlag`.** When it is not positive, FreeSurfer ignores the stored
geometry and uses 1 mm voxels, coronal LIA direction cosines and a zero
centre; so does this reader (unlike `nibabel`, whose default direction
cosines differ). The raw flag is kept, privately, as `_good_ras`. A
written file always records its geometry, with the flag set.

**Metadata.** The MRI parameters of the footer (TR, flip angle, TE, TI,
FoV) and the raw trailing tags are kept on the object and written back,
so an MGH file round-trips. They are not part of the datamodel, so they
do not survive a conversion to another format.
"""

__all__ = ["MghImage"]

from ._image import MghImage
