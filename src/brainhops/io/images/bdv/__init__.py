r"""BigDataViewer images, which are read only.

BigDataViewer is the Fiji viewer of large, multi-view and multi-scale
images, and BigStitcher stitches and registers light-sheet acquisitions
in the same datasets. A dataset is an XML file, usually `dataset.xml`,
and an HDF5 file of voxels that the XML file names. Its layout and its
conventions are described in [`brainhops.io.common.bdv`][]. The classes
of this module need h5py and answer to the format hints `bdv` and
`bigdataviewer`.

```python
import brainhops.io as io

pyramid = io.images.load("dataset.xml")          # BdvMultiScaleImage
level = io.images.load("dataset.xml", level=2)   # BdvImage
view = io.images.load("dataset.xml", setup=1, timepoint=3)
level.data.shape                 # (x, y, z)
level.transformation             # voxel -> "global", the registration
level.transformations[0]         # voxel -> "physical", the voxel size
level.metadata.raw.setups        # the setups of the XML file
```

A view is one setup, such as a channel, an angle or a tile, at one time
point. The options `setup` and `timepoint` select it by its ids, and
default to the first setup and the first time point, which are setup 0
and time point 0 in the datasets that BigDataViewer and BigStitcher
write. [`BdvMultiScaleImage`][] reads every level of the view, finest
first, and [`BdvImage`][] reads the level that `level` selects. A view
with a single level is read as a `BdvImage`.

Each level maps its voxels to the `"global"` space of the dataset by the
registration of the view, which is the product of its transforms as
BigDataViewer composes them, and to a `"physical"` space by the voxel
size of the setup. Both include the downsampling of the level. The
global space has no unit, since BigStitcher measures it in units of the
smallest voxel size of the dataset.

Brainhops cannot write a BigDataViewer dataset: an image is saved to
another format, such as NIfTI. The registration of a view is read and
written by [`brainhops.io.transformations.bdv`][].
"""

__all__ = [
    "BdvImage",
    "BdvMetadata",
    "BdvMultiScaleImage",
    "BdvRaw",
]

from brainhops.io.common.bdv import BdvMetadata, BdvRaw

from ._image import BdvImage
from ._multiscale import BdvMultiScaleImage
