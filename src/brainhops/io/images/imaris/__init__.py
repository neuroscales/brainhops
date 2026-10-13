r"""Imaris images.

Imaris (`.ims`) is the file format of the Imaris and ImarisViewer
software of Oxford Instruments, common in light-sheet and confocal
microscopy. An Imaris file is an HDF5 file that stores an image at
several resolutions, with time points and channels. The classes of this
module only read files, with `h5py` (`pip install brainhops[imaris]`),
and answer to the format hints `imaris` and `ims`.

| Class                   | Reads                                       |
| ----------------------- | ------------------------------------------- |
| `ImarisMultiScaleImage` | every resolution level of a file, as a pyramid |
| `ImarisImage`           | one resolution level, the finest by default |

```python
import brainhops.io as io

pyramid = io.images.load("cells.ims")       # ImarisMultiScaleImage
pyramid.images[0].data.shape                # (x, y, z, t, c)
pyramid.images[0].transformation            # voxel -> physical, in um
level = io.images.load("cells.ims", level=2)  # ImarisImage
level.metadata.raw.info["Channel 0"]["Name"]  # a parameter of the file
```

A file with a single resolution level is read as an
[`ImarisImage`][brainhops.io.images.imaris.ImarisImage]. The data of an
image always have five axes, `(x, y, z, t, c)`, as the other brainhops
images order space, time and channels, and they are read lazily, one
dataset of the file per channel and time point. An image holds the
groups and parameters of its file as [`ImarisMetadata`][], whose record
is an [`ImarisRaw`][]. The layout and the conventions of the format, and
where they come from, are described in
[`brainhops.io.common.imaris`][brainhops.io.common.imaris].

Brainhops cannot write Imaris files, so an image is saved to another
format, such as NIfTI or OME-Zarr. A valid Imaris file needs every
resolution level, computed by the binning of Imaris, and the histograms
of every level, and the reference writer is ImarisWriter.
"""

__all__ = [
    "ImarisImage",
    "ImarisMetadata",
    "ImarisMultiScaleImage",
    "ImarisRaw",
]

from brainhops.io.common.imaris import ImarisMetadata, ImarisRaw

from ._image import ImarisImage
from ._multiscale import ImarisMultiScaleImage
