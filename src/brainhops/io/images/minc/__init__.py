r"""MINC images.

MINC is the volume format of the MNI and of pipelines such as CIVET and
BigBrain. Two containers share the `.mnc` extension and are recognized
from their content. The classes of this module only read files. Each of
them answers to the format hint `minc`, which is a format name that can
be attached to a path to choose its reader.

| Class        | Container                             | Hints             |
| ------------ | ------------------------------------- | ----------------- |
| `Minc1Image` | NetCDF classic (`CDF\x01`, `CDF\x02`) | `minc1`, `minc.1` |
| `Minc2Image` | HDF5, with a `/minc-2.0` group        | `minc2`, `minc.2` |

Files are read with `nibabel` (`pip install brainhops[minc]`), and MINC2
files also need `h5py`. A MINC1 file may be gzipped (`.mnc.gz`).
[`MincImage`][brainhops.io.images.minc.MincImage] reads either version and
returns an object of the matching class.

```python
import brainhops.io as io

image = io.images.load("t1.mnc")   # a Minc1Image or a Minc2Image
image.data.shape                   # F order: (x, y, z) for z,y,x files
image.transformation               # voxel -> MINC world (RAS, mm)
image.transformations[0]           # voxel -> physical (|step|, mm)
image.dimensions                   # the raw MINC dimensions, file order
image.vox2world                    # the (4, 4) matrix of the world affine
```

The voxels are in Fortran order, which reverses the file dimensions, and
the axes are named after the dimensions (`xspace` gives `x`, `time` gives
`t`). The world is the RAS world of MINC. The layout is described in
[`brainhops.io.common.minc`][brainhops.io.common.minc]. Since nibabel
cannot write MINC, an image is saved to another format, such as NIfTI.
Attributes other than the
[`dimensions`][brainhops.io.common.minc.MincReader.dimensions] are not
read.
"""

__all__ = ["MincImage", "Minc1Image", "Minc2Image"]

from ._image import Minc1Image, Minc2Image, MincImage
