"""
Readers for images stored in MINC files (`.mnc`), MINC1 and MINC2.

MINC is the volume format of the Montreal Neurological Institute and of
the MINC toolkit (MNI, CIVET, BigBrain and many rodent pipelines). It
comes in two containers, both named `.mnc`, which are recognised from
their content whatever the file name:

| Class        | Container                               | Hints            |
| ------------ | --------------------------------------- | ---------------- |
| `Minc1Image` | NetCDF classic (`CDF\\x01`, `CDF\\x02`) | `minc`, `minc.1` |
| `Minc2Image` | HDF5, with a `/minc-2.0` group          | `minc`, `minc.2` |

Both are read with `nibabel` (the `minc` extra, `pip install
brainhops[minc]`); MINC2 also needs `h5py`. A MINC1 file may be gzipped
(`.mnc.gz`). [`MincImage`][brainhops.io.images.minc.MincImage] reads
either and returns the matching class.

```python
import brainhops.io as io

image = io.images.load("t1.mnc")   # a Minc1Image or a Minc2Image
image.data.shape                   # F order: (x, y, z) for z,y,x files
image.transformation               # voxel -> MINC world (RAS, mm)
image.transformations[0]           # voxel -> physical (|step|, mm)
image.dimensions                   # the raw MINC dimensions, file order
image.vox2world                    # the (4, 4) matrix of the world affine
```

**Coordinate systems.** The voxels are presented in F order (the file's
dimension order reversed), and the voxel axes are named after the MINC
dimensions (`xspace` -> `x`, `yspace` -> `y`, `zspace` -> `z`, `time`
-> `t`), whatever order the file stores them in. The world space is
MINC's: right, anterior and superior along `x`, `y` and `z`, with each
dimension running along its direction cosines from its `start` by its
(possibly negative) `step`. See [`brainhops.io.common.minc`][] for the
file layout and the limitations.

**Writing.** `nibabel` cannot write MINC, so these classes are
read-only: save a MINC image to another format (e.g. NIfTI) instead.
Writing MINC would need another backend (`pyminc`, or the MINC
toolkit).

**Metadata.** The dimensions (`start`, `step`, direction cosines,
units) are kept on the object as
[`dimensions`][brainhops.io.common.minc.MincParser.dimensions]. The other
MINC attributes (patient, acquisition, history) are not read.
"""

__all__ = ["MincImage", "Minc1Image", "Minc2Image"]

from ._image import Minc1Image, Minc2Image, MincImage
