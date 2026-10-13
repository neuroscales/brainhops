"""N5 datasets and pyramids.

N5 is a chunked format of the Saalfeld lab for large arrays, such as
light-sheet or electron microscopy volumes, in which a container is a
directory tree and each dataset stores its blocks as separate files. The
classes of this module only read local containers, with
[TensorStore](https://google.github.io/tensorstore/)
(`pip install brainhops[n5]`), and answer to the format hint `n5`.

| Class               | Reads                                             |
| ------------------- | ------------------------------------------------- |
| `N5MultiScaleImage` | every level `s0`, `s1`, ... of a pyramid          |
| `N5Image`           | one dataset, or one level of a pyramid (`level=`) |

```python
import brainhops.io as io

pyramid = io.images.load("container.n5/raw")    # an N5MultiScaleImage
pyramid.images[0].data.shape                    # (x, y, z)
pyramid.images[1].transformation                # voxel -> physical
level = io.images.load("container.n5/raw", level=2)  # an N5Image
dataset = io.images.load("container.n5/raw/s0")      # an N5Image
dataset.metadata.raw.attributes                 # the attributes.json
io.save(dataset, "s0.nii.gz")                   # to another format
```

A dataset or a pyramid is named by its directory or by its
`attributes.json` file. A pyramid of a single level is read as an
`N5Image`.

Each image holds the attributes as
[`N5Metadata`][brainhops.io.common.n5.N5Metadata], whose record is an
[`N5Raw`][brainhops.io.common.n5.N5Raw], and the samples of its dataset
in `raw`, a lazy array that TensorStore reads block by block when the
data are needed. The data are indexed in the order of the `dimensions`
of the dataset, `[x, y, z, ...]`, whose first dimension varies fastest
in a block. The datasets compressed with `raw`, `gzip`, `bzip2`, `xz`
and `blosc` are read, and the others, such as `lz4`, `zstd` or the
`compressionType` of version 0 of the specification, are refused when
they are loaded.

The specification of N5 defines no spatial attribute. Each image maps
its voxels to a `"physical"` space with the `pixelResolution`,
`resolution`, `unit`, `downsamplingFactors`, `offset` and `scales`
attributes that the n5-viewer, n5-ij, BigDataViewer and Paintera write,
and with the identity when there is none. The conventions are
described, with their sources, in
[`brainhops.io.common.n5`][brainhops.io.common.n5].

N5 containers are not written: the images have no writer, and an image
is saved to another format.
"""

__all__ = [
    "N5Image",
    "N5Metadata",
    "N5MultiScaleImage",
    "N5Raw",
]

from brainhops.io.common.n5 import N5Metadata, N5Raw

from ._image import N5Image
from ._multiscale import N5MultiScaleImage
