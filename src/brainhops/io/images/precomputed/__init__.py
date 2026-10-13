"""Neuroglancer precomputed volumes.

The precomputed format of Neuroglancer serves large volumes, such as
BigBrain or connectomics data, as a directory of chunks with an `info`
file, local or behind a URL. The classes of this module only read
volumes, with [TensorStore](https://google.github.io/tensorstore/)
(`pip install brainhops[precomputed]`), and answer to the format hints
`precomputed` and `neuroglancer`.

| Class                        | Reads                                  |
| ---------------------------- | -------------------------------------- |
| `PrecomputedMultiScaleImage` | every scale, as a multiscale image     |
| `PrecomputedImage`           | one scale, the finest unless `level=`  |

```python
import brainhops.io as io

pyramid = io.images.load("volume")           # a PrecomputedMultiScaleImage
pyramid.images[0].data.shape                 # (x, y, z), or (x, y, z, c)
pyramid.images[1].transformation             # voxel -> physical, in nm
level = io.images.load("volume", level=2)    # a PrecomputedImage
level.metadata.raw.scales[2].encoding        # "raw", "jpeg", ...
io.save(level, "level2.nii.gz")              # to another format
```

A volume is named by its directory or by its `info` file. A remote
volume is named by its URL without the `precomputed://` prefix of
Neuroglancer, such as `gs://bucket/volume` or `https://host/volume`,
and TensorStore 0.1.45 reads `gs`, `http` and `https` URLs. A volume of
a single scale is read as a `PrecomputedImage`.

Each image holds the `info` file as
[`PrecomputedMetadata`][brainhops.io.common.precomputed.PrecomputedMetadata],
whose record is a
[`PrecomputedRaw`][brainhops.io.common.precomputed.PrecomputedRaw], and
the voxels of its scale in `raw`, a lazy array that TensorStore reads
chunk by chunk when the data are needed. The data are indexed
`[x, y, z]`, with a channel axis `c` last when the volume has several
channels, and the first voxel of the data is the first voxel of the
scale. The scales encoded as `raw`, `jpeg` and `compressed_segmentation`
are read, sharded or not, and a scale encoded otherwise (`png`,
`compresso`, `jxl` or `crackle`) is refused when it is loaded.

Each scale maps its voxels to the `"physical"` space of Neuroglancer, in
nanometres, by an [`Affine`][brainhops.datamodel.transformations.Affine]
that scales by the `resolution` and moves by
`(voxel_offset + 0.5) * resolution`. The half voxel is there because an
integer index of brainhops is the centre of a voxel, while Neuroglancer
puts the corner of the voxel `voxel_offset` at
`voxel_offset * resolution`. The conventions are described, with their
sources, in
[`brainhops.io.common.precomputed`][brainhops.io.common.precomputed].

Precomputed volumes are not written: the images have no writer, and an
image is saved to another format.
"""

__all__ = [
    "PrecomputedImage",
    "PrecomputedMetadata",
    "PrecomputedMultiScaleImage",
    "PrecomputedRaw",
]

from brainhops.io.common.precomputed import (
    PrecomputedMetadata,
    PrecomputedRaw,
)

from ._image import PrecomputedImage
from ._multiscale import PrecomputedMultiScaleImage
