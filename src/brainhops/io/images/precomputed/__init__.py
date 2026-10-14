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

Each image holds the `info` file as [`PrecomputedMetadata`][], whose
record is a [`PrecomputedRaw`][], and the voxels of its scale in
`rawdata`, a lazy array that TensorStore reads chunk by chunk when the
data are needed. The data are indexed
`[x, y, z]`, with a channel axis `c` last when the volume has several
channels, and the first voxel of the data is the first voxel of the
scale. The scales encoded as `rawdata`, `jpeg` and
`compressed_segmentation` are read, sharded or not, and a scale encoded
otherwise (`png`, `compresso`, `jxl` or `crackle`) is refused when it is
loaded.

Each scale maps its voxels to the `"physical"` space of Neuroglancer, in
nanometres, by an [`Affine`][brainhops.datamodel.transformations.Affine]
that scales by the `resolution` and moves by
`(voxel_offset + 0.5) * resolution`. The half voxel is there because an
integer index of brainhops is the centre of a voxel, while Neuroglancer
puts the corner of the voxel `voxel_offset` at
`voxel_offset * resolution`. The conventions are described, with their
sources, in the section on the format below.

Precomputed volumes are not written: the images have no writer, and an
image is saved to another format.

## The format

The precomputed format of [Neuroglancer](https://github.com/google/neuroglancer)
serves large volumes, such as connectomics or BigBrain data, as a
directory of small files that a viewer fetches over HTTP. A volume is a
directory that holds a JSON file named `info` and one subdirectory of
chunks for each resolution, called a scale:

```text
volume/
    info
    8_8_8/
        0-64_0-64_0-64
        0-64_0-64_64-128
        ...
    16_16_16/
        0.shard
        1.shard
```

The `info` file says what the voxels are and lists the scales, finest
first:

```json
{
  "@type": "neuroglancer_multiscale_volume",
  "type": "image",
  "data_type": "uint8",
  "num_channels": 1,
  "scales": [
    {
      "key": "8_8_8",
      "size": [6446, 6643, 8090],
      "resolution": [8, 8, 8],
      "voxel_offset": [0, 0, 0],
      "chunk_sizes": [[64, 64, 64]],
      "encoding": "raw"
    }
  ]
}
```

`type` is `"image"` or `"segmentation"`, and a segmentation has a
single channel. `data_type` is one of `uint8`, `int8`, `uint16`,
`int16`, `uint32`, `int32`, `uint64` and `float32`. Each scale has a
`key`, the path of its chunks relative to the directory of `info`, a
`size` in voxels, a `resolution`, which is the size of a voxel in
nanometres, an optional `voxel_offset` in voxels, which defaults to 0,
and a list of `chunk_sizes`, of which a reader uses the first. All the
vectors are `[x, y, z]`.

The scale is cut into a grid of chunks, and the chunk at grid position
`g` holds the voxels from `voxel_offset + g * chunk_size` up to
`voxel_offset + min((g + 1) * chunk_size, size)`, end excluded. Without
sharding, the chunk is a file in the directory of the scale, named by
those bounds as `<xBegin>-<xEnd>_<yBegin>-<yEnd>_<zBegin>-<zEnd>`, such
as `0-64_64-128_0-64`. With a `sharding`
member, the chunks are gathered into `<shard>.shard` files, where each
chunk is found by the compressed Morton code of `g`, through a shard
index and a minishard index (see [`PrecomputedSharding`][]). In both
layouts, the data of a chunk is a `[x, y, z, channel]` array in Fortran
order, so `x` varies fastest. The `rawdata` encoding stores it as
little-endian values without a header, and the `jpeg`,
`compressed_segmentation`, `compresso`, `png`, `jxl` and `crackle`
encodings compress it.

The voxel `i` of a scale covers the interval
`[voxel_offset + i, voxel_offset + i + 1)` in units of its resolution,
so its centre lies at `(voxel_offset + i + 0.5) * resolution`
nanometres. Neuroglancer places the scales in one frame in this way, and
it reads the voxel under a point by rounding the coordinate down, so the
corner of a voxel, and not its centre, sits at an integer coordinate.
The edges of the scales coincide when their `voxel_offset * resolution`
agree.

The conventions above were checked against the specification at
google/neuroglancer@abbdfc5ea97b81c0c0bf04a414f5c59f4d84d27f: the pages
`src/datasource/precomputed/volume.md` and `sharded.md`, the types of
`src/util/data_type.ts` and the encodings of
`src/datasource/precomputed/base.ts`. The page `volume.md` does not say
where the centre of a voxel lies. That convention comes from the code
of the viewer at the same commit: `getInfo` and `getSources` in
`src/datasource/precomputed/frontend.ts`, which map a scale onto the
frame of the first scale by its resolution and its `voxel_offset`, and
`getDataValue` and `getInterpolatedDataValue` in
`src/sliceview/volume/frontend.ts`, which round a position down to find
its voxel and subtract 0.5 before they interpolate.
"""

__all__ = [
    "PrecomputedImage",
    "PrecomputedMetadata",
    "PrecomputedMultiScaleImage",
    "PrecomputedRaw",
    "PrecomputedScale",
    "PrecomputedSharding",
]

from ._image import PrecomputedImage
from ._metadata import PrecomputedMetadata
from ._multiscale import PrecomputedMultiScaleImage
from ._raw import PrecomputedRaw, PrecomputedScale, PrecomputedSharding
