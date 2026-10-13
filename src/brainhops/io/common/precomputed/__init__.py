"""
Reading of Neuroglancer precomputed volumes.

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
order, so `x` varies fastest. The `raw` encoding stores it as
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

The `info` file is held as a [`PrecomputedRaw`][] record, and
[`PrecomputedMetadata`][] reads that record without reading any chunk.
[`PrecomputedImage`][brainhops.io.images.precomputed.PrecomputedImage]
reads one scale lazily, and
[`PrecomputedMultiScaleImage`][brainhops.io.images.precomputed.PrecomputedMultiScaleImage]
reads them all.

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
    "PrecomputedMetadata",
    "PrecomputedRaw",
    "PrecomputedScale",
    "PrecomputedSharding",
]

from ._metadata import PrecomputedMetadata
from ._raw import PrecomputedRaw, PrecomputedScale, PrecomputedSharding
