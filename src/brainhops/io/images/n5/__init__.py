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

Each image holds the attributes as [`N5Metadata`][], whose record is an
[`N5Raw`][], and the samples of its dataset in `raw`, a lazy array that
TensorStore reads block by block when the data are needed. The data are
indexed in the order of the `dimensions` of the dataset,
`[x, y, z, ...]`, whose first dimension varies fastest in a block. The datasets
compressed with `raw`, `gzip`, `bzip2`, `xz` and `blosc` are read, and
the others, such as `lz4`, `zstd` or the `compressionType` of version 0
of the specification, are refused when they are loaded.

The specification of N5 defines no spatial attribute. Each image maps
its voxels to a `"physical"` space with the `pixelResolution`,
`resolution`, `unit`, `downsamplingFactors`, `offset` and `scales`
attributes that the n5-viewer, n5-ij, BigDataViewer and Paintera write,
and with the identity when there is none. The conventions are
described, with their sources, in the section on the format below.

N5 containers are not written: the images have no writer, and an image
is saved to another format.

## The format

A container is a directory tree, and every directory in it is a group. A
group may hold its attributes in a JSON file named `attributes.json`,
and a dataset is a group whose attributes describe an array, which is
cut into blocks that are stored as separate files:

```text
container.n5/
    attributes.json          {"n5": "4.0.0"}
    raw/
        attributes.json      {"resolution": [4, 4, 40], "multiScale": true}
        s0/
            attributes.json  {"dimensions": [1000, 800, 50],
                              "blockSize": [64, 64, 64],
                              "dataType": "uint8",
                              "compression": {"type": "gzip", "level": -1}}
            0/0/0
            0/1/0
            ...
        s1/
            attributes.json  {..., "downsamplingFactors": [2, 2, 1]}
```

The attributes of a dataset give its `dimensions`, its `blockSize`, its
`dataType` (`uint8` to `uint64`, `int8` to `int64`, `float32`,
`float64` or `object`) and its `compression`, an object whose `type` is
`raw`, `gzip`, `bzip2`, `lz4`, `xz` or a scheme of an extension, such
as `blosc`. Version 0 of the specification named the scheme in a
`compressionType` string instead. The block at grid position
`(i, j, k)` is the file `i/j/k` below the dataset, and a block that is
missing reads as zeros, since datasets are sparse. A block starts with a
big-endian header: a mode (`uint16`, 0 by default, 1 for a block of
variable length and 2 for serialized objects), the number of
dimensions (`uint16`), the size of the block along each dimension
(`uint32` each, smaller than `blockSize` for a block at an upper edge)
and, in mode 1, the number of elements (`uint32`). The samples follow,
big-endian and compressed.

The first dimension varies fastest in a block. The specification does
not say so, but the readers that brainhops relies on agree:
TensorStore decodes a block in Fortran order with the dimensions in the
order of `dimensions`, and zarr-python reverses `dimensions`,
`blockSize` and the grid positions to present the array in C order. In
the usual order of the Java tools, `dimensions` therefore lists `x`
first, which is the order of the data of brainhops, and the reverse of
the order of a Zarr array in C order.

The specification defines no spatial attribute, so the voxel size and
the position of a dataset are conventions of its tools. brainhops reads
the following attributes, which apply to the leading dimensions in the
same order as `dimensions`:

- `pixelResolution`, the size of a voxel of the finest level, as an
  object `{"dimensions": [rx, ry, rz], "unit": "um"}` or as a bare
  array, which has no unit (the n5-viewer, n5-ij and n5-universe);
- `resolution`, the same size as a bare array, with the unit of a
  `unit` string when there is one (Paintera, BigDataViewer and
  n5-universe), read when `pixelResolution` is absent;
- `downsamplingFactors`, the factors of a level from the finest one
  (the n5-viewer, n5-ij, Paintera and BigDataViewer);
- `scales`, the factors of every level, finest first, on the group of a
  pyramid (the deprecated layout of the n5-viewer);
- `offset`, a position in the units of the size (Paintera and
  n5-universe).

A dataset downsampled by `f` from a finest level of voxel size `r` has
voxels of size `r * f`, and its voxel `i` lies at
`r * (f * i + (f - 1) / 2) + o`, so that an integer index is the centre
of a voxel, as in brainhops, and the voxels of every level cover the
same extent. The origin `o` is the `offset` of the group of a pyramid,
or 0. A dataset that gives its own `offset` gives the position of its
first voxel instead, so its voxel `i` lies at `r * f * i + offset`.
Without these attributes, the voxel size is 1 and has no unit, so the
geometry of a dataset without spatial attributes stays unknown, as for
a raster image. A unit that names no physical length, such as `pixel`,
gives no unit. An attribute that breaks these conventions is ignored
with an [`N5AttributeWarning`][].

The attributes of a dataset that is read on its own are its own, as
the n5-universe library reads them. A pyramid is a group whose
datasets are `s0`, `s1`, ..., read up to the first one that is missing.
For a level of a pyramid, an attribute that the dataset does not give
is read from the group: the voxel size from its `pixelResolution` or
`resolution`, the factors from its `scales`, and its `offset` when the
voxel size comes from the group, as Paintera and the deprecated layout
of the n5-viewer store them. A level that gives no factors, while its
group gives no `scales`, has factors of 1. brainhops does not infer
factors of `2**k` from the name `s<k>`, which n5-universe does.

Other spatial conventions are not read: the `affineTransform` of the
n5-viewer, the `transform` of COSEM, the `pixelWidth` and `xOrigin`
attributes of the ImageJ layout of n5-ij, the `multiscales` of
OME-NGFF, the channel groups `c0`, `c1`, ... of the n5-viewer (each of
which is a pyramid of its own), and the registrations of BigDataViewer,
which live in its XML file.

The specification was checked at
saalfeldlab/n5@c489d1653e05b3bac38f433037befd79f62586c4 (tag n5-3.5.1):
the section "File-system specification" of `README.md`, version 4.0.0,
and `DatasetAttributes.java`, `DataType.java` and
`DefaultBlockReader.java`. The order of the dimensions was checked in
`tensorstore/driver/n5/metadata.cc` of
google/tensorstore@97c57499888348dca3d41e6d66fb9a69de1ba046 (tag
v0.1.45), whose `DecodeChunk` reads a block in Fortran order, and in
`zarr/n5.py` of
zarr-developers/zarr-python@f1978dde52b11a7803a1a06937e69948acc8fe84
(tag v2.18.3), whose `N5Store` reverses the dimensions. The spatial
attributes were checked in these sources:

- the section "Container structure" of `README.md` of
  saalfeldlab/n5-viewer@36e75fd88ebcbc88a64da9fb082a28f9b46ded21, which
  defines `downsamplingFactors` and `pixelResolution` on each level and
  marks the `scales` and `pixelResolution` of the group as deprecated,
  and `N5ExportMetadataReader.java` at the same commit, which reads them;
- `N5SingleScaleMetadataParser.java` (`buildTransform`, and a bare
  `pixelResolution` without a unit) and
  `N5GenericSingleScaleMetadataParser.java` (`resolution`, `unit`, and
  an `offset` that replaces the translation) of
  saalfeldlab/n5-universe@b4ba6f5a0d3de7fef4a1a3f1c4aaaef10e23958d;
- `N5ScalePyramidExporter.java` (`buildN5VMetadata`) of
  saalfeldlab/n5-ij@248569553f26e7a9522e6d4472b3d0b5267b8cba, which
  writes the `pixelResolution` of the finest level with the factors of
  averaging, or the voxel size of each level with factors of 1;
- the section on supported data of `README.md` of
  saalfeldlab/paintera@a0ad809d3f9460ade96ba7b705c55341267c4fce, which
  puts `resolution` and `offset` on the group of a pyramid marked
  `"multiScale": true` and ignores those of its datasets, and
  `N5Helpers.considerDownsampling` of
  saalfeldlab/paintera@b629633508f653e1936e3bea96834dcd6f6592e0 (tag
  paintera-0.23.0), which shifts the levels by half a voxel per factor;
- `BDV N5 format.md` of
  bigdataviewer/bigdataviewer-core@dfe1a66fd05613dbceaee9211d4266cee3520f98,
  which writes `"multiScale": true` and `resolution` on each time point
  and `downsamplingFactors` on each level, for Paintera.

Some conventions are not settled by these sources. Paintera can reverse
`resolution` and `offset`, because some tools wrote them in the reverse
order (its option `--reverse-array-attributes`), and brainhops reads
them in the order of `dimensions`, as the Java tools write them. The
older n5-viewer read a bare `pixelResolution` in micrometres, and
n5-universe reads it without a unit, which brainhops follows.
"""

__all__ = [
    "N5AttributeWarning",
    "N5Image",
    "N5Metadata",
    "N5MultiScaleImage",
    "N5Raw",
]

from ._image import N5Image
from ._metadata import N5Metadata
from ._multiscale import N5MultiScaleImage
from ._raw import N5Raw
from ._spatial import N5AttributeWarning
