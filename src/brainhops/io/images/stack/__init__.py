r"""Image stacks: folders of numbered two-dimensional images.

Light-sheet microscopes, electron microscopes and many export tools write
a volume as one image file per slice. This module reads such a folder as
one three-dimensional image, a
[`StackImage`][brainhops.io.images.stack.StackImage], and writes an image
as such a folder. The files are TIFF files, read and written with
tifffile (the `tiff` extra), or images in a format that Pillow reads and
writes (the `pillow` extra), such as PNG or JPEG.

A folder is read as a stack only when it is asked for, with the format
hint `"stack"` or with `StackImage.load`. The stack is never recognized
from its path, so `io.load` and `io.images.load` choose it only with the
hint, `io.sniff` never names it, and a single TIFF or Pillow file is
read by its own format.

```python
import brainhops.io as io
from brainhops.io.images.stack import StackImage

image = io.load("slices/", hint="stack")         # a StackImage
image = StackImage.load("slices/")               # the same
image = io.load("slices/", hint="stack", pattern="ch1_*.tif")
image = io.load("slices/", hint="stack", spacing=2.0, unit="um")
image = io.load("slices/", hint="stack", axis="y")  # coronal slices
image.data.shape                                 # (x, y, z) or (x, y, z, c)
image.metadata.raw.names                         # the files, in order
image.save("copy/")                              # a folder of the same files
image.save("out/slice.png")                      # slice_0.png, slice_1.png...
io.save(image, "volume.nii.gz")                  # or any other format
StackImage.from_instance(io.load("volume.nii.gz")).save("slices/")
```

## Files

A stack is a local folder. Its files are, by default, every file of the
folder whose extension is that of a TIFF file or of a format that the
Pillow reader claims, and subfolders are not searched. The option
`pattern` gives a glob pattern relative to the folder instead, such as
`"slice_*.png"` or `"ch1/*.tif"`, and every file that it selects must be
an image file. A pattern cannot reach outside the folder. Hidden files,
whose names start with a dot, are never taken. The format of a file is
told from its extension, and TIFF files need tifffile while other images
need Pillow. A folder that holds a Zarr store is not a stack.

The names of the files must differ in one run of digits only, which is
the number of the file, as in `z0.tif`, `z1.tif`, ..., `z10.tif`. The
files are sorted by that number, compared as an integer, so `z2.tif`
comes before `z10.tif`, whether or not the numbers are padded with zeros.
The numbers may start at any value, and a number may not be repeated.
Names that differ in two numbers, such as `t01_z005.tif`, are refused,
and `pattern` then selects the files of one stack.

Each file holds one plane, rows and columns, with at most one channel
axis, such as the samples of an RGB image. A TIFF file must hold a single
series, and a Pillow file a single frame. Every file must have the same
format, number of rows, number of columns, number of channels and data
type, so that they can be stacked. A file that breaks one of these rules
raises a `ParserContentError` that names it.

## Missing slices

The stack has one slice for every number from the first file to the
last. A number that no file carries, such as 4 among `z3.tif` and
`z5.tif`, is a missing slice, which the record lists in
[`StackRaw.missing`][brainhops.io.images.stack.StackRaw.missing]. Its
pixels are the value `fill` of the reader, which is NaN for a floating
type and 0 for any other type by default. Reading a missing slice reads
no file. Duplicated numbers and names that follow no single numbering
are still refused. An untouched stack is written without its missing
slices, while assigned data covers every slice, so a stack with
assigned data writes a file for each missing slice, which holds the
values that filled it.

## Data

The files are stacked along the axis that the option `axis` names, the
slowest axis, z, by default. The columns of a file lie along the first of
the two other axes and its rows along the second, which gives the
following orders.

| `axis` | A file is | Stored volume  | Data           |
|--------|-----------|----------------|----------------|
| `"z"`  | `(x, y)`  | `(z, y, x[, c])` | `(x, y, z[, c])` |
| `"y"`  | `(x, z)`  | `(y, z, x[, c])` | `(x, y, z[, c])` |
| `"x"`  | `(y, z)`  | `(x, z, y[, c])` | `(x, y, z[, c])` |

The stored volume is the one that tifffile would spell `ZYX` for the
default axis, and the data of the image is its view in the brainhops
order. Within a file, row zero is the top row, so the axis along the
rows points down, as for a single TIFF or Pillow image. The file of the
first number is the first slice.

After a read, `image.raw` is a lazy array that reads a file only when one
of its pixels is needed, with the reader of the single-file format:
`read_series` of the TIFF reader and `read_pillow` of the Pillow reader.
With NumPy as the array backend, `image.data` reads every file. With
Dask, `image.data` is a Dask array with one chunk per slice, and a file
is read only when its chunk is computed.

## Geometry

The folder has no header, so it records no origin and no orientation.
The geometry of the stack is held by its record, in three fields:
`pixel_size`, the size of a pixel along the columns and the rows of the
files, `spacing`, the distance from one slice to the next, and `unit`.
The image has a single transformation, a `Scaling` from the voxels to the
`"physical"` system, which these fields scale. A size that the record
does not know is 1 along that axis, without a unit.

* The pixel size is the one that every TIFF file records, read as the
  TIFF reader reads it: from OME-XML, from ImageJ metadata, or from
  resolution tags. Pillow files record no size that the Pillow reader
  uses by default, so their pixel size is unknown.
* The spacing is the `PhysicalSizeZ` of the OME-XML that every file
  records, or else their ImageJ `spacing`, and a negative ImageJ spacing
  reverses the stacking axis. Other files do not record it.
* A size that the files do not all record the same is unknown, and a
  warning is issued.
* The options `pixel_size`, `spacing` and `unit` of the reader set the
  fields instead. `pixel_size` is one size, or a pair `(columns, rows)`.
  The sizes that the files record are converted to `unit`, and the sizes
  of the options are in `unit`, or else in the unit that the files
  record.

## Metadata

`image.metadata` is a
[`StackMetadata`][brainhops.io.images.stack.StackMetadata], whose record,
a [`StackRaw`][brainhops.io.images.stack.StackRaw], lists the files as
[`StackMember`][brainhops.io.images.stack.StackMember] objects, in order,
and holds the stacking axis and the geometry. Each member gives the name
of its file relative to the folder, its number, its axes, shape and type,
and its header: the `TiffMetadata` that the TIFF reader parses, with the
tags, the OME-XML and the ImageJ metadata, or the `PillowHeader` of a
Pillow file, with its mode and the `info` dictionary of Pillow. Reading
the metadata alone, with `StackMetadata.load(folder)` or
`MetadataFormat.load(folder, hint="stack")`, reads the header of every
file but none of their pixels, and takes the same options as the image,
except `fill`.

## Writing

`image.save(path)` writes one file per slice, through the TIFF and Pillow
writers of single files, and the path decides the names of the files.

* A path whose name ends with the extension of a TIFF file or of a format
  that Pillow writes names the files: the number of each slice is
  appended to the name before its extension, after an underscore, and
  padded with zeros to the digits of the largest number. `out.tif` thus
  gives `out_0.tif`, `out_1.tif`, ..., `out_9.tif` for ten slices, and
  `out_000.tif` to `out_119.tif` for 120. The files are written in the
  folder of the path, whose other files are kept.
* Any other path is the folder of the stack. The files keep the names of
  the record, and a new stack is named `slice_<number>.tif`. The folder
  is written beside its target and then moved in its place. An existing
  folder is replaced only if it holds nothing but image files, and a
  folder from which the image reads some files only if it holds nothing
  else, so that the files of another channel, which a pattern left out,
  are never removed.

The numbers are those of the record, or 0, 1, 2 and so on for an image
without a record. A missing slice is not written. An image that was read
and not changed is written by copying its files, so that each file keeps
its bytes when it keeps its format. Otherwise, each slice is encoded by
the writer of its format. A TIFF file records the pixel size when it has
a unit of length, and the magnitude of the spacing when it has no
channels, as the `PhysicalSizeZ` of a volume of one slice. A file with
channels is written as a plane, and records no spacing, and a Pillow file
records no size. The option `axis` of the writer cuts the volume along
another axis.

Writing follows the rules of the format records: the image is
consolidated into a copy, as by `StackImage.consolidate`, which is
written, and the image itself is not changed. A slice of assigned data
is written even where the record has no file, since the data of the
model takes precedence over the record. When the files that are replaced
are files that the lazy array reads, the image is consolidated in place,
and its lazy array then reads the new files. Bytes and streams cannot
hold a folder, so they are refused. `io.save` chooses a format from the
extension of the path, so it saves a stack to `volume.tif` as one
multi-page TIFF file, and a folder is written with
`StackImage.from_instance(image).save(folder)`.
"""

__all__ = [
    "StackImage",
    "StackMember",
    "StackMetadata",
    "StackRaw",
]

from ._image import StackImage
from ._metadata import StackMetadata
from ._raw import StackMember, StackRaw
