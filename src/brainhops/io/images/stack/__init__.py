r"""Image stacks: folders of numbered two-dimensional images.

Light-sheet microscopes, electron microscopes and many export tools write
a volume as one image file per slice. This module reads such a folder as
one three-dimensional image, a
[`StackImage`][brainhops.io.images.stack.StackImage]. The files
are TIFF files, read with tifffile (the `tiff` extra), or images in a
format that Pillow reads (the `pillow` extra), such as PNG or JPEG.
Image stacks are read only.

```python
import brainhops.io as io

image = io.load("slices/")                       # a StackImage
image = io.load("slices/", hint="stack")      # the same, explicitly
image = io.load("slices/", pattern="ch1_*.tif")  # a subset of the files
image = io.load("slices/", pixel_size={"z": 2.0}, unit="um")
image.data.shape                                 # (x, y, z) or (x, y, z, c)
image.metadata.raw.names                         # the files, in order
io.save(image, "volume.nii.gz")                  # or any other format
```

## Files

A stack is a local folder. Its files are, by default, every file of
the folder whose extension is that of a TIFF file or of a format that the
Pillow reader claims, and subfolders are not searched. The option
`pattern` gives a glob pattern relative to the folder instead, such as
`"slice_*.png"` or `"ch1/*.tif"`, and every file that it selects must be
an image file. A pattern cannot reach outside the folder. Hidden files,
whose names start with a dot, are never taken. The format of a file is
told from its extension, and TIFF files need tifffile while other images
need Pillow. A folder that holds a Zarr store is never read as a
stack.

The names of the files must differ in one run of digits only, which is
the number of the file, as in `z0.tif`, `z1.tif`, ..., `z10.tif`. The
files are sorted by that number, compared as an integer, so `z2.tif`
comes before `z10.tif`, whether or not the numbers are padded with zeros.
The numbers must follow each other without a gap and without a
repetition, and they may start at any value. Names that differ in two
numbers, such as `t01_z005.tif`, are refused, and `pattern` then selects
the files of one stack.

Each file holds one plane, rows and columns, with at most one channel
axis, such as the samples of an RGB image. A TIFF file must hold a single
series, and a Pillow file a single frame. Every file must have the same
format, number of rows, number of columns, number of channels and data
type, so that they can be stacked. A file that breaks one of these rules
raises a `ParserContentError` that names it.

## Data

The files are stacked along the slowest axis of the volume, which is z.
The stored volume is therefore `(z, y, x)`, or `(z, y, x, c)` with
channels, as tifffile spells it `ZYX` or `ZYXS`, and the data of the
image is its Fortran-ordered view `(x, y, z)` or `(x, y, z, c)`. Within a
file, row zero is the top row, so y points down, as for a single TIFF or
Pillow image. The first file in the order of the numbers is the slice
`z = 0`.

After a read, `image.raw` is a lazy array that reads a file only when one
of its pixels is needed, with the reader of the single-file format:
`read_series` of the TIFF reader and `read_pillow` of the Pillow reader.
With NumPy as the array backend, `image.data` reads every file. With
Dask, `image.data` is a Dask array with one chunk per file, and a file is
read only when its chunk is computed.

## Geometry

The folder has no header, so it records no origin, no orientation and,
usually, no spacing between the slices. The image has a single
transformation, a `Scaling` from the voxels to the `"physical"` system,
as for a single TIFF or Pillow image.

* The size of a pixel along x and y is the one that every TIFF file
  records, read as the TIFF reader reads it: from OME-XML, from ImageJ
  metadata, or from resolution tags. Pillow files record no size that
  the Pillow reader uses by default, so their pixel size is unknown.
* The spacing of the slices is the ImageJ `spacing` that every file
  records, and a negative spacing flips z. Other files do not record it.
* A size that the files do not record, or on which they disagree, is
  unknown. The scaling is then 1 along that axis, without a unit, and a
  warning is issued when the files disagree.
* `pixel_size` and `unit` set the sizes instead, as for every raster
  reader. `pixel_size` is one size for every spatial axis, one size per
  axis in the order x, y and z, or a mapping that sets the named axes
  only, so that `pixel_size={"z": 2.0}` gives only the spacing of the
  slices. Without `unit`, such a size is in the unit that the files
  record.

## Metadata

`image.metadata` is a
[`StackMetadata`][brainhops.io.images.stack.StackMetadata],
whose record, a
[`StackRaw`][brainhops.io.images.stack.StackRaw], lists the
files as [`StackMember`][brainhops.io.images.stack.StackMember]
objects, in order. Each member gives the name of its file relative to the
folder, its number, its axes, shape and type, and its header: the
`TiffMetadata` that the TIFF reader parses, with the tags, the OME-XML
and the ImageJ metadata, or the `PillowHeader` of a Pillow file, with its
mode and the `info` dictionary of Pillow. Reading the metadata alone,
with `StackMetadata.load(folder)`, reads the header of every file but
none of their pixels.

## Writing

Image stacks are read only. A folder of files has no header that could
record what the image holds beyond its pixels, and writing it would mean
replacing many files that the lazy array reads. An image is saved to
another format instead, such as NIfTI or a multi-page TIFF, and
[`save`][brainhops.io.save] refuses to write it as a stack.
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
