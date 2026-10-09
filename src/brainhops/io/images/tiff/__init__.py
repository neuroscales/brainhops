r"""TIFF images, read and written with tifffile.

This module handles plain TIFF, BigTIFF, OME-TIFF, ImageJ hyperstacks and
pyramidal TIFF, and needs the `tiff` extra (see [Without
tifffile](#without-tifffile)).

```python
from brainhops.io.images import load
from brainhops.io import save

image = load("stack.ome.tif")        # TiffImage
image.data.shape                     # (x, y, z, c)
image.transformation                 # Scaling onto "physical", in µm
pyramid = load("slide.ome.tif")      # TiffMultiScaleImage, if pyramidal
level = load("slide.ome.tif", level=2)
save(image, "copy.ome.tif")
```

## Data

tifffile assembles the pages of a file into one or more series (an image, a
stack or a hyperstack), with axes such as `TZCYXS` taken from the metadata. The
reader returns one series (`series=`, the first by default) as a transposed
view in the brainhops order: spatial axes x, y and z, then time, channels and
other axes. An RGB image is thus `(x, y, c)`, and an ImageJ `TZCYX` hyperstack
is `(x, y, z, t, c)`.

Pixels are read when `image.data` is first accessed. A local, uncompressed and
contiguous series (as ImageJ, tifffile and most microscopes write them) is
memory-mapped copy-on-write, unless `mmap=False`. When dask is the array
backend, or with `lazy=True`, the data is a dask array over the Zarr store of
tifffile (which needs dask and zarr), suited to large compressed or tiled
series. Otherwise, the series is read in full.

## Geometry

Pixel coordinates are 0-based, with integers at pixel centres, and the first
row of the file is the top of the image. This convention is not encoded as an
orientation. The image has a single transformation, which scales the `"pixel"`
or `"voxel"` system to the `"physical"` one. The size along each axis is taken
from the first of the following sources that records it:

1. OME-XML: `PhysicalSizeX`, `PhysicalSizeY` and `PhysicalSizeZ` in their units
   (micrometres if the unit is absent, as the schema says), and `TimeIncrement`
   (seconds). The origin is given by the `PositionX`, `PositionY` and
   `PositionZ` of the first plane (the plane whose Z, C and T indices are all
   0), and this origin turns the transformation into an affine one with that
   translation. A position without a unit, or in the OME "reference frame", is
   taken in the unit of the pixel size. Positions apply only to the axes whose
   size is known.
2. ImageJ: the pixel size is `1 / XResolution` (and `1 / YResolution`) in the
   `unit` of the hyperstack (`"micron"`, `"um"` and `"\u00B5m"` are
   micrometres), the slice step is `spacing` (negative when the slices run
   backwards, which flips z), and the frame interval is `finterval`.
3. Resolution tags: `XResolution` and `YResolution` count pixels per
   `ResolutionUnit`. This unit is the inch (also when the tag is absent), the
   centimetre, or one of the millimetre and micrometre units that tifffile
   adds. Unit 1 (none) records only an aspect ratio, so the size stays unknown.

Otherwise, the size is unknown and the transformation is the identity, with no
unit. Missing resolution tags (1 pixel per inch for tifffile) and placeholders
(72 or 96 dpi, or 1 pixel per inch or centimetre) also leave the size unknown.
`pixel_size=` and `unit=` override the file as for every raster image, and
`origin=` overrides the positions (`False` ignores them).

## Pyramids

A pyramidal series (an OME-TIFF or plain TIFF with SubIFD levels, or any
pyramid that tifffile recognises) is read as a
[`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage] of
[`TiffImage`][brainhops.io.images.tiff.TiffImage] levels, finest first, each
read when its data is accessed. `load(file, level=k)` reads a single
`TiffImage` instead.

The pixel size of a level is the full-resolution pixel size multiplied by the
downsampling factor `f` of the level. This factor is the ratio of the
full-resolution shape to the shape of the level, without rounding. Pixel `i`
of the level is centred at the full-resolution coordinate
`f * i + (f - 1) / 2`, so that the edges of all levels coincide, as in block
averaging. OME-Zarr expresses the same placement as a translation of
`(f - 1) / 2` pixels. All levels map to the same `"physical"` system, so the
pyramid has no transformation of its own.

## Metadata

The image keeps `dialect` (`"ome"`, `"imagej"` or `None`), `ome_xml`,
`imagej_metadata`, `tags` (resolution, description, software, date, artist,
copyright and others), `series`, `level`, `n_series`, `n_levels` and
`storage_axes` (the axes of tifffile, such as `"TZCYX"`).

## Writing

Images are written with tifffile, in one of three dialects:

* OME-TIFF is written when the name ends with `.ome.tif`, `.ome.tiff`,
  `.ome.btf`, `.ome.tf2` or `.ome.tf8`. It is also written when the image was
  read from an OME-TIFF, or when its pixel size or time interval is known,
  provided that its axes need no letter outside `TZCYXS`. The file records
  `PhysicalSizeX`, `Y` and `Z`, `TimeIncrement` and, if there is a
  translation, plane positions. The image name and channel names of a source
  OME-TIFF are written back. Flips are not stored, so a negative scale is
  written as its magnitude.
* ImageJ is written when the image was read from an ImageJ hyperstack and
  ImageJ can store it (axes in the order `TZCYXS`, and type `uint8`, `uint16`
  or `float32`). Unless the name asks for OME-TIFF, this rule takes
  precedence over the second OME-TIFF rule. The file records the unit, the
  signed spacing, the frame interval and the rest of the ImageJ metadata
  (display ranges, LUTs, labels and so on) that still applies.
* Plain TIFF is written otherwise. The axes are recorded in the tifffile
  metadata, and a known pixel size is recorded in resolution tags, in pixels
  per centimetre.

`dialect=` chooses a dialect explicitly. OME-TIFF files also carry resolution
tags, for readers that do not read OME-XML. The axes keep the order in which
they were read, or else follow the order `TZCYX`, and the channel axis of an
RGB or RGBA `uint8` image is stored as samples (`S`). The data type is stored
as is. BigTIFF is used when the data exceeds 4 GiB, or with `bigtiff=True`,
and other keywords are passed to tifffile, such as `compression="zlib"` or
`tile=(256, 256)`.

A `TiffMultiScaleImage` is written as a pyramid in which the coarser levels
are SubIFDs of the first level. The placement of the levels is not stored, and
it is derived again from their shapes when the file is read. The software,
date, artist, copyright and a few other tags are written back.

## Without tifffile

Without tifffile, this module is not registered, and TIFF files are read by the
[Pillow reader][brainhops.io.images.pillow.PillowImage] if Pillow is installed.
The Pillow reader reads one page at a time (`frame=`), and it uses the
resolution tags as the pixel size only with `dpi=True`. OME-XML, ImageJ
metadata, stacks and pyramids need tifffile.
"""

__all__ = ["TiffImage", "TiffMultiScaleImage"]

from ._image import TiffImage
from ._multiscale import TiffMultiScaleImage
