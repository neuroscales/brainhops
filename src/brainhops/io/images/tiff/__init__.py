"""
TIFF images -- plain TIFF, BigTIFF, OME-TIFF, ImageJ hyperstacks and
pyramidal TIFF -- read and written with
[tifffile](https://github.com/cgohlke/tifffile).

This reader requires the `tiff` extra (`pip install brainhops[tiff]`).
Without tifffile, it is not registered (see
[Without tifffile](#without-tifffile)).

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

A TIFF file holds one or more *series* -- an image, a stack or a
hyperstack -- that tifffile assembles from its pages, each with the axes
it reads from the file's metadata (`TZCYXS`, ...). The reader returns one
series (`series=`, the first by default) transposed to the brainhops order
-- a view, not a copy -- so that the spatial axes `x, y[, z]` come first,
then time `t`, then channels `c`, then any other axis. An RGB image is
`(x, y, c)`; an ImageJ hyperstack `TZCYX` is `(x, y, z, t, c)`.

The pixels are read when `image.data` is first accessed, not when the
image is loaded:

* **memory-mapped** (copy-on-write) when the file is local and the series
  is stored uncompressed and contiguously, which is how ImageJ, tifffile
  and most microscopes write it -- nothing is read until the array is
  indexed;
* **lazily**, as a dask array over tifffile's Zarr store, when dask is the
  array backend or with `lazy=True` (this needs dask and zarr) -- the
  path for large compressed or tiled files;
* **in full** otherwise.

`mmap=False` turns memory-mapping off.

## Geometry

The index space is 0-based, and an integer index is the *centre* of a
pixel. The first row of the file is the top of the picture, so `y` points
down; this is not encoded as an orientation.

The image carries one transformation, a scaling from its `"pixel"` (or
`"voxel"`) coordinate system to a `"physical"` one. Its sizes come from
the first of these that records them, axis by axis:

1. **OME-XML**: `PhysicalSizeX/Y/Z` in their units (micrometres when the
   unit is absent, as the schema says), and `TimeIncrement` (seconds).
   The position of the first plane (`PositionX/Y/Z` of the plane whose
   Z, C and T indices are all 0) is the origin: the transformation is an
   affine with that translation. A position with no unit (or in OME's
   `"reference frame"`) is taken in the unit of the pixel size, and a
   position is used only along an axis whose size is known.
2. **ImageJ**: the pixel size is `1 / XResolution` (and `1 /
   YResolution`) in the hyperstack's `unit` (`"micron"`, `"um"` and
   `"\\u00B5m"` are micrometres), the slice step is `spacing` -- negative
   when the slices run backwards, which flips `z` -- and the frame
   interval is `finterval`.
3. **Resolution tags**: `XResolution` and `YResolution` per
   `ResolutionUnit` -- inch (also when the unit tag is absent),
   centimetre, or tifffile's millimetre and micrometre. A unit of 1 (none)
   gives an aspect ratio only, so the size is unknown.

When none records a size, it is **unknown**: the scaling is the identity
and the physical axes have no unit. Missing resolution tags (which
tifffile reports as 1 pixel per inch), and placeholders -- 72 or 96 dpi,
or one pixel per inch or centimetre -- are unknown too.

Every part of it can be overridden: `pixel_size=` (one size, one per
spatial axis, or a mapping by name) and `unit=` as for every raster image,
and `origin=` (`False` to ignore the file's positions).

## Pyramids

A pyramidal series -- OME-TIFF or plain TIFF whose levels are stored as
SubIFDs, or any pyramid tifffile recognizes -- is read as a
[`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage],
whose levels are [`TiffImage`][brainhops.io.images.tiff.TiffImage]s,
finest first, each read only when its data is accessed. `load(file,
level=k)` reads a single level as a `TiffImage` instead.

A level's pixel size is the full-resolution pixel size times its
downsampling factor, the ratio of the shapes (not rounded). Levels are
aligned by extent: pixel `i` of a level downsampled by `f` is centred on
the full-resolution pixel coordinate `f * i + (f - 1) / 2`, so the edges
of every level coincide. This is the convention of block-averaged
pyramids, and the one an OME-Zarr pyramid states with a translation of
`(f - 1) / 2` pixels. Every level maps onto the same `"physical"` system,
and the pyramid's own transformation is the identity.

## Metadata

What the file records besides the pixels is kept on the image, not in the
data model: `dialect` (`"ome"`, `"imagej"`, or `None`), `ome_xml`,
`imagej_metadata`, `tags` (resolution, description, software, date,
artist, copyright, ...), `series`, `level`, `n_series`, `n_levels` and
`storage_axes` (tifffile's axes of the series, such as `"TZCYX"`).

## Writing

An image is written with tifffile, in one of three dialects:

* **OME-TIFF** when the file name ends in `.ome.tif` or `.ome.tiff`, when
  the image was read from an OME-TIFF, or when its pixel size is known:
  `PhysicalSizeX/Y/Z` in micrometres, `TimeIncrement`, and plane positions
  when the transformation has a translation. The image name and channel
  names of an OME-TIFF it was read from are written back. OME-TIFF stores
  no flip: a negative scale is written as its magnitude.
* **ImageJ** when the image was read from an ImageJ hyperstack and ImageJ
  can store it (axes in the order `TZCYXS`, `uint8`, `uint16` or
  `float32`): unit, spacing (signed), frame interval, and the rest of the
  ImageJ metadata (display ranges, LUTs, labels, ...) when it still
  applies.
* **Plain TIFF** otherwise, with the axes in tifffile's own metadata and,
  if the pixel size is known (with `dialect="plain"`), the resolution tags
  in centimetres.

`dialect=` chooses one explicitly. When writing OME-TIFF, the resolution
tags are written too, for readers that know no OME. The axes are stored in
the order they were read in, or else `TZCYX`, with an RGB(A) `uint8`
channel axis stored as samples (`S`). The data type is stored as it is.
BigTIFF is used when the data does not fit in 4 GiB (or with
`bigtiff=True`). Other keywords go to tifffile (`compression="zlib"`,
`tile=(256, 256)`, ...). A `TiffMultiScaleImage` is written as a pyramid,
its levels as SubIFDs of the first (OME-TIFF or plain TIFF); the placement
of the levels is not stored, and is derived from their shapes again when
the file is read. The software, date, artist, copyright and a few other
tags read from a file are written back.

## Without tifffile

When tifffile is not installed, this module is not registered, and a TIFF
file is read by the Pillow raster reader
([`PillowImage`][brainhops.io.images.pillow.PillowImage]) instead, if
Pillow is installed: one page (`frame=`) at a time, with the resolution
tags as the pixel size only with `dpi=True`. OME-XML, ImageJ metadata,
stacks and pyramids need tifffile.
"""

__all__ = ["TiffImage", "TiffMultiScaleImage"]

from ._image import TiffImage
from ._multiscale import TiffMultiScaleImage
