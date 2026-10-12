"""
JPEG 2000 images, read as multiscale images and written with Pillow.

JPEG 2000 files are either JP2 files (`.jp2`, `.jpx`, `.jpf`) or raw
codestreams (`.j2k`, `.j2c`, `.jpc`). They are decoded and encoded by
[Pillow](https://python-pillow.org), which wraps
[OpenJPEG](https://www.openjpeg.org), so this module needs the `pillow`
extra (`pip install brainhops[pillow]`).

```python
from brainhops.io.images import load
from brainhops.io import save

pyramid = load("slide.jp2")          # Jpeg2000MultiScaleImage
pyramid.images[2].data.shape         # level 2, reduced by 4: (x, y[, c])
level = load("slide.jp2", level=2)   # Jpeg2000Image, which reads level 2
pyramid.metadata.raw.n_levels        # the record of the file
save(level, "copy.j2k")              # a raw codestream, lossless
```

## Resolution levels

A JPEG 2000 codestream is a wavelet decomposition, so an image coded
with `N` decomposition levels can be decoded at `N + 1` resolutions: the
full one, and reductions by 2, 4, up to `2**N`. Such a file is read as
a [`Jpeg2000MultiScaleImage`][], whose levels are [`Jpeg2000Image`][]
objects, finest first. A level is decoded at its own resolution, and
only when its data is first needed. `load(file, level=r)` reads a single
level as a [`Jpeg2000Image`][], and a file with a single resolution is
read as one.

The size of level `r` is the size that the codestream defines,
`ceil(X1 / 2**r) - ceil(X0 / 2**r)`, where `X0` and `X1` are the edges
of the image area on the reference grid of the codestream. This is
`ceil(width / 2**r)` when the image area has no offset.

!!! note "No region decoding"
    Pillow decodes a whole level at once. A region of a level cannot be
    decoded alone, and every quality layer is decoded.

!!! note "Image offsets"
    Pillow cannot decode a reduced level of an image whose area has an
    offset on the reference grid, so such a file is read at full
    resolution, as a [`Jpeg2000Image`][], and reading the data of one of
    its reduced levels raises an error.

!!! warning "Decompression bombs"
    Pillow checks the size of the full-resolution image when it opens a
    file, and refuses an image of more than twice
    `PIL.Image.MAX_IMAGE_PIXELS` pixels, even when only a coarse level is
    asked for. To read a larger image that you trust, raise the limit
    with `PIL.Image.MAX_IMAGE_PIXELS = None`.

## Data

An image holds the samples of its level in `raw`, a lazy array that
Pillow decodes as `(rows, columns)`, or `(rows, columns, samples)` for
several samples per pixel. Its `data` is a read-only view of these
samples, indexed `(x, y)` or `(x, y, c)` in Fortran order, as for every
raster image. The values are those that the codestream holds. A
component of 1 to 8 bits is read as `uint8` (`int8` if it is signed),
and a component of 9 to 16 bits as `uint16` (`int16`), because the
scaling of Pillow to 8 or 16 bits and its offset of signed samples are
undone. Pillow reads colour images of more than 8 bits per component at
8 bits only, so they are refused rather than read at a lower precision.
A palette is looked up, which gives RGB or RGBA data.

## Geometry

The index space is 0-based, an integer index is the centre of a pixel,
and the first row is the top of the picture, so `y` points down, as for
every raster image.

An image has one transformation onto a `"physical"` system. At full
resolution, it is a scaling by the pixel size. The pixel size is taken
from the resolution box of a JP2 file: from the capture resolution, or
else from the default display resolution, converted from grid points
per metre to millimetres. A placeholder, 72 or 96 dots per inch or one
per inch, gives no size, nor does a file without the box. The scaling is
then the identity, onto axes without a unit. The options `pixel_size=`
and `unit=` override the pixel size, as for every raster image, and the
transformations that they give are assigned to the image.

Level `r` is reduced by `f = 2**r`. The wavelet keeps the even samples of
each level, so the pyramid is aligned on its first pixel rather than on
its extent: pixel `i` of level `r` is centred on the full-resolution
pixel `f * i + s`. The shift `s = f * ceil(X0 / f) - X0` is zero unless
the image area has an offset `X0` that is not a multiple of `f`. The
transformation of the level is a scaling by `f` times the pixel size,
with a translation of `s` pixels when `s` is not zero. The levels of a
TIFF pyramid, by contrast, are aligned on their extent. The offset of
the image area itself is kept in the record, and is not a translation,
so the first full-resolution pixel is at the origin.

## Metadata

What the file says beside the coded image is held in the
[`Jpeg2000Metadata`][] of the image, whose record is a
[`Jpeg2000Raw`][]. The record holds the container, the size and the
offset of the image area and of the tiles, the bit depth, the signedness
and the subsampling of each component, the number of resolution levels
and of quality layers, whether the wavelet is the reversible one, the
comments, the capture and display resolutions, the palette, and the
top-level XML, UUID and UUID info boxes, such as GMLJP2, GeoJP2 and XMP.
These boxes are not interpreted, but they are written back. The two
classes of the record are defined in
[`brainhops.io.common.jpeg2000`][brainhops.io.common.jpeg2000] and
exported here as well.

## Writing

`raw` and the record describe the file as it was read. Data that are
assigned are pending until the image is consolidated or written, as for
every image format. A full-resolution image or a pyramid that is read and
written again without changes gives the same bytes. When only the
comments, the pixel size, the boxes or the container of such an image
change, the coded image of the file is copied as it is, and only the
boxes and the comments are written again. A reduced level is always
encoded again, because it is not the image that its file codes.

Otherwise the image is encoded with Pillow, losslessly by default, with
the reversible 5-3 wavelet and one quality layer, as a JP2 file or a raw
codestream, as the name of the file says. Pillow writes 8-bit grey, grey
and alpha, RGB and RGBA images, and 16-bit grey images. The number of
resolution levels is the number of levels that the file the image was
read from has at and below the level of the image, or else six, which is
the default of OpenJPEG, and fewer for a small image.
The option `num_resolutions=` sets it. The other options of the JPEG
2000 writer of Pillow are passed on, such as `quality_mode`,
`quality_layers`, `irreversible` and `tile_size`.

In a JP2 file, the pixel size is written as the capture resolution, when
it is known. The comments and the boxes of the record are written into
the new file. A raw codestream stores no resolution and no box, so it
stores the comments alone, and its pixel size is lost.

A pyramid is written as its full-resolution level, with as many
resolution levels as it has. The codec computes the coarser levels
again, so they must have the shapes that it gives them. A coarser level
that holds assigned data, or assigned transformations other than those
that the first level gives it, is refused, because the file cannot store
them.

## Pillow

The Pillow raster reader
([`PillowImage`][brainhops.io.images.pillow.PillowImage]) also reads
JPEG 2000 files, at full resolution only. It scores them `WEAK`, below
these readers, and does not claim their extensions, so `hint="pillow"`
asks for it.
"""

__all__ = [
    "Jpeg2000Image",
    "Jpeg2000Metadata",
    "Jpeg2000MultiScaleImage",
    "Jpeg2000Raw",
]

from brainhops.io.common.jpeg2000 import Jpeg2000Metadata, Jpeg2000Raw

from ._image import Jpeg2000Image
from ._multiscale import Jpeg2000MultiScaleImage
