"""
JPEG 2000 images -- JP2 files (`.jp2`, `.jpx`, `.jpf`) and raw codestreams
(`.j2k`, `.j2c`, `.jpc`) -- as multiscale images, read and written with
[Pillow](https://python-pillow.org), which wraps
[OpenJPEG](https://www.openjpeg.org).

This reader requires the `pillow` extra (`pip install brainhops[pillow]`).

```python
from brainhops.io.images import load
from brainhops.io import save

pyramid = load("slide.jp2")          # Jp2MultiScaleImage
pyramid.images[2].data.shape         # level 2, reduced by 4: (x, y[, c])
level = load("slide.jp2", level=2)   # Jp2Image, decoding level 2 only
save(level, "copy.j2k")              # raw codestream, lossless
```

## Resolution levels

A JPEG 2000 codestream is a wavelet decomposition: an image coded with
`N` decomposition levels can be decoded at `N + 1` resolutions, the full
one and reductions by 2, 4, ..., `2**N`. Such a file is read as a
multiscale image
([`Jp2MultiScaleImage`][brainhops.io.images.jpeg2000.Jp2MultiScaleImage]
or [`J2kMultiScaleImage`][brainhops.io.images.jpeg2000.J2kMultiScaleImage]),
whose levels are single-scale images
([`Jp2Image`][brainhops.io.images.jpeg2000.Jp2Image] or
[`J2kImage`][brainhops.io.images.jpeg2000.J2kImage]), finest first. A
level is decoded -- at its own, reduced, resolution -- only when its data
is first accessed. `load(file, level=r)` reads a single level as a
single-scale image, and a file with a single resolution is read as one.

The size of level `r` is the one the codestream defines: `ceil(X1 / 2**r)
- ceil(X0 / 2**r)`, where `X0` and `X1` are the edges of the image area on
the reference grid (`XOsiz` and `Xsiz`), which is `ceil(width / 2**r)`
when the image area has no offset.

!!! note "No region decoding"
    Pillow decodes a whole level at once: there is no decoding of a
    region of interest, and no access to the quality layers (every layer
    is decoded).

!!! note "Image offsets"
    Pillow cannot decode a reduced level of an image whose area has an
    offset on the reference grid (`XOsiz` or `YOsiz` not zero): such a file
    is read at full resolution, as a single-scale image, and asking for
    one of its reduced levels raises an error.

!!! warning "Decompression bombs"
    Pillow checks the size of the *full-resolution* image when it opens a
    file, and refuses one of more than twice `PIL.Image.MAX_IMAGE_PIXELS`
    pixels, even when only a coarse level is asked for. To read a larger
    image you trust, raise the limit: `PIL.Image.MAX_IMAGE_PIXELS = None`.

## Data

The pixels are returned F-ordered, as a view of what Pillow decodes:
`(x, y)` for a single component, and `(x, y, c)` with a channel axis
when there are several (grey + alpha, RGB, RGBA). The values are those
the codestream holds: an image of 1 to 8 bits per component is read as
`uint8` (`int8` if signed), and one of 9 to 16 bits as `uint16` (`int16`)
-- Pillow's scaling of other bit depths to 8 or 16 bits, and its offset of
signed samples, are undone. Pillow reads colour images of more than 8
bits per component only at 8 bits, so they are refused rather than read
at a lower precision.

## Geometry

The index space is 0-based, an integer index is the *centre* of a pixel,
and the first row is the top of the picture (`y` points down), as for
every raster image.

The image carries one transformation onto a `"physical"` system. At full
resolution it is a scaling by the pixel size. The pixel size is taken
from the resolution box of a JP2 file (`res `): the *capture* resolution
(`resc`), or else the default *display* resolution (`resd`), in grid
points per metre, converted to millimetres. A placeholder -- 72 or 96
dots per inch, or one per inch -- is unknown, as is an absent box; the
scaling is then the identity, onto axes with no unit. `pixel_size=` and
`unit=` override it, as for every raster image.

Level `r` is reduced by `f = 2**r`. The low-pass samples the wavelet
transform keeps are the even samples of each level, so the pyramid is
aligned on its first pixel, not on its extent: pixel `i` of level `r` is
centred on the full-resolution pixel `f * i + s`, where `s = f *
ceil(X0 / f) - X0` is zero unless the image area has an offset `X0` that
is not a multiple of `f`. Its transformation is a scaling by `f` times
the pixel size, with a translation of `s` pixels when `s` is not zero.
(The levels of a TIFF pyramid are, instead, aligned on their extent.)
The offset of the image area itself is kept in the header, and is not a
translation: the first full-resolution pixel is at the origin.

## Metadata

What the file records besides the pixels is kept in `header`, a
`Jpeg2000Header`: the container (`"jp2"` or `"j2k"`), the size and offset
of the image area and of the tiles, the bit depth, signedness and
subsampling of each component, the number of resolution levels and
quality layers, whether the wavelet is the reversible (lossless) one, the
comments, the capture and display resolutions, and the top-level XML
(GMLJP2, ...), UUID (GeoJP2, XMP, ...) and UUID info boxes. They are not
interpreted, but are written back.

## Writing

An image is written with Pillow, **losslessly** by default (the
reversible 5-3 wavelet, one quality layer), as a JP2 file
([`Jp2Image`][brainhops.io.images.jpeg2000.Jp2Image], `.jp2`) or a raw
codestream ([`J2kImage`][brainhops.io.images.jpeg2000.J2kImage], `.j2k`).
Pillow writes 8-bit grey, grey + alpha, RGB and RGBA images, and 16-bit
grey ones. The number of resolution levels is that of the file the image
was read from, or else six (OpenJPEG's default), fewer if the image is too
small; `num_resolutions=` sets it. Other keywords go to Pillow
(`quality_mode`, `quality_layers`, `irreversible`, `tile_size`, ...).

In a JP2 file, the pixel size is written as the capture resolution, when
it is known; the resolutions, XML and UUID boxes and the comment of the
file the image was read from are written back. A raw codestream stores
none of these.

A multiscale image is written as its full-resolution level, with as many
resolution levels as it has: the codec computes the coarser levels again,
so they must have the shapes it gives them.

## Pillow

The Pillow raster reader
([`PillowImage`][brainhops.io.images.pillow.PillowImage]) also reads JPEG
2000 files, at full resolution only. It scores them `WEAK`, below these
readers, and does not claim their extensions; `hint="pillow"` asks for it.
"""

__all__ = [
    "Jpeg2000Image",
    "Jp2Image",
    "J2kImage",
    "Jpeg2000MultiScaleImage",
    "Jp2MultiScaleImage",
    "J2kMultiScaleImage",
]

from ._image import (
    J2kImage,
    J2kMultiScaleImage,
    Jp2Image,
    Jp2MultiScaleImage,
    Jpeg2000Image,
    Jpeg2000MultiScaleImage,
)
