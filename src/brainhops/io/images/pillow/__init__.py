"""
Two-dimensional raster images -- PNG, JPEG, BMP, GIF, WebP, PNM, JPEG 2000,
TGA, ... -- read and written with [Pillow](https://python-pillow.org).

This reader requires the `pillow` extra (`pip install brainhops[pillow]`).

```python
from brainhops.io.images import load
from brainhops.io import save

image = load("photo.png")            # PillowImage
image.data.shape                     # (width, height, 3): x, y, c
save(image, "photo.jpg", quality=95)
```

## Data

Pillow decodes an image into rows of pixels, `(rows, columns[,
samples])`. The reader returns that array transposed to the brainhops
order -- a view, not a copy -- so that `data[x, y]` is the pixel in
column `x` and row `y`, as for every image in brainhops (see
[`brainhops.io.base.raster`][]):

| Pillow mode                     | `data`                                |
| ------------------------------- | ------------------------------------- |
| `1` (bilevel)                   | `bool`, `(x, y)`                      |
| `L` (grey)                      | `uint8`, `(x, y)`                     |
| `LA` (grey + alpha)             | `uint8`, `(x, y, 2)`                  |
| `RGB`, `YCbCr`, `LAB`, `HSV`    | `uint8`, `(x, y, 3)`                  |
| `RGBA`, `CMYK`                  | `uint8`, `(x, y, 4)`                  |
| `I;16` (16-bit grey)            | `uint16`, `(x, y)`                    |
| `I` (32-bit grey)               | `int32`, `(x, y)`                     |
| `F` (floating point)            | `float32`, `(x, y)`                   |
| `P`, `PA` (palette)             | `uint8`, `(x, y, 3)` or `(x, y, 4)`   |

The channels of a multi-sample image lie on a channel axis `c`, after the
spatial axes. The colour space is the one the file stores (`mode` tells
which), and is not converted, except for a palette image, whose colours
are looked up (`palette=False` keeps the indices instead). Pillow reduces
16-bit-per-channel colour PNGs to 8 bits per channel.

## Geometry

A raster file stores no origin and no orientation. The index space is
0-based, and an integer index is the *centre* of a pixel. The first row of
the file is the top of the picture, so `y` points down; this is a
convention of the format, which the reader does not encode as an
orientation. Nor is the EXIF orientation tag of a photograph applied: the
pixels are returned as stored (apply `PIL.ImageOps.exif_transpose` yourself
if you need them as displayed).

The image carries one transformation, a scaling from its `"pixel"`
coordinate system, whose axes count samples, to a `"physical"` one. By
default, the physical size of a pixel is **unknown**: the scaling is the
identity and the physical axes have no unit. Most files record a
resolution in dots per inch (`info["dpi"]`: the PNG `pHYs` chunk, the JPEG
JFIF density or EXIF resolution, the BMP header), but it describes a screen
or a printer far more often than the scene, so it is used only when the
caller asks:

* `load(file, dpi=True)` takes the pixel size from the file's resolution,
  `25.4 / dpi` millimetres, unless it is missing or a placeholder (72 or
  96 dpi), in which case the size stays unknown.
* `load(file, dpi=300)` or `dpi=(300, 150)` uses that resolution instead.
* `load(file, pixel_size=0.01, unit="mm")` (or `pixel_size=(sx, sy)`) sets
  the pixel size directly, and wins over the resolution. `unit` alone
  converts a size from `dpi` to another unit of length.

A PNG file may also record a pixel size in an `sCAL` chunk, which ITK
reads and writes as its pixel spacing. ITK does not convert its spacing
(conventionally in millimetres) to the unit the chunk names, so that unit
cannot be relied on, and the chunk is not applied: it is kept as
`info["sCAL"] = (unit, x, y)` (unit 1 is the metre, 2 the radian), and can
be passed on as `load(file, pixel_size=info["sCAL"][1:], unit="mm")`.

When an image is written, its resolution is recorded in the formats that
store one (PNG, JPEG, BMP, TIFF) if its preferred transformation is a
scaling onto axes measured in a unit of length; a translation, which no
raster format stores, is dropped. Otherwise, a resolution the image was
read with is written back. `dpi=` overrides this (`dpi=False` records
none).

## Frames

A multi-frame file (an animated GIF, PNG or WebP, a multi-picture JPEG) is
read one frame at a time: the first by default, or `load(file, frame=i)`.
The number of frames is `image.n_frames`. Pillow composites the frames of
an animated GIF, so frame `i` is what is displayed at step `i`.

## Metadata

What the file records besides the pixels is kept on the image, not in the
data model: `image_format` (Pillow's name for the format, such as
`"PNG"`), `mode` (the mode as stored, such as `"P"`), `info` (Pillow's
metadata dictionary), `frame` and `n_frames`. The ICC profile and the
EXIF block are written back when the image is saved.

## Writing

The data must be `(x, y)` or `(x, y, c)` with one to four channels (more
axes are accepted only if they are singletons). The data type is stored as
it is, never converted:

| `data`                          | Pillow mode | Formats that store it       |
| ------------------------------- | ----------- | --------------------------- |
| `bool`, `(x, y)`                | `1`         | PNG, BMP, GIF, TIFF, ...    |
| `uint8`, `(x, y)`               | `L`         | all                         |
| `uint8`, `(x, y, 2)`            | `LA`        | PNG, WebP, ...              |
| `uint8`, `(x, y, 3)`            | `RGB`       | all                         |
| `uint8`, `(x, y, 4)`            | `RGBA`      | PNG, WebP, TIFF, ...        |
| `uint16`, `(x, y)`              | `I;16`      | PNG, TIFF                   |
| `int32` / `float32`, `(x, y)`   | `I` / `F`   | TIFF                        |

Anything else -- `float64`, `int16`, several channels of 16 bits, more
than four channels, a 3D volume -- raises
[`WriterError`][brainhops.io.base.parsers.WriterError], as does a mode the
chosen format cannot store (16 bits in JPEG, say). Convert the data first.
The format is chosen from the file extension, or with `format=` (default
PNG when writing to an unnamed stream). Other keywords are passed on to
Pillow's `Image.save`, such as `quality` for JPEG. JPEG and lossy WebP do
not store the data exactly.

## Limits

* **Decompression bombs.** Pillow refuses to decode an image of more than
  twice `PIL.Image.MAX_IMAGE_PIXELS` pixels (about 179 million pixels by
  default), raising `PIL.Image.DecompressionBombError`, and warns above it.
  To read a larger image you trust, set `PIL.Image.MAX_IMAGE_PIXELS` to a
  larger value, or to `None`.
* **No lazy access.** Pillow decodes a whole frame at once: the data is
  read in full when the image is loaded.
* **TIFF.** Pillow reads TIFF, but TIFF files (`.tif`, `.tiff`) are left to
  the dedicated TIFF reader, which reads stacks, pyramids and their
  metadata; this reader claims TIFF content only as a fallback.
"""

__all__ = ["PillowImage"]

from ._image import PillowImage
