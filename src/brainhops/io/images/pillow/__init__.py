"""Raster images, read and written with Pillow.

This module covers PNG, JPEG, BMP, GIF, WebP, PNM, JPEG 2000, TGA and other
formats, and needs the `pillow` extra.

```python
from brainhops.io.images import load
from brainhops.io import save

image = load("photo.png")            # PillowImage
image.data.shape                     # (width, height, 3): x, y, c
save(image, "photo.jpg", quality=95)
```

## Data

Pillow decodes images as `(rows, columns[, samples])`, and the reader returns a
transposed view in the brainhops order, so that `data[x, y]` is the pixel of
column `x` and row `y`.

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

Channels come after the spatial axes. Colours stay in their stored colour
space, which `mode` names, except that palette colours are looked up
(`palette=False` keeps the indices). Pillow reduces 16-bit colour PNG files to
8 bits.

## Geometry

Raster files store no origin or orientation. Pixel coordinates are 0-based,
with integers at pixel centres, and the first row of the file is the top of the
image. This convention is not encoded as an orientation, and the EXIF
orientation is not applied (see `PIL.ImageOps.exif_transpose`).

The only transformation scales pixels to `"physical"`. Its pixel size is
unknown by default, which makes it the identity, with no unit. The resolution
that a file records (`info["dpi"]`) usually describes a screen or a printer, so
it is used only on request:

* `load(file, dpi=True)` makes pixels `25.4 / dpi` millimetres wide, unless the
  resolution is missing or a placeholder (72 or 96 dpi).
* `load(file, dpi=300)` or `dpi=(300, 150)` uses that resolution instead.
* `load(file, pixel_size=0.01, unit="mm")`, or `pixel_size=(sx, sy)`, takes
  precedence over any resolution. `unit` alone converts a size derived from a
  resolution.

ITK writes its spacing (conventionally in millimetres) to the PNG `sCAL` chunk
without converting it to the unit of the chunk, so that unit is unreliable. The
chunk is therefore kept as `info["sCAL"] = (unit, x, y)` (unit 1 is metres, 2
radians) but not applied; it can be passed on as
`load(file, pixel_size=info["sCAL"][1:], unit="mm")`.

On writing, formats that store a resolution (PNG, JPEG, BMP and TIFF) record
the one of the preferred transformation if it scales onto axes with a unit of
length, dropping any translation; otherwise, a resolution read with the image
is written back. `dpi=` overrides both, and `dpi=False` records none.

## Frames

Multi-frame files (animated GIF, PNG and WebP, multi-picture JPEG) are read one
frame at a time, the first by default or `load(file, frame=i)`, and
`image.n_frames` counts them. Animated GIF frames are composited, so frame `i`
is the picture displayed at step `i`.

## Metadata

The image keeps `image_format` (such as `"PNG"`), `mode` (as stored, such as
`"P"`), the Pillow `info` dictionary, `frame` and `n_frames`. The ICC profile
and EXIF block are written back on save.

## Writing

The data is `(x, y)` or `(x, y, c)` with 1 to 4 channels, plus singleton axes
only, and its type is stored as is:

| `data`                          | Pillow mode | Formats that store it       |
| ------------------------------- | ----------- | --------------------------- |
| `bool`, `(x, y)`                | `1`         | PNG, BMP, GIF, TIFF, ...    |
| `uint8`, `(x, y)`               | `L`         | all                         |
| `uint8`, `(x, y, 2)`            | `LA`        | PNG, WebP, ...              |
| `uint8`, `(x, y, 3)`            | `RGB`       | all                         |
| `uint8`, `(x, y, 4)`            | `RGBA`      | PNG, WebP, TIFF, ...        |
| `uint16`, `(x, y)`              | `I;16`      | PNG, TIFF                   |
| `int32` / `float32`, `(x, y)`   | `I` / `F`   | TIFF                        |

Any other data (`float64`, `int16`, several 16-bit channels, more than four
channels, a volume) raises a
[`WriterError`][brainhops.io.base.parsers.WriterError], as does a mode the
format cannot store, such as 16-bit JPEG. The format follows from the extension
or `format=` (PNG for an unnamed stream), and other keywords go to
`Image.save`, such as `quality` for JPEG. JPEG and lossy WebP are not exact.

## Limits

* Pillow refuses images larger than twice `PIL.Image.MAX_IMAGE_PIXELS` (about
  179 million pixels) with a `PIL.Image.DecompressionBombError`, and warns
  above the limit itself. Raising `MAX_IMAGE_PIXELS`, or setting it to `None`,
  allows a trusted larger image.
* The whole frame is decoded on load; there is no lazy access.
* `.tif` and `.tiff` files are left to the dedicated TIFF reader. This reader
  claims TIFF only as a fallback, when tifffile (the `tiff` extra) is not
  installed.
"""

__all__ = ["PillowImage"]

from ._image import PillowImage
