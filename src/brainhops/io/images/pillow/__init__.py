"""Raster images, read and written with [Pillow](https://python-pillow.org).

This module covers PNG, JPEG, BMP, GIF, WebP, PNM, TGA and other formats, and
needs the `pillow` extra.

```python
from brainhops.io.images import load
from brainhops.io import save

image = load("photo.png")            # PillowImage
image.data.shape                     # (width, height, 3): x, y, c
image.metadata.raw.format            # "PNG", from the header of the file
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

Channels come after the spatial axes. Colours stay in the colour space in
which they are stored, which the `mode` of the record names. Palette images are
the exception: their colours are looked up in the palette, unless
`palette=False` keeps the indices. Pillow reduces 16-bit colour PNG files to 8
bits.

After a read, `raw` is a lazy array of the frame, `(rows, columns[,
samples])`, which Pillow decodes only when `data` is first accessed. Its shape
and type are known from the header, so `image.shape` decodes nothing. The data
that it gives is read-only, since the writer writes `raw` and would lose an
edit made in place: assign `data` to change the pixels.

## Geometry

Raster files store no origin or orientation. Pixel coordinates are 0-based,
with integers at pixel centres, and the first row of the file is the top of the
image. This convention is not encoded as an orientation, and the EXIF
orientation is not applied (see `PIL.ImageOps.exif_transpose`).

The image has a single transformation, which scales pixels to the `"physical"`
system. The pixel size is unknown by default, which makes the transformation
the identity, with no unit. The resolution that a file records
(`info["dpi"]`) usually describes a screen or a printer, so it is used only on
request:

* `load(file, dpi=True)` makes pixels `25.4 / dpi` millimetres wide, unless the
  resolution is missing or a placeholder (72 or 96 dpi).
* `load(file, dpi=300)` or `dpi=(300, 150)` uses that resolution instead.
* `load(file, pixel_size=0.01, unit="mm")`, or `pixel_size=(sx, sy)`, takes
  precedence over any resolution. `unit` alone converts a size derived from a
  resolution.

With `dpi=True`, the image keeps the option and decodes the pixel size from the
resolution of its record, so an unchanged image is still written as the bytes
of its file. A resolution, a pixel size or a unit given to the reader are
assigned to the image as its transformations, and an image whose
transformations differ from those of its record is encoded again when it is
written.

ITK writes its spacing (conventionally in millimetres) to the PNG `sCAL` chunk
without converting it to the unit of the chunk, so the unit of the chunk is
unreliable. The chunk is therefore kept as `info["sCAL"] = (unit, x, y)`, where
unit 1 means metres and unit 2 means radians, but it is not applied. Its sizes
can be passed on as `load(file, pixel_size=info["sCAL"][1:], unit="mm")`.

When an image is written in a format that stores a resolution (PNG, JPEG, BMP
and TIFF), the resolution is computed from the preferred transformation if
that transformation scales onto axes with a unit of length, and any
translation is ignored. Otherwise, the resolution read with the image is
written back. `dpi=` overrides both, and `dpi=False` records no resolution.

## Frames

Multi-frame files (animated GIF, PNG and WebP, multi-picture JPEG) are read one
frame at a time. The first frame is read by default, `load(file, frame=i)`
reads frame `i`, and `image.metadata.raw.n_frames` gives the number of frames.
Animated GIF frames are composited, so frame `i` is the picture displayed at
step `i`. An unchanged image is written back with all the frames of its file,
and a changed image is written as one frame.

## Metadata

The header of the file is the record of the image,
[`PillowRaw`][brainhops.io.images.pillow.PillowRaw], held by its
[`PillowMetadata`][brainhops.io.images.pillow.PillowMetadata]. It gives the
format (such as `"PNG"`), the mode of the frame as stored (such as `"P"`), its
size, the Pillow `info` dictionary as read with the header, the index of the
frame and the number of frames:

```python
image.metadata.raw.format            # "PNG"
image.metadata.raw.mode              # "P"
image.metadata.raw.info["dpi"]       # (300.0, 300.0)
image.metadata.raw.n_frames          # 1
```

`PillowMetadata.load(file)` reads the record without decoding the pixels. The
image keeps the reader options `frame`, `palette` and `dpi`. The ICC profile
and EXIF block are written back when the image is encoded again.

## Writing

An image that was read and is written again in the format of its file, without
changes and without options, is written as the bytes of its file, so it keeps
them exactly, whatever the format and even when Pillow cannot write that
format. Any change of the data or of the geometry, any option, or another
format encodes the image again with Pillow, and the consolidated copy of the
image then reads back the bytes that Pillow wrote.

The data to write is `(x, y)`, or `(x, y, c)` with 1 to 4 channels, and any
other axis must be a singleton. The data type is stored as is:

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
format cannot store, such as 16-bit JPEG. The format is given by `format=` or,
failing that, by the extension of the file, and a stream without a name is
written in the format of the record, or as PNG. Other keywords are passed to
`Image.save`, such as `quality` for JPEG. Pillow ignores a keyword that it does
not read for the format, so the writer refuses such a keyword with a
`TypeError`. JPEG and lossy WebP do not preserve the pixel values exactly.

## Limits

* Pillow refuses images larger than twice `PIL.Image.MAX_IMAGE_PIXELS` (about
  179 million pixels) with a `PIL.Image.DecompressionBombError`, and warns
  above the limit itself. Raising `MAX_IMAGE_PIXELS`, or setting it to `None`,
  allows a trusted larger image.
* The whole frame is decoded when `data` is first read, so there is no access
  to a part of the frame.
* A stream given to the reader must stay open while the data may be read.
* `.tif` and `.tiff` files are left to the dedicated TIFF reader. The Pillow
  reader claims TIFF only as a fallback, when tifffile (the `tiff` extra) is
  not installed.
* JPEG 2000 files (`.jp2`, `.j2k` and the like) are left to the dedicated
  reader of [`brainhops.io.images.jpeg2000`][], which reads their resolution
  levels and their resolution boxes. The Pillow reader claims them only as a
  fallback, at full resolution, or when `hint="pillow"` asks for it.
"""

__all__ = ["PillowImage", "PillowMetadata", "PillowRaw"]

from ._image import PillowImage
from ._metadata import PillowMetadata
from ._raw import PillowRaw
