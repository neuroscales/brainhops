"""Whole-slide images, read with OpenSlide.

[OpenSlide](https://openslide.org/) reads, but does not write, the pyramidal
images of slide scanners. The `openslide` extra installs openslide-python and
openslide-bin, which provides the OpenSlide library as wheels. On a platform
for which openslide-bin has no wheel, the OpenSlide library can be installed
by the system's package manager instead, with openslide-python on top of it.

```python
from brainhops.io.images import load

slide = load("CMU-1.svs")            # AperioMultiScaleImage
slide.images[2].data.shape           # (x, y, c), read region by region
slide.images[2].data[:256, :256]     # reads that region only
slide.metadata.raw.properties["openslide.mpp-x"]  # every property
level = load("CMU-1.svs", level=2)   # AperioImage: one level
load("slide.tif", hint="openslide")  # a generic tiled TIFF, by OpenSlide
```

## Formats

Each vendor, as `OpenSlide.detect_format` names it, has a single-scale class
and a multiscale class, which share the extensions and hints of the vendor:

* `aperio`: [`AperioImage`][] and [`AperioMultiScaleImage`][]; extensions
  `.svs`; hints `aperio`, `svs`.
* `hamamatsu`: [`HamamatsuImage`][] and [`HamamatsuMultiScaleImage`][];
  extensions `.ndpi`, `.vms`, `.vmu`; hints `hamamatsu`, `ndpi`, `vms`, `vmu`.
* `mirax`: [`MiraxImage`][] and [`MiraxMultiScaleImage`][]; extensions `.mrxs`;
  hints `mirax`, `mrxs`, `3dhistech`.
* `leica`: [`LeicaImage`][] and [`LeicaMultiScaleImage`][]; extensions `.scn`;
  hints `leica`, `scn`.
* `philips`: [`PhilipsImage`][] and [`PhilipsMultiScaleImage`][]; extensions
  `.tiff`; hints `philips`.
* `ventana`: [`VentanaImage`][] and [`VentanaMultiScaleImage`][]; extensions
  `.bif`, `.tif`; hints `ventana`, `bif`.
* `sakura`: [`SakuraImage`][] and [`SakuraMultiScaleImage`][]; extensions
  `.svslide`; hints `sakura`, `svslide`.
* `trestle`: [`TrestleImage`][] and [`TrestleMultiScaleImage`][]; extensions
  `.tif`; hints `trestle`.
* `zeiss`: [`ZeissImage`][] and [`ZeissMultiScaleImage`][]; extensions `.czi`;
  hints `zeiss`, `czi`.
* `dicom`: [`DicomWsiImage`][] and [`DicomWsiMultiScaleImage`][]; extensions
  `.dcm`; hints `dicom-wsi`.
* `generic-tiff`: [`GenericTiffImage`][] and [`GenericTiffMultiScaleImage`][];
  extensions `.tif`, `.tiff`; hints `generic-tiff`.

Every hint is also accepted as `openslide.<hint>`, and `hint="openslide"`
selects all of these formats. A vendor class reads only the slides that
OpenSlide attributes to its vendor. The vendors share the two bases
[`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage] and
[`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage],
which read a slide of any vendor, and each vendor class only declares its
extensions and hints and sniffs its own vendor. The levels of a pyramid of a
vendor are objects of the single-scale class of that vendor.

OpenSlide opens files only by name. A local file is therefore sniffed (its
format is detected from its content) and read in place. A remote file, a
stream or bytes are first copied to a temporary file, and they are read only
when a hint asks for one of these formats, because they are not sniffed. The
multi-file formats (MIRAX, Hamamatsu VMS, DICOM WSI and Trestle) cannot be
read from such a copy, since the copy lacks their other files.

## Which reader reads a TIFF-based slide

SVS, NDPI, Philips, SCN, BIF, Trestle and generic slides are TIFF files, which
the [TIFF reader][brainhops.io.images.tiff] reads too. When several formats
can read a file, the format with the highest sniffing score reads it. A slide
of a known vendor scores `CERTAIN` with OpenSlide, while the TIFF reader
scores it 0.9 as a pyramid and `LIKELY` as a single-scale image. OpenSlide
therefore reads vendor slides whenever it is installed, and the same holds
for the single-scale classes when `level=` is given. A generic tiled TIFF, on
the other hand, scores only `MAYBE` with OpenSlide. The TIFF reader therefore
keeps such a file, together with its OME-XML, ImageJ and resolution metadata,
unless `hint="openslide"` is given. Finally, when `level=` is not given, the
single-scale class of a vendor scores a pyramid at 0.8 times the vendor
score, so that the multiscale class of the same vendor reads it.

## Data and geometry

A level is an F-ordered `(x, y, c)` RGB `uint8` array whose row 0 is the top of
the slide. OpenSlide decodes RGBA pixels, whose alpha channel only marks the
pixels outside the scanned area. These pixels are composited onto the
background colour of the slide (`openslide.background-color`, white by
default), so that the image is RGB. Indexing `image.data` reads only the
indexed region, while `numpy.asarray(image.data)` reads the whole level. With
`lazy=True`, or when dask is the array backend, the data is instead a dask
array whose chunks are whole tiles.

The transformation of a level scales its pixels to the `"physical"` system.
The scale is the full-resolution pixel size, which the slide records in
micrometres as `openslide.mpp-x` and `openslide.mpp-y`, multiplied by the
downsampling factor `f` of the level. When the slide records no pixel size,
the transformation is the identity, with no unit. Pixel `i` of a level is
centred at the full-resolution coordinate `f * i + (f - 1) / 2`, so that every
level covers the whole slide, as in TIFF and OME-Zarr pyramids. The bounds of
the scanned area (`openslide.bounds-*`) are kept as metadata only, and they
do not change the geometry.

## Metadata

A slide and each of its levels hold the same
[`OpenSlideMetadata`][brainhops.io.images.openslide.OpenSlideMetadata],
whose record is an
[`OpenSlideRaw`][brainhops.io.images.openslide.OpenSlideRaw]. The record holds
what OpenSlide reports when it opens the slide: every property
(`properties`), the shape and the downsampling factor of each level
(`levels`, one
[`OpenSlideLevelRaw`][brainhops.io.images.openslide.OpenSlideLevelRaw]
each), and the names of the label, macro, thumbnail and other images of the
slide (`associated_images`), which `associated_image(name)` reads. The record
also gives the `vendor`, the `level_downsamples`, the `background_color` and
the `bounds` of the slide, read from its properties. A single-scale image
keeps the index of its level in `level`. `OpenSlideMetadata.load(path)` reads
the record alone and keeps no open file.

## Consolidating and writing

OpenSlide cannot write slides, so these classes have no writer. An image or a
pyramid is saved to another format, for example with
`TiffImage.from_any(level).save(path)`, or with `io.save` under a name whose
format can hold it. Data assigned to an image, or to a level of a pyramid, is
pending until `consolidate()` moves it into `rawdata`. The metadata and
the transformations are kept, since no record can be encoded, and a
pyramid refuses to consolidate a level that is not a level of its slide,
such as an image of another class, with `WriterError`.

## Without OpenSlide

Without openslide-python or the OpenSlide library, the formats of this module
are not registered, and a request for one of them by hint (`"openslide"`,
`"svs"`, ...) says what to install. TIFF-based slides are then read by the
TIFF reader, if tifffile is installed.
"""

__all__ = [
    "OpenSlideReader",
    "OpenSlideImage",
    "OpenSlideLevelRaw",
    "OpenSlideMetadata",
    "OpenSlideMultiScaleImage",
    "OpenSlideRaw",
    "AperioImage",
    "AperioMultiScaleImage",
    "HamamatsuImage",
    "HamamatsuMultiScaleImage",
    "MiraxImage",
    "MiraxMultiScaleImage",
    "LeicaImage",
    "LeicaMultiScaleImage",
    "PhilipsImage",
    "PhilipsMultiScaleImage",
    "VentanaImage",
    "VentanaMultiScaleImage",
    "SakuraImage",
    "SakuraMultiScaleImage",
    "TrestleImage",
    "TrestleMultiScaleImage",
    "ZeissImage",
    "ZeissMultiScaleImage",
    "DicomWsiImage",
    "DicomWsiMultiScaleImage",
    "GenericTiffImage",
    "GenericTiffMultiScaleImage",
]

from ._base import OpenSlideImage, OpenSlideMultiScaleImage, OpenSlideReader
from ._formats import (
    AperioImage,
    AperioMultiScaleImage,
    DicomWsiImage,
    DicomWsiMultiScaleImage,
    GenericTiffImage,
    GenericTiffMultiScaleImage,
    HamamatsuImage,
    HamamatsuMultiScaleImage,
    LeicaImage,
    LeicaMultiScaleImage,
    MiraxImage,
    MiraxMultiScaleImage,
    PhilipsImage,
    PhilipsMultiScaleImage,
    SakuraImage,
    SakuraMultiScaleImage,
    TrestleImage,
    TrestleMultiScaleImage,
    VentanaImage,
    VentanaMultiScaleImage,
    ZeissImage,
    ZeissMultiScaleImage,
)
from ._metadata import OpenSlideMetadata
from ._raw import OpenSlideLevelRaw, OpenSlideRaw
