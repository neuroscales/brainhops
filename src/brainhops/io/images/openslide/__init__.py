"""
Whole-slide images -- Aperio SVS, Hamamatsu NDPI/VMS/VMU, MIRAX, Leica
SCN, Philips TIFF, Ventana BIF, Sakura SVSlide, Trestle, Zeiss CZI, DICOM
WSI and generic tiled TIFF -- read with [OpenSlide](https://openslide.org/)
(read only).

This reader requires the `openslide` extra
(`pip install brainhops[openslide]`), which installs `openslide-python`
and `openslide-bin`, the wheels of the OpenSlide C library (Linux, macOS
and Windows). Where `openslide-bin` has no wheel, install the OpenSlide
library with the system's package manager (`apt install libopenslide0`,
`brew install openslide`, `conda install -c conda-forge openslide`) and
`openslide-python` alone. Without them, these formats are not registered
(see [Without OpenSlide](#without-openslide)).

```python
from brainhops.io.images import load

slide = load("CMU-1.svs")            # AperioMultiScaleImage
slide.images[2].data.shape           # (x, y, c), read region by region
slide.images[2].data[:256, :256]     # reads that region only
slide.properties["openslide.mpp-x"]  # every OpenSlide property
level = load("CMU-1.svs", level=2)   # AperioImage: one level
load("slide.tif", hint="openslide")  # a generic tiled TIFF, by OpenSlide
```

## Formats

OpenSlide reads every format through one library, but each is a class of
its own, with its own extensions and hints, so that it can be asked for by
name. The vendor of a file is the one `OpenSlide.detect_format` reports:

Vendor (`VENDOR`): classes; extensions; hints.

* `aperio`: `AperioImage`, `AperioMultiScaleImage`; extensions `.svs`; hints
  `aperio`, `svs`.
* `hamamatsu`: `HamamatsuImage`, `HamamatsuMultiScaleImage`; extensions
  `.ndpi`, `.vms`, `.vmu`; hints `hamamatsu`, `ndpi`, `vms`, `vmu`.
* `mirax`: `MiraxImage`, `MiraxMultiScaleImage`; extensions `.mrxs`; hints
  `mirax`, `mrxs`, `3dhistech`.
* `leica`: `LeicaImage`, `LeicaMultiScaleImage`; extensions `.scn`; hints
  `leica`, `scn`.
* `philips`: `PhilipsImage`, `PhilipsMultiScaleImage`; extensions `.tiff`;
  hints `philips`.
* `ventana`: `VentanaImage`, `VentanaMultiScaleImage`; extensions `.bif`,
  `.tif`; hints `ventana`, `bif`.
* `sakura`: `SakuraImage`, `SakuraMultiScaleImage`; extensions `.svslide`;
  hints `sakura`, `svslide`.
* `trestle`: `TrestleImage`, `TrestleMultiScaleImage`; extensions `.tif`; hints
  `trestle`.
* `zeiss`: `ZeissImage`, `ZeissMultiScaleImage`; extensions `.czi`; hints
  `zeiss`, `czi`.
* `dicom`: `DicomWsiImage`, `DicomWsiMultiScaleImage`; extensions `.dcm`; hints
  `dicom-wsi`.
* `generic-tiff`: `GenericTiffImage`, `GenericTiffMultiScaleImage`; extensions
  `.tif`, `.tiff`; hints `generic-tiff`.

Every hint is also qualified by `openslide` (`"openslide.aperio"`), and
`hint="openslide"` alone selects all of them. A file is attributed to a
vendor's class only if OpenSlide detects that vendor: a class asked for by
hint refuses a slide of another vendor.

OpenSlide opens files by name. A local file (or an open file whose name is
a local file) is sniffed and read in place. A remote file, a stream or
bytes are copied into a temporary file first (deleted with the image),
and are only read when asked for by hint, since nothing is sniffed in
memory; a format made of several files (MIRAX, VMS, DICOM, Trestle)
cannot be read that way.

## Which reader reads a TIFF-based slide

SVS, NDPI, Philips, Leica SCN, Ventana BIF, Trestle and generic slides
are TIFF files, which the [TIFF reader][brainhops.io.images.tiff] reads
too. Formats are chosen by their sniffing scores first, so:

1. A slide of a known vendor scores `CERTAIN` (1.0) with its OpenSlide
   multiscale class. The TIFF reader scores a vendor whole-slide pyramid
   (tifffile's `is_svs`, `is_ndpi`, `is_philips`, `is_scn` or `is_bif`)
   0.9 rather than `CERTAIN`, and a single-scale TIFF image 0.75
   (`LIKELY`). So OpenSlide reads vendor slides when it is installed, and
   the TIFF reader reads them otherwise.
2. With `level=`, the OpenSlide single-scale class scores `CERTAIN`, the
   TIFF multiscale reader declines, and the single-scale TIFF reader
   scores `LIKELY`.
3. A generic tiled TIFF (no vendor) scores only `MAYBE` (0.5), so the TIFF
   reader, which also reads its OME-XML, ImageJ and resolution metadata,
   keeps it; `hint="openslide"` asks for OpenSlide instead.
4. Without a level, the single-scale class of a pyramid scores 0.8 times
   its vendor's score, below its multiscale class; a slide with a single
   level is read as a single-scale image.

## Data

Each level is F-ordered `(x, y, c)`: RGB `uint8` samples, row 0 at the
top (`y` points down; this is not encoded as an orientation). OpenSlide
decodes RGBA; the alpha channel only marks pixels outside the scanned
area, which are composited onto the slide's background colour
(`openslide.background-color`, white by default), so the image is RGB.

Pixels are never read in full unless asked for. `image.data` is a lazy
array that reads, with `read_region`, only the region it is indexed with
(`numpy.asarray(image.data)` reads the whole level); with `lazy=True`, or
when dask is the array backend, it is a dask array whose chunks are whole
tiles (at least 1024 pixels wide).

## Geometry

The index space is 0-based, and an integer is the centre of a pixel. The
image carries one transformation, a scaling from its `"pixel"` system to a
`"physical"` one, whose size is the full-resolution pixel size from
`openslide.mpp-x` / `openslide.mpp-y` (micrometres), times the level's
downsampling factor (`level_downsamples`). When the slide records no
pixel size it is unknown: the identity, in no unit. `pixel_size=` and
`unit=` override it, as for every raster image.

A level covers the same extent as the full resolution: pixel `i` of a
level downsampled by `f` is centred on the full-resolution pixel
coordinate `f * i + (f - 1) / 2` (a translation of `(f - 1) / 2`
full-resolution pixels), as in TIFF and OME-Zarr pyramids. A pyramid's
levels all map onto the same `"physical"` system, and its own
transformation is the identity. The bounds of a sparse slide
(`openslide.bounds-*`) are kept as metadata, not applied: every level
covers the whole slide.

## Metadata

What the slide records besides the pixels is kept on the image:
`vendor`, `properties` (every OpenSlide property, standard and
vendor-specific), `n_levels`, `level_downsamples`, `background_color`,
`bounds`, `level` (single-scale), and `associated_images`, the names of
the label, macro, thumbnail, ... images, which `associated_image(name)`
reads as `(x, y, c)` RGB arrays.

## Without OpenSlide

When `openslide-python` or the OpenSlide library is missing, this module
is not registered; asking for it by hint (`"openslide"`, `"svs"`, ...)
says what to install, and the TIFF-based slides are read by the
[TIFF reader][brainhops.io.images.tiff], if tifffile is installed.
"""

__all__ = [
    "OpenSlideFormat",
    "OpenSlideImage",
    "OpenSlideMultiScaleImage",
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

from ._base import OpenSlideFormat, OpenSlideImage, OpenSlideMultiScaleImage
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
