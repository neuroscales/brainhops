"""Whole-slide images, read with OpenSlide.

[OpenSlide](https://openslide.org) reads, but does not write, the pyramidal
images of slide scanners. The `openslide` extra installs openslide-python and
the openslide-bin wheels; elsewhere, the OpenSlide library can come from the
system, with openslide-python on top of it.

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

Each vendor, as `OpenSlide.detect_format` names it, has a single-scale and a
multiscale class, with its own extensions and hints:

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
OpenSlide attributes to its vendor.

OpenSlide opens files by name, so local files are sniffed and read in place.
Remote files, streams and bytes are copied to a temporary file and read only by
hint; the multi-file formats (MIRAX, Hamamatsu VMS, DICOM WSI and Trestle)
cannot be read from such a copy.

## Which reader reads a TIFF-based slide

SVS, NDPI, Philips, SCN, BIF, Trestle and generic slides are TIFF files, which
the [TIFF reader][brainhops.io.images.tiff] reads too. A slide of a known
vendor scores `CERTAIN` with OpenSlide, against 0.9 for the TIFF pyramid and
`LIKELY` for a single-scale TIFF, so OpenSlide wins whenever it is installed;
with `level=`, the single-scale classes compete in the same way. A generic
tiled TIFF scores only `MAYBE` with OpenSlide, so the TIFF reader keeps it,
along with its OME-XML, ImageJ and resolution metadata, unless
`hint="openslide"` is given. Without `level=`, a single-scale class scores a
pyramid at 0.8 times the vendor score, below the multiscale class.

## Data and geometry

A level is an F-ordered `(x, y, c)` RGB `uint8` array whose row 0 is the top of
the slide. The RGBA pixels of OpenSlide are composited onto the background
colour of the slide (`openslide.background-color`, white by default).
`image.data` reads only the region that is indexed, while
`numpy.asarray(image.data)` reads the whole level. With `lazy=True`, or when
dask is the array backend, the data is a dask array of whole tiles.

Each level is scaled to `"physical"` by the pixel size of `openslide.mpp-x` and
`openslide.mpp-y` (in micrometres) times its downsampling factor `f`, or by the
identity, with no unit, when the slide records no size. Pixel `i` is centred at
the full-resolution coordinate `f * i + (f - 1) / 2`, so that every level
covers the whole slide, as in TIFF and OME-Zarr pyramids. The bounds of the
scan (`openslide.bounds-*`) are kept as metadata only.

## Metadata

Images have the attributes `vendor`, `properties` (every OpenSlide property),
`n_levels`, `level_downsamples`, `background_color`, `bounds`, `level`
(single-scale images only) and `associated_images`, the names of the images
that `associated_image(name)` reads.

## Without OpenSlide

Without openslide-python or the OpenSlide library, this module is not
registered, and a request by hint (`"openslide"`, `"svs"`, ...) says what to
install. TIFF-based slides then fall back to the TIFF reader, if tifffile is
installed.
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
