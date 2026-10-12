"""Whole-slide image formats, one per OpenSlide vendor.

Each vendor has a format class, from which a single-scale image class and a
multiscale image class derive. The formats differ only in their vendor name
(as `OpenSlide.detect_format` reports it), their extensions and their hints.
"""

__all__ = [
    "AperioFormat",
    "AperioImage",
    "AperioMultiScaleImage",
    "HamamatsuFormat",
    "HamamatsuImage",
    "HamamatsuMultiScaleImage",
    "MiraxFormat",
    "MiraxImage",
    "MiraxMultiScaleImage",
    "LeicaFormat",
    "LeicaImage",
    "LeicaMultiScaleImage",
    "PhilipsFormat",
    "PhilipsImage",
    "PhilipsMultiScaleImage",
    "VentanaFormat",
    "VentanaImage",
    "VentanaMultiScaleImage",
    "SakuraFormat",
    "SakuraImage",
    "SakuraMultiScaleImage",
    "TrestleFormat",
    "TrestleImage",
    "TrestleMultiScaleImage",
    "ZeissFormat",
    "ZeissImage",
    "ZeissMultiScaleImage",
    "DicomWsiFormat",
    "DicomWsiImage",
    "DicomWsiMultiScaleImage",
    "GenericTiffFormat",
    "GenericTiffImage",
    "GenericTiffMultiScaleImage",
]

import typing_extensions as tx

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.images.openslide._base import (
    _LEVEL_CLASSES,
    OpenSlideFormat,
    OpenSlideImage,
    OpenSlideMultiScaleImage,
)

# ----------------------------------------------------------------------
#   APERIO SVS
# ----------------------------------------------------------------------


class AperioFormat(OpenSlideFormat):
    """Aperio ScanScope Virtual Slide (SVS), now from Leica.

    An SVS file is a tiled TIFF whose first `ImageDescription` starts with
    `Aperio`. Besides the pyramid, it holds a thumbnail, a label and a macro
    image.
    """

    VENDOR = "aperio"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svs",)
    HINTS = ("aperio", "svs")


@register_format
class AperioImage(AperioFormat, OpenSlideImage):
    """A single level of an Aperio SVS slide.

    See [`OpenSlideImage`][].
    """


@register_format
class AperioMultiScaleImage(AperioFormat, OpenSlideMultiScaleImage):
    """An Aperio SVS slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   HAMAMATSU
# ----------------------------------------------------------------------


class HamamatsuFormat(OpenSlideFormat):
    """Hamamatsu NanoZoomer slides.

    NDPI files are TIFF-like files with 64-bit offsets. VMS and VMU slides are
    an index file together with JPEG or raw files in the same directory.
    """

    VENDOR = "hamamatsu"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ndpi", ".vms", ".vmu")
    HINTS = ("hamamatsu", "ndpi", "vms", "vmu")


@register_format
class HamamatsuImage(HamamatsuFormat, OpenSlideImage):
    """A single level of a Hamamatsu slide.

    See [`OpenSlideImage`][].
    """


@register_format
class HamamatsuMultiScaleImage(HamamatsuFormat, OpenSlideMultiScaleImage):
    """A Hamamatsu slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   MIRAX
# ----------------------------------------------------------------------


class MiraxFormat(OpenSlideFormat):
    """3DHISTECH MIRAX slides.

    A slide is a `.mrxs` file together with a directory of the same name, which
    holds `Slidedat.ini` and the data files.
    """

    VENDOR = "mirax"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mrxs",)
    HINTS = ("mirax", "mrxs", "3dhistech")


@register_format
class MiraxImage(MiraxFormat, OpenSlideImage):
    """A single level of a MIRAX slide.

    See [`OpenSlideImage`][].
    """


@register_format
class MiraxMultiScaleImage(MiraxFormat, OpenSlideMultiScaleImage):
    """A MIRAX slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   LEICA SCN
# ----------------------------------------------------------------------


class LeicaFormat(OpenSlideFormat):
    """Leica SCN slides.

    An SCN file is a BigTIFF whose `ImageDescription` is Leica XML that
    describes the collection and its images.
    """

    VENDOR = "leica"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".scn",)
    HINTS = ("leica", "scn")


@register_format
class LeicaImage(LeicaFormat, OpenSlideImage):
    """A single level of a Leica SCN slide.

    See [`OpenSlideImage`][].
    """


@register_format
class LeicaMultiScaleImage(LeicaFormat, OpenSlideMultiScaleImage):
    """A Leica SCN slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   PHILIPS TIFF
# ----------------------------------------------------------------------


class PhilipsFormat(OpenSlideFormat):
    """Philips TIFF slides, exported from iSyntax.

    A Philips TIFF is a tiled TIFF whose `ImageDescription` is Philips
    DataObject XML.
    """

    VENDOR = "philips"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tiff",)
    HINTS = ("philips",)


@register_format
class PhilipsImage(PhilipsFormat, OpenSlideImage):
    """A single level of a Philips TIFF slide.

    See [`OpenSlideImage`][].
    """


@register_format
class PhilipsMultiScaleImage(PhilipsFormat, OpenSlideMultiScaleImage):
    """A Philips TIFF slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   VENTANA BIF
# ----------------------------------------------------------------------


class VentanaFormat(OpenSlideFormat):
    """Roche (Ventana) BIF slides.

    A BIF file is a tiled BigTIFF with Ventana XMP metadata, whose tiles may
    overlap.
    """

    VENDOR = "ventana"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".bif", ".tif")
    HINTS = ("ventana", "bif")


@register_format
class VentanaImage(VentanaFormat, OpenSlideImage):
    """A single level of a Ventana BIF slide.

    See [`OpenSlideImage`][].
    """


@register_format
class VentanaMultiScaleImage(VentanaFormat, OpenSlideMultiScaleImage):
    """A Ventana BIF slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   SAKURA SVSLIDE
# ----------------------------------------------------------------------


class SakuraFormat(OpenSlideFormat):
    """Sakura VisionTek slides (SVSlide).

    An SVSlide file is an SQLite database of JPEG tiles.
    """

    VENDOR = "sakura"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svslide",)
    HINTS = ("sakura", "svslide")


@register_format
class SakuraImage(SakuraFormat, OpenSlideImage):
    """A single level of a Sakura SVSlide file.

    See [`OpenSlideImage`][].
    """


@register_format
class SakuraMultiScaleImage(SakuraFormat, OpenSlideMultiScaleImage):
    """A Sakura SVSlide file, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   TRESTLE TIFF
# ----------------------------------------------------------------------


class TrestleFormat(OpenSlideFormat):
    """Trestle slides.

    A Trestle slide is a tiled TIFF with Trestle metadata, accompanied by
    sidecar files (`.slidedat`, `.lgm` and others) in the same directory.
    """

    VENDOR = "trestle"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif",)
    HINTS = ("trestle",)


@register_format
class TrestleImage(TrestleFormat, OpenSlideImage):
    """A single level of a Trestle TIFF slide.

    See [`OpenSlideImage`][].
    """


@register_format
class TrestleMultiScaleImage(TrestleFormat, OpenSlideMultiScaleImage):
    """A Trestle TIFF slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   ZEISS CZI
# ----------------------------------------------------------------------


class ZeissFormat(OpenSlideFormat):
    """Zeiss CZI slides, the brightfield RGB mosaics that OpenSlide 4 reads."""

    VENDOR = "zeiss"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".czi",)
    HINTS = ("zeiss", "czi")


@register_format
class ZeissImage(ZeissFormat, OpenSlideImage):
    """A single level of a Zeiss CZI slide.

    See [`OpenSlideImage`][].
    """


@register_format
class ZeissMultiScaleImage(ZeissFormat, OpenSlideMultiScaleImage):
    """A Zeiss CZI slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   DICOM WSI
# ----------------------------------------------------------------------


class DicomWsiFormat(OpenSlideFormat):
    """DICOM whole-slide images.

    A DICOM WSI is a series of `.dcm` files in one directory, one per level and
    one per associated image. Any one of the files can be given to OpenSlide.
    """

    VENDOR = "dicom"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dcm",)
    HINTS = ("dicom-wsi",)


@register_format
class DicomWsiImage(DicomWsiFormat, OpenSlideImage):
    """A single level of a DICOM whole-slide image.

    See [`OpenSlideImage`][].
    """


@register_format
class DicomWsiMultiScaleImage(DicomWsiFormat, OpenSlideMultiScaleImage):
    """A DICOM whole-slide image, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# ----------------------------------------------------------------------
#   GENERIC TILED TIFF
# ----------------------------------------------------------------------


class GenericTiffFormat(OpenSlideFormat):
    """Tiled TIFF files of no known vendor.

    The levels of such a slide are the tiled directories that follow the first
    one. tifffile reads these files too, with more metadata, so this format
    scores only `MAYBE`. OpenSlide is therefore chosen only by hint
    (`hint="openslide"`), or when neither tifffile nor Pillow is installed.
    """

    VENDOR = "generic-tiff"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif", ".tiff")
    HINTS = ("generic-tiff",)

    SCORE = Confidence.MAYBE


@register_format
class GenericTiffImage(GenericTiffFormat, OpenSlideImage):
    """A single level of a generic tiled TIFF slide.

    See [`OpenSlideImage`][].
    """


@register_format
class GenericTiffMultiScaleImage(GenericTiffFormat, OpenSlideMultiScaleImage):
    """A generic tiled TIFF slide, as a pyramid.

    See [`OpenSlideMultiScaleImage`][].
    """


# Register the single-scale class of each vendor, which the multiscale class
# of the same vendor uses to build its levels.
_LEVEL_CLASSES.update(
    {
        cls.VENDOR: cls
        for cls in (
            AperioImage,
            HamamatsuImage,
            MiraxImage,
            LeicaImage,
            PhilipsImage,
            VentanaImage,
            SakuraImage,
            TrestleImage,
            ZeissImage,
            DicomWsiImage,
            GenericTiffImage,
        )
    }
)
