"""
One format class per OpenSlide vendor, each with a single-scale (one
level) and a multiscale (pyramid) image.

They differ only in the vendor they accept (`VENDOR`, as
`OpenSlide.detect_format` names it), their extensions and their hints.
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

# dependencies
import typing_extensions as tx

# internals
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
    """
    Leica (Aperio) ScanScope Virtual Slide: a tiled TIFF whose first
    `ImageDescription` starts with `Aperio`, with a thumbnail, a label and
    a macro image.
    """

    VENDOR = "aperio"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svs",)
    HINTS = ("aperio", "svs")


@register_format
class AperioImage(AperioFormat, OpenSlideImage):
    """One level of a Aperio SVS slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class AperioMultiScaleImage(AperioFormat, OpenSlideMultiScaleImage):
    """A Aperio SVS slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   HAMAMATSU
# ----------------------------------------------------------------------


class HamamatsuFormat(OpenSlideFormat):
    """
    Hamamatsu NanoZoomer slides: NDPI (a TIFF-like file with 64-bit
    offsets), and VMS/VMU (an index file and the JPEG or raw files it
    names, all in one directory).
    """

    VENDOR = "hamamatsu"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ndpi", ".vms", ".vmu")
    HINTS = ("hamamatsu", "ndpi", "vms", "vmu")


@register_format
class HamamatsuImage(HamamatsuFormat, OpenSlideImage):
    """One level of a Hamamatsu slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class HamamatsuMultiScaleImage(HamamatsuFormat, OpenSlideMultiScaleImage):
    """A Hamamatsu slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   MIRAX
# ----------------------------------------------------------------------


class MiraxFormat(OpenSlideFormat):
    """
    3DHISTECH MIRAX slides: an `.mrxs` file and a directory of the same
    name holding `Slidedat.ini` and the data files.
    """

    VENDOR = "mirax"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mrxs",)
    HINTS = ("mirax", "mrxs", "3dhistech")


@register_format
class MiraxImage(MiraxFormat, OpenSlideImage):
    """One level of a MIRAX slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class MiraxMultiScaleImage(MiraxFormat, OpenSlideMultiScaleImage):
    """A MIRAX slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   LEICA SCN
# ----------------------------------------------------------------------


class LeicaFormat(OpenSlideFormat):
    """
    Leica SCN slides: a BigTIFF whose `ImageDescription` is the Leica
    XML describing the collection and its images.
    """

    VENDOR = "leica"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".scn",)
    HINTS = ("leica", "scn")


@register_format
class LeicaImage(LeicaFormat, OpenSlideImage):
    """One level of a Leica SCN slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class LeicaMultiScaleImage(LeicaFormat, OpenSlideMultiScaleImage):
    """A Leica SCN slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   PHILIPS TIFF
# ----------------------------------------------------------------------


class PhilipsFormat(OpenSlideFormat):
    """
    Philips TIFF slides (exported from iSyntax): a tiled TIFF whose
    `ImageDescription` is the Philips `DataObject` XML.
    """

    VENDOR = "philips"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tiff",)
    HINTS = ("philips",)


@register_format
class PhilipsImage(PhilipsFormat, OpenSlideImage):
    """One level of a Philips TIFF slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class PhilipsMultiScaleImage(PhilipsFormat, OpenSlideMultiScaleImage):
    """A Philips TIFF slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   VENTANA BIF
# ----------------------------------------------------------------------


class VentanaFormat(OpenSlideFormat):
    """
    Roche (Ventana) BIF slides: a tiled BigTIFF with Ventana XMP
    metadata, whose tiles may overlap.
    """

    VENDOR = "ventana"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".bif", ".tif")
    HINTS = ("ventana", "bif")


@register_format
class VentanaImage(VentanaFormat, OpenSlideImage):
    """One level of a Ventana BIF slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class VentanaMultiScaleImage(VentanaFormat, OpenSlideMultiScaleImage):
    """A Ventana BIF slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   SAKURA SVSLIDE
# ----------------------------------------------------------------------


class SakuraFormat(OpenSlideFormat):
    """
    Sakura VisionTek slides: an SQLite database of JPEG tiles.
    """

    VENDOR = "sakura"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svslide",)
    HINTS = ("sakura", "svslide")


@register_format
class SakuraImage(SakuraFormat, OpenSlideImage):
    """One level of a Sakura SVSlide slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class SakuraMultiScaleImage(SakuraFormat, OpenSlideMultiScaleImage):
    """A Sakura SVSlide slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   TRESTLE TIFF
# ----------------------------------------------------------------------


class TrestleFormat(OpenSlideFormat):
    """
    Trestle slides: a tiled TIFF with Trestle metadata, and sidecar files
    (`.slidedat`, `.lgm`, ...) beside it.
    """

    VENDOR = "trestle"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif",)
    HINTS = ("trestle",)


@register_format
class TrestleImage(TrestleFormat, OpenSlideImage):
    """One level of a Trestle TIFF slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class TrestleMultiScaleImage(TrestleFormat, OpenSlideMultiScaleImage):
    """A Trestle TIFF slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   ZEISS CZI
# ----------------------------------------------------------------------


class ZeissFormat(OpenSlideFormat):
    """
    Zeiss CZI slides (brightfield RGB mosaics), as OpenSlide 4 reads
    them.
    """

    VENDOR = "zeiss"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".czi",)
    HINTS = ("zeiss", "czi")


@register_format
class ZeissImage(ZeissFormat, OpenSlideImage):
    """One level of a Zeiss CZI slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class ZeissMultiScaleImage(ZeissFormat, OpenSlideMultiScaleImage):
    """A Zeiss CZI slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   DICOM WSI
# ----------------------------------------------------------------------


class DicomWsiFormat(OpenSlideFormat):
    """
    DICOM whole-slide images: a series of `.dcm` files (one per level,
    plus associated images) in one directory; OpenSlide is given any one
    of them.
    """

    VENDOR = "dicom"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dcm",)
    HINTS = ("dicom-wsi",)


@register_format
class DicomWsiImage(DicomWsiFormat, OpenSlideImage):
    """One level of a DICOM WSI slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class DicomWsiMultiScaleImage(DicomWsiFormat, OpenSlideMultiScaleImage):
    """A DICOM WSI slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# ----------------------------------------------------------------------
#   GENERIC TILED TIFF
# ----------------------------------------------------------------------


class GenericTiffFormat(OpenSlideFormat):
    """
    A tiled TIFF of no known vendor, whose levels are the following
    tiled directories. tifffile reads these too, with more metadata, so
    this format only scores `MAYBE` and is chosen by hint
    (`hint="openslide"`), or when tifffile and Pillow are absent.
    """

    VENDOR = "generic-tiff"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif", ".tiff")
    HINTS = ("generic-tiff",)

    SCORE = Confidence.MAYBE


@register_format
class GenericTiffImage(GenericTiffFormat, OpenSlideImage):
    """One level of a generic tiled TIFF slide (see
    [`OpenSlideImage`][brainhops.io.images.openslide.OpenSlideImage])."""


@register_format
class GenericTiffMultiScaleImage(GenericTiffFormat, OpenSlideMultiScaleImage):
    """A generic tiled TIFF slide, as a pyramid (see
    [`OpenSlideMultiScaleImage`][brainhops.io.images.openslide.OpenSlideMultiScaleImage])."""


# The single-scale class each multiscale class builds its levels with.
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
