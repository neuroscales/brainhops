"""Whole-slide image formats, one pair of classes per OpenSlide vendor.

Each vendor that OpenSlide reads has a single-scale class, which reads one
level of a slide, and a multiscale class, which reads the whole pyramid.
The vendors share everything else: the two bases of `_base`, the record
[`OpenSlideRaw`][brainhops.io.images.openslide.OpenSlideRaw] and the
metadata
[`OpenSlideMetadata`][brainhops.io.images.openslide.OpenSlideMetadata].
Each class therefore declares only its extensions and its hints, and sniffs
the slides of its own vendor, as `OpenSlide.detect_format` names it. A
multiscale class also decodes its levels as objects of the single-scale
class of its vendor.
"""

__all__ = [
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

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core._internal_api import smartproperty
from brainhops.datamodel.images import SingleScaleImage
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence

# this format
from ._base import (
    OpenSlideImage,
    OpenSlideMultiScaleImage,
    decode_levels,
    sniff_level,
    sniff_pyramid,
)

# ----------------------------------------------------------------------
#   APERIO SVS
# ----------------------------------------------------------------------


@register_format
class AperioImage(OpenSlideImage):
    """A single level of an Aperio ScanScope Virtual Slide (SVS).

    An SVS file, now from Leica, is a tiled TIFF whose first
    `ImageDescription` starts with `Aperio`. Besides the pyramid, it
    holds a thumbnail, a label and a macro image.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"aperio"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`AperioMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svs",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("aperio", "svs")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is an Aperio slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`AperioMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Aperio
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "aperio", Confidence.CERTAIN, error, level
        )


@register_format
class AperioMultiScaleImage(OpenSlideMultiScaleImage):
    """An Aperio SVS slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"aperio"`, as pyramids whose levels are [`AperioImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svs",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("aperio", "svs")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`AperioImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(AperioImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is an Aperio pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Aperio
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "aperio", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   HAMAMATSU
# ----------------------------------------------------------------------


@register_format
class HamamatsuImage(OpenSlideImage):
    """A single level of a Hamamatsu NanoZoomer slide.

    NDPI files are TIFF-like files with 64-bit offsets. VMS and VMU
    slides are an index file together with JPEG or raw files in the same
    directory.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"hamamatsu"`, and [`OpenSlideImage`][] describes the
    rest. The whole pyramid is read by [`HamamatsuMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ndpi", ".vms", ".vmu")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "hamamatsu",
        "ndpi",
        "vms",
        "vmu",
    )

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Hamamatsu slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`HamamatsuMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Hamamatsu
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "hamamatsu", Confidence.CERTAIN, error, level
        )


@register_format
class HamamatsuMultiScaleImage(OpenSlideMultiScaleImage):
    """A Hamamatsu NanoZoomer slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"hamamatsu"`, as pyramids whose levels are [`HamamatsuImage`][]
    objects, and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ndpi", ".vms", ".vmu")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = (
        "hamamatsu",
        "ndpi",
        "vms",
        "vmu",
    )

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`HamamatsuImage`][] objects decoded from
        `rawdata` and the record on first access and cached, apart from
        an assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(HamamatsuImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Hamamatsu pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Hamamatsu
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "hamamatsu", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   MIRAX
# ----------------------------------------------------------------------


@register_format
class MiraxImage(OpenSlideImage):
    """A single level of a 3DHISTECH MIRAX slide.

    A slide is a `.mrxs` file together with a directory of the same
    name, which holds `Slidedat.ini` and the data files. OpenSlide
    recognises the slide by the extension `.mrxs`.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"mirax"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`MiraxMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mrxs",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mirax", "mrxs", "3dhistech")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a MIRAX slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`MiraxMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a MIRAX
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(filename, "mirax", Confidence.CERTAIN, error, level)


@register_format
class MiraxMultiScaleImage(OpenSlideMultiScaleImage):
    """A 3DHISTECH MIRAX slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"mirax"`, as pyramids whose levels are [`MiraxImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mrxs",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mirax", "mrxs", "3dhistech")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`MiraxImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(MiraxImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a MIRAX pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a MIRAX
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "mirax", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   LEICA SCN
# ----------------------------------------------------------------------


@register_format
class LeicaImage(OpenSlideImage):
    """A single level of a Leica SCN slide.

    An SCN file is a BigTIFF whose `ImageDescription` is Leica XML that
    describes the collection and its images.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"leica"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`LeicaMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".scn",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("leica", "scn")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Leica slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`LeicaMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Leica
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(filename, "leica", Confidence.CERTAIN, error, level)


@register_format
class LeicaMultiScaleImage(OpenSlideMultiScaleImage):
    """A Leica SCN slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"leica"`, as pyramids whose levels are [`LeicaImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".scn",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("leica", "scn")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`LeicaImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(LeicaImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Leica pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Leica
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "leica", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   PHILIPS TIFF
# ----------------------------------------------------------------------


@register_format
class PhilipsImage(OpenSlideImage):
    """A single level of a Philips TIFF slide, exported from iSyntax.

    A Philips TIFF is a tiled TIFF whose `ImageDescription` is Philips
    DataObject XML.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"philips"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`PhilipsMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tiff",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("philips",)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Philips slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`PhilipsMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Philips
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "philips", Confidence.CERTAIN, error, level
        )


@register_format
class PhilipsMultiScaleImage(OpenSlideMultiScaleImage):
    """A Philips TIFF slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"philips"`, as pyramids whose levels are [`PhilipsImage`][]
    objects, and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tiff",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("philips",)

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`PhilipsImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(PhilipsImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Philips pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Philips
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "philips", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   VENTANA BIF
# ----------------------------------------------------------------------


@register_format
class VentanaImage(OpenSlideImage):
    """A single level of a Roche (Ventana) BIF slide.

    A BIF file is a tiled BigTIFF with Ventana XMP metadata, whose tiles
    may overlap.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"ventana"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`VentanaMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".bif", ".tif")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("ventana", "bif")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Ventana slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`VentanaMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Ventana
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "ventana", Confidence.CERTAIN, error, level
        )


@register_format
class VentanaMultiScaleImage(OpenSlideMultiScaleImage):
    """A Roche (Ventana) BIF slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"ventana"`, as pyramids whose levels are [`VentanaImage`][]
    objects, and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".bif", ".tif")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("ventana", "bif")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`VentanaImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(VentanaImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Ventana pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Ventana
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "ventana", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   SAKURA SVSLIDE
# ----------------------------------------------------------------------


@register_format
class SakuraImage(OpenSlideImage):
    """A single level of a Sakura VisionTek slide (SVSlide).

    An SVSlide file is an SQLite database of JPEG tiles.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"sakura"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`SakuraMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svslide",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("sakura", "svslide")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Sakura slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`SakuraMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Sakura
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "sakura", Confidence.CERTAIN, error, level
        )


@register_format
class SakuraMultiScaleImage(OpenSlideMultiScaleImage):
    """A Sakura VisionTek slide (SVSlide), as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"sakura"`, as pyramids whose levels are [`SakuraImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".svslide",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("sakura", "svslide")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`SakuraImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(SakuraImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Sakura pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Sakura
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "sakura", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   TRESTLE TIFF
# ----------------------------------------------------------------------


@register_format
class TrestleImage(OpenSlideImage):
    """A single level of a Trestle slide.

    A Trestle slide is a tiled TIFF with Trestle metadata, accompanied
    by sidecar files (`.slidedat`, `.lgm` and others) in the same
    directory.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"trestle"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`TrestleMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("trestle",)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Trestle slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`TrestleMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Trestle
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "trestle", Confidence.CERTAIN, error, level
        )


@register_format
class TrestleMultiScaleImage(OpenSlideMultiScaleImage):
    """A Trestle slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"trestle"`, as pyramids whose levels are [`TrestleImage`][]
    objects, and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("trestle",)

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`TrestleImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(TrestleImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Trestle pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Trestle
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "trestle", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   ZEISS CZI
# ----------------------------------------------------------------------


@register_format
class ZeissImage(OpenSlideImage):
    """A single level of a Zeiss CZI slide.

    OpenSlide 4 reads the brightfield RGB mosaics of CZI files.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"zeiss"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`ZeissMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".czi",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("zeiss", "czi")

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Zeiss slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`ZeissMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Zeiss
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(filename, "zeiss", Confidence.CERTAIN, error, level)


@register_format
class ZeissMultiScaleImage(OpenSlideMultiScaleImage):
    """A Zeiss CZI slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"zeiss"`, as pyramids whose levels are [`ZeissImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".czi",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("zeiss", "czi")

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`ZeissImage`][] objects decoded from `rawdata`
        and the record on first access and cached, apart from an
        assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(ZeissImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a Zeiss pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a Zeiss
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "zeiss", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   DICOM WSI
# ----------------------------------------------------------------------


@register_format
class DicomWsiImage(OpenSlideImage):
    """A single level of a DICOM whole-slide image.

    A DICOM whole-slide image is a series of `.dcm` files in one
    directory, one per level and one per associated image. Any one of
    the files can be given to OpenSlide.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"dicom"`, and [`OpenSlideImage`][] describes the rest.
    The whole pyramid is read by [`DicomWsiMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dcm",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("dicom-wsi",)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a DICOM slide.

        The score is `CERTAIN` when a level is requested or when the
        slide has a single level, and 0.8 times that otherwise, so that
        [`DicomWsiMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a DICOM
            slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(filename, "dicom", Confidence.CERTAIN, error, level)


@register_format
class DicomWsiMultiScaleImage(OpenSlideMultiScaleImage):
    """A DICOM whole-slide image, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"dicom"`, as pyramids whose levels are [`DicomWsiImage`][] objects,
    and [`OpenSlideMultiScaleImage`][] describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".dcm",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("dicom-wsi",)

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`DicomWsiImage`][] objects decoded from
        `rawdata` and the record on first access and cached, apart from
        an assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(DicomWsiImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a DICOM pyramid.

        The score is `CERTAIN` for a slide of several levels when no
        level is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a DICOM
            slide. A slide of a single level, or a request for a level,
            scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "dicom", Confidence.CERTAIN, error, level
        )


# ----------------------------------------------------------------------
#   GENERIC TILED TIFF
# ----------------------------------------------------------------------


@register_format
class GenericTiffImage(OpenSlideImage):
    """A single level of a tiled TIFF file of no known vendor.

    The levels of such a slide are the tiled directories that follow the
    first one and are marked as reduced-resolution images. tifffile
    reads these files too, with more metadata, so this class scores only
    `MAYBE`. OpenSlide is therefore chosen only by hint
    (`hint="openslide"`), or when neither tifffile nor Pillow is
    installed.

    The class reads one level of the slides that OpenSlide attributes to
    the vendor `"generic-tiff"`, and [`OpenSlideImage`][] describes the
    rest. The whole pyramid is read by [`GenericTiffMultiScaleImage`][].
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif", ".tiff")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("generic-tiff",)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a generic TIFF slide.

        The score is `MAYBE` when a level is requested or when the slide
        has a single level, and 0.8 times that otherwise, so that
        [`GenericTiffMultiScaleImage`][] reads a pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a generic
            TIFF slide.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(
            filename, "generic-tiff", Confidence.MAYBE, error, level
        )


@register_format
class GenericTiffMultiScaleImage(OpenSlideMultiScaleImage):
    """A generic tiled TIFF slide, as a pyramid.

    The class reads the slides that OpenSlide attributes to the vendor
    `"generic-tiff"`, as pyramids whose levels are
    [`GenericTiffImage`][] objects, and [`OpenSlideMultiScaleImage`][]
    describes the rest.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tif", ".tiff")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("generic-tiff",)

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are [`GenericTiffImage`][] objects decoded from
        `rawdata` and the record on first access and cached, apart from
        an assigned list. Each level holds its array of `rawdata`, the
        metadata of the pyramid and its index in `level`. Assigning a
        list replaces the stored one, and assigning `None` decodes the
        levels again.
        """
        return decode_levels(GenericTiffImage, self.rawdata, self.metadata)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a file is a generic TIFF pyramid.

        The score is `MAYBE` for a slide of several levels when no level
        is requested, and `NO` otherwise.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a generic
            TIFF slide. A slide of a single level, or a request for a
            level, scores `NO` without an error.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(
            filename, "generic-tiff", Confidence.MAYBE, error, level
        )
