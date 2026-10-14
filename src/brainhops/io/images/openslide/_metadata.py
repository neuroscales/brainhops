"""The metadata of a whole-slide image that OpenSlide reads."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
)
from brainhops.io.metadata import MetadataFormat

# this format
from ._data import _Slide, detect, local_path, open_handle
from ._raw import OpenSlideRaw, read_record


@register_format
class OpenSlideMetadata(MetadataFormat, BinaryFileReader):
    """Metadata of a whole-slide image, held as an [`OpenSlideRaw`][] record.

    The metadata of a slide is what OpenSlide reports about it when it opens
    it: its properties, the dimensions and the downsampling factor of each
    level, and the names of its associated images. One class reads the
    slides of every vendor, because OpenSlide describes them all in the
    same way. The images and pyramids of
    [`brainhops.io.images.openslide`][] hold it, and every level of a
    pyramid holds the same metadata.

    Reading the metadata opens the slide with OpenSlide, reads its
    description and closes it, so the metadata keeps no open file. It never
    calls `read_region`. OpenSlide still reads the raw tiles of the coarsest
    level of a TIFF-based slide when it opens it, to compute the property
    `openslide.quickhash-1`, unless they exceed 5 MiB. OpenSlide opens slides
    by name only, so a stream or bytes without a local file are first copied
    to a temporary file, which is deleted once the metadata is read. The
    metadata cannot be written, because OpenSlide cannot write slides.

    A slide of a known vendor scores `CERTAIN`. A generic tiled TIFF scores
    only `MAYBE`, because the TIFF reader reads it with more metadata, as
    it does for the images of the package.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".svs",
        ".ndpi",
        ".vms",
        ".vmu",
        ".mrxs",
        ".scn",
        ".tiff",
        ".bif",
        ".tif",
        ".svslide",
        ".czi",
        ".dcm",
    )
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("openslide",)

    raw: NoRepr[tx.Optional[OpenSlideRaw]] = None
    """The record of the slide, or `None` for metadata without a record."""

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a local file is a whole-slide image.

        OpenSlide detects the vendor of the file from its content, and from
        its extension for a MIRAX slide.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a slide that
            OpenSlide reads.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `CERTAIN` for a slide of a known vendor, `MAYBE` for a generic
            tiled TIFF, and `NO` otherwise.
        """
        local = local_path(filename)
        if local is None:
            if error:
                if error is True:
                    error = SnifferExistsError
                raise error(f"No such local file: {filename}")
            return Confidence.NO
        vendor, _ = detect(local)
        if not vendor:
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("OpenSlide does not read this file.")
            return Confidence.NO
        if vendor == "generic-tiff":
            return Confidence.MAYBE
        return Confidence.CERTAIN

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that an open file is a whole-slide image.

        The file is sniffed through the name of the local file that it was
        opened from. A stream without such a name scores `NO`, so it is read
        only by hint.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a slide.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        local = local_path(file)
        if local is None:
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("OpenSlide only sniffs local files.")
            return Confidence.NO
        return cls.sniff_filename(local, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `NO`, because content in memory is read only by hint.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error, which is always raised when set.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        if error:
            if error is True:
                error = SnifferContentError
            raise error("OpenSlide only sniffs local files.")
        return Confidence.NO

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "OpenSlideMetadata":
        """Read the metadata of a slide from a file.

        A local file is opened in place, and a remote file is first copied
        to a temporary file.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        OpenSlideMetadata
            The metadata, which holds the record of the slide.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If the file does not exist.
        ParserContentError
            If OpenSlide cannot read the file.
        """
        _no_options(cls, kwargs)
        local = local_path(filename)
        if local is not None:
            return cls.from_raw(_read_local(local))
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with filename.open("rb") as f:
            content = f.read()
        return cls.from_raw(_read_content(content, _to_filename(filename)))

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "OpenSlideMetadata":
        """Read the metadata of a slide from an open file.

        A local file is opened by name, and any other stream is copied to a
        temporary file. The position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        OpenSlideMetadata
            The metadata, which holds the record of the slide.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If OpenSlide cannot read the file.
        """
        _no_options(cls, kwargs)
        local = local_path(file)
        if local is not None:
            return cls.from_raw(_read_local(local))
        with preserve_position(file):
            content = file.read()
        return cls.from_raw(_read_content(content, _to_filename(file)))

    @classmethod
    def from_bytes(
        cls, content: path.BinaryContentLike, **kwargs
    ) -> "OpenSlideMetadata":
        """Read the metadata of a slide from bytes, through a temporary copy.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            Refused, since the metadata has no option.

        Returns
        -------
        OpenSlideMetadata
            The metadata, which holds the record of the slide.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If OpenSlide cannot read the content.
        """
        _no_options(cls, kwargs)
        return cls.from_raw(_read_content(bytes(content), None))


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader of metadata was given.

    Raises
    ------
    TypeError
        If any option is given.
    """
    if kwargs:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(kwargs))}."
        )


def _read_local(filename: str) -> OpenSlideRaw:
    """Read the record of a local slide, and close the slide."""
    handle = open_handle(filename)
    try:
        return read_record(handle)
    finally:
        handle.close()


def _read_content(content: bytes, name: tx.Optional[str]) -> OpenSlideRaw:
    """Read the record of a slide through a temporary copy of its content.

    The copy keeps the extension of `name`. It is deleted, and the slide
    closed, when the slide that reads it is garbage-collected, which is
    when this function returns.
    """
    slide = _Slide.from_content(content, name)
    return read_record(slide.handle)
