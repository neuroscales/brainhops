"""The metadata of an image sequence."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops._core import path
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import BinaryFileReader, ParserTypeError
from brainhops.io.metadata import MetadataFormat

# this format
from ._raw import (
    NOT_A_STREAM,
    SequenceRaw,
    refuse_options,
    refuse_stream,
    sniff_folder,
    sniff_score,
)


@register_format
class SequenceMetadata(MetadataFormat, BinaryFileReader):
    """Metadata of an image sequence, which is the list of its files.

    An image sequence is a folder of numbered two-dimensional images that
    form one volume. Its metadata holds a [`SequenceRaw`][] record, which
    lists the files and holds the header of each one. Reading the metadata
    reads the header of every file, but never their pixels, and the
    metadata keeps no open file. Image sequences cannot be written, so the
    metadata cannot be written either.

    A folder is recognized when it is not a Zarr store and its image
    files are numbered without a gap (see
    [`brainhops.io.images.sequence`][]). A file, a stream or bytes are
    never recognized, so this class never takes a file from the TIFF and
    Pillow readers. The format hint `"sequence"` selects this class.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("sequence",)

    raw: NoRepr[tx.Optional[SequenceRaw]] = None
    """The record of the sequence, or `None` for metadata without a record.
    """

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        pattern: tx.Optional[str] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a path is the folder of a sequence.

        Only the names of the files are checked, so no file is opened.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        error : bool or type[Exception], default=False
            Whether to raise an error when the path is not the folder of a
            sequence.
        pattern : str, optional
            The glob pattern that selects the files of the sequence.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            `LIKELY` for a folder that can hold a sequence, and `NO`
            otherwise.
        """
        return sniff_score(sniff_folder(filename, pattern), error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Refuse a stream, which cannot hold a folder.

        Parameters
        ----------
        file : IO
            The stream.
        error : bool or type[Exception], default=False
            Whether to raise an error, since a stream is never a sequence.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_stream(error)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Refuse bytes, which cannot hold a folder.

        Parameters
        ----------
        content : bytes
            The bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error, since bytes are never a sequence.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        return refuse_stream(error)

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        pattern: tx.Optional[str] = None,
        **kwargs,
    ) -> "SequenceMetadata":
        """Read the metadata of the image sequence in a folder.

        The record is read with [`SequenceRaw.from_folder`][], which reads
        the header of every file but not their pixels.

        Parameters
        ----------
        filename : path-like
            The path of the folder.
        pattern : str, optional
            The glob pattern, relative to the folder, that selects the
            files of the sequence. By default, every image file of the
            folder is taken.
        **kwargs : Any
            Other options, which are refused.

        Returns
        -------
        SequenceMetadata
            The metadata, which holds the record of the sequence.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserExistsError
            If the folder does not exist.
        ParserContentError
            If the folder does not hold a sequence that can be read.
        """
        refuse_options(cls, kwargs)
        return cls.from_raw(SequenceRaw.from_folder(filename, pattern))

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> "SequenceMetadata":
        """Refuse a stream, which cannot hold the files of a sequence.

        Parameters
        ----------
        file : IO
            The stream.
        **kwargs : Any
            Options of the reader, which are ignored.

        Raises
        ------
        ParserTypeError
            Always, since a sequence is read from the path of its folder.
        """
        raise ParserTypeError(NOT_A_STREAM)
