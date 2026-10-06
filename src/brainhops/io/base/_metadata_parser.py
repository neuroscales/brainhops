"""
Reading the metadata of a file without its data.

A file stores its metadata in a raw record (a NIfTI header, the
attributes of a Zarr array, the JSON of an x5 node, ...), which the
metadata class of the format decodes with `from_raw` and encodes with
`to_raw` and `update_raw`. [`MetadataParser`][] adds the file side: its
`from_*` methods read the raw record of a file, and nothing else, then
build the metadata with `from_raw`. The metadata class of a format lists
it first among its bases, as the image class of a format lists its
parser.

The parsers own no registry: the dispatcher among the formats is
[`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata],
whose `load` is what
[`Metadata.load`][brainhops.datamodel.metadata.Metadata.load] calls.
"""

__all__ = ["Hdf5MetadataParser", "MetadataParser"]

# stdlib
import contextlib
from io import BytesIO

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import preserve_position

from .parsers import (
    Confidence,
    FileParser,
    ParserExistsError,
    ParserNotImplementedError,
    SnifferContentError,
    WriterNotImplementedError,
)


class MetadataParser(FileParser):
    """
    Reads the metadata of a file of one format.

    A format implements `from_fileobj`, which reads the raw record of an
    open file and builds the metadata with `from_raw`, and the sniffers
    that recognise its files; a path is opened by `from_filename`, in
    binary mode, and handed to `from_fileobj`. A format that reads paths
    otherwise (MGH reads its tags lazily from a path) overrides
    `from_filename` too. A format whose record is an object of its own on
    disk (the attributes of a Zarr array) also overrides `to_file`, which
    writes the encoded record (`to_raw`) back. Every other format refuses
    to write: its record is written by the writer of its images or
    transformations, along with the data.

    The parser of a format owns no registry: the class of the format
    also derives from
    [`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata],
    and registers into its registry with
    [`register_format`][brainhops.io.base.register_format].
    """

    _READ_MODE = "rb"

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of a file held in memory.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs
            Options of the format's reader.

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.
        """
        return cls.from_fileobj(BytesIO(content), **kwargs)

    def to_file(self, file: path.FileLike, **kwargs: tx.Any) -> None:
        """
        Write the metadata into the raw record of a file.

        A format whose record is an object of its own on disk overrides
        this method: it encodes the fields with `to_raw`, over a copy of
        the record this metadata was read from, and writes the record
        into the file, which must exist already.

        Parameters
        ----------
        file : str, path-like or file object
            The file.
        **kwargs
            Options of `to_raw` (`image=`, `on_loss=`).

        Raises
        ------
        WriterNotImplementedError
            By default: the record is written along with the data, by the
            writer of the format's images or transformations.
        """
        raise WriterNotImplementedError(
            f"{type(self).__name__} is written along with the data: save "
            f"the image or the transformation that holds it."
        )


class Hdf5MetadataParser(MetadataParser):
    """
    The metadata parser of a format stored in an HDF5 file.

    A format implements `sniff_h5`, a score, and `from_h5`, the
    metadata, both on an open `h5py.File`, as the formats of
    [`Hdf5Parser`][brainhops.io.base.hdf5.Hdf5Parser] do. `h5py` is
    optional: it is imported when a file is sniffed or read, and without
    it no file is recognised.
    """

    @classmethod
    def sniff_h5(cls, h5file: tx.Any) -> float:
        """
        Score how confident the format is that an open HDF5 file is one
        of its files.

        Parameters
        ----------
        h5file : h5py.File
            The open file.

        Returns
        -------
        float
            The confidence, in `[0, 1]`. By default, 0.
        """
        return Confidence.NO

    @classmethod
    def from_h5(cls, h5file: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        **kwargs
            Options of the format's reader.

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.

        Raises
        ------
        ParserNotImplementedError
            By default: a format implements this method.
        """
        raise ParserNotImplementedError(
            f"{cls.__name__} cannot read the metadata of a file on its own."
        )

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the format is that a file is one of its files.

        Parameters
        ----------
        file : str, path-like, file object or h5py.File
            The file.
        error : bool or type, optional
            Raise an error (this one, or `SnifferContentError` for `True`)
            instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        score = Confidence.NO
        with contextlib.suppress(Exception):
            with _open_h5(file) as h5file:
                score = cls.sniff_h5(h5file)
        if not score and error:
            raise (SnifferContentError if error is True else error)(
                f"Not a {cls.__name__} HDF5 file: {file}"
            )
        return score

    sniff_filename = sniff_file
    sniff_fileobj = sniff_file

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs: tx.Any,
    ) -> float:
        """
        Score how confident the format is that bytes hold one of its
        files.

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type, optional
            Raise an error instead of returning 0.
        **kwargs
            Ignored.

        Returns
        -------
        float
            The confidence, in `[0, 1]`.
        """
        return cls.sniff_file(BytesIO(content), error=error)

    @classmethod
    def from_file(cls, file: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of a file: a path, an open binary stream or an
        open `h5py.File`.

        Parameters
        ----------
        file : str, path-like, file object or h5py.File
            The file.
        **kwargs
            Options of the format's reader (`from_h5`).

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike) and not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        with _open_h5(file) as h5file:
            return cls.from_h5(h5file, **kwargs)

    from_filename = from_file

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of an open binary stream.

        Parameters
        ----------
        file : file object
            The stream. Its position is restored.
        **kwargs
            Options of the format's reader (`from_h5`).

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.
        """
        with _open_h5(file) as h5file:
            return cls.from_h5(h5file, **kwargs)


# ----------------------------------------------------------------------
#   PRIVATE
# ----------------------------------------------------------------------


@contextlib.contextmanager
def _open_h5(file: tx.Any) -> tx.Iterator[tx.Any]:
    """Open a path or a binary stream as an `h5py.File`, for reading."""
    import h5py

    if isinstance(file, h5py.File):
        yield file
        return
    if isinstance(file, (str, path.PathLike)):
        with h5py.File(str(file), "r") as h5file:
            yield h5file
        return
    with preserve_position(file), h5py.File(file, "r") as h5file:
        yield h5file
