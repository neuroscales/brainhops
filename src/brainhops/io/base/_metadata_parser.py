"""
Reading the metadata of a file without its data.

A file stores its metadata in a raw record (a NIfTI header, the
attributes of a Zarr array, the JSON of an x5 node, ...), which the
metadata class of the format decodes with `from_raw` and encodes with
`to_raw` and `update_raw`. [`MetadataParser`][] adds the file side: its
`from_*` methods read the raw record of a file, and nothing else, then
defer to `from_raw`; its `to_*` methods encode the fields with `to_raw`,
then write the record. The metadata class of a format inherits from it,
as the image class of a format inherits from its parser.

`MetadataParser` is also a dispatcher, with a registry of its own:
`MetadataParser.load(path)`, which is what
[`Metadata.load`][brainhops.datamodel.metadata.Metadata.load] calls,
sniffs the file and hands it to the metadata class of its format. The
registry is separate from that of
[`FileBasedObject`][brainhops.io.base.FileBasedObject], so that the
generic `brainhops.io.load` never returns metadata where an image or a
transformation was asked for.
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

from ._base import FormatDispatcher, format_registry
from .parsers import (
    Confidence,
    ParserExistsError,
    ParserNotImplementedError,
    SnifferContentError,
    WriterNotImplementedError,
)


@format_registry
class MetadataParser(FormatDispatcher):
    """
    Reads and writes the raw record of a metadata class, and dispatches
    among the formats that do.

    A format implements `_read_raw`, which reads the raw record from a
    path or from an open file, and the sniffers that recognise its files.
    Its `from_file`, `from_fileobj` and `from_bytes` then read the record
    and build the metadata with `from_raw`. A format whose record is an
    object of its own on disk (the attributes of a Zarr array) also
    implements `_write_raw`, and its `to_file` writes the encoded record
    (`to_raw`) back. Every other format refuses to write: its record is
    written by the writer of its images or transformations, along with
    the data.

    The class is a dispatcher: `MetadataParser.load(path)` picks the
    registered format that best matches the file, by name and by
    content, as `brainhops.io.load` does, and `hint=` restricts the
    candidates. A format registers with
    [`register_format`][brainhops.io.base.register_format].
    """

    # --- reading ------------------------------------------------------

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of a file, from a path or an open file.

        Parameters
        ----------
        file : str, path-like or file object
            The file.
        **kwargs
            Options of the format's reader (on a dispatcher, also `hint=`).

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.

        Raises
        ------
        ParserExistsError
            If the path does not exist.
        ParserContentError
            On a dispatcher, if no registered format reads the file.
        """
        if cls._is_dispatcher():
            return super().from_file(file, **kwargs)
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, path.PathLike) and not path.exists(file):
            raise ParserExistsError(f"No such file: {file}")
        return cls._from_record(cls._read_raw(file, **kwargs))

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs: tx.Any) -> tx.Any:
        """
        Read the metadata of an open file.

        Parameters
        ----------
        file : file object
            The file, open for reading in binary mode. Its position is
            restored afterwards.
        **kwargs
            Options of the format's reader.

        Returns
        -------
        Metadata
            The metadata of the file, with its raw record.
        """
        if cls._is_dispatcher():
            return super().from_fileobj(file, **kwargs)
        with preserve_position(file):
            return cls._from_record(cls._read_raw(file, **kwargs))

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
        if cls._is_dispatcher():
            return super().from_bytes(content, **kwargs)
        return cls.from_fileobj(BytesIO(content), **kwargs)

    @classmethod
    def _read_raw(cls, file: path.FileLike, **kwargs: tx.Any) -> tx.Any:
        """
        Read the raw record of a file, and nothing else.

        A format overrides this hook.

        Parameters
        ----------
        file : path-like or file object
            The file: a path, or a file open for reading in binary mode.
        **kwargs
            Options of the format's reader.

        Returns
        -------
        object
            The raw record.

        Raises
        ------
        ParserNotImplementedError
            By default: the format cannot read its metadata on its own.
        """
        raise ParserNotImplementedError(
            f"{cls.__name__} cannot read the metadata of a file on its own."
        )

    @classmethod
    def _from_record(cls, raw: tx.Any) -> tx.Any:
        """The metadata of a raw record that was just read: `from_raw`,
        for the metadata class of a format."""
        return cls.from_raw(raw)

    # --- writing ------------------------------------------------------

    def to_file(self, file: path.FileLike, **kwargs: tx.Any) -> None:
        """
        Write the metadata into the raw record of a file.

        The fields are encoded with `to_raw`, over a copy of the record
        this metadata was read from, and the record is written into the
        file, which must exist already.

        Parameters
        ----------
        file : str, path-like or file object
            The file.
        **kwargs
            Options of `to_raw` (`image=`, `on_loss=`).

        Raises
        ------
        WriterNotImplementedError
            If the format writes its record along with the data only.
        """
        self._write_raw(self.to_raw(**kwargs), file)

    def _write_raw(self, raw: tx.Any, file: path.FileLike) -> None:
        """
        Write a raw record into a file.

        A format whose record is an object of its own on disk overrides
        this hook.

        Parameters
        ----------
        raw : object
            The encoded record.
        file : path-like or file object
            The file.

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

    A format implements `_sniff_h5(h5file)`, a score, and
    `_read_raw_h5(h5file, **kwargs)`, the raw record, both on an open
    `h5py.File`. `h5py` is optional: it is imported when a file is
    sniffed or read, and without it no file is recognised.
    """

    @classmethod
    def _sniff_h5(cls, h5file: tx.Any) -> float:
        """Score an open HDF5 file. A format overrides this hook."""
        return Confidence.NO

    @classmethod
    def _read_raw_h5(cls, h5file: tx.Any, **kwargs: tx.Any) -> tx.Any:
        """Read the raw record of an open HDF5 file. A format overrides
        this hook."""
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
        file : str, path-like or file object
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
        if cls._is_dispatcher():
            return super().sniff_file(file, error=error, **kwargs)
        score = Confidence.NO
        with contextlib.suppress(Exception):
            with _open_h5(file) as h5file:
                score = cls._sniff_h5(h5file)
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
        if cls._is_dispatcher():
            return super().sniff_bytes(content, error=error, **kwargs)
        return cls.sniff_file(BytesIO(content), error=error)

    @classmethod
    def _read_raw(cls, file: path.FileLike, **kwargs: tx.Any) -> tx.Any:
        with _open_h5(file) as h5file:
            return cls._read_raw_h5(h5file, **kwargs)


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
