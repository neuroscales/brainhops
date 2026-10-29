"""The readers and writers of metadata that delegate to the record."""

__all__ = ["MetadataReader", "MetadataWriter"]

import typing_extensions as tx

from brainhops._core import path
from brainhops.datamodel.metadata._report import takes_error_policy
from brainhops.io.base.parsers import (
    FileReader,
    FileWriter,
    WriterError,
    _delegated_from_fileobj,
)

# ----------------------------------------------------------------------
#   MODES
# ----------------------------------------------------------------------


class _RecordMode:
    """Mode of a class that reads or writes files through its record.

    The adapters of `io/base/parsers.py` open a file in the mode that a
    class declares in `_READ_MODE` or `_WRITE_MODE`, such as `"rt"` for
    text or `"rb"` for binary. A metadata class that delegates to its
    record class has the mode of that class, which this descriptor
    returns. A class without a record class has the default mode.
    """

    def __init__(self, default: str) -> None:
        self.default = default

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: tx.Any, owner: type) -> str:
        raw = getattr(owner, "RAW", None)
        if raw is None:
            return self.default
        return getattr(raw, self.name)


# ----------------------------------------------------------------------
#   READER
# ----------------------------------------------------------------------


class MetadataReader(FileReader):
    """Reader of the metadata of a format whose record class reads files.

    The metadata of many formats holds a record, which is the header of
    the file as the format stores it, and the class of that record can
    already score and read a file. Such a format names its record class
    in the constant `RAW` of
    [`MetadataFormat`][brainhops.io.metadata.MetadataFormat] and lists
    this class among its bases, before `MetadataFormat`:

    ```python
    class LtaMetadata(MetadataReader, MetadataWriter, MetadataFormat):
        RAW = LtaRaw
    ```

    Each `sniff_*` method then passes its input and its keyword arguments
    to the method of the same name of the record class, and each `from_*`
    method does the same and builds the metadata from the record with
    `from_raw`. The record class therefore decides how each input is read,
    whether it is text or binary, and which options it takes. A format
    overrides a method of this class only when its metadata reads that
    input in another way than its record does.
    """

    _READ_MODE = _RecordMode("r")

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file, as the record class scores it.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not of the format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open file, as the record class scores it.

        Parameters
        ----------
        file : IO
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not of the format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the content of a file, as the record class scores it.

        Parameters
        ----------
        content : bytes
            The content of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not of the
            format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_bytes(content, error=error, **kwargs)

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the text of a file, as the record class scores it.

        Parameters
        ----------
        text : str
            The text of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the text is not of the format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_text(text, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the lines of a file, as the record class scores them.

        Parameters
        ----------
        lines : Iterable[str]
            The lines of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the lines are not of the
            format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_lines(lines, error=error, **kwargs)

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the first line of a file, as the record class scores it.

        Parameters
        ----------
        line : str
            The line.
        error : bool or type[Exception], default=False
            Whether to raise an error when the line is not of the format.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.RAW.sniff_line(line, error=error, **kwargs)

    # ---- from --------------------------------------------------------
    # `from_fileobj` is marked as delegated to the record class, so that
    # the adapters of `io/base/parsers.py` take it for a reader of
    # streams only when the record class reads streams. Otherwise a
    # `from_bytes` inherited from the adapters would fall back to it, and
    # the fallback would loop.

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read the metadata from a path, through the record class.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_filename(filename, **kwargs))

    @classmethod
    @_delegated_from_fileobj(lambda cls: cls.RAW)
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read the metadata from an open file, through the record class.

        Parameters
        ----------
        file : IO
            The open file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_fileobj(file, **kwargs))

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read the metadata from the content of a file.

        The content is read by the record class.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_bytes(content, **kwargs))

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Read the metadata from the text of a file, through the record.

        Parameters
        ----------
        text : str
            The text of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_text(text, **kwargs))

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read the metadata from the lines of a file, through the record.

        Parameters
        ----------
        lines : Iterable[str]
            The lines of the file.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_lines(lines, **kwargs))

    @classmethod
    def from_line(cls, line: str, **kwargs) -> tx.Self:
        """Read the metadata from a single line, through the record.

        Parameters
        ----------
        line : str
            The line.
        **kwargs : Any
            Options of the reader of the record.

        Returns
        -------
        Self
            The metadata.
        """
        return cls.from_raw(cls.RAW.from_line(line, **kwargs))


# ----------------------------------------------------------------------
#   WRITER
# ----------------------------------------------------------------------


class MetadataWriter(FileWriter):
    """Writer of the metadata of a format whose record class writes files.

    This class is the counterpart of [`MetadataReader`][]. Each `to_*`
    method passes its keyword arguments to the method of the same name
    of the record that `_record_to_write` returns, so the file that
    the metadata writes is the file that its record writes. The record
    is written as it is: the fields of the metadata that changed since
    the record was read are not encoded into it, which is the job of
    `consolidate`.

    `to_filename` is the exception. It encodes the content with
    `to_bytes` before it opens the file, and then replaces a local file
    at once, so that an error while the content is encoded leaves the
    file as it was.
    """

    _WRITE_MODE = _RecordMode("w")

    def _record_to_write(self) -> tx.Any:
        """Return the record that the metadata writes.

        By default, metadata without a record cannot be written, because
        an empty record does not describe a valid file in most formats. A
        format whose empty record is a valid file, such as one that holds
        the identity, overrides this method to return a new record.

        Returns
        -------
        object
            The record, an instance of `RAW`.

        Raises
        ------
        WriterError
            If the metadata holds no record.
        """
        if self.raw is None:
            raise WriterError(
                f"This {type(self).__name__} holds no record, so it has "
                f"no file to write."
            )
        return self.raw

    @takes_error_policy
    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the file of the record to an open file.

        Parameters
        ----------
        file : IO
            The open file.
        **kwargs : Any
            Options of the writer of the record.
        """
        self._record_to_write().to_fileobj(file, **kwargs)

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the content of the file of the record.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        bytes
            The content of the file.
        """
        return self._record_to_write().to_bytes(**kwargs)

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the file of the record.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        str
            The text of the file.
        """
        return self._record_to_write().to_text(**kwargs)

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Return the lines of the file of the record.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        Iterator[str]
            The lines of the file, without line endings.
        """
        return self._record_to_write().to_lines(**kwargs)

    def to_line(self, **kwargs) -> str:
        """Return the single line of the file of the record.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer of the record.

        Returns
        -------
        str
            The line.
        """
        return self._record_to_write().to_line(**kwargs)
