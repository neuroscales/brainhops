# stdlib
from collections.abc import Iterable

# dependencies
import typing_extensions as _tx

# core
from brainhops._core import path, peek

# ----------------------------------------------------------------------
#   EXCEPTIONS
# ----------------------------------------------------------------------


# ---- sniff -----------------------------------------------------------


class SnifferError(Exception):
    """Base class for sniffer errors."""
    pass


class SnifferTypeError(SnifferError, TypeError):
    """Raised when a sniffer encounters an unexpected type."""
    pass


class SnifferExistsError(SnifferError, FileNotFoundError):
    """Raised when a sniffer encounters an unexpected content."""
    pass


class SnifferContentError(SnifferError, TypeError):
    """Raised when a sniffer encounters an unexpected content."""
    pass


class SnifferNotImplementedError(SnifferError, NotImplementedError):
    """Raised when a sniffer function is not implemented."""
    pass


# ---- from ------------------------------------------------------------


class ParserError(Exception):
    """Base class for parser errors."""
    pass


class ParserTypeError(ParserError, TypeError):
    """Raised when a parser encounters an unexpected type."""
    pass


class ParserExistsError(ParserError, FileNotFoundError):
    """Raised when a parser encounters an unexpected content."""
    pass


class ParserContentError(ParserError, TypeError):
    """Raised when a parser encounters an unexpected content."""
    pass


class ParserNotImplementedError(ParserError, NotImplementedError):
    """Raised when a parser function is not implemented."""
    pass


# ---- to ------------------------------------------------------------


class WriterError(ParserError):
    """Base class for writer errors."""
    pass


class WriterNotImplementedError(WriterError, NotImplementedError):
    """Raised when a writer function is not implemented."""
    pass


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


# ---- sniff -----------------------------------------------------------


class FileSniffer:

    _READ_MODE = "r"

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given file is of the type that this parser can
        handle.

        Parameters
        ----------
        file : FileOrContentLike
            The file to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the file cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the file is of the correct type, False otherwise.
        """
        kwargs['error'] = error

        if isinstance(file, str) and path.Path(file).exists():
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return cls.sniff_file(file, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_file(file, **kwargs)

        if isinstance(file, (bytes, bytearray)):
            return cls.sniff_bytes(file, **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff file of type {type(file)}")
        return False

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given file is of the type that this parser can
        handle.

        Parameters
        ----------
        file : FileLike
            The file to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the file cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the file is of the correct type, False otherwise.
        """
        kwargs['error'] = error

        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            if not file.exists():
                return False
            with file.open(cls._READ_MODE) as f:
                return cls.sniff_file(f, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_content(file.read(), **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff file of type {type(file)}")
        return False

    @classmethod
    def sniff_content(
        cls,
        content: path.ContentLike,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given content is of the type that this parser
        can handle.

        Parameters
        ----------
        content : ContentLike
            The content to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the content is of the correct type, False otherwise.
        """
        kwargs['error'] = error

        if isinstance(content, (bytes, bytearray)):
            return cls.sniff_bytes(content, **kwargs)

        if isinstance(content, str):
            return cls.sniff_text(content, **kwargs)

        if isinstance(content, Iterable):
            return cls.sniff_lines(content, **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff content of type {type(content)}")
        return False

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given file is of the type that this parser can handle.

        Parameters
        ----------
        content : BinaryContentLike
            The content to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the content is of the correct type, False otherwise.
        """
        raise SnifferNotImplementedError(
            f"sniff_bytes() is not available in parser of type {cls.__name__}"
        )

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given text is of the type that this parser can handle.

        Parameters
        ----------
        text : str
            The text to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the text is of the correct type, False otherwise.
        """
        kwargs['error'] = error
        return cls.sniff_lines(text.splitlines(), **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: _tx.Iterable[str],
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given lines are of the type that this parser
        can handle.

        Parameters
        ----------
        lines : Iterable[str]
            The lines to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the lines are of the correct type, False otherwise.
        """
        kwargs['error'] = error
        if not isinstance(lines, peek.peekable_lines):
            lines = peek.peekable_lines(lines)
        return cls.sniff_line(lines.peek(), **kwargs)

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        """
        Determine if the given line is of the type that this parser can handle.

        Parameters
        ----------
        line : str
            The line to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        bool
            True if the line is of the correct type, False otherwise.
        """
        raise SnifferNotImplementedError(
            f"sniff_line() is not available in parser of type {cls.__name__}"
        )


# ---- from ------------------------------------------------------------


class FileParser(FileSniffer):

    @classmethod
    def from_(cls, other: path.FileOrContentLike, **kwargs) -> _tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        Parameters
        ----------
        other : FileOrContentLike
            Input file, or its content.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(other, str) and path.Path(other).exists():
            other = path.Path(other)

        if isinstance(other, path.PathLike):
            return cls.from_file(other, **kwargs)

        if hasattr(other, "read"):
            return cls.from_file(other, **kwargs)

        if isinstance(other, (bytes, bytearray)):
            return cls.from_bytes(other, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(other)}")

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> _tx.Self:
        """
        Build an object from a file (path or file-like object).

        Parameters
        ----------
        file : FileLike
            The file to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            if not file.exists():
                return False
            with file.open(cls._READ_MODE) as f:
                return cls.from_file(f, **kwargs)

        if hasattr(file, "read"):
            return cls.from_content(file.read(), **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_content(cls, content: path.ContentLike, **kwargs) -> _tx.Self:
        """
        Build an object from a file content (bytes, str, or iterable of lines).

        Parameters
        ----------
        content : ContentLike
            The content to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(content, (bytes, bytearray)):
            return cls.from_bytes(content, **kwargs)

        if isinstance(content, str):
            return cls.from_text(content, **kwargs)

        if isinstance(content, Iterable):
            return cls.from_lines(content, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse content of type {type(content)}")

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> _tx.Self:
        """
        Build an object from a binary representation of a file.

        Parameters
        ----------
        content : BinaryContentLike
            The content to sniff.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        raise ParserNotImplementedError(
            f"from_bytes() is not available in parser of type {cls.__name__}"
        )

    @classmethod
    def from_text(cls, text: str, **kwargs) -> _tx.Self:
        """
        Build an object from a text representation of a file.

        Parameters
        ----------
        text : str
            The text to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        return cls.from_lines(text.splitlines(), **kwargs)

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str], **kwargs) -> _tx.Self:
        """
        Build an object from an iterable of lines
        (e.g., the content of a file).

        Parameters
        ----------
        lines : Iterable[str]
            The lines to sniff.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        if not isinstance(lines, peek.peekable_lines):
            lines = peek.peekable_lines(lines)
        return cls.from_line(lines.peek(), **kwargs)

    @classmethod
    def from_line(cls, line: str, **kwargs) -> _tx.Self:
        """
        Build an object from a single line of text.

        Parameters
        ----------
        line : str
            The line to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        raise ParserNotImplementedError(
            f"from_line() is not available in parser of type {cls.__name__}"
        )


# ---- to --------------------------------------------------------------


class FileParserWriter(FileParser):

    _WRITE_MODE = "w"

    def to(self, file: path.FileLike, **kwargs) -> None:
        """
        Write the object to a file (path or file-like object).

        Parameters
        ----------
        file : FileLike
            The file to write to.
        **kwargs
            Parser-specific options.
        """
        if isinstance(file, str) and path.Path(file).exists():
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return self.to_file(file, **kwargs)

        if hasattr(file, "write"):
            return self.to_file(file, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """
        Write the object to a file (path or file-like object).

        Parameters
        ----------
        file : FileLike
            The file to write to.
        **kwargs
            Parser-specific options.
        """
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            if not file.exists():
                return False
            with file.open(self._WRITE_MODE) as f:
                return self.to_file(f, **kwargs)

        if hasattr(file, "writelines"):
            file.writelines(self.to_lines(**kwargs))

        elif hasattr(file, "write"):
            if "b" in self._WRITE_MODE:
                file.write(self.to_bytes(**kwargs))
            else:
                file.write(self.to_text(**kwargs))

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    def to_bytes(self, **kwargs) -> bytes:
        """
        Return a binary version of the file.

        Parameters
        ----------
        **kwargs
            Parser-specific options.

        Returns
        -------
        bytes
            A binary version of the file.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"to_bytes() is not available in writer of type {cls.__name__}"
        )

    def to_text(self, **kwargs) -> str:
        """
        Return a text version of the file.

        Parameters
        ----------
        **kwargs
            Parser-specific options.

        Returns
        -------
        str
            A text version of the file.
        """
        return "\n".join(self.to_lines(**kwargs))

    def to_lines(self, **kwargs) -> _tx.Iterator[str]:
        """
        Return a text version of the file as an iterable of lines.

        Parameters
        ----------
        **kwargs
            Parser-specific options.

        Returns
        -------
        Iterator[str]
            An iterable of lines representing the object.
        """
        yield self.to_line(**kwargs)

    def to_line(self, **kwargs) -> str:
        """
        Return a line representing the object.

        Parameters
        ----------
        **kwargs
            Parser-specific options.

        Returns
        -------
        str
            A line representing the object.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"to_line() is not available in writer of type {cls.__name__}"
        )

# ----------------------------------------------------------------------
#   TEXT
# ----------------------------------------------------------------------


class TextFileSniffer(FileSniffer):

    _READ_MODE = "rt"

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: _tx.Union[bool, _tx.Type[Exception]] = False,
        **kwargs
    ) -> bool:
        kwargs['error'] = error
        encoding = kwargs.pop('encoding', 'utf-8')
        return cls.sniff_text(content.decode(encoding), **kwargs)


class TextFileParser(TextFileSniffer, FileParser):

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> _tx.Self:
        encoding = kwargs.pop('encoding', 'utf-8')
        return cls.from_text(content.decode(encoding), **kwargs)


class TextFileParserWriter(TextFileParser):

    def to_bytes(self, **kwargs) -> bytes:
        encoding = kwargs.pop('encoding', 'utf-8')
        return self.to_text(**kwargs).encode(encoding)

# ----------------------------------------------------------------------
#   BINARY
# ----------------------------------------------------------------------


class BinaryFileSniffer:

    _READ_MODE = "rb"


class BinaryFileParser(BinaryFileSniffer):

    ...


class BinaryFileParserWriter(BinaryFileParser):

    _WRITE_MODE = "wb"
