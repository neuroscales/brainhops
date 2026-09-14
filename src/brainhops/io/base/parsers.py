# stdlib
from collections.abc import Iterable

# dependencies
import typing_extensions as tx

# core
from brainhops._core import path, peek
from brainhops._core.streams import preserve_position

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


class AmbiguousFormatError(ParserError):
    """
    Raised when several parsers claim the same content, equally well.

    Reaching this means two parsers agree on the confidence score, the
    extension they matched, the constraints they declare and their
    explicit priority, yet build different objects. There is no
    meaningful way to choose between them, and picking one at random
    would silently return the wrong kind of object.

    The fix belongs in the parsers, not in the caller: give one of them a
    sniffer that can tell the two apart (a magic number, an intent code,
    a filename constraint), or set an explicit `PRIORITY`.
    """

    pass


# ---- to ------------------------------------------------------------


class WriterError(ParserError):
    """Base class for writer errors."""

    pass


class WriterNotImplementedError(WriterError, NotImplementedError):
    """Raised when a writer function is not implemented."""

    pass


class UnrepresentableTransformationError(WriterError):
    """
    Raised when a transformation cannot be encoded in the target format.

    A format that stores only affine geometry, such as NIfTI, cannot hold
    an arbitrary transformation. When the object being written carries one
    that the format has no way to represent, the writer raises this error
    rather than resampling the data or discarding the transformation. The
    message names the transformation that could not be written.
    """

    pass


# ----------------------------------------------------------------------
#   CONFIDENCE
# ----------------------------------------------------------------------


class Confidence:
    """
    Named confidence levels for sniffers.

    These are plain floats; a sniffer may return any value in `[0, 1]`.
    """

    NO: float = 0.0
    """The content is definitely not of this type."""

    WEAK: float = 0.25
    """The content can be read as this type, but probably is not one."""

    MAYBE: float = 0.5
    """The content is plausibly of this type."""

    LIKELY: float = 0.75
    """The content is very likely of this type."""

    CERTAIN: float = 1.0
    """The content is definitely of this type."""


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


# ---- sniff -----------------------------------------------------------


class FileSniffer:
    """
    A class that can sniff files to determine if they are of a certain type.
    """

    _READ_MODE: str = "r"

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """
    File extensions handled by this parser, e.g. `(".nii", ".nii.gz")`.

    Used as a first, cheap dispatch pass. When several parsers match,
    the *longest* matching extension wins, so a parser declaring
    `".nii.gz"` takes precedence over one declaring `".gz"`.
    """

    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """
    Filename prefixes required by this parser, e.g. `("y_", "iy_")`.

    An empty tuple means "no constraint". A parser that constrains the
    prefix is more specific than one that does not, and wins ties.

    Declaring `EXTENSIONS` and `PREFIXES` separately states the
    cross-product implicitly, which is how these conventions actually
    work: SPM's four names are `{y_, iy_}` x `{.nii, .nii.gz}`.
    """

    # TODO: neither attribute can express an *infix* constraint, which
    # some conventions need -- BIDS suffixes (`*_T1w.nii.gz`) and
    # FreeSurfer names (`lh.white`, `rh.pial`) among them. If such a
    # format shows up, add an optional `PATTERNS` of globs as an escape
    # hatch *alongside* these two rather than replacing them: globs
    # would force the cross-product above to be enumerated, which is
    # more verbose for the common case. Specificity generalizes
    # cleanly -- count the literal, non-wildcard characters matched,
    # which is what `_match_name` already measures for the simple case.

    PRIORITY: tx.ClassVar[int] = 0
    """
    Explicit tie-breaker, consulted only when specificity cannot decide.
    Higher wins. Leave at `0` unless two parsers genuinely collide.
    """

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the file is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error

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
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the file is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error

        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            if not file.exists():
                if error:
                    if error is True:
                        error = SnifferExistsError
                    raise error(f"No such file: {file}")
                return Confidence.NO
            with file.open(cls._READ_MODE) as f:
                return cls.sniff_fileobj(f, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_fileobj(file, **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff file of type {type(file)}")
        return False

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Determine if the given file-like object is of the type that this
        parser can handle.

        Parameters
        ----------
        file : IO
            A file object open for reading.
        error : bool | type[Exception], optional
            If not False, raise an error if the file cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        float
            Confidence that the file is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error
        with preserve_position(file):
            return cls.sniff_content(file.read(), **kwargs)

    @classmethod
    def sniff_content(
        cls,
        content: path.ContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the content is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error

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
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the content is of this type, in `[0, 1]`.
        """
        raise SnifferNotImplementedError(
            f"sniff_bytes() is not available in parser of type {cls.__name__}"
        )

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the text is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error
        return cls.sniff_lines(text.splitlines(), **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the lines is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error
        if not isinstance(lines, peek.peekable_lines):
            lines = peek.peekable_lines(lines)
        return cls.sniff_line(lines.peek(), **kwargs)

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
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
        float
            Confidence that the line is of this type, in `[0, 1]`.
        """
        raise SnifferNotImplementedError(
            f"sniff_line() is not available in parser of type {cls.__name__}"
        )


# ---- from ------------------------------------------------------------


class FileParser(FileSniffer):
    """A class that can read files of a certain type."""

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        This is the generic front door to the `from_*` family: it looks
        at what it was handed and calls the right one.

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
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
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
                raise ParserExistsError(f"No such file: {file}")
            with file.open(cls._READ_MODE) as f:
                return cls.from_fileobj(f, **kwargs)

        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """
        Build an object from a file-like object.

        Parameters
        ----------
        file : IO
            A file object open for reading.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        with preserve_position(file):
            return cls.from_content(file.read(), **kwargs)

    @classmethod
    def from_content(cls, content: path.ContentLike, **kwargs) -> tx.Self:
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
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
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
    def from_text(cls, text: str, **kwargs) -> tx.Self:
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
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
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
    def from_line(cls, line: str, **kwargs) -> tx.Self:
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
    """A class that can read and write files of a certain type."""

    _WRITE_MODE = "w"

    def save(self, file: path.FileLike, **kwargs) -> None:
        """
        Write the object to a file (path or file-like object).

        This is the generic front door to the `to_*` family. It is named
        `save` rather than `to` because `to` already means something else
        on the data models these parsers are mixed into: `Transformation.to`
        converts an object to another *type*. A writer's `to` was shadowed
        by it on every writable transformation.

        Parameters
        ----------
        file : FileLike
            The file to write to.
        **kwargs
            Parser-specific options.
        """
        # NOTE: unlike `load`, we must not require the path to exist --
        # writing to a new file is the common case.
        if isinstance(file, str):
            file = path.Path(file)

        if isinstance(file, path.PathLike):
            return self.to_file(file, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

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
            with file.open(self._WRITE_MODE) as f:
                return self.to_fileobj(f, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """
        Write the object to a file-like object open for writing.

        Parameters
        ----------
        file : IO
            A file object open for writing.
        **kwargs
            Parser-specific options.
        """
        if "b" in self._WRITE_MODE:
            file.write(self.to_bytes(**kwargs))
        elif hasattr(file, "writelines"):
            # `to_lines` yields lines without their terminator.
            file.writelines(line + "\n" for line in self.to_lines(**kwargs))
        else:
            file.write(self.to_text(**kwargs))

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

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
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
    """
    A class that can sniff text files to determine if they are of a
    certain type.
    """

    _READ_MODE: str = "rt"

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        kwargs["error"] = error
        encoding = kwargs.pop("encoding", "utf-8")
        return cls.sniff_text(content.decode(encoding), **kwargs)


class TextFileParser(TextFileSniffer, FileParser):
    """A class that can read text files of a certain type."""

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        encoding = kwargs.pop("encoding", "utf-8")
        return cls.from_text(content.decode(encoding), **kwargs)


class TextFileParserWriter(TextFileParser, FileParserWriter):
    """A class that can read and write text files of a certain type."""

    def to_bytes(self, **kwargs) -> bytes:
        encoding = kwargs.pop("encoding", "utf-8")
        return self.to_text(**kwargs).encode(encoding)


# ----------------------------------------------------------------------
#   BINARY
# ----------------------------------------------------------------------


class BinaryFileSniffer(FileSniffer):
    """
    A class that can sniff binary files to determine if they are of a
    certain type.
    """

    _READ_MODE: str = "rb"


class BinaryFileParser(BinaryFileSniffer, FileParser):
    """A class that can read binary files of a certain type."""

    ...


class BinaryFileParserWriter(BinaryFileParser, FileParserWriter):
    """A class that can read and write binary files of a certain type."""

    _WRITE_MODE: str = "wb"
