"""The base classes that give a format the ability to sniff, read and
write itself, and the errors they raise."""

__all__ = [
    "Confidence",
    "FileSniffer",
    "FileParser",
    "FileParserWriter",
    "BinaryFileSniffer",
    "BinaryFileParser",
    "BinaryFileParserWriter",
    "TextFileSniffer",
    "TextFileParser",
    "TextFileParserWriter",
]

# stdlib
from collections.abc import Iterable
from contextvars import ContextVar
from io import BytesIO

# dependencies
import typing_extensions as tx

# core
from brainhops._core import path, peek
from brainhops._core.streams import preserve_position
from brainhops.errors import (
    AmbiguousFormatError,  # noqa: F401
    ParserContentError,  # noqa: F401
    ParserError,  # noqa: F401
    ParserExistsError,
    ParserNotImplementedError,
    ParserTypeError,
    SnifferContentError,
    SnifferError,  # noqa: F401
    SnifferExistsError,
    SnifferNotImplementedError,
    SnifferTypeError,
    UnrepresentableTransformationError,  # noqa: F401
    WriterError,  # noqa: F401
    WriterNotImplementedError,
)
from brainhops.io.base.specs import SourceSpec

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

        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_fileobj(file, **kwargs)

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

        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_fileobj(file, **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff file of type {type(file)}")
        return False

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Determine if the given filename is of the type that this parser
        can handle.

        Parameters
        ----------
        filename : FilenameLike
            The filename to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the filename cannot be sniffed.
        **kwargs
            Parser-specific options.

        Returns
        -------
        float
            Confidence that the filename is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error

        if isinstance(filename, str):
            filename = path.Path(filename)

        if isinstance(filename, path.PathLike):
            if not path.exists(filename):
                if error:
                    if error is True:
                        error = SnifferExistsError
                    raise error(f"No such file: {filename}")
                return Confidence.NO
            with filename.open(cls._READ_MODE) as f:
                return cls.sniff_fileobj(f, **kwargs)

        # Cannot parse this content -> return False or error
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(f"Cannot sniff filename of type {type(filename)}")
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


def _passthrough_from_fileobj(func: tx.Callable) -> tx.Callable:
    """
    Mark a `from_fileobj` implementation as one that does not read the
    stream itself, but ends up reading it whole and delegating to
    `from_bytes` (e.g., `FileParser.from_fileobj`, or a mixin that
    forwards to `super().from_fileobj`).

    Parameters
    ----------
    func : callable
        The function underlying a `from_fileobj` classmethod.

    Returns
    -------
    callable
        The same function, marked.
    """
    func._passthrough_from_fileobj = True
    return func


# Parser classes whose `from_bytes` is currently falling back to
# `from_fileobj`; a re-entry for the same class means that the
# `from_fileobj` chain fell back to `from_bytes` in turn.
_FROM_BYTES_FALLBACK: ContextVar[frozenset] = ContextVar(
    "_FROM_BYTES_FALLBACK", default=frozenset()
)


def _overrides_from_fileobj(cls: type) -> bool:
    """
    Check whether a parser class implements `from_fileobj` itself, rather
    than inheriting the default that delegates to `from_bytes`.

    The MRO is walked, skipping implementations marked with
    `_passthrough_from_fileobj` (the `FileParser` default and the
    `FormatDispatcher` forwarder), so that mixing in a forwarder does
    not count as an override.

    Parameters
    ----------
    cls : type
        A subclass of `FileParser`.

    Returns
    -------
    bool
        True if some class in the MRO of `cls` provides a `from_fileobj`
        that does not fall back to `from_bytes`.
    """
    for klass in cls.__mro__:
        func = klass.__dict__.get("from_fileobj")
        if func is None:
            continue
        func = getattr(func, "__func__", func)
        if not getattr(func, "_passthrough_from_fileobj", False):
            return True
    return False


class FileParser(FileSniffer):
    """A class that can read files of a certain type."""

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        This is the generic front door to the `from_*` family: it looks
        at what it was handed and calls the right one.

        A `str` is always a path, whether or not the file exists, so a
        missing file raises `FileNotFoundError` whichever way its path
        was spelled. Text held in memory is read with `from_text` or
        `from_content`.

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

        Raises
        ------
        ParserExistsError
            If `other` is a path to a file that does not exist. It is a
            `FileNotFoundError`.
        """
        if isinstance(other, str):
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
    def from_spec(cls, spec: SourceSpec, **kwargs) -> tx.Self:
        """Build an object from an unqualified structured source."""
        if spec.hints or spec.options:
            raise TypeError(
                "Structured hints and options require a format dispatcher, "
                "not a concrete parser."
            )
        return cls.load(spec.path, **kwargs)

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
        if isinstance(file, (str, path.PathLike)):
            return cls.from_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)

        # Cannot parse this content -> return False or error
        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """
        Build an object from a filename.

        Parameters
        ----------
        filename : FilenameLike
            The filename to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)

        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with filename.open(cls._READ_MODE) as f:
            return cls.from_fileobj(f, **kwargs)

    @classmethod
    @_passthrough_from_fileobj
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """
        Build an object from a file-like object.

        The default implementation reads the whole stream and hands its
        content to `from_content` (hence to `from_bytes` for binary
        streams). Parsers that only need part of the stream (e.g., a
        header) should override this method; `from_bytes` then falls back
        to it.

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

        If the class implements `from_fileobj` itself, the bytes are
        wrapped in an `io.BytesIO` stream and handed to `from_fileobj`.
        Otherwise, this raises `ParserNotImplementedError`: the default
        `from_fileobj` delegates to `from_bytes`, so falling back to it
        would recurse. A `from_fileobj` override that only forwards to
        `super().from_fileobj` should be decorated with
        `_passthrough_from_fileobj`; if it is not, the loop is still
        detected when `from_bytes` is re-entered for the same class, and
        `ParserNotImplementedError` is raised.

        Parameters
        ----------
        content : BinaryContentLike
            The content to parse.
        **kwargs
            Parser-specific options.

        Returns
        -------
        obj
            The parsed object.

        Raises
        ------
        ParserNotImplementedError
            If neither `from_bytes` nor `from_fileobj` is implemented.
        """
        active = _FROM_BYTES_FALLBACK.get()
        if cls not in active and _overrides_from_fileobj(cls):
            token = _FROM_BYTES_FALLBACK.set(active | {cls})
            try:
                return cls.from_fileobj(BytesIO(content), **kwargs)
            finally:
                _FROM_BYTES_FALLBACK.reset(token)
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
        return self.to_file(file, **kwargs)

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
        if isinstance(file, (str, path.PathLike)):
            return self.to_filename(file, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """
        Write the object to a filename.

        Parameters
        ----------
        filename : FilenameLike
            The filename to write to.
        **kwargs
            Parser-specific options.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)

        with filename.open(self._WRITE_MODE) as f:
            return self.to_fileobj(f, **kwargs)

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
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Determine if the given file-like object is of the type that this
        parser can handle.

        A text stream decodes as it is read, so content that is not text
        -- a binary file that shares an extension with a text format --
        fails there. That is a "no", not a failure to sniff.

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
        try:
            return super().sniff_fileobj(file, error=error, **kwargs)
        except UnicodeDecodeError as e:
            return _not_text(cls, error, e)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Determine if the given bytes are of the type that this parser can
        handle, by decoding them to text and delegating to `sniff_text`.
        Bytes that do not decode are not text, so they score `NO`.

        Parameters
        ----------
        content : BinaryContentLike
            The content to sniff.
        error : bool | type[Exception], optional
            If not False, raise an error if the content cannot be sniffed.
        **kwargs
            Parser-specific options, plus `encoding` (default `"utf-8"`)
            for decoding `content`.

        Returns
        -------
        float
            Confidence that the content is of this type, in `[0, 1]`.
        """
        kwargs["error"] = error
        encoding = kwargs.pop("encoding", "utf-8")
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError as e:
            return _not_text(cls, error, e)
        return cls.sniff_text(text, **kwargs)


def _not_text(
    cls: type, error: tx.Union[bool, tx.Type[Exception]], cause: Exception
) -> float:
    """Decline content that does not decode as text, or raise if the
    caller asked for it."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(
            f"{cls.__name__} reads text, but the content does not decode "
            f"as text."
        ) from cause
    return Confidence.NO


class TextFileParser(TextFileSniffer, FileParser):
    """A class that can read text files of a certain type."""

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """
        Build an object from bytes, by decoding them to text and
        delegating to `from_text`.

        Parameters
        ----------
        content : BinaryContentLike
            The content to parse.
        **kwargs
            Parser-specific options, plus `encoding` (default `"utf-8"`)
            for decoding `content`.

        Returns
        -------
        obj
            The parsed object.
        """
        encoding = kwargs.pop("encoding", "utf-8")
        return cls.from_text(content.decode(encoding), **kwargs)


class TextFileParserWriter(TextFileParser, FileParserWriter):
    """A class that can read and write text files of a certain type."""

    def to_bytes(self, **kwargs) -> bytes:
        """
        Convert the object to bytes, by converting it to text and
        encoding the result.

        Parameters
        ----------
        **kwargs
            Writer-specific options, plus `encoding` (default `"utf-8"`)
            for encoding the text.

        Returns
        -------
        bytes
            The byte representation of the object.
        """
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
