"""Base classes that let a format sniff, read and write itself.

The errors that these classes raise are also importable from here.
"""

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

from collections.abc import Iterable
from contextvars import ContextVar
from io import BytesIO

import typing_extensions as tx

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
    """Named confidence levels for sniffers.

    The levels are plain floats, and a sniffer may return any value between 0
    and 1.
    """

    NO: float = 0.0
    """The input is definitely not of this type."""

    WEAK: float = 0.25
    """The input can be read as this type but probably is not of it."""

    MAYBE: float = 0.5
    """The input is plausibly of this type."""

    LIKELY: float = 0.75
    """The input is very likely of this type."""

    CERTAIN: float = 1.0
    """The input is definitely of this type."""


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


# ---- sniff -----------------------------------------------------------


class FileSniffer:
    """Class that can sniff files to decide whether they are of its type.

    Sniffing a file means inspecting its content and returning a confidence
    score between 0 and 1. The `sniff*` methods form a chain in which each
    method turns its input into a simpler form and passes it on, from a path
    to an open file, then to content, and finally to [`sniff_bytes`][] or
    [`sniff_line`][], which a concrete format overrides.
    """

    _READ_MODE: str = "r"

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """File extensions that the format handles, such as `(".nii", ".nii.gz")`.

    A matching extension keeps the format among the candidates even if its
    sniffer declines, and the extension match is the next ranking criterion
    after the sniffer score. The longest matching extension wins, so `.nii.gz`
    beats `.gz`.
    """

    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """Required file name prefixes, such as `("y_", "iy_")`.

    An empty tuple means that any name is accepted. When prefixes are declared,
    a file name matches the format only if it starts with one of them, and a
    format with a matching prefix is more specific and wins ties. Extensions
    and prefixes are declared separately, and every prefix combines with every
    extension, as file naming conventions usually do: the four names of SPM
    deformation fields are `{y_, iy_}` times `{.nii, .nii.gz}`.
    """

    # TODO: neither attribute expresses infix constraints, such as the BIDS
    # suffix *_T1w.nii.gz or FreeSurfer's lh.white. If such constraints are
    # needed, add optional glob PATTERNS alongside them, and count specificity
    # as the number of literal characters matched, as _match_name already does
    # for extensions and prefixes.

    PRIORITY: tx.ClassVar[int] = 0
    """Tie-breaker used only when specificity cannot decide.

    The higher value wins. The priority should stay at 0 unless two parsers
    genuinely collide.
    """

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confidently a file or its content is of this type.

        A path is sniffed with [`sniff_filename`][], an open file with
        [`sniff_fileobj`][], and binary content with [`sniff_bytes`][].

        Parameters
        ----------
        file : FileOrContentLike
            The file or its content.
        error : bool or type of Exception, default=False
            If not false, raise when the input cannot be sniffed: `True` raises
            the default error, and an exception class raises that class.
        **kwargs : Any
            Parser options.

        Returns
        -------
        float
            The confidence, between 0 and 1. An unsupported input scores zero.
        """
        kwargs["error"] = error

        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_fileobj(file, **kwargs)

        if isinstance(file, (bytes, bytearray)):
            return cls.sniff_bytes(file, **kwargs)

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
        """Same as [`sniff`][], for a path or an open file."""
        kwargs["error"] = error

        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.sniff_fileobj(file, **kwargs)

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
        """Same as [`sniff`][], for a path.

        A missing file scores [`Confidence.NO`][], or raises
        [`SnifferExistsError`][] if `error` is set.
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
        """Same as [`sniff`][], for a file object open for reading.

        The whole stream is read and passed to [`sniff_content`][], and the
        position of the stream is restored afterwards.
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
        """Same as [`sniff`][], for bytes, text, or lines."""
        kwargs["error"] = error

        if isinstance(content, (bytes, bytearray)):
            return cls.sniff_bytes(content, **kwargs)

        if isinstance(content, str):
            return cls.sniff_text(content, **kwargs)

        if isinstance(content, Iterable):
            return cls.sniff_lines(content, **kwargs)

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
        """Same as [`sniff`][], for binary content.

        Raises
        ------
        SnifferNotImplementedError
            In the base class, which cannot sniff binary content.
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
        """Same as [`sniff`][], for text.

        The text is split into lines and passed to [`sniff_lines`][].
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
        """Same as [`sniff`][], for an iterable of lines.

        The first line is passed to [`sniff_line`][].
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
        """Same as [`sniff`][], for a single line.

        Raises
        ------
        SnifferNotImplementedError
            In the base class, which cannot sniff a line.
        """
        raise SnifferNotImplementedError(
            f"sniff_line() is not available in parser of type {cls.__name__}"
        )


# ---- from ------------------------------------------------------------


def _passthrough_from_fileobj(func: tx.Callable) -> tx.Callable:
    """Mark a `from_fileobj` as a passthrough.

    A passthrough implementation does not read the stream itself. Instead, it
    ends up passing the whole stream to `from_bytes`, as the default
    implementation does, or as a mixin does when it only forwards the call to
    `super()`. [`FileParser.from_bytes`][] never falls back to a passthrough,
    because the fallback would loop.
    """
    func._passthrough_from_fileobj = True
    return func


# The parser classes whose from_bytes is currently falling back to
# from_fileobj. If from_bytes is entered again for one of these classes, then
# from_fileobj has passed the content back to from_bytes, and the fallback
# would loop.
_FROM_BYTES_FALLBACK: ContextVar[frozenset] = ContextVar(
    "_FROM_BYTES_FALLBACK", default=frozenset()
)


def _overrides_from_fileobj(cls: type) -> bool:
    """Return whether a class in the MRO of `cls` defines `from_fileobj`.

    Implementations marked with `_passthrough_from_fileobj` do not count, so a
    mixin that only forwards the call is not an override.
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
    """Class that can read files of its type."""

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """Read an object from a path, an open file, or binary content.

        Paths and open files are read with [`from_file`][], and binary content
        with [`from_bytes`][]. A `str` is always a path, whether or not the
        file exists, so text held in memory must be read with [`from_text`][]
        or [`from_content`][].

        Raises
        ------
        ParserExistsError
            If the path names a file that does not exist.
        ParserTypeError
            If the input is of another type.
        """
        if isinstance(other, str):
            other = path.Path(other)

        if isinstance(other, path.PathLike):
            return cls.from_file(other, **kwargs)

        if hasattr(other, "read"):
            return cls.from_file(other, **kwargs)

        if isinstance(other, (bytes, bytearray)):
            return cls.from_bytes(other, **kwargs)

        raise ParserTypeError(f"Cannot parse file of type {type(other)}")

    @classmethod
    def from_spec(cls, spec: SourceSpec, **kwargs) -> tx.Self:
        """Read the file named by a specification without hints or options.

        Raises
        ------
        TypeError
            If the specification has hints or options, which only a dispatcher
            can apply.
        """
        if spec.hints or spec.options:
            raise TypeError(
                "Structured hints and options require a format dispatcher, "
                "not a concrete parser."
            )
        return cls.load(spec.path, **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Read an object from a path or an open file."""
        if isinstance(file, (str, path.PathLike)):
            return cls.from_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an object from a path.

        Raises
        ------
        ParserExistsError
            If the file does not exist.
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
        """Read an object from a file object open for reading.

        The default implementation reads the whole stream and passes it to
        [`from_content`][]. A parser that needs only part of the stream, such
        as a header, should override this method. [`from_bytes`][] then wraps
        binary content in a stream and reads it with the override.
        """
        with preserve_position(file):
            return cls.from_content(file.read(), **kwargs)

    @classmethod
    def from_content(cls, content: path.ContentLike, **kwargs) -> tx.Self:
        """Read an object from bytes, text, or lines."""
        if isinstance(content, (bytes, bytearray)):
            return cls.from_bytes(content, **kwargs)

        if isinstance(content, str):
            return cls.from_text(content, **kwargs)

        if isinstance(content, Iterable):
            return cls.from_lines(content, **kwargs)

        raise ParserTypeError(f"Cannot parse content of type {type(content)}")

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read an object from binary content.

        If the class implements [`from_fileobj`][] itself, the content is
        wrapped in an `io.BytesIO` and read with that method. Otherwise, this
        method raises, because the default `from_fileobj` would pass the
        content back to `from_bytes` and the fallback would recurse. An
        override that only forwards to `super().from_fileobj` should be marked
        with `_passthrough_from_fileobj`. If such an override is not marked,
        the loop is still detected when this method is entered again for the
        same class.

        Raises
        ------
        ParserNotImplementedError
            If neither this method nor `from_fileobj` is implemented.
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
        """Read an object from text.

        The text is split into lines and passed to [`from_lines`][].
        """
        return cls.from_lines(text.splitlines(), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read an object from an iterable of lines.

        The first line is passed to [`from_line`][].
        """
        if not isinstance(lines, peek.peekable_lines):
            lines = peek.peekable_lines(lines)
        return cls.from_line(lines.peek(), **kwargs)

    @classmethod
    def from_line(cls, line: str, **kwargs) -> tx.Self:
        """Read an object from a single line.

        Raises
        ------
        ParserNotImplementedError
            In the base class, which cannot read a line.
        """
        raise ParserNotImplementedError(
            f"from_line() is not available in parser of type {cls.__name__}"
        )


# ---- to --------------------------------------------------------------


class FileParserWriter(FileParser):
    """Class that can read and write files of its type."""

    _WRITE_MODE = "w"

    def save(self, file: path.FileLike, **kwargs) -> None:
        """Write the object to a path or an open file.

        The method is named `save` rather than `to` because the data models
        that parsers are mixed into already use `to` for conversion, as in
        `Transformation.to`, which would shadow the writer.
        """
        return self.to_file(file, **kwargs)

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the object to a path or a file object.

        A file object needs a `write` or a `writelines` method.

        Raises
        ------
        ParserTypeError
            If `file` is of another type.
        """
        if isinstance(file, (str, path.PathLike)):
            return self.to_filename(file, **kwargs)

        if hasattr(file, "write") or hasattr(file, "writelines"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the object to a path, opened in the class write mode."""
        if isinstance(filename, str):
            filename = path.Path(filename)

        with filename.open(self._WRITE_MODE) as f:
            return self.to_fileobj(f, **kwargs)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the object to a file object open for writing.

        If the class writes in binary mode, the result of [`to_bytes`][] is
        written. Otherwise, if the file object has a `writelines` method, the
        lines of [`to_lines`][] are written with that method, each followed by
        a newline. A file object without `writelines` receives the result of
        [`to_text`][] through `write`.
        """
        if "b" in self._WRITE_MODE:
            file.write(self.to_bytes(**kwargs))
        elif hasattr(file, "writelines"):
            file.writelines(line + "\n" for line in self.to_lines(**kwargs))
        else:
            file.write(self.to_text(**kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the binary content of the file.

        Raises
        ------
        WriterNotImplementedError
            In the base class, which cannot write binary content.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"to_bytes() is not available in writer of type {cls.__name__}"
        )

    def to_text(self, **kwargs) -> str:
        """Return the text of the file.

        The text is made of the lines of [`to_lines`][] joined by newlines.
        """
        return "\n".join(self.to_lines(**kwargs))

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Return the lines of the file, without terminators.

        The default implementation yields the result of [`to_line`][] as the
        only line.
        """
        yield self.to_line(**kwargs)

    def to_line(self, **kwargs) -> str:
        """Return the single line that represents the object.

        Raises
        ------
        WriterNotImplementedError
            In the base class, which cannot write a line.
        """
        cls = type(self)
        raise WriterNotImplementedError(
            f"to_line() is not available in writer of type {cls.__name__}"
        )


# ----------------------------------------------------------------------
#   TEXT
# ----------------------------------------------------------------------


class TextFileSniffer(FileSniffer):
    """Class that can sniff text files for its type."""

    _READ_MODE: str = "rt"

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Sniff a file object as text.

        A text stream decodes its content as it is read. Content that is not
        text, such as a binary file that has the extension of a text format,
        therefore scores [`Confidence.NO`][] instead of failing with a decoding
        error. If `error` is set, [`SnifferContentError`][] is raised instead.
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
        """Sniff binary content as text.

        The content is decoded with the `encoding` keyword, UTF-8 by default,
        and passed to `sniff_text`. Content that does not decode scores
        [`Confidence.NO`][], or raises [`SnifferContentError`][] if `error` is
        set.
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
    """Decline content that does not decode as text.

    The function returns [`Confidence.NO`][]. If `error` is set, the requested
    error is raised instead, with `cause` as its cause.
    """
    if error:
        if error is True:
            error = SnifferContentError
        raise error(
            f"{cls.__name__} reads text, but the content does not decode "
            f"as text."
        ) from cause
    return Confidence.NO


class TextFileParser(TextFileSniffer, FileParser):
    """Class that can read text files of its type."""

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read an object from bytes decoded as text.

        The `encoding` keyword selects the encoding, UTF-8 by default.
        """
        encoding = kwargs.pop("encoding", "utf-8")
        return cls.from_text(content.decode(encoding), **kwargs)


class TextFileParserWriter(TextFileParser, FileParserWriter):
    """Class that can read and write text files of its type."""

    def to_bytes(self, **kwargs) -> bytes:
        """Return the text of the file, encoded.

        The `encoding` keyword selects the encoding, UTF-8 by default.
        """
        encoding = kwargs.pop("encoding", "utf-8")
        return self.to_text(**kwargs).encode(encoding)


# ----------------------------------------------------------------------
#   BINARY
# ----------------------------------------------------------------------


class BinaryFileSniffer(FileSniffer):
    """Class that can sniff binary files for its type."""

    _READ_MODE: str = "rb"


class BinaryFileParser(BinaryFileSniffer, FileParser):
    """Class that can read binary files of its type."""

    ...


class BinaryFileParserWriter(BinaryFileParser, FileParserWriter):
    """Class that can read and write binary files of its type."""

    _WRITE_MODE: str = "wb"
