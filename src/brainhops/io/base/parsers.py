"""Independent sniffing, reading and writing capabilities for file formats.

Native format parsers compose readers and writers independently. Writers
do not inherit reading or sniffing. Dispatchers, which choose between
formats, have their own API in
[`FileFormat`][brainhops.io.base.FileFormat] and inherit none of these
classes.

The errors that these classes raise are also importable from here.
"""

__all__ = [
    "Confidence",
    "FileReader",
    "FileWriter",
    "BinaryFileReader",
    "BinaryFileWriter",
    "TextFileReader",
    "TextFileWriter",
    "disabled",
]

from collections.abc import Iterable
from contextvars import ContextVar
from io import BytesIO, TextIOBase

import typing_extensions as tx

from brainhops._core import path, peek
from brainhops._core.path import replace_file
from brainhops._core.streams import (
    compression_suffix,
    decompress,
    open_compressing,
    open_decompressed,
    preserve_position,
)
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
#   DISABLED METHODS
# ----------------------------------------------------------------------


class disabled:  # noqa: N801
    """A reader or writer method that a format cannot provide.

    Some formats cannot be read from every kind of input or written to
    every kind of output. An AFNI dataset, for example, is two files,
    which bytes alone cannot hold. Such a format assigns this object to
    the method in its class body, as in `from_bytes = disabled(
    ParserContentError, "...")`, instead of writing a method that only
    raises. Calling the method, on the class or on an instance, with any
    arguments, raises the error with the message.

    A `from_fileobj` that is disabled counts as an implementation for the
    input adapters, so `from_bytes` falls back to it and raises the same
    error, unless `from_bytes` is implemented or disabled as well.

    Parameters
    ----------
    error : type[Exception]
        The class of the error that the method raises.
    message : str
        The message of the error, which says what the format can read or
        write instead.
    """

    def __init__(self, error: tx.Type[Exception], message: str) -> None:
        self.error = error
        self.message = message
        self.name = "method"

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(
        self, obj: tx.Any, owner: tx.Optional[type] = None
    ) -> tx.Callable[..., tx.NoReturn]:
        def refuse(*args: tx.Any, **kwargs: tx.Any) -> tx.NoReturn:
            raise self.error(self.message)

        refuse.__name__ = refuse.__qualname__ = self.name
        refuse.__doc__ = self.message
        return refuse

    def __repr__(self) -> str:
        return f"disabled({self.error.__name__}, {self.message!r})"


# ----------------------------------------------------------------------
#   BASES
# ----------------------------------------------------------------------


# ---- sniff -----------------------------------------------------------


class _FileNamesMixin:
    """Constants that describe the names of the files of a format.

    Dispatch reads these constants on every registered format, including
    a format that writes files but does not read them. Both
    [`_SnifferMixin`][] and [`FileFormat`][brainhops.io.base.FileFormat]
    inherit them.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ()
    """File extensions that the format handles, such as `(".mgh", ".mgz")`.

    A matching extension keeps the format among the candidates even if its
    sniffer declines, and the extension match is the next ranking criterion
    after the sniffer score. The longest matching extension wins, so
    `.note.txt` beats `.txt`. Every format may be compressed, so the
    extensions leave out compression suffixes: a name that ends with
    `.gz`, `.bz2` or `.xz` also matches the extensions of the name without
    that suffix, and `x.nii.gz` matches `.nii`. An extension that implies
    compression without such a suffix, such as `.mgz`, is listed.
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


# The parser classes whose sniff_bytes is currently falling back to
# sniff_fileobj. If sniff_bytes is entered again for one of these classes,
# then sniff_fileobj has passed the content back to sniff_bytes, and the
# fallback would loop.
_SNIFF_BYTES_FALLBACK: ContextVar[frozenset] = ContextVar(
    "_SNIFF_BYTES_FALLBACK", default=frozenset()
)


class _SnifferMixin(_FileNamesMixin):
    """Default `sniff*` methods, which derive one from another.

    A class that scores files implements only its lowest entry point,
    such as `sniff_fileobj` or `sniff_line`, and this mixin derives the
    other `sniff*` methods from it. It chooses no format: [`FileReader`][]
    adds it to a reader of one format, whose `sniff*` methods return a
    confidence. A dispatcher, whose `sniff*` methods return the class
    that matched, does not inherit it.
    """

    _READ_MODE: str = "r"

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
        file = path.as_path_or_stream(file)

        if isinstance(file, path.Path):
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
        file = path.as_path_or_stream(file)

        if isinstance(file, path.Path):
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
        [`SnifferExistsError`][] if `error` is set. A compressed file is
        decompressed as it is read.
        """
        kwargs["error"] = error

        filename = path.as_path_or_stream(filename)

        if isinstance(filename, path.Path):
            if not filename.exists():
                if error:
                    if error is True:
                        error = SnifferExistsError
                    raise error(f"No such file: {filename}")
                return Confidence.NO
            with open_decompressed(filename, cls._READ_MODE) as f:
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

        Compressed content is decompressed first. The content is then
        wrapped in an `io.BytesIO` and scored with [`sniff_fileobj`][], so
        that a class that scores a stream also scores bytes. If that
        method only reads the stream back into bytes, as the default
        does, the content would come back to this method, which then
        raises instead.

        Raises
        ------
        SnifferNotImplementedError
            If neither this method nor `sniff_fileobj` is implemented.
        """
        active = _SNIFF_BYTES_FALLBACK.get()
        if cls not in active:
            token = _SNIFF_BYTES_FALLBACK.set(active | {cls})
            try:
                return cls.sniff_fileobj(
                    BytesIO(decompress(bytes(content))), error=error, **kwargs
                )
            finally:
                _SNIFF_BYTES_FALLBACK.reset(token)
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
    `super()`. [`FileReader.from_bytes`][] never falls back to a passthrough,
    because the fallback would loop.
    """
    func._passthrough_from_fileobj = True
    return func


def _delegated_from_fileobj(
    delegate: tx.Callable[[type], tx.Optional[type]],
) -> tx.Callable[[tx.Callable], tx.Callable]:
    """Mark a `from_fileobj` that hands the stream to another class.

    A metadata format that reads its header through its record class is
    an example: its `from_fileobj` reads the stream with the
    `from_fileobj` of the record class. Whether such a method reads the
    stream itself therefore depends on the class that it delegates to.
    `delegate(cls)` returns that class, or `None` when `cls` delegates to
    no class, in which case the method is a passthrough.

    Parameters
    ----------
    delegate : Callable[[type], type or None]
        The function that returns the class to which `cls` hands the
        stream.

    Returns
    -------
    Callable[[Callable], Callable]
        The decorator that marks the method.
    """

    def mark(func: tx.Callable) -> tx.Callable:
        func._delegated_from_fileobj = delegate
        return func

    return mark


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
    mixin that only forwards the call is not an override. An implementation
    marked with `_delegated_from_fileobj` counts when the class that it
    hands the stream to defines `from_fileobj`.
    """
    for klass in cls.__mro__:
        func = klass.__dict__.get("from_fileobj")
        if func is None:
            continue
        func = getattr(func, "__func__", func)
        delegate = getattr(func, "_delegated_from_fileobj", None)
        if delegate is not None:
            target = delegate(cls)
            if target is not None:
                return _overrides_from_fileobj(target)
            continue
        if not getattr(func, "_passthrough_from_fileobj", False):
            return True
    return False


class _ReaderMixin:
    """Default `load` and `from_*` methods, which derive one from another.

    A class that reads files implements only its lowest entry point, such
    as `from_fileobj` or `from_lines`, and this mixin derives the other
    `from_*` methods and `load` from it. The mixin chooses no format:
    [`FileReader`][] adds it to a reader of one format, and
    [`FileFormat`][brainhops.io.base.FileFormat] adds it to dispatchers
    and to the formats that they choose between.
    """

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """Read an object from a path, an open file, or binary content.

        Paths and open files are read with [`from_file`][], binary content
        with [`from_bytes`][], and a [`SourceSpec`][] with [`from_spec`][].
        A `str` is always a path, whether or not the file exists, so text
        held in memory must be read with [`from_text`][] or
        [`from_content`][]. The default [`from_filename`][],
        [`from_fileobj`][] and [`from_bytes`][] decompress a file or
        content that gzip, bzip2 or xz compresses, so that a format that
        does not override them reads compressed files too.

        Raises
        ------
        ParserExistsError
            If the path names a file that does not exist.
        ParserTypeError
            If the input is of another type.
        """
        if isinstance(other, SourceSpec):
            return cls.from_spec(other, **kwargs)

        other = path.as_path_or_stream(other)

        if isinstance(other, path.Path):
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
        file = path.as_path_or_stream(file)
        if isinstance(file, path.Path):
            return cls.from_filename(file, **kwargs)

        if hasattr(file, "read"):
            return cls.from_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot parse file of type {type(file)}")

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read an object from a path.

        The file is opened in the read mode of the class, and a compressed
        file is decompressed as it is read.

        Raises
        ------
        ParserExistsError
            If the file does not exist.
        """
        filename = path.Path(filename)

        if not filename.exists():
            raise ParserExistsError(f"No such file: {filename}")
        with open_decompressed(filename, cls._READ_MODE) as f:
            return cls.from_fileobj(f, **kwargs)

    @classmethod
    @_passthrough_from_fileobj
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read an object from a file object open for reading.

        The default implementation reads the whole stream and passes it to
        [`from_content`][], and [`from_bytes`][] decompresses binary content
        that is compressed. A parser that needs only part of the stream,
        such as a header, should override this method. [`from_bytes`][]
        then wraps binary content in a stream and reads it with the
        override. An override receives the stream as the caller gave it,
        compressed or not, and decompresses it itself, because a lazy array
        that keeps the stream must read it from the position where the
        file starts.
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

        Compressed content is decompressed first. If the class implements
        [`from_fileobj`][] itself, the content is
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
                return cls.from_fileobj(
                    BytesIO(decompress(bytes(content))), **kwargs
                )
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


class FileReader(_ReaderMixin, _SnifferMixin):
    """Reader of one format, with sniffing that returns a confidence score.

    A dispatcher such as [`FileFormat`][brainhops.io.base.FileFormat] uses
    the same mixins but does not inherit this class, so that
    `issubclass(cls, FileReader)` tells a reader of one format apart from
    a dispatcher.
    """


# ---- to --------------------------------------------------------------


class _WriterMixin:
    """Default `save` and `to_*` methods, which derive one from another.

    A class that writes files implements only its lowest entry point, such
    as `to_bytes` or `to_lines`, and this mixin derives the other `to_*`
    methods and `save` from it. It is the counterpart of
    [`_ReaderMixin`][], and is independent of reading and sniffing.
    """

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

        A file object needs a `write` method.

        Raises
        ------
        ParserTypeError
            If `file` is of another type.
        """
        file = path.as_path_or_stream(file)
        if isinstance(file, path.Path):
            return self.to_filename(file, **kwargs)

        if hasattr(file, "write"):
            return self.to_fileobj(file, **kwargs)

        raise ParserTypeError(f"Cannot write file of type {type(file)}")

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the object to a path, always as bytes.

        The content is encoded before the target changes, so that an
        error while it is encoded leaves the target as it was. A local
        file is written next to its target, which it then replaces at
        once. A remote file is written once the whole content is
        encoded. A text format is written with [`to_bytes`][], so that
        its line endings are written as they are, on every system. A
        binary format is written with [`to_fileobj`][], which a local
        file receives as it is written and a remote file receives once
        it is complete. A name that ends with `.gz`, `.bz2` or `.xz`
        compresses the file, whatever the format.
        """
        filename = path.Path(filename)
        suffix = compression_suffix(path.base_name(filename))

        def encode(file: tx.BinaryIO) -> None:
            with open_compressing(file, suffix) as stream:
                if "b" in self._WRITE_MODE:
                    self.to_fileobj(stream, **kwargs)
                else:
                    stream.write(self.to_bytes(**kwargs))

        if not filename.is_local():
            buffer = BytesIO()
            encode(buffer)
            filename.write_bytes(buffer.getvalue())
            return

        def write(name: path.Path) -> None:
            with name.open("xb") as f:
                encode(f)

        replace_file(filename, write)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the object to a file object open for writing.

        A binary format writes the result of [`to_bytes`][]. A text
        format writes the result of [`to_text`][] to a text stream, such
        as a file opened with `"w"` or an `io.StringIO`, and the result
        of [`to_bytes`][] to any other stream, such as a file opened with
        `"wb"` or an `io.BytesIO`.
        """
        if "b" not in self._WRITE_MODE and isinstance(file, TextIOBase):
            file.write(self.to_text(**kwargs))
        else:
            file.write(self.to_bytes(**kwargs))

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

        The text is made of the lines of [`to_lines`][], each followed by
        a newline.
        """
        return "".join(line + "\n" for line in self.to_lines(**kwargs))

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


class FileWriter(_WriterMixin):
    """Writer of one format.

    The writing methods come from [`_WriterMixin`][], as the reading
    methods of [`FileReader`][] come from [`_ReaderMixin`][]. A format
    can write files when `issubclass(cls, FileWriter)` holds, which is
    how [`save`][brainhops.io.save] chooses the formats that it writes
    with.
    """


# ----------------------------------------------------------------------
#   TEXT
# ----------------------------------------------------------------------


class _TextSnifferMixin(_SnifferMixin):
    """Default `sniff*` methods for a text format, which decode bytes."""

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

        Compressed content is decompressed first. The content is decoded
        with the `encoding` keyword, UTF-8 by default, and passed to
        `sniff_text`. Content that does not decode scores
        [`Confidence.NO`][], or raises [`SnifferContentError`][] if `error` is
        set.
        """
        kwargs["error"] = error
        encoding = kwargs.pop("encoding", "utf-8")
        try:
            text = decompress(bytes(content)).decode(encoding)
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


class _TextReaderMixin(_ReaderMixin):
    """Default `from_*` methods for a text format, which decode bytes."""

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read an object from bytes decoded as text.

        Compressed content is decompressed first. The `encoding` keyword
        selects the encoding, UTF-8 by default.
        """
        encoding = kwargs.pop("encoding", "utf-8")
        text = decompress(bytes(content)).decode(encoding)
        return cls.from_text(text, **kwargs)


class TextFileReader(_TextReaderMixin, _TextSnifferMixin, FileReader):
    """Read one text format and score its content."""


class TextFileWriter(FileWriter):
    """Output adapters for text, without any reading methods."""

    def to_bytes(self, **kwargs) -> bytes:
        """Return the text of the file, encoded.

        The `encoding` keyword selects the encoding, UTF-8 by default.
        """
        encoding = kwargs.pop("encoding", "utf-8")
        return self.to_text(**kwargs).encode(encoding)


# ----------------------------------------------------------------------
#   BINARY
# ----------------------------------------------------------------------


class BinaryFileReader(FileReader):
    """Class that can read binary files of its type."""

    _READ_MODE: str = "rb"


class BinaryFileWriter(FileWriter):
    """Output adapters for binary files, without any reading methods."""

    _WRITE_MODE: str = "wb"
