__all__ = [
    "format_registry",
    "register_format",
    "FileBasedObject",
    "WritableFileBasedObject",
    "TextFileBasedObject",
    "BinaryFileBasedObject",
    "WritableTextFileBasedObject",
    "WritableBinaryFileBasedObject",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.io.base._dispatch import Source, parse, sniff
from brainhops.io.base.parsers import (
    BinaryFileParser,
    BinaryFileParserWriter,
    FileParser,
    FileParserWriter,
    TextFileParser,
    TextFileParserWriter,
)

_T = tx.TypeVar("_T")


# ----------------------------------------------------------------------
#   REGISTRATION
# ----------------------------------------------------------------------


def format_registry(cls: tx.Type[_T]) -> tx.Type[_T]:
    """
    Give a class its own registry of file formats.

    A class decorated with `@format_registry` becomes a *dispatcher*:
    `sniff*` reports the best score among its registered formats, and
    `load`/`from_*` pick the best one and delegate to it. Use it on the
    base class of a kind of object -- `FileBasedImage`,
    `FileBasedTransformation` -- and on the root, `FileBasedObject`.

    A dispatcher is not itself a format, so it is never registered into
    its ancestors' registries; only `@register_format` classes are.
    """
    # NOTE *Why a decorator rather than a metaclass*
    #   The objects being parsed are also data models, whose metaclass
    #   comes from `bagof.magic`. A registry metaclass would have to
    #   derive from it to avoid a metaclass conflict, which in turn makes
    #   it run for the throwaway classes `bagof` builds while computing
    #   MROs. Class *keywords* are no better: `bagof`'s metaclass does not
    #   forward them to `__init_subclass__`, so `register=False` would be
    #   silently ignored. Decorators are independent of the metaclass, so
    #   they compose with any data model, and they make registration
    #   visible at the class that opts into it.

    # Assign to `cls.__dict__`, so that `_is_dispatcher` can tell a class
    # that *owns* a registry from one that merely inherits one.
    cls._REGISTRY = set()
    return cls


def register_format(cls: tx.Type[_T]) -> tx.Type[_T]:
    """
    Register a format into the registries of all its ancestors.

    A concrete reader is registered into *every* registry above it, so a
    `NiftiImage` is found both by `images.load` and by the generic
    `load`. Registering twice is a no-op, so re-importing is harmless.

    !!! note "Dispatchers are not formats"
        Registries are *flat*. A concrete format is registered into every
        registry above it, and dispatchers are registered into none, so
        each format is tried exactly once per `load`. Registering a
        dispatcher as a format would break that: every format below it
        would be tried twice -- once directly, once through the
        dispatcher -- and `FileBasedObject.load` would recurse into
        itself, since the dispatcher's registry would contain a class
        whose `load` re-enters dispatch.

    !!! note "Order does not matter"
        The registry is an unordered `set`, deliberately: registration
        order is import order, which is neither stable nor meaningful.
        Dispatch never falls back on it -- candidates that cannot be
        separated on merit raise `AmbiguousFormatError` instead.

    Parameters
    ----------
    cls : type
        The format to register.

    Returns
    -------
    cls : type
        The same class, unchanged.

    Raises
    ------
    TypeError
        If `cls` owns a registry of its own.
    """
    if "_REGISTRY" in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} owns a registry (@format_registry), so it "
            f"dispatches to formats rather than being one; it must not "
            f"also be decorated with @register_format."
        )
    for base in cls.__mro__[1:]:  # skip `cls` itself
        # `__dict__`, not `hasattr`: `hasattr` would also match registries
        # merely inherited by `base`, and we would be mutating whichever
        # ancestor's registry happened to be inherited.
        if "_REGISTRY" in base.__dict__:
            base._REGISTRY.add(cls)
    return cls


# ----------------------------------------------------------------------
#   BASE
# ----------------------------------------------------------------------


@format_registry
class FileBasedObject(FileParser):
    """
    An object that is stored in a file.

    Subclasses decorated with `@format_registry` become dispatchers for
    their own kind of object (images, transformations, ...). Concrete
    parsers decorated with `@register_format` land in *every* ancestor
    registry, so both the scoped and the generic entry points see them.

    !!! note "`sniff*` means something different on a dispatcher"
        A concrete format's `sniff*` answers "how confident am I that
        this is mine", a number in `[0, 1]`. A dispatcher has already
        compared those numbers, so it answers the question they were
        being compared to settle: *which* format is it, returning the
        class (or `None`). Ask that class to score itself if the number
        is what you wanted.

    On a dispatcher, `load`/`from_*` pick that same best parser and
    delegate to it. On a concrete parser, everything behaves as usual.
    """

    # ---- helpers -----------------------------------------------------

    @classmethod
    def _is_dispatcher(cls) -> bool:
        """Whether this class dispatches to a registry of its own."""
        return "_REGISTRY" in cls.__dict__

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff(file, error=error, **kwargs)
        return sniff(file, cls._REGISTRY, "sniff", error, f"{file}", **kwargs)

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_file(file, error=error, **kwargs)
        return sniff(
            file, cls._REGISTRY, "sniff_file", error, f"file: {file}", **kwargs
        )

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_fileobj(file, error=error, **kwargs)
        return sniff(
            file,
            cls._REGISTRY,
            "sniff_fileobj",
            error,
            f"file object: {file}",
            **kwargs,
        )

    @classmethod
    def sniff_content(
        cls,
        content: path.ContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_content(content, error=error, **kwargs)
        return sniff(
            content,
            cls._REGISTRY,
            "sniff_content",
            error,
            "input content",
            **kwargs,
        )

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_bytes(content, error=error, **kwargs)
        return sniff(
            content,
            cls._REGISTRY,
            "sniff_bytes",
            error,
            "input content",
            **kwargs,
        )

    @classmethod
    def sniff_text(
        cls,
        text: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_text(text, error=error, **kwargs)
        return sniff(
            text, cls._REGISTRY, "sniff_text", error, "input text", **kwargs
        )

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_lines(lines, error=error, **kwargs)
        return sniff(
            lines, cls._REGISTRY, "sniff_lines", error, "input lines", **kwargs
        )

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        if not cls._is_dispatcher():
            return super().sniff_line(line, error=error, **kwargs)
        return sniff(
            line, cls._REGISTRY, "sniff_line", error, "input line", **kwargs
        )

    # ---- from --------------------------------------------------------

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().load(other, **kwargs)
        return parse(Source(other), cls._REGISTRY, "load", "sniff", **kwargs)

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_file(file, **kwargs)
        return parse(
            Source(file), cls._REGISTRY, "from_file", "sniff_file", **kwargs
        )

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_fileobj(file, **kwargs)
        return parse(
            Source(file),
            cls._REGISTRY,
            "from_fileobj",
            "sniff_fileobj",
            **kwargs,
        )

    @classmethod
    def from_content(cls, content: path.ContentLike, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_content(content, **kwargs)
        return parse(
            Source(content),
            cls._REGISTRY,
            "from_content",
            "sniff_content",
            **kwargs,
        )

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_bytes(content, **kwargs)
        return parse(
            Source(content),
            cls._REGISTRY,
            "from_bytes",
            "sniff_bytes",
            **kwargs,
        )

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_text(text, **kwargs)
        return parse(
            Source(text),
            cls._REGISTRY,
            "from_text",
            "sniff_text",
            **kwargs,
        )

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_lines(lines, **kwargs)
        return parse(
            Source(lines),
            cls._REGISTRY,
            "from_lines",
            "sniff_lines",
            **kwargs,
        )

    @classmethod
    def from_line(cls, line: str, **kwargs) -> tx.Self:
        if not cls._is_dispatcher():
            return super().from_line(line, **kwargs)
        return parse(
            Source(line),
            cls._REGISTRY,
            "from_line",
            "sniff_line",
            **kwargs,
        )


@format_registry
class WritableFileBasedObject(FileParserWriter, FileBasedObject):
    """An object that is stored in a file and can be written back."""


@format_registry
class TextFileBasedObject(TextFileParser, FileBasedObject):
    """An object that is stored in a text file."""


@format_registry
class BinaryFileBasedObject(BinaryFileParser, FileBasedObject):
    """An object that is stored in a binary file."""


@format_registry
class WritableTextFileBasedObject(
    TextFileParserWriter, WritableFileBasedObject
):
    """An object that is stored in a text file and can be written back."""


@format_registry
class WritableBinaryFileBasedObject(
    BinaryFileParserWriter, WritableFileBasedObject
):
    """An object that is stored in a binary file and can be written back."""
