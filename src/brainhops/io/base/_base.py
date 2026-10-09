__all__ = [
    "format_registry",
    "FormatDispatcher",
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
from bagof.magic import Field, fields

# internals
from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._dispatch import Source, parse, sniff
from brainhops.io.base.parsers import (
    BinaryFileParser,
    BinaryFileParserWriter,
    FileParser,
    FileParserWriter,
    TextFileParser,
    TextFileParserWriter,
    _passthrough_from_fileobj,
)
from brainhops.io.base.specs import SourceSpec

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


class FormatDispatcher(FileParser):
    """
    The dispatching behaviour of a class that owns a registry of formats.

    A class decorated with `@format_registry` dispatches: its `sniff*`
    methods identify the registered format that would read a file, and
    its `load` and `from_*` methods pick that format and delegate to it.
    On a class that owns no registry, a concrete format, every method
    behaves as the parser method it overrides.

    This mixin owns no registry itself, so that a kind of object that
    the generic `load` must not return, such as the metadata of a file
    (`FileBasedMetadata`), can dispatch among its own formats without being
    registered into the registry of [`FileBasedObject`][].

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
        """On a dispatcher, identify which registered format would read
        `file`. On a concrete format, score how confident it is that
        `file`, in any supported form, is its own."""
        if not cls._is_dispatcher():
            return super().sniff(file, error=error, **kwargs)
        return sniff(
            Source(file), cls._REGISTRY, "sniff", error, f"{file}", **kwargs
        )

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """On a dispatcher, identify which registered format would read
        the file (path or file-like object). On a concrete format, score
        how confident it is that the file is its own."""
        if not cls._is_dispatcher():
            return super().sniff_file(file, error=error, **kwargs)
        return sniff(
            Source(file),
            cls._REGISTRY,
            "sniff_file",
            error,
            f"file: {file}",
            **kwargs,
        )

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """On a dispatcher, identify which registered format would read
        the open file object. On a concrete format, score how confident
        it is that the file object is its own."""
        if not cls._is_dispatcher():
            return super().sniff_fileobj(file, error=error, **kwargs)
        return sniff(
            Source(file),
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
        """On a dispatcher, identify which registered format would read
        the content (text or bytes). On a concrete format, score how
        confident it is that the content is its own."""
        if not cls._is_dispatcher():
            return super().sniff_content(content, error=error, **kwargs)
        return sniff(
            Source.content(content),
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
        """On a dispatcher, identify which registered format would read
        the bytes. On a concrete format, score how confident it is that
        the bytes are its own."""
        if not cls._is_dispatcher():
            return super().sniff_bytes(content, error=error, **kwargs)
        return sniff(
            Source.content(content),
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
        """On a dispatcher, identify which registered format would read
        the text. On a concrete format, score how confident it is that
        the text is its own."""
        if not cls._is_dispatcher():
            return super().sniff_text(text, error=error, **kwargs)
        return sniff(
            Source.content(text),
            cls._REGISTRY,
            "sniff_text",
            error,
            "input text",
            **kwargs,
        )

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """On a dispatcher, identify which registered format would read
        the lines. On a concrete format, score how confident it is that
        the lines are its own."""
        if not cls._is_dispatcher():
            return super().sniff_lines(lines, error=error, **kwargs)
        return sniff(
            Source.content(lines),
            cls._REGISTRY,
            "sniff_lines",
            error,
            "input lines",
            **kwargs,
        )

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """On a dispatcher, identify which registered format would read
        the line. On a concrete format, score how confident it is that
        the line is its own."""
        if not cls._is_dispatcher():
            return super().sniff_line(line, error=error, **kwargs)
        return sniff(
            Source.content(line),
            cls._REGISTRY,
            "sniff_line",
            error,
            "input line",
            **kwargs,
        )

    # ---- from --------------------------------------------------------

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from `other`. On a concrete format,
        build an instance of this class from `other`, in any supported
        form."""
        if isinstance(other, SourceSpec):
            return cls.from_spec(other, **kwargs)
        if not cls._is_dispatcher():
            return super().load(other, **kwargs)
        return parse(Source(other), cls._REGISTRY, "load", "sniff", **kwargs)

    @classmethod
    def from_spec(cls, spec: SourceSpec, **kwargs) -> tx.Self:
        """Load a structured source specification through this dispatcher."""
        if not cls._is_dispatcher():
            return super().from_spec(spec, **kwargs)
        return parse(
            Source(spec.path),
            cls._REGISTRY,
            "load",
            "sniff",
            hints=spec.hints,
            options=spec.options,
            **kwargs,
        )

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the file (path or file-like
        object). On a concrete format, build an instance of this class
        from the file."""
        if not cls._is_dispatcher():
            return super().from_file(file, **kwargs)
        return parse(
            Source(file),
            cls._REGISTRY,
            "from_file",
            "sniff_file",
            **kwargs,
        )

    @classmethod
    @_passthrough_from_fileobj
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the open file object. On a concrete
        format, build an instance of this class from the file object."""
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
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the content (text or bytes). On a
        concrete format, build an instance of this class from the
        content."""
        if not cls._is_dispatcher():
            return super().from_content(content, **kwargs)
        return parse(
            Source.content(content),
            cls._REGISTRY,
            "from_content",
            "sniff_content",
            **kwargs,
        )

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the bytes. On a concrete format,
        build an instance of this class from the bytes."""
        if not cls._is_dispatcher():
            return super().from_bytes(content, **kwargs)
        return parse(
            Source.content(content),
            cls._REGISTRY,
            "from_bytes",
            "sniff_bytes",
            **kwargs,
        )

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the text. On a concrete format,
        build an instance of this class from the text."""
        if not cls._is_dispatcher():
            return super().from_text(text, **kwargs)
        return parse(
            Source.content(text),
            cls._REGISTRY,
            "from_text",
            "sniff_text",
            **kwargs,
        )

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the lines. On a concrete format,
        build an instance of this class from the lines."""
        if not cls._is_dispatcher():
            return super().from_lines(lines, **kwargs)
        return parse(
            Source.content(lines),
            cls._REGISTRY,
            "from_lines",
            "sniff_lines",
            **kwargs,
        )

    @classmethod
    def from_line(cls, line: str, **kwargs) -> tx.Self:
        """On a dispatcher, pick the best-matching registered format and
        build an instance of it from the line. On a concrete format,
        build an instance of this class from the line."""
        if not cls._is_dispatcher():
            return super().from_line(line, **kwargs)
        return parse(
            Source.content(line),
            cls._REGISTRY,
            "from_line",
            "sniff_line",
            **kwargs,
        )


@format_registry
class FileBasedObject(FormatDispatcher):
    """
    An object that is stored in a file.

    Subclasses decorated with `@format_registry` become dispatchers for
    their own kind of object (images, transformations, ...). Concrete
    parsers decorated with `@register_format` land in *every* ancestor
    registry, so both the scoped and the generic entry points see them.
    The dispatching itself is described in `FormatDispatcher`.
    """

    # FIXME: `FileBasedObject` inherits from `FormatDispatcher` but not
    # from `FileParser`, whereas `WritableFileBasedObject` inherits from
    # `FileParserWriter`. This is asymmetric and counter inttuitive.


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


# ----------------------------------------------------------------------
#   DATA MODELS
# ----------------------------------------------------------------------


class _FileBasedModelMixin:
    """
    Gives a file-based data model a `from_any` that reads files.

    A data model's `from_any` builds an instance from a mapping, from
    an instance of a similar class, or from constructor arguments. A
    file-based one can also be built from the file that stores it, so
    this mixin tries that first: a path, an open file, bytes or a
    structured source are read with `load`, and everything else is left
    to the data model.

    It sits where the data models meet the file formats, ahead of the
    data model in the bases (`FileBasedImage`, `FileBasedTransformation`),
    so that the data models themselves never deal with files.

    !!! note "Why this is not part of `FileBasedObject`"
        To win, this `from_any` must come before
        `DataModelBase.from_any` in the MRO, and `FileBasedObject`
        comes after `DataModelBase` in the MRO of every file-based class.
        Nor can it be moved ahead. A format's parser base declares
        itself a data model first, as in
        `NiftiParser(DataModelBase, BinaryFileParserWriter)` and
        `ZarrParser(DataModelBase, FileParser)`. Its parser chain then
        leads, through `FileParserWriter`, to `FileBasedObject`. Listing
        `FileBasedObject` first in `FileBasedImage`'s bases therefore
        makes the MRO of `NiftiImage` inconsistent: "Cannot create a
        consistent method resolution order (MRO) for bases
        DataModelBase, FileParserWriter, Image".

        Listing the data models last everywhere does give a consistent
        MRO, with `FileBasedObject` ahead of `DataModelBase`. That means
        reordering the parser bases (`NiftiParser`, `ZarrParser`, the
        ITK and FLIRT parsers), the dispatchers and the formats. But the
        file machinery is itself a generic data model
        (`FileBasedTransformation` is a `Transformation`). Moved ahead of
        the specific data model a format refines, its declarations of a
        field shadow the specific ones:

        - `NiftiVoxelToRAS`, `NiftiRASToVoxel` and
          `NiftiRASCoordinatesField` lose their voxel and RAS endpoint
          defaults.
        - The ITK formats no longer take their chain as their first
          positional argument.

        A mixin that is not a parser has none of these constraints, and
        goes ahead of the data model.

    !!! note "Every string is a file"
        No file-based class takes a string as its first constructor
        argument, so a string always names a file, or a store, and is
        read with `load`. A file that cannot be read fails there, rather
        than being handed to the constructor as if it were data.
    """

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance from a file, or from anything the data model
        reads.

        A path (`str` or `os.PathLike`), an open file, `bytes` or a
        structured source ([`SourceSpec`][brainhops.io.base.SourceSpec])
        is read with `load`: on a dispatcher such as `FileBasedImage`,
        the best-matching registered format reads it, and on a concrete
        format, that format does. Any other value is handed to the data
        model's own `from_any`, which reads a mapping field by field,
        copies an instance of a similar class, and passes anything else
        to the constructor.

        Parameters
        ----------
        other : Any
            A file, its content, a mapping, or an instance of a similar
            class.
        *args
            Constructor arguments. A file is read with keyword options
            only.
        **kwargs
            Format-specific options when reading a file, and field
            values otherwise.

        Returns
        -------
        obj
            The object that was built.

        Raises
        ------
        TypeError
            If positional arguments come with a file to read.
        """
        if not _is_file_or_content(other):
            return super().from_any(other, *args, **kwargs)
        if args:
            raise TypeError(
                f"{cls.__name__}.from_any() reads a file with keyword "
                f"options only, but was given {len(args)} positional "
                f"argument(s)."
            )
        return cls.load(other, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance from an instance of a similar class.

        The data model copies the fields both classes share, by name.
        A field that a file format declares for its own use -- such as
        the `nibabel` `image` and `header` of the NIfTI and MGH formats
        -- is only copied from an object of that same format: from any
        other object, a field of the same name holds something else
        (a NIfTI image is no MGH image), so this class's default is
        kept instead. Saving a NIfTI image to MGH, or the converse,
        therefore converts the data model only, and the format-specific
        state is rebuilt by the writer.
        """
        for field in _foreign_format_fields(cls, other):
            if field.factory is True:
                default = field.default()
            else:
                default = field.default
            kwargs.setdefault(field.public_name, default)
        return super().from_instance(other, *args, **kwargs)


def _foreign_format_fields(cls: type, other: tx.Any) -> tx.List[tx.Any]:
    """
    The fields of `cls` that only file formats declare, and that `other`
    does not inherit from any of the formats that declare them.

    A field is format-specific when every data model of the MRO that
    declares it is a file parser; one that a plain data model declares
    too (`data`, `transformations`, ...) is shared by every format.
    """
    if isinstance(other, cls):
        return []
    shared: tx.Set[str] = set()
    owners: tx.Dict[str, tx.List[type]] = {}
    for klass in cls.__mro__:
        if not (isinstance(klass, type) and issubclass(klass, DataModelBase)):
            continue
        names = [field.name for field in fields(klass)]
        if issubclass(klass, FileParser):
            for name in names:
                owners.setdefault(name, []).append(klass)
        else:
            shared.update(names)
    no_default = Field().default
    return [
        field
        for field in fields(cls)
        if field.init
        and field.kw
        and field.name not in shared
        and field.default is not no_default
        and not any(isinstance(other, k) for k in owners.get(field.name, ()))
    ]


def _is_file_or_content(other: tx.Any) -> bool:
    """Whether `load` reads `other` as a file or as file content."""
    return isinstance(
        other, (str, path.PathLike, bytes, bytearray, SourceSpec)
    ) or hasattr(other, "read")
