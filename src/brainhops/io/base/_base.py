__all__ = [
    "format_registry",
    "Format",
    "FormatDispatcher",
    "register_format",
    "FileBasedObject",
    "WritableFileBasedObject",
    "TextFileBasedObject",
    "BinaryFileBasedObject",
    "WritableTextFileBasedObject",
    "WritableBinaryFileBasedObject",
]

import typing_extensions as tx
from bagof.magic import Field, fields

from brainhops._core import path
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base._dispatch import Source, parse, sniff
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    FileReader,
    FileWriter,
    TextFileReader,
    TextFileWriter,
    _passthrough_from_fileobj,
)
from brainhops.io.base.specs import SourceSpec

_T = tx.TypeVar("_T")


# ----------------------------------------------------------------------
#   REGISTRATION
# ----------------------------------------------------------------------


def format_registry(cls: tx.Type[_T]) -> tx.Type[_T]:
    """Give a class its own format registry, which makes it a dispatcher.

    A dispatcher is a class that does not read a format itself but chooses
    among the formats registered under it. Its `sniff*` methods return the
    best-scoring registered format, and its `load` and `from_*` methods choose
    that format and delegate to it. The decorator is applied to the root
    [`FileBasedObject`][] and to the base class of each kind of object, such
    as `FileBasedImage` and `FileBasedTransformation`. A dispatcher is never
    added to the registries of its ancestors; only classes decorated with
    [`register_format`][] are.
    """
    # Registration uses a decorator rather than a metaclass because data
    # models already use the bagof.magic metaclass. A derived metaclass would
    # also run on the throwaway classes that bagof builds to compute MROs, and
    # class keywords do not work either, because that metaclass does not
    # forward them to __init_subclass__.

    # The registry is set in cls.__dict__ so that a class that owns a registry
    # can be told apart from a class that inherits one.
    cls._REGISTRY = set()
    return cls


def register_format(cls: tx.Type[_T]) -> tx.Type[_T]:
    """Register a concrete format into the registry of every ancestor.

    A format such as `NiftiImage` is thereby found both by `images.load` and by
    the generic [`load`][brainhops.io.base.load]. Registration is idempotent,
    so re-importing a module is harmless.

    !!! note "Dispatchers are not formats"
        Registries are flat. A format is in the registry of every ancestor,
        whereas a dispatcher is in no registry at all, so each format is tried
        once per load. If a dispatcher were registered, its formats would be
        tried twice, once directly and once through the dispatcher, and
        `FileBasedObject.load` would recurse into itself.

    !!! note "Order does not matter"
        A registry is an unordered set, because registration follows import
        order, which is not stable. Dispatch never relies on registration
        order: candidates that cannot be told apart raise
        [`AmbiguousFormatError`][brainhops.errors.AmbiguousFormatError].

    Raises
    ------
    TypeError
        If `cls` owns a registry.
    """
    if "_REGISTRY" in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} owns a registry (@format_registry), so it "
            f"dispatches to formats rather than being one; it must not "
            f"also be decorated with @register_format."
        )
    for base in cls.__mro__[1:]:
        # The test uses __dict__ rather than hasattr, because hasattr would
        # also see registries that `base` merely inherits.
        if "_REGISTRY" in base.__dict__:
            base._REGISTRY.add(cls)
    return cls


# ----------------------------------------------------------------------
#   BASE
# ----------------------------------------------------------------------


@format_registry
class Format:
    """Public format membership, independent of reading and writing.

    Concrete public formats register here with `register_format`. Reading
    public objects inherit `FileBasedObject`; writing adds `FileWriter`.
    Generic saving selects writers from this registry. Generic loading uses
    the narrower `FileBasedObject` registry, so write-only formats and
    standalone metadata readers are not candidates. Native parser
    representations do not inherit this class.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = ()
    PREFIXES: tx.ClassVar[tx.Tuple[str, ...]] = ()
    PRIORITY: tx.ClassVar[int] = 0


class FormatDispatcher(FileReader):
    """Dispatch between the formats registered under a dispatcher.

    On a class decorated with [`format_registry`][], the `sniff*` methods
    identify the registered format that matches the input, and `load` and the
    `from_*` methods choose a format and delegate to it. On a class without a
    registry, which is a concrete format, every method behaves as the
    [`FileReader`][] method it overrides.

    The mixin owns no registry itself. A kind that the generic `load` must
    never return, such as `FileBasedMetadata`, can therefore dispatch between
    its own formats without joining the registry of [`FileBasedObject`][].

    !!! note "`sniff*` means something different on a dispatcher"
        A concrete format returns its confidence, between 0 and 1, that the
        input is in its format. A dispatcher has already compared those
        confidences, so it returns which format matched, or `None`. That format
        can be asked to score the input itself.
    """

    # ---- helpers -----------------------------------------------------

    @classmethod
    def _is_dispatcher(cls) -> bool:
        return "_REGISTRY" in cls.__dict__

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """Identify the format of a file or its content.

        A dispatcher returns the registered format that would read the input,
        and a concrete format scores the input.
        """
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
        """Same as [`sniff`][], for a path or an open file."""
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
        """Same as [`sniff`][], for an open file object."""
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
        """Same as [`sniff`][], for text or binary content."""
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
        """Same as [`sniff`][], for binary content."""
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
        """Same as [`sniff`][], for text."""
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
        """Same as [`sniff`][], for lines of text."""
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
        """Same as [`sniff`][], for a single line."""
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
        """Read an object from a file or its content.

        A dispatcher reads the input with the best-matching registered format,
        and a concrete format reads it as an instance of its own class. A
        [`SourceSpec`][] is read with [`from_spec`][].
        """
        if isinstance(other, SourceSpec):
            return cls.from_spec(other, **kwargs)
        if not cls._is_dispatcher():
            return super().load(other, **kwargs)
        return parse(Source(other), cls._REGISTRY, "load", "sniff", **kwargs)

    @classmethod
    def from_spec(cls, spec: SourceSpec, **kwargs) -> tx.Self:
        """Read a structured source specification.

        The specification's path is read with its hints and options.
        """
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
        """Read an object from a path or an open file.

        A dispatcher reads the file through the best-matching format.
        """
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
        """Same as [`from_file`][], for an open file object."""
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
        """Same as [`from_file`][], for text or binary content."""
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
        """Same as [`from_file`][], for binary content."""
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
        """Same as [`from_file`][], for text."""
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
        """Same as [`from_file`][], for lines of text."""
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
        """Same as [`from_file`][], for a single line."""
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
class FileBasedObject(FormatDispatcher, Format):
    """Root of all objects stored in files.

    Subclasses decorated with [`format_registry`][] dispatch between the
    formats of their kind, such as images or transformations. Formats decorated
    with [`register_format`][] are added to every ancestor registry, so both
    the scoped and the generic entry points find them. [`FormatDispatcher`][]
    describes the dispatch.
    """


@format_registry
class WritableFileBasedObject(FileBasedObject, FileWriter):
    """Compatibility base for public formats that both read and write.

    New formats may combine `FileBasedObject` and `FileWriter` directly.
    A write-only public format combines `Format` and `FileWriter` instead.
    """


@format_registry
class TextFileBasedObject(TextFileReader, FileBasedObject):
    """Object stored in a text file."""


@format_registry
class BinaryFileBasedObject(BinaryFileReader, FileBasedObject):
    """Object stored in a binary file."""


@format_registry
class WritableTextFileBasedObject(
    TextFileReader, TextFileWriter, WritableFileBasedObject
):
    """Object stored in a writable text file."""


@format_registry
class WritableBinaryFileBasedObject(
    BinaryFileReader, BinaryFileWriter, WritableFileBasedObject
):
    """Object stored in a writable binary file."""


# ----------------------------------------------------------------------
#   DATA MODELS
# ----------------------------------------------------------------------


class _FileBasedModelMixin:
    """Give file-based data models a `from_any` that reads files first.

    A path, an open file, binary content or a [`SourceSpec`][] is read with
    `load`, and any other value is passed to the data model. The mixin is
    listed before the data model in the bases of `FileBasedImage` and
    `FileBasedTransformation`, so that data models never deal with files.

    !!! note "Why this is not part of `FileBasedObject`"
        To take effect, this `from_any` must precede `DataModelBase.from_any`
        in the MRO, but `FileBasedObject` comes after the data model in every
        file-based class. Parser bases declare the data model first, as in
        `NiftiParser(DataModelBase, BinaryFileParser, BinaryFileWriter)`.
        Their chain leads to `FileBasedObject`, so listing it first makes
        the MRO of `NiftiImage` inconsistent. Listing data models last would
        require reordering every parser, dispatcher and format, and the file
        machinery's own field declarations would then shadow the specific ones:
        for example, `NiftiVoxelToRAS` would lose its voxel and RAS endpoint
        defaults. A mixin that is not a parser has no such constraints.

    !!! note "Every string is a file"
        No file-based class takes a string as the first constructor argument,
        so a string always names a file or a store. A file that cannot be read
        fails in `load`, rather than being passed to the constructor as data.
    """

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build an object from a file or any value the data model takes.

        A path, an open file (any object with a `read` method), `bytes`,
        `bytearray` or a [`SourceSpec`][] is read with `load`. A dispatcher
        then chooses the best format, and a concrete format reads the input as
        an instance of its own class. Any other value is passed to the
        `from_any` of the data model, which maps a mapping field by field,
        copies an instance of a similar class, or calls the constructor.

        Parameters
        ----------
        other : object
            A file, its content, a mapping, or an instance of a similar class.
        *args : Any
            Constructor arguments. A file is read with keyword options only.
        **kwargs : Any
            Format options when reading a file, and field values otherwise.

        Raises
        ------
        TypeError
            If positional arguments accompany a file.
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
        """Copy an instance, resetting fields that belong to another format.

        The data model copies the fields of `other` by name. Fields declared by
        a file format, such as the nibabel image and header of NIfTI and MGH,
        are copied only from an object of the same format; otherwise the class
        default is kept, because the same field name holds something different.
        Converting NIfTI to MGH on save therefore converts the data model only,
        and the writer rebuilds the format state.
        """
        for field in _foreign_format_fields(cls, other):
            if field.factory is True:
                default = field.default()
            else:
                default = field.default
            kwargs.setdefault(field.public_name, default)
        return super().from_instance(other, *args, **kwargs)


def _foreign_format_fields(cls: type, other: tx.Any) -> tx.List[tx.Any]:
    """Return the fields of `cls` that belong to a different file format.

    A field belongs to file formats when, in the MRO of `cls`, only file
    parsers declare it. A field that a plain data model also declares is
    shared by all formats and is never returned. A format field is
    also left out when `other` is an instance of one of the formats that
    declare it, because `other` then holds a meaningful value for that field.
    Only keyword fields that the constructor accepts and that have a default
    are returned.
    """
    if isinstance(other, cls):
        return []
    shared: tx.Set[str] = set()
    owners: tx.Dict[str, tx.List[type]] = {}
    for klass in cls.__mro__:
        if not (isinstance(klass, type) and issubclass(klass, DataModelBase)):
            continue
        names = [field.name for field in fields(klass)]
        if issubclass(klass, (Format, FileReader, FileWriter)):
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
    return isinstance(
        other, (str, path.PathLike, bytes, bytearray, SourceSpec)
    ) or hasattr(other, "read")
