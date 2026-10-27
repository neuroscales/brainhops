__all__ = [
    "format_registry",
    "FileFormat",
    "register_format",
]

import typing_extensions as tx
from bagof.magic import Field, fields

from brainhops._core import backends, path
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import Image
from brainhops.io.base._dispatch import Source, parse, sniff
from brainhops.io.base.parsers import (
    FileReader,
    FileWriter,
    _FileNamesMixin,
    _passthrough_from_fileobj,
    _ReaderMixin,
    _SnifferMixin,
    _TextReaderMixin,
    _WriterMixin,
)
from brainhops.io.base.specs import SourceSpec

_T = tx.TypeVar("_T")


# ----------------------------------------------------------------------
#   REGISTRATION
# ----------------------------------------------------------------------


def format_registry(
    cls: tx.Optional[tx.Type[_T]] = None, *, isolated: bool = False
) -> tx.Any:
    """Give a class its own format registry, which makes it a dispatcher.

    A dispatcher is a class that does not read a format itself but chooses
    among the formats registered under it. Its `sniff*` methods return the
    best-scoring registered format, and its `load` and `from_*` methods choose
    that format and delegate to it. The decorator is applied to the root
    [`FileFormat`][] and to the base class of each kind of object, such
    as `ImageFormat` and `TransformationFormat`. A dispatcher is never
    added to the registries of its ancestors; only classes decorated with
    [`register_format`][] are.

    Set `isolated=True` for a standalone family such as metadata. Its
    concrete formats register within that family and nested registries,
    but registration stops before reaching its ancestors.
    """
    if cls is None:
        return lambda cls: format_registry(cls, isolated=isolated)

    # Registration uses a decorator rather than a metaclass because data
    # models already use the bagof.magic metaclass. A derived metaclass would
    # also run on the throwaway classes that bagof builds to compute MROs, and
    # class keywords do not work either, because that metaclass does not
    # forward them to __init_subclass__.

    # The registry is set in cls.__dict__ so that a class that owns a registry
    # can be told apart from a class that inherits one.
    cls._REGISTRY = set()
    cls._REGISTRY_ISOLATED = isolated
    return cls


def is_format_registry(cls: type) -> bool:
    """Tell whether a class owns a format registry.

    A class owns a registry when it is decorated with
    [`format_registry`][], which makes it a dispatcher. A format registered
    under that class inherits the registry attribute but does not own it,
    so the test looks in the class's own `__dict__` rather than using
    `hasattr`.

    Parameters
    ----------
    cls : type
        The class to test.

    Returns
    -------
    bool
        Whether `cls` was decorated with [`format_registry`][].
    """
    return "_REGISTRY" in cls.__dict__


def register_format(cls: tx.Type[_T]) -> tx.Type[_T]:
    """Register a concrete format into ancestor registries up to isolation.

    A format such as `NiftiImage` is thereby found both by `images.load` and by
    the generic [`load`][brainhops.io.base.load]. Registration is idempotent,
    so re-importing a module is harmless.

    !!! note "Dispatchers are not formats"
        Registries are flat. A format is in the registry of every ancestor,
        whereas a dispatcher is in no registry at all, so each format is tried
        once per load. If a dispatcher were registered, its formats would be
        tried twice, once directly and once through the dispatcher, and
        `FileFormat.load` would recurse into itself.

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
    if is_format_registry(cls):
        raise TypeError(
            f"{cls.__name__} owns a registry (@format_registry), so it "
            f"dispatches to formats rather than being one; it must not "
            f"also be decorated with @register_format."
        )
    pending = list(cls.__bases__)
    visited = set()
    while pending:
        base = pending.pop()
        if base in visited:
            continue
        visited.add(base)
        if is_format_registry(base):
            base._REGISTRY.add(cls)
            if base.__dict__.get("_REGISTRY_ISOLATED", False):
                continue
        pending.extend(base.__bases__)
    return cls


# ----------------------------------------------------------------------
#   BASE
# ----------------------------------------------------------------------


@format_registry
class FileFormat(_FileNamesMixin):
    """Dispatch between the formats registered under a dispatcher.

    On a class decorated with [`format_registry`][], the `sniff*` methods
    identify the registered format that matches the input, and `load` and the
    `from_*` methods choose a format and delegate to it.

    A concrete format lists its reader and its writer before this class,
    as in `class NiftiImage(BinaryFileReader, BinaryFileWriter,
    ImageFormat, SingleScaleImage)`, so that its `sniff*`, `load` and
    `from_*` methods are those of its reader. The methods of this class
    therefore only ever run on a dispatcher, and they raise `TypeError`
    on any other class. This class does not inherit
    [`FileReader`][brainhops.io.base.parsers.FileReader], whose `sniff*`
    methods return a confidence rather than a class.

    This class owns the root registry used by generic loading and saving.
    A standalone family, such as metadata, uses
    `@format_registry(isolated=True)` to keep its formats out of ancestor
    registries while retaining the same dispatch API.

    !!! note "`sniff*` means something different on a dispatcher"
        A concrete format returns its confidence, between 0 and 1, that the
        input is in its format. A dispatcher has already compared those
        confidences, so it returns which format matched, or `None`. That format
        can be asked to score the input itself.
    """

    # ---- helpers -----------------------------------------------------

    @classmethod
    def _is_dispatcher(cls) -> bool:
        return is_format_registry(cls)

    @classmethod
    def _check_dispatcher(cls, name: str) -> None:
        """Refuse a dispatch method called on a class that is no dispatcher.

        A concrete format that lists its reader before its format base
        never reaches the methods of this class. One that reaches them
        lists no reader, or lists it after its format base.

        Parameters
        ----------
        name : str
            The name of the method that was called.

        Raises
        ------
        TypeError
            If `cls` owns no format registry.
        """
        if not cls._is_dispatcher():
            raise TypeError(
                f"{cls.__name__}.{name}() needs a reader or a format "
                f"registry: {cls.__name__} has no registry, and lists no "
                f"reader before its format base."
            )

    @classmethod
    def _reader_formats(cls) -> tx.Set[type]:
        """Select readable candidates without requiring reader inheritance.

        A public format can implement a read route directly or inherit it
        from its native parser. Writers whose only read routes are the
        defaults of the reader mixins are export-only and must not be
        selected by sniffing or loading.
        """
        adapters = (FileFormat, _ReaderMixin, _TextReaderMixin)
        routes = (
            "load",
            "from_spec",
            "from_file",
            "from_filename",
            "from_fileobj",
            "from_content",
            "from_bytes",
            "from_text",
            "from_lines",
            "from_line",
        )
        return {
            candidate
            for candidate in cls._REGISTRY
            if not issubclass(candidate, FileWriter)
            or any(
                name in base.__dict__
                for base in candidate.__mro__
                if base not in adapters
                for name in routes
            )
        }

    # ---- sniff -------------------------------------------------------

    @classmethod
    def sniff(
        cls,
        file: path.FileOrContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """Identify the registered format of a file or its content.

        The result is the registered format that would read the input.
        """
        cls._check_dispatcher("sniff")
        return sniff(
            Source(file),
            cls._reader_formats(),
            "sniff",
            error,
            f"{file}",
            **kwargs,
        )

    @classmethod
    def sniff_file(
        cls,
        file: path.FileLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """Same as [`sniff`][], for a path or an open file."""
        cls._check_dispatcher("sniff_file")
        return sniff(
            Source(file),
            cls._reader_formats(),
            "sniff_file",
            error,
            f"file: {file}",
            **kwargs,
        )

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> tx.Optional[type]:
        """Identify the registered format that matches a path."""
        cls._check_dispatcher("sniff_filename")
        return sniff(
            Source(filename),
            cls._reader_formats(),
            "sniff_filename",
            error,
            f"file: {filename}",
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
        cls._check_dispatcher("sniff_fileobj")
        return sniff(
            Source(file),
            cls._reader_formats(),
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
        cls._check_dispatcher("sniff_content")
        return sniff(
            Source.content(content),
            cls._reader_formats(),
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
        cls._check_dispatcher("sniff_bytes")
        return sniff(
            Source.content(content),
            cls._reader_formats(),
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
        cls._check_dispatcher("sniff_text")
        return sniff(
            Source.content(text),
            cls._reader_formats(),
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
        cls._check_dispatcher("sniff_lines")
        return sniff(
            Source.content(lines),
            cls._reader_formats(),
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
        cls._check_dispatcher("sniff_line")
        return sniff(
            Source.content(line),
            cls._reader_formats(),
            "sniff_line",
            error,
            "input line",
            **kwargs,
        )

    # ---- from --------------------------------------------------------

    @classmethod
    def load(cls, other: path.FileOrContentLike, **kwargs) -> tx.Self:
        """Read an object from a file or its content.

        The input is read with the best-matching registered format. A
        [`SourceSpec`][] is read with [`from_spec`][].
        """
        if isinstance(other, SourceSpec):
            return cls.from_spec(other, **kwargs)
        cls._check_dispatcher("load")
        return parse(
            Source(other), cls._reader_formats(), "load", "sniff", **kwargs
        )

    @classmethod
    def from_spec(cls, spec: SourceSpec, **kwargs) -> tx.Self:
        """Read a structured source specification.

        The specification's path is read with its hints and options.
        """
        cls._check_dispatcher("from_spec")
        return parse(
            Source(spec.path),
            cls._reader_formats(),
            "load",
            "sniff",
            hints=spec.hints,
            options=spec.options,
            **kwargs,
        )

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        """Read an object from a path or an open file.

        The file is read with the best-matching registered format.
        """
        cls._check_dispatcher("from_file")
        return parse(
            Source(file),
            cls._reader_formats(),
            "from_file",
            "sniff_file",
            **kwargs,
        )

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for a path."""
        cls._check_dispatcher("from_filename")
        return parse(
            Source(filename),
            cls._reader_formats(),
            "from_filename",
            "sniff_filename",
            **kwargs,
        )

    @classmethod
    @_passthrough_from_fileobj
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for an open file object."""
        cls._check_dispatcher("from_fileobj")
        return parse(
            Source(file),
            cls._reader_formats(),
            "from_fileobj",
            "sniff_fileobj",
            **kwargs,
        )

    @classmethod
    def from_content(cls, content: path.ContentLike, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for text or binary content."""
        cls._check_dispatcher("from_content")
        return parse(
            Source.content(content),
            cls._reader_formats(),
            "from_content",
            "sniff_content",
            **kwargs,
        )

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for binary content."""
        cls._check_dispatcher("from_bytes")
        return parse(
            Source.content(content),
            cls._reader_formats(),
            "from_bytes",
            "sniff_bytes",
            **kwargs,
        )

    @classmethod
    def from_text(cls, text: str, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for text."""
        cls._check_dispatcher("from_text")
        return parse(
            Source.content(text),
            cls._reader_formats(),
            "from_text",
            "sniff_text",
            **kwargs,
        )

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Same as [`from_file`][], for lines of text."""
        cls._check_dispatcher("from_lines")
        return parse(
            Source.content(lines),
            cls._reader_formats(),
            "from_lines",
            "sniff_lines",
            **kwargs,
        )

    @classmethod
    def from_line(cls, line: str, **kwargs) -> tx.Self:
        """Same as [`from_file`][], for a single line."""
        cls._check_dispatcher("from_line")
        return parse(
            Source.content(line),
            cls._reader_formats(),
            "from_line",
            "sniff_line",
            **kwargs,
        )


# ----------------------------------------------------------------------
#   DATA MODELS
# ----------------------------------------------------------------------


class _FileBasedModelMixin:
    """Give file-based data models a `from_any` that reads files first.

    A path, an open file, binary content or a [`SourceSpec`][] is read with
    `load`, and any other value is passed to the data model. This mixin,
    inherited by `ImageFormat`, `TransformationFormat` and
    `MetadataFormat`, must precede `DataModelBase` in each concrete class's
    MRO. The format dispatcher
    supplies file handling; the concrete class supplies its data model.
    Keeping this mixin separate from the reader mixins lets native parsers
    retain their specialized methods without hiding file-aware construction.

    !!! note "Every string is a file"
        No file-based class takes a string as the first constructor argument,
        so a string always names a file or a store. A file that cannot be read
        fails in `load`, rather than being passed to the constructor as data.

    !!! warning "The format base comes before the data model"
        A file format lists its format base, such as `TransformationFormat`
        or a class derived from it, before the class of its data model, such
        as `Affine` or `ImmutableSequence`. Bases that only read or write,
        such as `BinaryFileReader`, may come first. The format then decides
        whether to convert or to copy another object before the data model
        copies any field. A conversion that received the fields that the data
        model had already copied would build the wrong map. A class that
        breaks the rule raises `TypeError` when it is defined.
    """

    def __init_subclass__(cls, **kwargs: tx.Any) -> None:
        super().__init_subclass__(**kwargs)
        _check_base_order(cls)

    @classmethod
    def _data_model(cls) -> type:
        """Return the data model of the format, such as `SingleScaleImage`.

        This is the first class in the MRO that is a data model but not a
        parser, a writer or a format.
        """
        for base in cls.__mro__:
            if issubclass(base, DataModelBase) and not issubclass(
                base, (FileFormat, FileReader, FileWriter)
            ):
                return base
        return cls

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
        a file format, such as the `rawdata` array and the metadata of
        NIfTI and MGH, are copied only from an object of the same
        format; otherwise the class default is kept, because the same
        field name holds something different.
        Converting NIfTI to MGH on save therefore converts the data model only,
        and the writer rebuilds the format state.

        A format that decodes its data or its transformations from its
        file stores them in the private fields `_data` and
        `_transformations`, where `None` means that they are decoded.
        Within the format, the stored values are copied, so a copy of an
        image whose data was not assigned holds the same `rawdata` as
        `other`, which stays lazy if it is a proxy, and decodes the same
        record. From another class, the data, the levels of a pyramid and
        the transformations that `other` reports are copied, decoded if
        need be, because its `rawdata` and its record mean nothing to this
        format. A format that takes `metadata` over receives the metadata
        of `other`, which its field converts into the metadata of the
        format, without the record of `other`.

        Parameters
        ----------
        other : object
            The instance to copy.
        *args : Any
            Constructor arguments, which take precedence over the fields
            of `other`.
        **kwargs : Any
            Constructor arguments, which take precedence over the fields
            of `other`.

        Returns
        -------
        Self
            The copy.
        """
        for field in _foreign_format_fields(cls, other):
            if field.factory is True:
                default = field.default()
            else:
                default = field.default
            kwargs.setdefault(field.public_name, default)
        # The fields `_data`, `_images` and `_transformations` of another
        # class, if it has them, are its own stored values, so the decoded
        # views are read instead. A transformation model stores its data as
        # `_data` itself, and its own `from_instance` converts the data of
        # another class, so only a format that takes `data` over reads it
        # here. A pyramid that takes `images` over reads its levels from
        # the file, so the levels of another pyramid are read here.
        for name in ("data", "images"):
            if not isinstance(other, cls) and _takes_over(cls, name):
                # The data are read as the other object stores them, not
                # converted with the backend setting in effect.
                with backends.backend("native"):
                    value = getattr(other, name, None)
                if value is not None:
                    kwargs.setdefault(name, value)
        # The metadata of a data model, generic or of another format, is
        # given to the field that this format takes it over with, which
        # converts it into the metadata of this format.
        if not isinstance(other, cls) and _takes_over(cls, "metadata"):
            metadata = getattr(other, "metadata", None)
            if metadata is not None:
                kwargs.setdefault("metadata", metadata)
        if not isinstance(other, cls) and any(
            field.name == "_transformations"
            and field.public_name == "transformations"
            for field in fields(cls)
        ):
            transformations = getattr(other, "transformations", None)
            if transformations is not None:
                kwargs.setdefault("transformations", transformations)
        return super().from_instance(other, *args, **kwargs)


def _check_base_order(cls: type) -> None:
    """Refuse a format whose bases are not listed in the expected order.

    A concrete format lists its reader and its writer first, then its
    format base, then its data model, as in
    `class NiftiImage(BinaryFileReader, BinaryFileWriter, ImageFormat,
    SingleScaleImage)`. The reader and the writer are listed before the
    format base, because the reading methods of `FileFormat` only
    dispatch, and the format base must come before
    the data model, so that the format decides whether to convert another
    object before the data model copies its fields. A root of the
    formats, such as `TransformationFormat`, lists `FileFormat` among its
    own bases and is not checked.

    Parameters
    ----------
    cls : type
        The class to check.

    Raises
    ------
    TypeError
        If a class of the data model comes before the first format class
        in the method resolution order of `cls`, or if `cls` lists a
        reader or a writer after a format base.
    """
    # `bagof.magic` builds a stand-in class of this name only to compute an
    # MRO, then builds the class itself, which is checked.
    if FileFormat in cls.__bases__ or cls.__name__ == "__magic_discard__":
        return
    before = []
    for base in cls.__mro__[1:]:
        if issubclass(base, FileFormat):
            break
        if issubclass(base, DataModelBase) and not issubclass(
            base, (FileReader, FileWriter)
        ):
            before.append(base.__name__)
    if before:
        raise TypeError(
            f"{cls.__name__} lists its data model ({', '.join(before)}) "
            f"before its format base. List the format base first, so that "
            f"the format decides whether to convert another object before "
            f"the data model copies its fields."
        )
    io_bases = (_ReaderMixin, _SnifferMixin, _WriterMixin)
    late = []
    after_format = False
    for base in cls.__bases__:
        if issubclass(base, FileFormat):
            after_format = True
        elif after_format and issubclass(base, io_bases):
            late.append(base.__name__)
    if late:
        raise TypeError(
            f"{cls.__name__} lists its reader or writer ({', '.join(late)}) "
            f"after its format base. List the reader and the writer first."
        )


def _foreign_format_fields(cls: type, other: tx.Any) -> tx.List[tx.Any]:
    """Return the fields of `cls` that belong to a different file format.

    A field belongs to file formats when, in the MRO of `cls`, only file
    parsers declare it. A field that a plain data model also declares is
    shared by all formats and is never returned. Fields are compared by
    their public names, so a field of the data model that a format takes
    over under a private name, such as `_transformations`, is still
    shared. A format field is also left out when `other` is an instance
    of one of the formats that declare it, because `other` then holds a
    meaningful value for that field. Only keyword fields that the
    constructor accepts and that have a default are returned.

    A dispatcher is not counted among the formats that declare a field.
    `MetadataFormat` declares the `raw` field of every metadata format, but
    the record of one format means nothing to another, so `raw` is reset
    when metadata is copied from another format.
    """
    if isinstance(other, cls):
        return []
    shared: tx.Set[str] = set()
    owners: tx.Dict[str, tx.List[type]] = {}
    for klass in cls.__mro__:
        if not (isinstance(klass, type) and issubclass(klass, DataModelBase)):
            continue
        if is_format_registry(klass):
            continue
        names = [field.public_name for field in fields(klass)]
        if issubclass(klass, (FileFormat, FileReader, FileWriter)):
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
        and field.public_name not in shared
        and field.default is not no_default
        and not any(
            isinstance(other, k) for k in owners.get(field.public_name, ())
        )
    ]


def _takes_over(cls: type, name: str) -> bool:
    """Tell whether a format stores a field of its data model privately.

    A format takes a field over when it declares the field `_<name>` with
    the public name `name`, while its data model, the first class in its
    MRO that is a data model but not a parser, declares the field under
    the name `name`. An image format that takes `data` over is an example.
    The image model declares its data as `_data` with the public name
    `data`, so that `data` can return the lazy array of a file when no
    array was assigned, and it counts as declaring `data`. A
    transformation model declares `_data` itself, and its own
    `from_instance` converts the data of another class, so a
    transformation format does not take `data` over.
    """
    if not any(
        field.name == "_" + name and field.public_name == name
        for field in fields(cls)
    ):
        return False
    for klass in cls.__mro__:
        if not (isinstance(klass, type) and issubclass(klass, DataModelBase)):
            continue
        if issubclass(klass, (FileFormat, FileReader, FileWriter)):
            continue
        if issubclass(klass, Image) and name == "data":
            return True
        return any(field.name == name for field in fields(klass))
    return False


def _is_file_or_content(other: tx.Any) -> bool:
    return isinstance(
        other, (str, path.PathLike, bytes, bytearray, SourceSpec)
    ) or hasattr(other, "read")
