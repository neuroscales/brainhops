"""
The registries of file formats, and the formats known before they are
imported.

A format registers itself as its module is imported (`@register_format`),
but importing every format to find out which one reads a file is what
makes `import brainhops.io.images` slow: the formats need the data model
and their optional dependencies. So the packages of each kind *declare*
their formats instead, in an `_entries` module that says, for each one,
where it lives, which hints, extensions and prefixes it answers to, and
what it needs installed. A `FormatEntry` stands for a format until it is
needed, and is resolved (its module imported) then.

Kept apart from `_base` and `_dispatch`, and free of any import beyond
the standard library and `brainhops._core.dependencies`, so that the
packages of each kind can declare their formats without importing the
machinery that reads them.
"""

__all__ = [
    "FormatEntry",
    "FormatRegistry",
    "declare",
    "declared",
    "ensure_declared",
    "key_of",
    "registry_for",
]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

# The keys of the dispatchers that the declared formats register into
# (see `key_of`).
OBJECT = "brainhops.io.base._base:FileBasedObject"
WRITABLE = "brainhops.io.base._base:WritableFileBasedObject"
IMAGE = "brainhops.io.images.base._base:FileBasedImage"
WRITABLE_IMAGE = "brainhops.io.images.base._base:WritableFileBasedImage"
TRANSFORMATION = (
    "brainhops.io.transformations.base._base:FileBasedTransformation"
)
WRITABLE_TRANSFORMATION = (
    "brainhops.io.transformations.base._base:WritableFileBasedTransformation"
)

# Each optional package a format may need, by the name a user installs,
# with the flag of `brainhops._core.dependencies` that says whether it is
# installed.
_FLAGS = {
    "nibabel": "HAS_NIBABEL",
    "h5py": "HAS_H5PY",
    "Pillow": "HAS_PILLOW",
    "tifffile": "HAS_TIFFFILE",
    "openslide-python": "HAS_OPENSLIDE",
    "abczarr": "HAS_ABCZARR",
}

# The packages whose `__init__` declares the formats of their kind.
_KINDS = ("brainhops.io.images", "brainhops.io.transformations")


def key_of(cls: type) -> str:
    """The name a class is declared under: `module:qualname`."""
    return f"{cls.__module__}:{cls.__qualname__}"


class FormatEntry:
    """
    A format, as it is known before its module is imported.

    `EXTENSIONS`, `PREFIXES` and `PRIORITY` are spelt as on the format
    class, so that what matches a file name (`_match_name`) works on an
    entry or a class alike; once the format is resolved, they are read
    from the class.

    Parameters
    ----------
    module : str
        The module that defines the format.
    qualname : str
        The format's qualified name in that module.
    hints : str
        The hints the format answers to (`format_hints`), separated by
        spaces.
    extensions, prefixes, priority
        The format's `EXTENSIONS`, `PREFIXES` and `PRIORITY`.
    dispatchers : tuple[str]
        The keys of the dispatchers whose registries it belongs to.
    requires : tuple[str]
        The optional packages it needs, as a user installs them.
    extra : str
        The extra of `brainhops` that installs them.
    check : callable, optional
        A further test of whether it can be used, run the first time
        the question is asked (whether abczarr has a driver).
    """

    def __init__(
        self,
        module: str,
        qualname: str,
        hints: str = "",
        extensions: tx.Tuple[str, ...] = (),
        prefixes: tx.Tuple[str, ...] = (),
        priority: int = 0,
        dispatchers: tx.Tuple[str, ...] = (),
        requires: tx.Tuple[str, ...] = (),
        extra: tx.Optional[str] = None,
        check: tx.Optional[tx.Callable[[], bool]] = None,
    ) -> None:
        self.module = module
        self.qualname = qualname
        self._hints = frozenset(hints.split()) if hints else None
        self._extensions = tuple(extensions)
        self._prefixes = tuple(prefixes)
        self._priority = priority
        self.dispatchers = tuple(dispatchers)
        self.requires = tuple(requires)
        self.extra = extra
        self.check = check
        self._cls: tx.Optional[type] = None
        self._usable: tx.Optional[bool] = None
        self.error: tx.Optional[Exception] = None

    @classmethod
    def from_class(cls, klass: type) -> "FormatEntry":
        """The entry of a format that was registered without being
        declared, such as a format of another package."""
        entry = cls(klass.__module__, klass.__qualname__)
        entry._cls = klass
        return entry

    def __repr__(self) -> str:
        return f"FormatEntry({self.key!r})"

    @property
    def key(self) -> str:
        """The name it is declared under: `module:qualname`."""
        return f"{self.module}:{self.qualname}"

    @property
    def __name__(self) -> str:
        """The format's name, as error messages give it."""
        if self._cls is not None:
            return self._cls.__name__
        return self.qualname.rsplit(".", 1)[-1]

    @property
    def hints(self) -> tx.FrozenSet[str]:
        """The hints the format answers to."""
        if self._hints is None:
            # Only a format registered without being declared has no
            # hints of its own, and it is a class already.
            from brainhops.io.base.specs import format_hints

            self._hints = format_hints(self._cls)
        return self._hints

    @property
    def EXTENSIONS(self) -> tx.Tuple[str, ...]:
        """The extensions the format declares."""
        if self._cls is not None:
            return self._cls.EXTENSIONS
        return self._extensions

    @property
    def PREFIXES(self) -> tx.Tuple[str, ...]:
        """The prefixes the format requires."""
        if self._cls is not None:
            return self._cls.PREFIXES
        return self._prefixes

    @property
    def PRIORITY(self) -> int:
        """The format's explicit priority."""
        if self._cls is not None:
            return self._cls.PRIORITY
        return self._priority

    @property
    def missing(self) -> tx.Optional[str]:
        """The first package the format needs that is not installed."""
        from brainhops._core import dependencies

        for package in self.requires:
            if not getattr(dependencies, _FLAGS[package]):
                return package
        return None

    @property
    def available(self) -> bool:
        """
        Whether the format takes part in dispatch: what it needs is
        installed, and, until it is registered, `check` passes.

        Being installed only says that a package is there to import, so
        a format may still fail to resolve; `resolve` says so.
        """
        if self._cls is not None:
            # A format module imports its dependency only once a file is
            # read, so its class may be registered without it.
            return self.missing is None
        if self._usable is None:
            self._usable = self.missing is None and (
                self.check is None or bool(self.check())
            )
        return self._usable

    def resolve(self) -> tx.Optional[type]:
        """
        The format class, imported if it was not yet, or `None` if it
        is not available or failed to import (see `error`).
        """
        if self._cls is not None:
            return self._cls
        if self.error is not None or not self.available:
            return None
        try:
            # Importing the module registers the format, which sets
            # `_cls` (see `FormatRegistry.add`).
            importlib.import_module(self.module)
        except (ImportError, OSError) as e:
            # OSError: a binding whose native library is missing.
            self.error = e
            return None
        if self._cls is None:
            self.error = ImportError(
                f"{self.module} does not register {self.qualname} with "
                f"@register_format, but it is declared as a format."
            )
        return self._cls


class FormatRegistry:
    """
    The formats a dispatcher chooses between.

    It holds entries, most of which are not imported yet, and is ordered
    by declaration. Iterating it gives the format classes, importing
    them; `entries()` gives the entries, without importing anything.
    A class is in it once it is registered, so `cls in registry` never
    imports.
    """

    def __init__(self, key: str, owner: tx.Optional[type] = None) -> None:
        self.key = key
        self.owner = owner
        # A declared format is filed under its key, so that its entry,
        # declared before the class exists, is found when it registers.
        # Any other format is filed under its class, so that two classes
        # of the same name (as tests build) are two formats.
        self._entries: tx.Dict[tx.Any, FormatEntry] = {}

    def __repr__(self) -> str:
        return f"FormatRegistry({self.key!r}, {len(self._entries)} formats)"

    def _find(self, cls: tx.Any) -> tx.Optional[tx.Any]:
        """The key `cls` is filed under, if it is registered."""
        if cls in self._entries:
            return cls
        key = key_of(cls) if isinstance(cls, type) else None
        entry = self._entries.get(key)
        if entry is not None and entry._cls is cls:
            return key
        return None

    def add(self, cls: type) -> None:
        """Register a format class; registering it again is a no-op."""
        entry = _ENTRIES.get(key_of(cls))
        if entry is None:
            if cls not in self._entries:
                self._entries[cls] = FormatEntry.from_class(cls)
            return
        entry._cls = cls
        self._entries.setdefault(entry.key, entry)

    def discard(self, cls: type) -> None:
        """Unregister a format class, if it is registered."""
        key = self._find(cls)
        if key is not None:
            del self._entries[key]

    def entries(self) -> tx.List[FormatEntry]:
        """Every format, declared or registered, without importing any."""
        ensure_declared()
        return list(self._entries.values())

    def __contains__(self, cls: tx.Any) -> bool:
        if isinstance(cls, FormatEntry):
            return self._entries.get(cls.key) is cls
        return self._find(cls) is not None

    def __iter__(self) -> tx.Iterator[type]:
        """The classes of the available formats, imported as needed."""
        for entry in self.entries():
            if entry.available:
                cls = entry.resolve()
                if cls is not None:
                    yield cls

    def __len__(self) -> int:
        return sum(1 for entry in self.entries() if entry.available)

    def __bool__(self) -> bool:
        return any(entry.available for entry in self.entries())


# Every declared format, by key.
_ENTRIES: tx.Dict[str, FormatEntry] = {}
# The registry of every dispatcher, by key.
_REGISTRIES: tx.Dict[str, FormatRegistry] = {}
_DECLARED = False


def registry_for(key: str, owner: tx.Optional[type] = None) -> FormatRegistry:
    """
    The registry of the dispatcher `key`, holding the formats declared
    into it.

    `owner` is the dispatcher class claiming it (`@format_registry`). A
    registry is made for a dispatcher before the class exists when its
    formats are declared first. Another class of the same name, as a
    test builds one in a function, gets a registry of its own.
    """
    registry = _REGISTRIES.get(key)
    if owner is not None and registry is not None:
        if registry.owner is not None and registry.owner is not owner:
            registry = None
    if registry is None:
        registry = _REGISTRIES[key] = FormatRegistry(key)
        for entry in _ENTRIES.values():
            if key in entry.dispatchers:
                registry._entries[entry.key] = entry
    if owner is not None:
        registry.owner = owner
    return registry


def declare(entries: tx.Iterable[FormatEntry]) -> None:
    """Declare formats into the registries of their dispatchers.
    Declaring a format again is a no-op."""
    new = {}
    for entry in entries:
        if entry.key not in _ENTRIES:
            new[entry.key] = _ENTRIES[entry.key] = entry
    # A format registered before it was declared (its module imported
    # ahead of its package's declarations) is filed under its class:
    # file it under its entry instead, or it would be there twice.
    for registry in _REGISTRIES.values():
        for cls in [k for k in registry._entries if isinstance(k, type)]:
            entry = new.get(key_of(cls))
            if entry is not None:
                entry._cls = cls
                del registry._entries[cls]
                registry._entries[entry.key] = entry
    for entry in new.values():
        for key in entry.dispatchers:
            registry_for(key)._entries.setdefault(entry.key, entry)


def declared() -> tx.List[FormatEntry]:
    """Every declared format, whatever its dispatcher."""
    ensure_declared()
    return list(_ENTRIES.values())


def ensure_declared() -> None:
    """
    Declare the formats of every kind.

    The packages of each kind (`brainhops.io.images`, ...) declare their
    formats as they are imported, so a dispatcher asked for its formats
    imports them first: `FileBasedObject.load` finds every format even
    when nothing but `brainhops.io.base` was imported.
    """
    global _DECLARED
    if not _DECLARED:
        for package in _KINDS:
            importlib.import_module(package)
        _DECLARED = True
