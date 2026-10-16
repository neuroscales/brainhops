"""Format registries that can hold formats whose module is not imported yet.

A format whose implementation is costly to import can be registered
before its module is imported. Its package defines a stub, which is a
small class that holds what dispatch needs to know about the format
before it reads anything: its hints, its file extensions and prefixes,
its priority, and the families that it belongs to, which are the
dispatchers that the stub derives from. The stub is registered with the
import path of the format, and the format derives from the stub, so that
it inherits all of these. The first time dispatch needs the format, it
imports the module of the format and replaces the stub with the format in
every registry. This step is called resolving the stub.

A registry is a set of formats. Iterating over a registry, or testing
whether it holds a format, resolves the stubs that it holds, so that code
that reads a registry always sees formats. Dispatch reads the stubs
instead, and resolves only those that may read the file at hand.
"""

__all__ = ["Candidates", "FormatRegistry", "is_stub", "register_stub"]

import importlib
import importlib.util
import sys
from collections.abc import MutableSet

import typing_extensions as tx

from brainhops.io.base.specs import format_hints

# The class attributes that a format must inherit from its stub unchanged,
# because dispatch reads them from the stub before the format is imported.
_NAMING = ("EXTENSIONS", "PREFIXES", "PRIORITY")

# The stubs that are not resolved yet. Each stub is mapped to the module
# and the qualified name of its format, and to the registries that hold it.
_PENDING: tx.Dict[type, tx.Tuple[str, str, tx.List["FormatRegistry"]]] = {}

# The stubs that are resolved, mapped to their format.
_RESOLVED: tx.Dict[type, type] = {}


def is_stub(entry: type) -> bool:
    """Tell whether a registry entry is a stub that is not resolved yet.

    Parameters
    ----------
    entry : type
        A format or a stub.

    Returns
    -------
    bool
        Whether `entry` is a stub whose format is not imported yet.
    """
    return entry in _PENDING


def resolve(entry: type) -> type:
    """Return the format of a stub, and import it if needed.

    A format that is not a stub is returned as it is. The first time a
    stub is resolved, the module of its format is imported, the format is
    checked against the stub, and the stub is replaced with the format in
    every registry that holds it.

    Parameters
    ----------
    entry : type
        A format or a stub.

    Returns
    -------
    type
        The format.

    Raises
    ------
    TypeError
        If the format does not derive from its stub, if it owns a
        registry, or if it changes what dispatch reads from the stub.
    """
    if entry in _RESOLVED:
        return _RESOLVED[entry]
    if entry not in _PENDING:
        return entry
    module, qualname, _ = _PENDING[entry]
    cls = importlib.import_module(module)
    for name in qualname.split("."):
        cls = getattr(cls, name)
    # Importing the module may have resolved the stub already.
    if entry in _RESOLVED:
        return _RESOLVED[entry]
    _check(entry, cls)
    _, _, registries = _PENDING.pop(entry)
    _RESOLVED[entry] = cls
    for registry in registries:
        registry._replace(entry, cls)
    return cls


def _check(stub: type, cls: tx.Any) -> None:
    """Check that a format can stand in for its stub."""
    name = f"{stub.__module__}.{stub.__qualname__}"
    if not isinstance(cls, type) or not issubclass(cls, stub) or cls is stub:
        raise TypeError(
            f"The stub {name} is registered for {cls!r}, which does not "
            f"derive from it."
        )
    if "_REGISTRY" in cls.__dict__:
        raise TypeError(
            f"The stub {name} is registered for {cls.__name__}, which owns "
            f"a registry (@format_registry), so it dispatches to formats "
            f"rather than being one."
        )
    changed = [
        attr for attr in _NAMING if getattr(cls, attr) != getattr(stub, attr)
    ]
    if format_hints(cls) != format_hints(stub):
        changed.append("HINTS")
    if changed:
        raise TypeError(
            f"{cls.__name__} changes {', '.join(changed)}, which dispatch "
            f"reads from its stub {name} before {cls.__name__} is imported. "
            f"Declare them on the stub instead."
        )


def _targets(stub: type, cls: tx.Any) -> bool:
    """Tell whether a pending stub is registered for `cls`."""
    module, qualname, _ = _PENDING[stub]
    return (
        getattr(cls, "__module__", None) == module
        and getattr(cls, "__qualname__", None) == qualname
    )


def register_stub(
    stub: type, target: str, registries: tx.Iterable["FormatRegistry"]
) -> None:
    """Register a stub in registries, in place of the format it stands for.

    Parameters
    ----------
    stub : type
        The stub.
    target : str
        The import path of the format, written `"module:qualname"`. A
        module that starts with a dot is relative to the package of the
        stub.
    registries : iterable of FormatRegistry
        The registries that hold the stub.

    Raises
    ------
    ValueError
        If `target` is not written `"module:qualname"`.
    """
    module, sep, qualname = target.partition(":")
    if not sep or not module or not qualname:
        raise ValueError(
            f"The format of a stub is written 'module:qualname', not "
            f"{target!r}."
        )
    if module.startswith("."):
        package = sys.modules[stub.__module__].__package__
        module = importlib.util.resolve_name(module, package)
    registries = list(registries)
    _PENDING[stub] = (module, qualname, registries)
    for registry in registries:
        registry._stubs.add(stub)


class FormatRegistry(MutableSet):
    """Set of the formats registered under a dispatcher.

    The registry holds formats and stubs. Iterating over it, and testing
    whether it holds a format, resolve the stubs that it holds, so these
    operations only see formats. [`entries`][] lists the formats and the
    stubs without importing anything.
    """

    def __init__(self) -> None:
        self._formats: tx.Set[type] = set()
        self._stubs: tx.Set[type] = set()

    @classmethod
    def _from_iterable(cls, it: tx.Iterable[type]) -> tx.Set[type]:
        # The set operators of a registry return a plain set of formats.
        return set(it)

    def __contains__(self, cls: object) -> bool:
        if cls in self._formats or cls in self._stubs:
            return True
        for stub in list(self._stubs):
            if _targets(stub, cls):
                return resolve(stub) is cls
        return False

    def __iter__(self) -> tx.Iterator[type]:
        for stub in list(self._stubs):
            resolve(stub)
        return iter(set(self._formats))

    def __len__(self) -> int:
        return len(self._formats) + len(self._stubs)

    def __repr__(self) -> str:
        names = sorted(cls.__name__ for cls in self.entries())
        return f"{type(self).__name__}({{{', '.join(names)}}})"

    def add(self, cls: type) -> None:
        """Add a format to the registry.

        Parameters
        ----------
        cls : type
            The format.
        """
        self._formats.add(cls)

    def discard(self, cls: type) -> None:
        """Remove a format, or the stub that stands for it, if present.

        Parameters
        ----------
        cls : type
            The format or its stub.
        """
        self._formats.discard(cls)
        self._stubs.discard(cls)
        for stub in list(self._stubs):
            if _targets(stub, cls):
                self._stubs.discard(stub)

    def entries(self) -> tx.Set[type]:
        """Return the formats and the stubs, without importing anything.

        Returns
        -------
        set of type
            The formats, and the stubs that are not resolved yet.
        """
        return self._formats | self._stubs

    def _replace(self, stub: type, cls: type) -> None:
        # A stub that was removed from the registry is not replaced.
        if stub in self._stubs:
            self._stubs.discard(stub)
            self._formats.add(cls)


class Candidates:
    """Formats that a dispatcher chooses between, some of them stubs.

    Dispatch reads the hints and the file names of the stubs, and
    resolves only the stubs that may read the input. A predicate, such as
    whether the format is a reader, can only be tested on a format, so it
    is applied to each stub once it is resolved.

    Parameters
    ----------
    entries : iterable of type
        The formats and the stubs.
    accept : callable, optional
        A predicate that a format must satisfy to be a candidate.
    """

    def __init__(
        self,
        entries: tx.Iterable[type],
        accept: tx.Optional[tx.Callable[[type], bool]] = None,
    ) -> None:
        self._entries = set(entries)
        self._accept = accept

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, cls: object) -> bool:
        return cls in self.resolve()

    def __iter__(self) -> tx.Iterator[type]:
        return iter(self.resolve())

    def entries(self) -> tx.Set[type]:
        """Return the formats and the stubs, without importing anything.

        Returns
        -------
        set of type
            The formats, and the stubs that are not resolved yet.
        """
        return {_RESOLVED.get(entry, entry) for entry in self._entries}

    def resolve(
        self, keep: tx.Optional[tx.Callable[[type], bool]] = None
    ) -> tx.Set[type]:
        """Return the candidate formats, and resolve the stubs needed.

        A format that is already imported is always returned, if it
        satisfies the predicate of the candidates. A stub is resolved, and
        its format returned, only if `keep` accepts the stub.

        Parameters
        ----------
        keep : callable, optional
            A predicate on stubs. All stubs are resolved if it is `None`.

        Returns
        -------
        set of type
            The candidate formats.
        """
        formats = set()
        for entry in self.entries():
            if keep is not None and is_stub(entry) and not keep(entry):
                continue
            cls = resolve(entry)
            if self._accept is None or self._accept(cls):
                formats.add(cls)
        return formats

    def pending(self, keep: tx.Callable[[type], bool]) -> bool:
        """Tell whether a stub that `keep` rejects is still unresolved.

        Parameters
        ----------
        keep : callable
            The predicate on stubs that a first pass used.

        Returns
        -------
        bool
            Whether a full pass would consider formats that the first pass
            did not.
        """
        return any(
            is_stub(entry) and not keep(entry) for entry in self.entries()
        )
