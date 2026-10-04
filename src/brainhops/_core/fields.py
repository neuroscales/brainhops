"""
Field descriptors: [`LazyField`][], a field whose value is loaded on
first access, and [`Lazy`][], the pending value it loads.
"""

__all__ = ["Lazy", "LazyField"]

# dependencies
import typing_extensions as tx


class Lazy:
    """
    A value that is not known yet, and how to get it.

    `load` is called, with no argument, the first time the value is
    needed (see [`LazyField`][]). A `load` that reads a file should be
    picklable (a module-level function or a `functools.partial` of one),
    so that an object still waiting for it can be pickled.
    """

    __slots__ = ("load",)

    def __init__(self, load: tx.Callable[[], tx.Any]) -> None:
        self.load = load

    def __repr__(self) -> str:
        return f"Lazy({self.load!r})"


class LazyField:
    """
    A data descriptor for a field whose value may still be pending.

    The value lives in the instance `__dict__`, under the field's name,
    as for a plain attribute. When it is a [`Lazy`][], the first read
    loads it: `load()` is called, its result goes through `prepare` (if
    given) and is assigned through `setattr`, so that the class converts
    it as it would any value, and `on_load(obj, name, value)` (if given)
    is called with what was stored. An assignment over a pending value
    loads it first, so `on_load` sees the value being replaced.

    Install it on a class after the class is built (a `Magic` class keeps
    its own field table, so the defaults and the constructor are not
    affected), and inject a pending value with
    `obj.__dict__[name] = Lazy(load)`, which bypasses conversion.

    Parameters
    ----------
    name : str, optional
        The attribute name; set by `__set_name__` when omitted.
    default : object, optional
        What a read returns when the instance holds no value at all.
    prepare : callable, optional
        Maps a loaded value before it is assigned.
    on_load : callable, optional
        Called as `on_load(obj, name, value)` once a value is loaded.
    """

    def __init__(
        self,
        name: tx.Optional[str] = None,
        *,
        default: tx.Any = None,
        prepare: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None,
        on_load: tx.Optional[tx.Callable[[tx.Any, str, tx.Any], None]] = None,
    ) -> None:
        self.name = name
        self.default = default
        self.prepare = prepare
        self.on_load = on_load

    def __set_name__(self, owner: type, name: str) -> None:
        if self.name is None:
            self.name = name

    def __get__(self, obj: tx.Any, owner: tx.Optional[type] = None) -> tx.Any:
        if obj is None:
            return self
        try:
            value = obj.__dict__[self.name]
        except KeyError:
            return self.default
        if isinstance(value, Lazy):
            value = self._load(obj, value)
        return value

    def __set__(self, obj: tx.Any, value: tx.Any) -> None:
        pending = obj.__dict__.get(self.name)
        if isinstance(pending, Lazy) and not isinstance(value, Lazy):
            self._load(obj, pending)
        obj.__dict__[self.name] = value

    def pending(self, obj: tx.Any) -> bool:
        """Whether `obj` still waits for its value."""
        return isinstance(obj.__dict__.get(self.name), Lazy)

    def _load(self, obj: tx.Any, pending: Lazy) -> tx.Any:
        # Unset first, so that the assignment below does not load again.
        del obj.__dict__[self.name]
        value = pending.load()
        if self.prepare is not None:
            value = self.prepare(value)
        setattr(obj, self.name, value)
        value = obj.__dict__.get(self.name, self.default)
        if self.on_load is not None:
            self.on_load(obj, self.name, value)
        return value
