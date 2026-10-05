"""
Field descriptors: [`LazyField`][], a field whose value is loaded on
first access, and [`Lazy`][], the pending value it loads.
"""

__all__ = ["Lazy", "LazyField"]

# dependencies
import typing_extensions as tx


class Lazy:
    """
    A value that is not known yet, together with the function that
    computes it.

    A [`LazyField`][] calls `load`, with no argument, the first time the
    value is read. When `load` reads a file, it should be picklable (a
    module-level function, or a `functools.partial` of one), so that an
    object that is still waiting for its value can be pickled.
    """

    __slots__ = ("load",)

    def __init__(self, load: tx.Callable[[], tx.Any]) -> None:
        """
        Parameters
        ----------
        load : callable
            The function that computes the value. It takes no argument.
        """
        self.load = load

    def __repr__(self) -> str:
        return f"Lazy({self.load!r})"


class LazyField:
    """
    A data descriptor for an attribute whose value may still be pending.

    The value lives in the instance `__dict__`, under the attribute's
    name, as the value of a plain attribute does. When that value is a
    [`Lazy`][], the first read loads it: `load()` is called, its result
    goes through `prepare` (when one is given), and the outcome is
    assigned with `setattr`, so that the class converts it as it would
    convert any assigned value. `on_load(obj, name, value)` is then called
    with the value that was stored. Assigning a new value over a pending
    one loads the pending value first, so that `on_load` sees the value
    being replaced.

    The descriptor is installed on a class after the class is built. On a
    `Magic` class, the field table is built before the descriptor is
    installed, so the defaults and the constructor are not affected. A
    pending value is injected with `obj.__dict__[name] = Lazy(load)`,
    which bypasses conversion.

    Examples
    --------
    ```pycon
    >>> from brainhops._core.fields import Lazy, LazyField
    >>> class Record:
    ...     pass
    >>> Record.history = LazyField("history", default=())
    >>> record = Record()
    >>> record.history
    ()
    >>> def read_history():
    ...     print("reading")
    ...     return ("acquired", "resliced")
    >>> record.__dict__["history"] = Lazy(read_history)
    >>> Record.history.pending(record)
    True
    >>> record.history
    reading
    ('acquired', 'resliced')
    >>> record.history
    ('acquired', 'resliced')
    ```
    """

    def __init__(
        self,
        name: tx.Optional[str] = None,
        *,
        default: tx.Any = None,
        prepare: tx.Optional[tx.Callable[[tx.Any], tx.Any]] = None,
        on_load: tx.Optional[tx.Callable[[tx.Any, str, tx.Any], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        name : str, optional
            The name of the attribute. When it is omitted, `__set_name__`
            sets it from the class body the descriptor is assigned in.
        default : object, optional
            What a read returns when the instance holds no value at all.
        prepare : callable, optional
            A function applied to a loaded value before it is assigned.
        on_load : callable, optional
            A function called as `on_load(obj, name, value)` once a value
            has been loaded and stored.
        """
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
        """
        Whether an instance is still waiting for its value.

        Parameters
        ----------
        obj : object
            The instance to inspect.

        Returns
        -------
        bool
            `True` when the instance holds a [`Lazy`][] value that has
            not been loaded yet.
        """
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
