import typing_extensions as tx

#: The containers that read as "no value supplied" when `empty_as_unset` is
#: set. Only a sized container is listed, so an array -- whose truth value
#: is ambiguous, and which raises rather than answering -- is never treated
#: as empty.
_EMPTY_TYPES = (list, tuple, dict, set, frozenset)


def _is_unset(value: tx.Any) -> bool:
    if value is None:
        return True
    return isinstance(value, _EMPTY_TYPES) and len(value) == 0


def smartproperty(
    fget: tx.Union[tx.Callable[[tx.Self], tx.Any], str, None] = None,
    fset: tx.Optional[tx.Callable[[tx.Self, tx.Any], None]] = None,
    fdel: tx.Optional[tx.Callable[[tx.Self], None]] = None,
    doc: tx.Optional[str] = None,
    empty_as_unset: bool = False,
) -> property:
    # A property that reads and writes its value from the "private"
    # attribute corresponding to its name (`f"_{func.__name__}"`).
    # If the private attribute is None, the decorated function is called to
    # compute the value and stores it in `f"_cache_{func.__name__}"` for
    # future access.
    #
    # `empty_as_unset` stores an empty container as `None`, so it reads as
    # "no value supplied" and the value is computed instead. This is needed
    # when the property stands in for an inherited field whose default is an
    # empty container: the constructor writes that default through this
    # setter, which would otherwise shadow the reader for the object's whole
    # life.

    if fget is None:
        # Applied with options -- `@smartproperty(empty_as_unset=True)` --
        # rather than directly: return the decorator the function is handed
        # to.
        def decorate(func: tx.Callable) -> property:
            return smartproperty(
                func, fset, fdel, doc, empty_as_unset=empty_as_unset
            )

        return decorate

    if isinstance(fget, str):
        name = fget
        fget = None
    else:
        name = fget.__name__

    private = "_" + name
    cache = "_cache_" + name
    make_fn = fget

    def fget(self: tx.Self) -> tx.Any:
        value = getattr(self, private, None)
        if value is not None:
            return value
        value = getattr(self, cache, None)
        if make_fn is not None and value is None:
            value = make_fn(self)
            setattr(self, cache, value)
        return value

    fget.__name__ = name

    if fset is None:
        def fset(self: tx.Self, value: tx.Any) -> None:
            if empty_as_unset and _is_unset(value):
                value = None
            setattr(self, private, value)
            if value is not None and hasattr(self, cache):
                delattr(self, cache)

    fset.__name__ = name

    return property(fget, fset, fdel, doc)
