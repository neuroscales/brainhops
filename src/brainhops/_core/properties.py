# dependencies
import typing_extensions as tx

# typing
_Getter = tx.Callable[[tx.Self], tx.Any]
_Setter = tx.Callable[[tx.Self, tx.Any], None]
_Deleter = tx.Callable[[tx.Self], None]
_IsUnset = tx.Callable[[tx.Any], bool]


#: The containers that read as "no value supplied" when `empty_as_unset` is
#: set. Only a sized container is listed, so an array -- whose truth value
#: is ambiguous, and which raises rather than answering -- is never treated
#: as empty.
_EMPTY_TYPES = (list, tuple, dict, set, frozenset)


# --- lazyproperty -----------------------------------------------------


@tx.overload
def lazyproperty(fget: _Getter) -> property:
    """Bare decorator."""


@tx.overload
def smartproperty(
    *,
    empty_as_unset: bool = False,
) -> tx.Callable[[_Getter], property]:
    """Decorator factory (with options)."""


@tx.overload
def lazyproperty(
    fget: None,
    doc: tx.Optional[str] = None,
    *,
    empty_as_unset: bool = False,
) -> tx.Callable[[_Getter], property]:
    """Functional decorator factory."""


@tx.overload
def lazyproperty(
    fget: _Getter,
    doc: tx.Optional[str] = None,
    *,
    empty_as_unset: bool = False,
) -> property:
    """
    Functional decorator.

    A property that computes its value on first access and caches it.

    Parameters
    ----------
    fget : callable, optional
        The function that computes the property value.
    doc : str, optional
        The docstring for the property.
    empty_as_unset : bool, default=False
        Whether to treat an empty container as "no value supplied" and
        compute the value instead. This is needed when the property stands
        in for an inherited field whose default is an empty container: the
        constructor writes that default through this setter, which would
        otherwise shadow the reader for the object's whole life.

    Returns
    -------
    property
        The property object.
    """


def lazyproperty(fget=None, doc=None, empty_as_unset=False):
    return smartproperty(
        fget, fset=False, cache=True, doc=doc, empty_as_unset=empty_as_unset
    )


# --- smartproperty ----------------------------------------------------


@tx.overload
def smartproperty(fget: _Getter) -> property:
    """Bare decorator."""


@tx.overload
def smartproperty(
    *,
    empty_as_unset: bool = False,
    cache: bool = False,
) -> tx.Callable[[_Getter], property]:
    """Decorator factory (with options)."""


@tx.overload
def smartproperty(
    fget: None,
    fset: tx.Union[_Setter, bool, None] = None,
    fdel: tx.Optional[_Deleter] = None,
    doc: tx.Optional[str] = None,
    *,
    empty_as_unset: bool = False,
    cache: bool = False,
) -> tx.Callable[[_Getter], property]:
    """Functional decorator factory."""


@tx.overload
def smartproperty(
    fget: tx.Union[_Getter, str],
    fset: tx.Union[_Setter, bool, None] = None,
    fdel: tx.Optional[_Deleter] = None,
    doc: tx.Optional[str] = None,
    *,
    empty_as_unset: bool = False,
    cache: bool = False,
) -> property:
    """
    Functional decorator.

    A property that can:

    * read and write its value from a "private" attribute,
    * compute its value on first access if the private attribute is None, and
    * cache the computed value for future access.

    Parameters
    ----------
    fget : callable | str, optional
        The function that computes the property value, or the name of
        the property.
    fset : callable | bool, optional
        The function that sets the property value, or `False` to make the
        property read-only. If `None` or `True`, a default setter is
        created that writes the value to a private attribute and deletes
        the cached value, if any.
    fdel : callable, optional
        The function that deletes the property value.
        If `None` or `False`, the property cannot be deleted.
        If `True`, a default deleter is created that deletes the private
        and/or cached attribute, if any.
    doc : str, optional
        The docstring for the property.
    empty_as_unset : bool, default=False
        Whether to treat an empty container as "no value supplied" and
        compute the value instead. This is needed when the property stands
        in for an inherited field whose default is an empty container: the
        constructor writes that default through this setter, which would
        otherwise shadow the reader for the object's whole life.
    cache : bool, default=False
        Whether to cache the computed value for future access.

    Returns
    -------
    property
        The property object.
    """


def smartproperty(
    fget=None,
    fset=None,
    fdel=None,
    doc=None,
    empty_as_unset=False,
    cache=False,
):

    if fget is None:
        # Applied with options -- `@smartproperty(empty_as_unset=True)` --
        # rather than directly: return the decorator the function is handed
        # to.
        def decorate(func: tx.Callable) -> property:
            return smartproperty(
                func,
                fset,
                fdel,
                doc,
                empty_as_unset=empty_as_unset,
                cache=cache,
            )

        return decorate

    if isinstance(fget, str):
        name = fget
        fget = None
    else:
        name = fget.__name__

    is_unset = _make_is_unset(empty_as_unset)
    fget = _make_fget(name, fget, is_unset, set=fset is not False, cache=cache)

    if not callable(fset):
        fset = _make_fset(
            name, is_unset, settable=fset is not False, cacheable=cache
        )

    return property(fget, fset, fdel, doc)


def _make_fget(
    name: str,
    make_fn: tx.Optional[_Getter],
    isunset_fn: _IsUnset,
    set: bool = True,
    cache: bool = False,
) -> _Getter:
    if set and cache:
        return _make_fget_settable_cacheable(name, make_fn, isunset_fn)
    if set:
        return _make_fget_settable(name, make_fn, isunset_fn)
    if cache:
        return _make_fget_cacheable(name, make_fn)
    return _make_fget_fallback(name, make_fn)


def _make_fget_settable(
    name: str, make_fn: tx.Optional[_Getter], isunset_fn: _IsUnset
) -> _Getter:
    # Return a getter function that reads the value from the private
    # attribute corresponding to `name`, falling back to `make_fn` when
    # no value was supplied. The fallback is recomputed on every access:
    # a property that should compute once asks for `cache=True`.
    private_name = "_" + name

    def fget(self: tx.Self) -> tx.Any:
        value = getattr(self, private_name, None)
        if not isunset_fn(value):
            return value
        if make_fn is None:
            return value
        return make_fn(self)

    fget.__name__ = name
    return fget


def _make_fget_cacheable(name: str, make_fn: _Getter) -> _Getter:
    # Return a getter function that computes its value with `make_fn` if
    # necessary and caches it for future access.
    cache_name = "_cache_" + name

    def fget(self: tx.Self) -> tx.Any:
        value = getattr(self, cache_name, None)
        if value is not None:
            return value
        value = make_fn(self)
        setattr(self, cache_name, value)
        return value

    fget.__name__ = name
    return fget


def _make_fget_settable_cacheable(
    name: str, make_fn: _Getter, isunset_fn: _IsUnset
) -> _Getter:
    # Return a getter function that reads the value from the private
    # attribute corresponding to `name`, computing it with `make_fn` if
    # necessary and caching it for future access. If the value is set,
    # it is read from the private attribute instead of computed.
    private_name = "_" + name
    cache_name = "_cache_" + name

    def fget(self: tx.Self) -> tx.Any:
        value = getattr(self, private_name, None)
        if not isunset_fn(value):
            return value
        value = getattr(self, cache_name, None)
        if value is not None:
            return value
        value = make_fn(self)
        setattr(self, cache_name, value)
        return value

    fget.__name__ = name
    return fget


def _make_fget_fallback(name: str, make_fn: _Getter) -> _Getter:
    # Return a getter function that computes the value with `make_fn`
    # and returns it.

    def fget(self: tx.Self) -> tx.Any:
        return make_fn(self)

    fget.__name__ = name
    return fget


def _make_fset(
    name: str,
    isunset_fn: _IsUnset,
    settable: bool = True,
    cacheable: bool = False,
) -> _Setter:
    if settable and cacheable:
        return _make_fset_settable_cacheable(name, isunset_fn)
    if settable:
        return _make_fset_settable(name, isunset_fn)
    return None


def _make_fset_settable(name: str, isunset_fn: _IsUnset) -> _Setter:
    # Return a setter function that writes the value to the private
    # attribute corresponding to `name`. A value that reads as unset is
    # normalized to None, so the getter falls back instead of serving it.
    private_name = "_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        if isunset_fn(value):
            value = None
        setattr(self, private_name, value)

    fset.__name__ = name
    return fset


def _make_fset_settable_cacheable(name: str, isunset_fn: _IsUnset) -> _Setter:
    # Return a setter function that writes the value to the private
    # attribute corresponding to `name`, and deletes the cached value,
    # if any.
    private_name = "_" + name
    cache_name = "_cache_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        if isunset_fn(value):
            value = None
        setattr(self, private_name, value)
        if hasattr(self, cache_name):
            delattr(self, cache_name)

    fset.__name__ = name
    return fset


def _make_is_unset(empty_is_unset: bool) -> _IsUnset:
    if empty_is_unset:
        return _is_none_or_empty
    return _is_none


def _is_none_or_empty(value: tx.Any) -> bool:
    if value is None:
        return True
    return isinstance(value, _EMPTY_TYPES) and len(value) == 0


def _is_none(value: tx.Any) -> bool:
    return value is None
