# dependencies
import typing_extensions as tx

# typing
_Getter = tx.Callable[[tx.Self], tx.Any]
_Setter = tx.Callable[[tx.Self, tx.Any], None]
_Deleter = tx.Callable[[tx.Self], None]
_IsUnset = tx.Callable[[tx.Any], bool]
_UnsetForm = tx.Union[None, tx.Literal["empty"], _IsUnset]
UnsetLike = tx.Union[_UnsetForm, tx.Tuple[_UnsetForm, ...]]
"""When a stored value reads as unset: `None`, `"empty"`, a predicate, or
a tuple of them (see [`smartproperty`][])."""


#: The containers that read as unset under `unset="empty"`. Only a sized
#: container is listed, so an array -- whose truth value is ambiguous, and
#: which raises rather than answering -- is never treated as empty.
_EMPTY_TYPES = (list, tuple, dict, set, frozenset)


# --- lazyproperty -----------------------------------------------------


@tx.overload
def lazyproperty(fget: _Getter) -> property:
    """Bare decorator."""


@tx.overload
def lazyproperty(
    *,
    unset: UnsetLike = None,
) -> tx.Callable[[_Getter], property]:
    """Decorator factory (with options)."""


@tx.overload
def lazyproperty(
    fget: None,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
) -> tx.Callable[[_Getter], property]:
    """Functional decorator factory."""


@tx.overload
def lazyproperty(
    fget: _Getter,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
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
    unset : None, "empty", callable, or tuple of them, default=None
        When the stored value reads as no value, so that the value is
        computed instead. See [`smartproperty`][].

    Returns
    -------
    property
        The property object.
    """


def lazyproperty(fget=None, doc=None, unset=None):
    return smartproperty(fget, fset=False, cache=True, doc=doc, unset=unset)


# --- smartproperty ----------------------------------------------------


@tx.overload
def smartproperty(fget: _Getter) -> property:
    """Bare decorator."""


@tx.overload
def smartproperty(
    *,
    unset: UnsetLike = None,
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
    unset: UnsetLike = None,
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
    unset: UnsetLike = None,
    cache: bool = False,
) -> property:
    """
    Functional decorator.

    A property that can:

    * read and write its value from a "private" attribute,
    * compute its value on access if the private attribute is unset, and
    * cache the computed value for future access.

    Parameters
    ----------
    fget : callable | str, optional
        The function that computes the property value, or the name of
        the property.
    fset : callable | bool, optional
        The function that sets the property value, or `False` to make the
        property read-only. If `None` or `True`, a default setter is
        created that writes the value, as given, to a private attribute
        and deletes the cached value, if any.
    fdel : callable, optional
        The function that deletes the property value.
        If `None` or `False`, the property cannot be deleted.
        If `True`, a default deleter is created that deletes the private
        and/or cached attribute, if any.
    doc : str, optional
        The docstring for the property.
    unset : None, "empty", callable, or tuple of them, default=None
        When the getter computes the value instead of returning the one
        stored. `None` means a stored `None`, which is the default;
        `"empty"` means a stored empty container (a `list`, `tuple`,
        `dict`, `set` or `frozenset`); a callable `unset(value) -> bool`
        says it for any value. A tuple means any of them:
        `unset=(None, "empty")` computes the value when nothing, or an
        empty container, is stored -- which a property needs when it
        stands in for an inherited field whose default is an empty
        container, since the constructor writes that default through the
        setter. Whatever `unset` says, the setter stores the value as
        given: only the getter reads it as unset.
    cache : bool, default=False
        Whether to cache the computed value for future access.

    Returns
    -------
    property
        The property object.

    Raises
    ------
    TypeError, ValueError
        If `unset` is not one of the forms above.
    """


def smartproperty(
    fget=None,
    fset=None,
    fdel=None,
    doc=None,
    unset=None,
    cache=False,
):
    # Read `unset` once, when the property is declared, so that a wrong one
    # is refused there and the getter tests one predicate.
    is_unset = _unset_predicate(unset)

    if fget is None:
        # Applied with options -- `@smartproperty(unset=...)` -- rather
        # than directly: return the decorator the function is handed to.
        def decorate(func: tx.Callable) -> property:
            return smartproperty(
                func, fset, fdel, doc, unset=unset, cache=cache
            )

        return decorate

    if isinstance(fget, str):
        name = fget
        fget = None
    else:
        name = fget.__name__

    fget = _make_fget(name, fget, is_unset, set=fset is not False, cache=cache)

    if not callable(fset):
        fset = _make_fset(name, settable=fset is not False, cacheable=cache)

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
    settable: bool = True,
    cacheable: bool = False,
) -> _Setter:
    if settable and cacheable:
        return _make_fset_settable_cacheable(name)
    if settable:
        return _make_fset_settable(name)
    return None


def _make_fset_settable(name: str) -> _Setter:
    # Return a setter function that writes the value, as given, to the
    # private attribute corresponding to `name`. Whether it reads as unset
    # is the getter's question.
    private_name = "_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        setattr(self, private_name, value)

    fset.__name__ = name
    return fset


def _make_fset_settable_cacheable(name: str) -> _Setter:
    # Return a setter function that writes the value, as given, to the
    # private attribute corresponding to `name`, and deletes the cached
    # value, if any.
    private_name = "_" + name
    cache_name = "_cache_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        setattr(self, private_name, value)
        if hasattr(self, cache_name):
            delattr(self, cache_name)

    fset.__name__ = name
    return fset


# --- smartsetter ------------------------------------------------------


@tx.overload
def smartsetter(fset: _Setter) -> property:
    """Bare decorator: the name is the function's."""


@tx.overload
def smartsetter(name: str) -> tx.Callable[[_Setter], property]:
    """Decorator factory: the name is given."""


def smartsetter(fset):
    """
    A property whose setter is the decorated function.

    The getter reads the value from the "private" attribute of the
    property's name (`_<name>`), as [`smartproperty`][] does, and returns
    `None` when nothing is stored. The decorated function is the whole
    setter: it stores the value itself, so that it may check it first and
    clear what depends on it after.

    ```python
    @smartsetter
    def data(self, value):
        self._data = value
        self._forget_views()
    ```

    Parameters
    ----------
    fset : callable | str
        The setter, whose name names the private attribute the getter
        reads; or that name, given explicitly, in which case the setter is
        the function the result decorates (`@smartsetter("data")` reads
        `_data`, whatever the setter is called). The class binds the
        property under the setter's name either way.

    Returns
    -------
    property
        The property object (or, given a name, the decorator that makes
        it).
    """
    if isinstance(fset, str):
        name = fset

        def decorate(func: _Setter) -> property:
            return _smartsetter(name, func)

        return decorate
    return _smartsetter(fset.__name__, fset)


def _smartsetter(name: str, fset: _Setter) -> property:
    fget = _make_fget_settable(name, None, _is_none)
    return property(fget, fset, None, fset.__doc__)


# --- unset ------------------------------------------------------------


def _unset_predicate(unset: UnsetLike) -> _IsUnset:
    """The one predicate `unset` stands for (see `smartproperty`)."""
    forms = unset if isinstance(unset, tuple) else (unset,)
    if not forms:
        raise ValueError(
            "unset=() names no form: give None, 'empty', a predicate, or a "
            "tuple of them."
        )
    tests = tuple(_unset_form(form) for form in forms)
    if len(tests) == 1:
        return tests[0]

    def is_unset(value: tx.Any) -> bool:
        return any(test(value) for test in tests)

    return is_unset


def _unset_form(form: tx.Any) -> _IsUnset:
    if form is None:
        return _is_none
    if isinstance(form, str):
        if form == "empty":
            return _is_empty
        raise ValueError(
            f"unset={form!r} is not a form of unset: the one string it takes "
            f"is 'empty'."
        )
    if callable(form):
        return form
    raise TypeError(
        f"unset takes None, 'empty', a predicate `(value) -> bool`, or a "
        f"tuple of them, not a {type(form).__name__}."
    )


def _is_none(value: tx.Any) -> bool:
    return value is None


def _is_empty(value: tx.Any) -> bool:
    return isinstance(value, _EMPTY_TYPES) and len(value) == 0
