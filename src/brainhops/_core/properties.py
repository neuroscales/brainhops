"""Properties that compute their value.

A [`smartproperty`][] stores its value in a private attribute and computes
the value when nothing is stored. A [`lazyproperty`][] computes its value
on first access and caches it.
"""

__all__ = ["lazyproperty", "smartproperty", "UnsetLike"]

import typing_extensions as tx

_Getter = tx.Callable[[tx.Self], tx.Any]
_Setter = tx.Callable[[tx.Self, tx.Any], None]
_Deleter = tx.Callable[[tx.Self], None]
_IsUnset = tx.Callable[[tx.Any], bool]
_UnsetForm = tx.Union[None, tx.Literal["empty"], _IsUnset]
UnsetLike = tx.Union[_UnsetForm, tx.Tuple[_UnsetForm, ...]]
"""Accepted values of the `unset` argument of [`smartproperty`][].

The value decides when a stored value reads as unset. It is `None`,
`"empty"`, a predicate, or a tuple of these forms.
"""


# The containers listed here read as unset under unset="empty" when they
# are empty. Arrays are not listed, because the truth value of an array is
# ambiguous and testing it would raise an error.
_EMPTY_TYPES = (list, tuple, dict, set, frozenset)

_Names: tx.TypeAlias = tx.Union[str, tx.Iterable[str]]


class Invalidator:
    """Source of the names of the properties that an assignment invalidates.

    An invalidator tells a property setter which cached properties become
    stale when the property is assigned. The names are read from the object
    whose property was set, rather than fixed when the property is
    declared, because they are defined by the class of that object. As a
    result, a subclass that derives more cached views from the same data
    invalidates all of them. Each subclass of `Invalidator` decides where
    the names are looked up, and calling the invalidator on an object
    returns them as a tuple.
    """

    def __init__(self, resolve: tx.Callable[[object], _Names]) -> None:
        self.resolve = resolve

    def __call__(self, obj: object) -> tx.Tuple[str, ...]:
        invalidated = self.resolve(obj)
        if isinstance(invalidated, str):
            invalidated = (invalidated,)
        return tuple(invalidated)


class InvalidatorInAttribute(Invalidator):
    """Invalidator that reads the names from an attribute.

    The attribute is given by name, which may be a dotted path.
    """

    def __init__(self, attrname: str) -> None:
        self.attrname = attrname.split(".")
        super().__init__(self._getattr)

    def _getattr(self, obj: object) -> _Names:
        node = obj
        for attr in self.attrname:
            node = getattr(node, attr)
        return node


class InvalidatorInMethod(Invalidator):
    """Invalidator that reads the names by calling a method."""

    def __init__(self, methodname: str) -> None:
        self.methodname = methodname
        super().__init__(self._callmethod)

    def _callmethod(self, obj: object) -> _Names:
        method = getattr(obj, self.methodname)
        return method()


# --- lazyproperty -----------------------------------------------------


@tx.overload
def lazyproperty(fget: _Getter) -> property: ...


@tx.overload
def lazyproperty(
    *,
    unset: UnsetLike = None,
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def lazyproperty(
    fget: None,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def lazyproperty(
    fget: _Getter,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
) -> property:
    """Create a property that is computed on first access and then cached.

    The property is a read-only [`smartproperty`][] with `cache=True`.

    Parameters
    ----------
    fget
        Function that computes the value.
    doc
        Docstring of the property.
    unset
        Values that read as unset, so that the value is computed instead.
        See [`smartproperty`][].

    Returns
    -------
    property
        The property, or a decorator that makes one when `fget` is not
        given.
    """


def lazyproperty(fget=None, doc=None, unset=None):
    return smartproperty(fget, fset=False, cache=True, doc=doc, unset=unset)


# --- smartproperty ----------------------------------------------------


@tx.overload
def smartproperty(fget: _Getter) -> property: ...


@tx.overload
def smartproperty(
    *,
    unset: UnsetLike = None,
    cache: bool = False,
    invalidates: tx.Union[str, tx.Iterable[str], None] = (),
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def smartproperty(
    fget: None,
    fset: tx.Union[_Setter, bool, None] = None,
    fdel: tx.Optional[_Deleter] = None,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
    cache: bool = False,
    invalidates: tx.Union[str, tx.Iterable[str], None] = (),
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def smartproperty(
    fget: tx.Union[_Getter, str],
    fset: tx.Union[_Setter, bool, None] = None,
    fdel: tx.Optional[_Deleter] = None,
    doc: tx.Optional[str] = None,
    *,
    unset: UnsetLike = None,
    cache: bool = False,
    invalidates: tx.Union[str, tx.Iterable[str], None] = (),
) -> property:
    """Create a property backed by a private attribute.

    A property named `name` reads and writes the attribute `_<name>`. When
    the stored value reads as unset, the value is computed by `fget`
    instead, and with `cache=True` the computed value is stored in
    `_cache_<name>` so that it is computed only once. A value that has been
    set is always returned rather than computed.

    Parameters
    ----------
    fget
        Function that computes the value, or the name of the property when
        there is nothing to compute.
    fset
        Setter of the property. With `None` or `True`, a default setter
        stores the value as given in the private attribute and deletes the
        cached value, if there is one. With `False`, the property cannot be
        assigned, but a value stored in the private attribute, for example
        by a constructor, is still read.
    fdel
        Deleter of the property, passed to [`property`][property] as it is.
    doc
        Docstring of the property.
    unset
        Values that read as unset. By default, only a stored `None` reads
        as unset. With `"empty"`, an empty list, tuple, dict, set or
        frozenset reads as unset, and with a callable, any value for which
        the callable returns true does. A tuple of these forms matches a
        value when any of its forms matches. A property that stands in for
        an inherited field whose default is an empty container needs
        `unset=(None, "empty")`, because the constructor writes that
        default through the setter.
    cache
        Whether to cache the computed value.
    invalidates
        Names of other properties whose cached values the setter deletes, or
        an [`Invalidator`][] that reads these names from the object.

    Returns
    -------
    property
        The property, or a decorator that makes one when `fget` is not
        given.

    Raises
    ------
    ValueError
        If `unset` is an empty tuple or a string other than `"empty"`.
    TypeError
        If `unset` has any other invalid type.
    """


def smartproperty(
    fget=None,
    fset=None,
    fdel=None,
    doc=None,
    unset=None,
    cache=False,
    invalidates=(),
):
    # The form of unset is resolved once, at declaration, so that a wrong
    # value is refused there and the getter tests a single predicate.
    is_unset = _unset_predicate(unset)

    if fget is None:
        # When only options are given, as in @smartproperty(unset=...), the
        # call returns a decorator that receives the getter.
        def decorate(func: tx.Callable) -> property:
            return smartproperty(
                func,
                fset,
                fdel,
                doc,
                unset=unset,
                cache=cache,
                invalidates=invalidates,
            )

        return decorate

    if not callable(invalidates):
        if isinstance(invalidates, str):
            invalidates = (invalidates,)
        invalidates = tuple(invalidates or ())

    if isinstance(fget, str):
        name = fget
        fget = None
    else:
        name = fget.__name__

    fget = _make_fget(name, fget, is_unset, set=fset is not False, cache=cache)

    if not callable(fset):
        fset = _make_fset(name, settable=fset is not False, cacheable=cache)
    if invalidates and fset is not None:
        fset = _wrap_fset_invalidator(fset, invalidates)

    return property(fget, fset, fdel, doc)


def _make_fget(
    name: str,
    make_fn: tx.Optional[_Getter],
    isunset_fn: _IsUnset,
    set: bool = True,
    cache: bool = False,
) -> _Getter:
    # Only caching changes how the value is read, so the set argument does
    # not select a different getter. With fset=False, the property cannot be
    # assigned, but a constructor may still write the private field _<name>,
    # and the read-only property must then return that value.
    if cache:
        return _make_fget_settable_cacheable(name, make_fn, isunset_fn)
    return _make_fget_settable(name, make_fn, isunset_fn)


def _make_fget_settable(
    name: str, make_fn: tx.Optional[_Getter], isunset_fn: _IsUnset
) -> _Getter:
    # Without a cache, the value is computed again on every access.
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


def _make_fget_settable_cacheable(
    name: str, make_fn: tx.Optional[_Getter], isunset_fn: _IsUnset
) -> _Getter:
    private_name = "_" + name
    cache_name = "_cache_" + name

    def fget(self: tx.Self) -> tx.Any:
        value = getattr(self, private_name, None)
        if not isunset_fn(value):
            return value
        value = getattr(self, cache_name, None)
        if value is not None:
            return value
        if make_fn is None:
            return value
        value = make_fn(self)
        setattr(self, cache_name, value)
        return value

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
    # The setter stores any value, because the getter decides whether the
    # stored value reads as unset.
    private_name = "_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        setattr(self, private_name, value)

    fset.__name__ = name
    return fset


def _make_fset_settable_cacheable(name: str) -> _Setter:
    private_name = "_" + name
    cache_name = "_cache_" + name

    def fset(self: tx.Self, value: tx.Any) -> None:
        setattr(self, private_name, value)
        if hasattr(self, cache_name):
            delattr(self, cache_name)

    fset.__name__ = name
    return fset


# --- invalidates ------------------------------------------------------


def _wrap_fset_invalidator(
    fset: _Setter, invalidates: tx.Tuple[str]
) -> _Setter:
    if not callable(invalidates):
        cache_names = tuple("_cache_" + name for name in invalidates)

        def fset_invalidator(self: tx.Self, value: tx.Any) -> None:
            fset(self, value)
            for cache_name in cache_names:
                # A property that was never read has nothing cached.
                self.__dict__.pop(cache_name, None)

    else:

        def fset_invalidator(self: tx.Self, value: tx.Any) -> None:
            fset(self, value)
            invalidated = invalidates(self)
            cache_names = tuple("_cache_" + name for name in invalidated)
            for cache_name in cache_names:
                self.__dict__.pop(cache_name, None)

    fset_invalidator.__name__ = fset.__name__
    return fset_invalidator


# --- smartsetter ------------------------------------------------------


@tx.overload
def smartsetter(fset: _Setter) -> property:
    """Use as a bare decorator, which takes the name from the function."""


@tx.overload
def smartsetter(name: str) -> tx.Callable[[_Setter], property]:
    """Use as a decorator factory with an explicit name."""


def smartsetter(fset):
    """Create a property whose setter is the decorated function.

    The getter reads `_<name>` as [`smartproperty`][] does and returns
    `None` when nothing is stored. The decorated function stores the value
    itself, so it can validate or normalise the value. To also clear
    dependent caches, the setter can be given to [`smartproperty`][] as
    `fset`, together with `invalidates`.

    ```python
    @smartsetter
    def degree(self, value):
        self._degree = InterpolationOrder(value)
    ```

    Parameters
    ----------
    fset : callable or str
        The setter, whose name gives the name of the private attribute, or
        an explicit name for that attribute. For example,
        `@smartsetter("data")` reads `_data` whatever the setter is called.
        In both cases, the class binds the property under the name of the
        setter.

    Returns
    -------
    property
        The property, or a decorator that makes one when a name is given.
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
    """Return the single predicate that a form of `unset` stands for.

    A tuple of forms becomes a predicate that is true when any form matches.

    Raises
    ------
    ValueError
        If the tuple is empty or a string other than `"empty"` is given.
    TypeError
        If a form is neither `None`, a string, nor callable.
    """
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
