"""Properties that compute their value.

A [`smartproperty`][] stores its value in a private attribute and computes
the value when nothing is stored. A [`lazyproperty`][] computes its value
on first access and caches it. Assigning either property, through any
setter, deletes its own cached value, and the cached values that it
names in `invalidates`. Code outside this module deletes a cached value
with [`drop_cache`][], and reads one without computing it with
[`cached_value`][].
"""

__all__ = [
    "cached_value",
    "drop_cache",
    "lazyproperty",
    "smartproperty",
    "UnsetLike",
]

import typing_extensions as tx

_Getter = tx.Callable[[tx.Self], tx.Any]
_Setter = tx.Callable[[tx.Self, tx.Any], None]
_Deleter = tx.Callable[[tx.Self], None]
_IsUnset = tx.Callable[[tx.Any], bool]
_UnsetForm = tx.Union[None, _IsUnset]
UnsetLike = tx.Union[_UnsetForm, tx.Tuple[_UnsetForm, ...]]
"""Accepted values of the `unset` argument of [`smartproperty`][].

The value decides when a stored value reads as unset. It is `None`, a
predicate, or a tuple of these forms.
"""

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


_InvalidatesLike = tx.Union[
    str, Invalidator, tx.Iterable[tx.Union[str, Invalidator]], None
]


# --- lazyproperty -----------------------------------------------------


@tx.overload
def lazyproperty(fget: _Getter) -> property: ...


@tx.overload
def lazyproperty(
    *,
    invalidates: _InvalidatesLike = (),
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def lazyproperty(
    fget: None,
    doc: tx.Optional[str] = None,
    *,
    invalidates: _InvalidatesLike = (),
) -> tx.Callable[[_Getter], property]: ...


@tx.overload
def lazyproperty(
    fget: _Getter,
    doc: tx.Optional[str] = None,
    *,
    invalidates: _InvalidatesLike = (),
) -> property:
    """Create a property that is computed on first access and then cached.

    A lazy property suits a view that is decoded or derived from other
    state, such as the array decoded from the record of a file. The
    property is a [`smartproperty`][] with `cache=True` and without a
    default setter. A value stored in the private attribute `_<name>`, for
    example by a constructor, is returned instead of being computed. A
    setter may be added with the `setter` decorator of the property, as in
    the example below. That setter deletes the cached value of the
    property before it runs, and the cached values named by `invalidates`
    after it runs.

    ```python
    @lazyproperty(invalidates="transformations")
    def data(self):
        return decode(self.raw)

    @data.setter
    def data(self, value):
        self.raw = encode(value)
    ```

    Parameters
    ----------
    fget : callable
        Function that computes the value.
    doc : str, optional
        Docstring of the property.
    invalidates : str or Invalidator or tuple, default=()
        Other properties whose cached values a setter deletes. See
        [`smartproperty`][].

    Returns
    -------
    property
        The property, or a decorator that makes one when `fget` is not
        given.
    """


def lazyproperty(fget=None, doc=None, *, invalidates=()):
    return smartproperty(
        fget, fset=False, cache=True, doc=doc, invalidates=invalidates
    )


# --- smartproperty ----------------------------------------------------


@tx.overload
def smartproperty(fget: _Getter) -> property: ...


@tx.overload
def smartproperty(
    *,
    unset: UnsetLike = None,
    cache: bool = False,
    invalidates: _InvalidatesLike = (),
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
    invalidates: _InvalidatesLike = (),
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
    invalidates: _InvalidatesLike = (),
) -> property:
    """Create a property backed by a private attribute.

    A property named `name` reads and writes the attribute `_<name>`. When
    the stored value reads as unset, the value is computed by `fget`
    instead, and with `cache=True` the computed value is stored in
    `_cache_<name>` so that it is computed only once. A value that has been
    set is always returned rather than computed.

    The common forms are listed below, from the simplest to the most
    specific.

    - `smartproperty("name")` declares a stored value that reads `None`
      when nothing is stored.
    - `@smartproperty` over a getter declares a stored value that is
      computed when nothing is stored.
    - `@smartproperty(cache=True)` over a getter declares a stored value
      that is computed once and cached when nothing is stored.
    - [`lazyproperty`][], with a setter added by `@name.setter`, declares
      a view decoded from other state, with a cache.
    - `invalidates=` names the other cached properties that become stale
      when the property is assigned.
    - `fset=` gives a setter defined elsewhere, for example a module-level
      function shared by several classes.
    - `fset=False` makes the property read-only.
    - `unset=` says which stored values read as unset.

    Every setter, whether it is the default setter, a setter given with
    `fset=`, or a setter added with `@name.setter`, deletes the cached
    value of the property before it runs, so a property never needs to
    name itself in `invalidates`. A setter may therefore store a value in
    its own cache, which is kept. After the setter runs, the cached values
    named by `invalidates` are deleted.

    Parameters
    ----------
    fget : callable or str
        Function that computes the value, or the name of the property when
        there is nothing to compute.
    fset : callable or bool, optional
        Setter of the property. With `None` or `True`, a default setter
        stores the value as given in the private attribute. With `False`,
        the property cannot be assigned, but a value stored in the private
        attribute, for example by a constructor, is still read.
    fdel : callable, optional
        Deleter of the property, passed to [`property`][property] as it is.
    doc : str, optional
        Docstring of the property.
    unset : UnsetLike, optional
        Values that read as unset. By default, only a stored `None` reads
        as unset. With a callable, any value for which the callable
        returns true reads as unset. A tuple of these forms matches a
        value when any of its forms matches. A property that stands in for
        an inherited field whose default is an empty container takes the
        field over under a private name, with
        `_x: ... = Inherit("x", default=None)` from
        [`bagof.magic`][bagof.magic], and keeps the default `unset=None`.
        The constructor then stores `None` when nothing is given, and an
        empty container assigned on purpose is used as it is.
    cache : bool, default=False
        Whether to cache the computed value.
    invalidates : str or Invalidator or tuple, default=()
        Other properties whose cached values the setter deletes. Each
        property is given by its name, or by an [`Invalidator`][] that
        reads the names from the object, and a tuple may mix both. A
        setter given later with the `setter` decorator of the property
        deletes the same cached values.

    Returns
    -------
    property
        The property, or a decorator that makes one when `fget` is not
        given.

    Raises
    ------
    ValueError
        If `unset` is an empty tuple.
    TypeError
        If `unset` has any other invalid type, or if an item of
        `invalidates` is neither a name nor an `Invalidator`.
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

    # Like unset, invalidates is resolved at declaration, so that a wrong
    # item is refused there.
    invalidates = _invalidated_items(invalidates)

    if isinstance(fget, str):
        name = fget
        fget = None
    else:
        name = fget.__name__

    fget = _make_fget(name, fget, is_unset, set=fset is not False, cache=cache)

    if not callable(fset):
        fset = _make_fset(name, settable=fset is not False)
    if fset is not None:
        fset = _wrap_fset(name, fset, cache, invalidates)

    return _SmartProperty(fget, fset, fdel, doc, name, cache, invalidates)


class _SmartProperty(property):
    """Property whose setters always delete the right cached values.

    The `setter` decorator of a plain [`property`][property] copies the
    getter and takes the new setter as it is. A setter written with
    `@name.setter` under a [`smartproperty`][] would therefore neither
    delete the cached value of the property nor the cached values named by
    `invalidates`. This class wraps such a setter in the same way as a
    setter given with `fset=`. To do so, the property keeps its name,
    whether it caches its value, and what it invalidates. The `getter` and
    `deleter` decorators keep them too, so that a later `setter` still
    applies them.
    """

    def __init__(
        self,
        fget: tx.Optional[_Getter] = None,
        fset: tx.Optional[_Setter] = None,
        fdel: tx.Optional[_Deleter] = None,
        doc: tx.Optional[str] = None,
        name: str = "",
        cache: bool = False,
        invalidates: tx.Tuple[tx.Union[str, Invalidator], ...] = (),
    ) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.name = name
        self.cache = cache
        self.invalidates = invalidates

    def getter(self, fget: _Getter) -> property:
        """Return a copy of the property with another getter.

        Parameters
        ----------
        fget : callable
            The new getter.

        Returns
        -------
        property
            The copy of the property.
        """
        return self._copy(fget, self.fset, self.fdel)

    def setter(self, fset: _Setter) -> property:
        """Return a copy of the property with another setter.

        The setter deletes the cached value of the property before it runs,
        and the cached values named by `invalidates` after it runs, as a
        setter given with `fset=` does.

        Parameters
        ----------
        fset : callable
            The new setter.

        Returns
        -------
        property
            The copy of the property.
        """
        fset = _wrap_fset(self.name, fset, self.cache, self.invalidates)
        return self._copy(self.fget, fset, self.fdel)

    def deleter(self, fdel: _Deleter) -> property:
        """Return a copy of the property with another deleter.

        Parameters
        ----------
        fdel : callable
            The new deleter.

        Returns
        -------
        property
            The copy of the property.
        """
        return self._copy(self.fget, self.fset, fdel)

    def _copy(
        self,
        fget: tx.Optional[_Getter],
        fset: tx.Optional[_Setter],
        fdel: tx.Optional[_Deleter],
    ) -> "_SmartProperty":
        return _SmartProperty(
            fget,
            fset,
            fdel,
            self.__doc__,
            self.name,
            self.cache,
            self.invalidates,
        )


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


def _make_fset(name: str, settable: bool = True) -> tx.Optional[_Setter]:
    # The cached value is deleted by the wrapper of every setter, so the
    # default setter only stores the value.
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


# --- invalidates ------------------------------------------------------


def _invalidated_items(
    invalidates: _InvalidatesLike,
) -> tx.Tuple[tx.Union[str, Invalidator], ...]:
    """Return the items of `invalidates` as a tuple.

    A single name or `Invalidator` becomes a tuple of one item.

    Raises
    ------
    TypeError
        If an item is neither a string nor an `Invalidator`.
    """
    if isinstance(invalidates, (str, Invalidator)):
        return (invalidates,)
    items = tuple(invalidates or ())
    for item in items:
        if not isinstance(item, (str, Invalidator)):
            raise TypeError(
                f"invalidates takes names, Invalidators, or a tuple of "
                f"them, not a {type(item).__name__}."
            )
    return items


def _wrap_fset(
    name: str,
    fset: _Setter,
    cache: bool,
    invalidates: tx.Tuple[tx.Union[str, Invalidator], ...],
) -> _Setter:
    # The cached value of the property is deleted before the setter runs,
    # so that a setter may fill it, as the `values` setter of a field does.
    # A setter with no cached value to delete is used as it is.
    if not cache and not invalidates:
        return fset
    own_cache = (name,) if cache else ()

    def fset_invalidator(self: tx.Self, value: tx.Any) -> None:
        drop_cache(self, *own_cache)
        fset(self, value)
        for item in invalidates:
            if isinstance(item, str):
                drop_cache(self, item)
            else:
                drop_cache(self, *item(self))

    fset_invalidator.__name__ = fset.__name__
    return fset_invalidator


# --- drop_cache -------------------------------------------------------


def drop_cache(obj: object, *names: str) -> None:
    """Delete the cached values of properties of an object.

    A cached [`smartproperty`][] or [`lazyproperty`][] deletes its cached
    value when it is assigned. This function is for code that changes the
    state from which a property is computed without assigning the
    property, for example a writer that points `raw` at the file it has
    just written. A property that has not cached a value is skipped.

    Parameters
    ----------
    obj : object
        The object whose cached values are deleted.
    *names : str
        The names of the properties.
    """
    for name in names:
        obj.__dict__.pop("_cache_" + name, None)


def cached_value(obj: object, name: str) -> tx.Any:
    """Return the cached value of a property of an object, or `None`.

    A cached [`smartproperty`][] or [`lazyproperty`][] stores the value
    that it computes apart from a value that is assigned to it. This
    function is for code that needs the computed value only when it has
    already been computed, for example a writer that compares a decoded
    view, which may have been changed in place, with a fresh decode,
    without computing that view itself.

    Parameters
    ----------
    obj : object
        The object whose cached value is read.
    name : str
        The name of the property.

    Returns
    -------
    Any
        The cached value, or `None` when the property has cached nothing.
    """
    return obj.__dict__.get("_cache_" + name)


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
        If the tuple is empty.
    TypeError
        If a form is neither `None` nor callable.
    """
    forms = unset if isinstance(unset, tuple) else (unset,)
    if not forms:
        raise ValueError(
            "unset=() names no form: give None, a predicate, or a tuple of "
            "them."
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
    if callable(form):
        return form
    raise TypeError(
        f"unset takes None, a predicate `(value) -> bool`, or a tuple of "
        f"them, not a {type(form).__name__}."
    )


def _is_none(value: tx.Any) -> bool:
    return value is None
