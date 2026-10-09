"""Helpers for the field tables of [`bagof.magic`][bagof.magic] classes."""

import typing_extensions as tx
from bagof.magic import Field

MISSING = Field().type  # bagof does not export its MISSING sentinel.


def field_default(cls: type, name: str, fallback: tx.Any = None) -> tx.Any:
    """Return the default that a class declares for a field.

    A default overridden by a subclass, for example with
    `on={"_log": True}`, lives in the field table, so `getattr(cls, name)`
    would return the default of the base class. The `fallback` is returned
    when there is no such field or no default.
    """
    field = getattr(cls, "__magic_fields__", {}).get(name)
    if field is None or field.default is MISSING:
        return fallback
    return field.default


def stores(cls: type, name: str) -> bool:
    """Return whether a class stores a field that its constructor sets.

    Derived names, such as a `ClassVar` that replaces a base field or a
    discarded `InitVar`, are not stored, so [`replace`][] cannot carry them
    over to another class.
    """
    field = getattr(cls, "__magic_fields__", {}).get(name)
    return field is not None and bool(field.init) and not field.var


def replace(obj: tx.Any, cls: tx.Optional[type], **changes: tx.Any) -> tx.Any:
    """Return a copy of a Magic object with some fields replaced.

    Unlike [`replace`][bagof.magic.replace], the copy may be an instance of
    another class `cls`; with `None`, the type is kept. Changes may name
    fields by their aliases, and fields that the object lacks take their
    defaults.

    Raises
    ------
    TypeError
        If a field is given twice, unknown, not accepted by the
        constructor, or needed by the constructor but neither given nor
        stored.
    AttributeError
        If a field to carry over was never set and has no default.
    """
    _field_table(obj, "replace")  # Refuse non-Magic objects first.
    cls = cls or type(obj)
    # The fields of the target class decide what is accepted, so that a
    # rebuild as another class (Identity as Affine) takes what it accepts.
    keyed = _keyed(_fields_of(cls, "replace").values())
    given, arguments = {}, []
    aliases = {
        alias: field.public_name
        for field in keyed.values()
        for alias in field.aliases
    }
    for name, value in changes.items():
        preferred = aliases.get(name, name)
        if preferred in given:
            raise TypeError(
                f"{cls.__name__} got multiple values for argument "
                f"{preferred!r}"
            )
        given[preferred] = value
    for name, field in keyed.items():
        # The constructor does not accept this field, so it cannot be set.
        if not field.init:
            if name in given:
                raise TypeError(
                    f"{cls.__name__} does not take {name!r} when it is "
                    f"built, so replace() has no way to set it."
                )
            continue
        if name in given:
            value = given.pop(name)
        elif not field.var:
            value = _value(obj, field, cls, "replace")
        elif field.build:
            # An InitVar is not stored, so its default stands in. The
            # constructor resolves a factory itself, and _HasFactory mimics
            # what the constructor would receive.
            value = _HasFactory(field.factory)
        elif field.default is not MISSING:
            value = field.default
        else:
            raise TypeError(
                f"{cls.__name__} needs {name!r} to be built and does not "
                f"keep it afterwards, so replace() cannot reuse the one it "
                f"was built with: pass {name}= as well."
            )
        arguments.append((field, value))
    if given:
        named = ", ".join(repr(name) for name in given)
        raise TypeError(f"{cls.__name__} has no field named {named}.")
    # Positional-only fields cannot be named and keyword-only fields cannot
    # be counted, so each value is passed as its field allows. Positional-only
    # fields come first in the signature, in field order.
    positional, keyword = [], {}
    for field, value in arguments:
        if field.positional and not field.kw:
            positional.append(value)
        else:
            keyword[field.public_name] = value
    return cls(*positional, **keyword)


def _field_table(obj: tx.Any, caller: str) -> tx.Dict[str, Field]:
    """Return the field table of the class of an instance.

    The lookup on `type(obj)` reports a class passed by mistake, since its
    metaclass carries no table.
    """
    return _fields_of(type(obj), caller, obj)


def _fields_of(
    cls: type, caller: str, shown: tx.Any = None
) -> tx.Dict[str, Field]:
    """Return the field table of a Magic class, including pseudo-fields.

    Raises
    ------
    TypeError
        If the class is not a Magic class.
    """
    table = getattr(cls, "__magic_fields__", None)
    if table is None:
        raise TypeError(
            f"{caller}() needs an instance of a Magic class, and "
            f"{(shown if shown is not None else cls)!r} is not one."
        )
    return table


def _keyed(found: tx.Iterable[Field]) -> tx.Dict[str, Field]:
    """Map fields by their public name, which is unique within a class."""
    return {field.public_name: field for field in found}


def _value(obj: tx.Any, field: Field, cls: type, caller: str) -> tx.Any:
    """Return the value of a field, falling back on its default.

    Raises
    ------
    AttributeError
        If the field has neither a value nor a default.
    """
    has_value, value = _stored(obj, field)
    if has_value:
        return value
    if field.default is not MISSING:
        return field.default
    raise AttributeError(
        f"{type(obj).__name__}.{field.name} has never been given a "
        f"value, so {caller}() has nothing to report for "
        f"{cls.__name__}.{field.public_name}. Give the field a default, "
        f"or set it in __post_init__."
    )


def _stored(obj: tx.Any, field: Field) -> tx.Tuple[bool, tx.Any]:
    """Return whether a field holds a value on an object, and the value."""
    try:
        return True, getattr(obj, field.name)
    except AttributeError:
        return False, None


class _HasFactory:
    def __init__(self, factory: callable) -> None:
        self.factory = factory

    def __repr__(self) -> str:
        return "<factory>"

    def __call__(self) -> tx.Any:
        return self.factory()
