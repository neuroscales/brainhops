"""Magic extensions."""

import typing_extensions as tx
from bagof.magic import Field

MISSING = Field().type  # hack to access the unexported sentinel


def field_default(cls: type, name: str, fallback: tx.Any = None) -> tx.Any:
    """
    The default value a class declares for one of its fields.

    A default that a subclass overrides -- with `on={"_log": True}`, say
    -- is recorded in the field table and not bound as a class attribute,
    so `getattr(cls, name)` answers with the value the base class wrote.
    `fallback` stands in for a name that is not a field, or a field with
    no default.
    """
    field = getattr(cls, "__magic_fields__", {}).get(name)
    if field is None or field.default is MISSING:
        return fallback
    return field.default


def stores(cls: type, name: str) -> bool:
    """
    Whether `cls` keeps `name` as a field it is built with.

    A name the class derives instead -- a `ClassVar` standing in for a
    field of a base class, an `InitVar` used and thrown away -- is not
    stored under it, so `replace()` has nothing to carry over for it and
    a rebuild as another class has to be handed the value.
    """
    field = getattr(cls, "__magic_fields__", {}).get(name)
    return field is not None and bool(field.init) and not field.var


def replace(obj: tx.Any, cls: tx.Optional[type], **changes: tx.Any) -> tx.Any:
    """
    A reimplementation of `bagof.magic.replace` that allows the type of
    the returned object to be changed.
    """
    _field_table(obj, "replace")  # refuse a non-magic object here
    cls = cls or type(obj)
    # Every field of the *target* class, pseudo-fields included: a change
    # has to be turned down by name whether or not the name belongs to a
    # real field, and a class that cannot tell two of its fields apart
    # cannot say which one a change was meant for. The fields are the new
    # class's, not the old one's: a rebuild that changes the type -- an
    # `Identity` as an `Affine`, a tangent as the map it exponentiates --
    # takes what the new class accepts. A field the new class has and the
    # old object does not falls back to its default.
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
        # A field the constructor does not take -- one that can be
        # passed neither by position nor by name, which is what `NoInit`
        # says -- has no way in.
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
            # An InitVar is used during construction and not stored, so
            # there is nothing to carry over: its default stands in.
            # The constructor resolves a factory itself, and this is
            # what it would have been handed.
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
    # A positional-only field cannot be named and a keyword-only one
    # cannot be counted off, so each value goes back the way its own
    # field is allowed to be passed. Positional-only fields come first
    # in the signature, in field order, which is the order they were
    # collected in.
    positional, keyword = [], {}
    for field, value in arguments:
        if field.positional and not field.kw:
            positional.append(value)
        else:
            keyword[field.public_name] = value
    return cls(*positional, **keyword)


def _field_table(obj: tx.Any, caller: str) -> tx.Dict[str, Field]:
    """Every field of an instance's class, pseudo-fields included.

    The lookup goes through `type(obj)` rather than `obj`, so a class
    handed in by mistake is reported instead of being read as if it were
    an instance: a class carries the table, its metaclass does not.
    """
    return _fields_of(type(obj), caller, obj)


def _fields_of(
    cls: type, caller: str, shown: tx.Any = None
) -> tx.Dict[str, Field]:
    """Every field a Magic class carries, pseudo-fields included."""
    table = getattr(cls, "__magic_fields__", None)
    if table is None:
        raise TypeError(
            f"{caller}() needs an instance of a Magic class, and "
            f"{(shown if shown is not None else cls)!r} is not one."
        )
    return table


def _keyed(found: tx.Iterable[Field]) -> tx.Dict[str, Field]:
    """Key fields by the name they are known by outside the class.

    One name reaches one field: a class with two fields under a single
    outside name is refused when it is built, so the keys here never
    collide.
    """
    return {field.public_name: field for field in found}


def _value(obj: tx.Any, field: Field, cls: type, caller: str) -> tx.Any:
    """The value a field holds, or a written error if it holds none.

    A rebuild as another class reads the fields that class declares, and
    the object need not carry all of them: one it does not falls back to
    the field's default, and only a field with no default is an error.
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
    """Return (has_value, value) for a field on an object.

    A field with no constructor parameter and no default only gets a
    value when set by hand, so (False, None) is a normal result.
    """
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
