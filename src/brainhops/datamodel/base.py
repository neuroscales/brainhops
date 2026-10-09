"""Base class of the data models and its converter."""

__all__ = ["DataModelBase", "DataModelConverter", "IdentityComparison"]

import re
from collections.abc import Mapping

import typing_extensions as tx
from bagof.converters import ConversionError, Converter, register_converter
from bagof.magic import Field, HideIfNone, Magic, fields


class IdentityComparison:
    """Mixin that makes a class and all its subclasses compare by identity.

    Equality never raises, and instances are hashable. Identity comparison
    is restored on every subclass, because a subclass that also derives
    from another data model may otherwise receive a value-based `__eq__`
    and `__hash__`.
    """

    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__

    def __init_subclass__(cls, **kwargs: tx.Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("__eq__", "__ne__", "__hash__"):
            setattr(cls, name, getattr(object, name))


class DataModelBase(
    Magic,
    convert=True,
    mapping=False,
    repr=HideIfNone,
    doc=True,
    pin_discriminant="pin+narrow",
):
    """Base class of all data models.

    Every data model converts the values of its fields to their declared
    types, and can be built from a mapping or from a similar instance with
    [`from_any`][].

    Some data models are polymorphic: calling a class such as `Axis`
    builds the subclass that the arguments select, according to the `on=`
    condition with which each subclass is declared. A polymorphic class is
    built only from arguments that match its own condition and refuses the
    others, so `OrientedAxis(orientation=None)` raises instead of building
    an oriented axis without an orientation. A field that a class declares
    itself keeps the type written there, so the fields that select a
    subclass are typed narrowly, as in `Literal["space"]`.
    """

    # The options given to the class statement apply to the whole hierarchy.

    @classmethod
    def from_dict(cls, other: tx.Mapping, *args, **kwargs) -> tx.Self:
        """Build an instance from a mapping.

        Only keys that name keyword fields or constructor-only keywords, such
        as the `matrix=` keyword of an affine, are used. Other keys are
        ignored, unlike in
        [`from_any`][brainhops.datamodel.base.DataModelBase.from_any]. A key
        that names a fixed field, that is, a field whose value the class sets
        itself, is checked against that value rather than passed on. The
        extra arguments are passed to the constructor and take precedence
        over the mapping.

        Raises
        ------
        ValueError
            If a fixed field is given a value other than `None` and the value
            that the class fixes.
        """

        def read(field: tx.Any) -> tx.Any:
            return other.get(field.public_name, _ABSENT)

        return _build(cls, read, args, kwargs, init_vars=True)

    @classmethod
    def from_instance(cls, other: tx.Self, *args, **kwargs) -> tx.Self:
        """Build an instance from an instance of a similar class.

        Only attributes that name keyword fields are used, and an attribute
        that is `None` leaves the default of the class in place. The extra
        arguments are passed to the constructor and take precedence. Unless
        `other` is already an instance of this class, attributes that name
        fixed fields are checked, so a generic axis oriented right to left
        cannot be read as a `LeftToRightAxis`.

        Raises
        ------
        ValueError
            If a fixed field of `other` is set to a value other than the value
            that the class fixes.
        """
        # An instance of this class already satisfies its fixed fields.
        check_fixed = not isinstance(other, cls)
        # A polymorphic class also fixes the fields it is selected on, such as
        # the orientation of `LeftToRightAxis`, although they go to the
        # constructor.
        selected_on = _selected_on(cls) if check_fixed else {}

        def read(field: tx.Any) -> tx.Any:
            if not field.init and not check_fixed:
                return _ABSENT
            value = getattr(other, field.name, None)
            if value is None:
                return _ABSENT
            specs = selected_on.get(field.name)
            if specs and field.init:
                _check_selected_on(cls, field, value, specs)
            return value

        return _build(cls, read, args, kwargs)

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Build an instance from a mapping, a similar instance or an argument.

        A similar instance is an instance of this class, of one of its
        parents within the data model, or of another class in the same
        polymorphic family. The family matters because calling a polymorphic
        class such as `Axis` builds the subclass that the arguments select, so
        an axis built as a generic `Axis` is often an instance of a sibling
        class such as `TimeAxis`. Any other object, including an instance of
        a parent outside the data model, is passed to the constructor.

        Raises
        ------
        TypeError
            If a mapping has a key that names no field. Unlike
            [`from_dict`][brainhops.datamodel.base.DataModelBase.from_dict],
            this method catches misspelled keys.
        """
        if isinstance(other, Mapping):
            _refuse_unknown_keys(cls, other)
            return cls.from_dict(other, *args, **kwargs)
        elif isinstance(other, cls):
            return cls.from_instance(other, *args, **kwargs)
        elif isinstance(other, DataModelBase) and issubclass(cls, type(other)):
            return cls.from_instance(other, *args, **kwargs)
        elif isinstance(other, DataModelBase) and _same_family(cls, other):
            return cls.from_instance(other, *args, **kwargs)
        else:
            return cls(other, *args, **kwargs)


# --- helpers ----------------------------------------------------------

# The source has no value for the field.
_ABSENT = object()

# Default of a field that has no default.
_NO_DEFAULT = Field().default


def _build(
    cls: tx.Type[DataModelBase],
    read: tx.Callable[[tx.Any], tx.Any],
    args: tx.Tuple[tx.Any, ...],
    kwargs: tx.Dict[str, tx.Any],
    init_vars: bool = False,
) -> tx.Any:
    """Build an instance of `cls` from the values returned by `read`.

    `read(field)` returns the value of the source, or `_ABSENT`. Keyword
    fields take that value unless `kwargs` already sets them. Fixed fields
    are checked before construction when their value can be read from the
    field, and after construction otherwise. With `init_vars`, the
    constructor-only keywords are read too, since a mapping can name them.
    """
    later = {}
    for field in _fields(cls, init_vars):
        value = read(field)
        if value is _ABSENT:
            continue
        if field.init and field.kw:
            kwargs.setdefault(field.public_name, value)
        elif not field.init and value is not None:
            expected = _fixed_value(field)
            if expected is _NO_DEFAULT:
                later[field] = value
            else:
                _check_fixed(cls, field, value, expected)
    obj = cls(*args, **kwargs)
    for field, value in later.items():
        _check_fixed(cls, field, value, getattr(obj, field.name))
    return obj


def _fields(cls: type, init_vars: bool = False) -> tx.Tuple[tx.Any, ...]:
    """Return the fields of `cls`, and its constructor-only keywords if asked.

    The `fields` function of bagof omits the `ClassVar` and `InitVar`
    pseudo-fields, so the constructor-only keywords are read from the field
    table instead. An `InitVar` is told apart from a `ClassVar` because it
    is an init field.
    """
    if not init_vars:
        return fields(cls)
    table = getattr(cls, "__magic_fields__", {})
    return tuple(f for f in table.values() if not f.var or f.init)


def _fixed_value(field: tx.Any) -> tx.Any:
    """Return the value of a fixed field, or `_NO_DEFAULT`."""
    if callable(field.factory):
        return field.factory()
    return field.default


def _check_fixed(
    cls: type, field: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    """Refuse a value that differs from the one `cls` fixes for a field.

    The value is converted with the converter of the field before the
    comparison, so that a mapping can describe a fixed data model.
    """
    if value is expected:
        return
    where = f"{cls.__name__}.{field.public_name}"
    converted = value
    if callable(field.converter):
        try:
            converted = field.converter(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{where} is always {expected!r}, and {value!r} could not "
                f"be converted to compare with it -- {e}"
            ) from e
    if converted is not expected and converted != expected:
        raise ValueError(
            f"{where} is always {expected!r}, so a value whose "
            f"{field.public_name} is {value!r} cannot be read as a "
            f"{cls.__name__}."
        )


# These private bagof attributes are read because bagof has no public
# accessor for them. The registration of a polymorphic class holds
# `(owners, specs, priority, ...)`. Without it, only the fields that are not
# passed to the constructor are checked.
_REGISTRATION = "__magic_registration__"

_OPTIONS = "__magic_options__"


def _is_polymorphic(cls: type) -> bool:
    """Return whether calling `cls` builds the subclass that is selected."""
    options = getattr(cls, _OPTIONS, None)
    return bool(getattr(options, "polymorphic", False))


def _same_family(cls: type, other: tx.Any) -> bool:
    """Return whether `other` shares a polymorphic base class with `cls`."""
    return any(
        isinstance(other, base)
        for base in cls.__mro__
        if base is not DataModelBase
        and issubclass(base, DataModelBase)
        and _is_polymorphic(base)
    )


def _selected_on(cls: type) -> tx.Dict[str, tx.List[tx.Any]]:
    """Return the `on=` conditions of `cls` and its parents, by field."""
    out: tx.Dict[str, tx.List[tx.Any]] = {}
    for base in cls.__mro__:
        registration = getattr(base, _REGISTRATION, None)
        try:
            specs = registration[1]
        except (TypeError, IndexError, KeyError):
            continue
        for spec in specs:
            name = getattr(spec, "name", None)
            if name is not None and callable(getattr(spec, "matches", None)):
                out.setdefault(name, []).append(spec)
    return out


def _matches(spec: tx.Any, value: tx.Any) -> bool:
    try:
        return bool(spec.matches(value))
    except (TypeError, ValueError, AttributeError):
        return False


def _check_selected_on(
    cls: type, field: tx.Any, value: tx.Any, specs: tx.Sequence[tx.Any]
) -> None:
    """Refuse a value for which `cls` would not be selected.

    This is the counterpart of `_check_fixed` for the fields that a
    polymorphic class is selected on: they are passed to the constructor,
    yet they are just as fixed.
    """
    failed = next((s for s in specs if not _matches(s, value)), None)
    if failed is None:
        return
    where = f"{cls.__name__}.{field.public_name}"
    expected = getattr(failed, "value", _NO_DEFAULT)
    if expected is _NO_DEFAULT:
        default = _fixed_value(field)
        if default is not None and all(_matches(s, default) for s in specs):
            expected = default
    if expected is not _NO_DEFAULT:
        raise ValueError(
            f"{where} is always {expected!r}, so a value whose "
            f"{field.public_name} is {value!r} cannot be read as a "
            f"{cls.__name__}."
        )
    # Keep the name of a predicate and drop its address.
    text = re.sub(r"<function (\S+) at 0x[0-9a-f]+>", r"\1", failed.text)
    raise ValueError(
        f"{where} cannot be {value!r}: a {cls.__name__} is what "
        f"{field.public_name}={text} selects, so this value cannot be read "
        f"as one."
    )


def _refuse_unknown_keys(cls: type, other: tx.Mapping) -> None:
    """Refuse a mapping with keys that name no field of `cls`."""
    known = {
        field.public_name
        for field in _fields(cls, init_vars=True)
        if not field.init or field.kw
    }
    unknown = [key for key in other if key not in known]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise TypeError(
            f"{cls.__name__} has no field named {names}; expected some "
            f"of {', '.join(repr(k) for k in sorted(known))}."
        )


# --- Converter --------------------------------------------------------
# The converter lets fields that are data models be set from mappings or
# compatible instances.


@register_converter(DataModelBase)
class DataModelConverter(Converter[DataModelBase, tx.Any]):
    """Converter to a [`DataModelBase`][], applied to fields of that type.

    An instance of the target class is returned unchanged. Any other value
    goes through [`DataModelBase.from_any`][], so a mapping or a parent
    instance is read field by field rather than passed to the constructor.
    A `TypeError` or `ValueError` raised during conversion becomes a
    [`ConversionError`][bagof.converters.ConversionError] that keeps the
    original message.
    """

    DEFAULT = DataModelBase

    def __call__(self, value: tx.Any) -> DataModelBase:
        """Convert a value to an instance of the target class."""
        # Unlike `Converter.__call__`, go through `from_any`, and keep the
        # message so that a field error says why the value was refused.
        if isinstance(value, self.origin):
            return value
        try:
            return self.origin.from_any(value)
        except ConversionError:
            raise
        except (TypeError, ValueError) as e:
            raise self.value_error(value, str(e)) from e
