"""The base class shared by every data model, and its converter."""

__all__ = ["DataModelBase", "DataModelConverter"]

# externals
from collections.abc import Mapping

import typing_extensions as tx
from bagof.converters import ConversionError, Converter, register_converter
from bagof.magic import HIDE_IF_NONE, Field, Magic, fields


class DataModelBase(
    Magic,
    convert=True,
    mapping=False,
    repr=HIDE_IF_NONE,
):
    """Base class for all data models."""

    # We use this base class to set options that we want to propagate to
    # all classes in the hierarchy.

    @classmethod
    def from_dict(cls, other: tx.Mapping, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from a dictionary-like object.

        Only keys in the dictionary that match keyword-like fields of
        this class will be used. Other keys are ignored, but see
        [`from_other`][brainhops.datamodel.base.DataModelBase.from_other],
        which refuses them.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the values in the dictionary.

        A key naming a field that this class fixes (a field that cannot
        be passed to its constructor) is checked instead of used: a
        dictionary that sets it to anything other than `None` or the
        value of this class is refused with a [`ValueError`][].
        """

        def read(field: tx.Any) -> tx.Any:
            return other.get(field.public_name, _ABSENT)

        return _build(cls, read, args, kwargs)

    @classmethod
    def from_instance(cls, other: tx.Self, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from an instance of a similar
        class.

        Only attributes of the other instance that match keyword-like
        fields of this class will be used. An attribute that is `None`
        is unset, and leaves the default of this class in place.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the attributes in the instance.

        Unless the other instance is already an instance of this class,
        an attribute naming a field that this class fixes (a field that
        cannot be passed to its constructor) is checked instead of used:
        an instance that sets it to anything other than `None` or the
        value of this class is refused with a [`ValueError`][]. A
        generic `Axis` whose orientation is right-to-left, for example,
        cannot be read as a `LeftToRightAxis`.
        """
        # An instance of this class (or of a subclass) is one already:
        # whatever it fixes, it fixes the way this class allows.
        check_fixed = not isinstance(other, cls)

        def read(field: tx.Any) -> tx.Any:
            if not field.init and not check_fixed:
                return _ABSENT
            value = getattr(other, field.name, None)
            return _ABSENT if value is None else value

        return _build(cls, read, args, kwargs)

    @classmethod
    def from_other(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from any object that can be
        interpreted as a dictionary, or an instance of a similar class,
        or an arguments to be passed to the constructor.

        A similar class is this class or one of its parents within the
        data model. Any other object, including an instance of a parent
        that is not a data model (such as a plain `object`), is passed
        to the constructor.

        Unlike [`from_dict`][brainhops.datamodel.base.DataModelBase.from_dict],
        a dictionary with a key that matches no field of this class is
        refused with a [`TypeError`][] naming the keys, so that a
        misspelt key is not silently dropped.
        """
        if isinstance(other, Mapping):
            _refuse_unknown_keys(cls, other)
            return cls.from_dict(other, *args, **kwargs)
        elif isinstance(other, cls):
            return cls.from_instance(other, *args, **kwargs)
        elif isinstance(other, DataModelBase) and issubclass(cls, type(other)):
            return cls.from_instance(other, *args, **kwargs)
        else:
            return cls(other, *args, **kwargs)


# A value read from nowhere: the source carries nothing for the field.
_ABSENT = object()

# What `Field.default` holds when a field has no default.
_NO_DEFAULT = Field().default


def _build(
    cls: tx.Type[DataModelBase],
    read: tx.Callable[[tx.Any], tx.Any],
    args: tx.Tuple[tx.Any, ...],
    kwargs: tx.Dict[str, tx.Any],
) -> tx.Any:
    """The body shared by `from_dict` and `from_instance`.

    `read(field)` returns the value that the source carries for `field`,
    or `_ABSENT`. A keyword-like field takes it, unless `kwargs` already
    sets that field. A field that the class fixes checks it instead:
    before the instance is built when the fixed value can be read off
    the field, and after otherwise.
    """
    later = {}
    for field in fields(cls):
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


def _fixed_value(field: tx.Any) -> tx.Any:
    """The value a fixed field takes, or `_NO_DEFAULT` if unknown."""
    if callable(field.factory):
        return field.factory()
    return field.default


def _check_fixed(
    cls: type, field: tx.Any, value: tx.Any, expected: tx.Any
) -> None:
    """Refuse `value` unless it is the value `cls` fixes for `field`.

    `value` is converted the way the field converts before it is
    compared, so that a dictionary can describe a fixed data model.
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


def _refuse_unknown_keys(cls: type, other: tx.Mapping) -> None:
    """Refuse a mapping with keys that match no field of `cls`."""
    known = {
        field.public_name
        for field in fields(cls)
        if not field.init or field.kw
    }
    unknown = [key for key in other if key not in known]
    if unknown:
        names = ", ".join(repr(key) for key in unknown)
        raise TypeError(
            f"{cls.__name__} has no field named {names}; expected some "
            f"of {', '.join(repr(k) for k in sorted(known))}."
        )


@register_converter(DataModelBase)
class DataModelConverter(Converter[DataModelBase, tx.Any]):
    """Converts a value to a [`DataModelBase`][] instance.

    This converter is registered for [`DataModelBase`][], and is used
    automatically wherever a field is typed with [`DataModelBase`][] or
    one of its subclasses and conversion is enabled. A value that is
    already an instance of the target type is returned unchanged. Any
    other value is converted through [`DataModelBase.from_other`][] of
    the target type, so that a mapping or an instance of a parent class
    is read field by field instead of being passed to the constructor
    as its first argument.

    As with the default converter, a [`TypeError`][] or
    [`ValueError`][] raised while converting surfaces as a
    [`ConversionError`][bagof.converters.ConversionError] (a
    `ValueConversionError`), with the original as its cause. Unlike the
    default converter, it keeps the original message.
    """

    DEFAULT = DataModelBase

    def __call__(self, value: tx.Any) -> DataModelBase:
        """Convert `value` to an instance of the target type."""
        # `Converter.__call__` would call the target class itself. Route
        # through `from_other` instead, with the same wrapping: a
        # `ConversionError` passes through untouched, and a plain
        # `TypeError` or `ValueError` becomes a `ValueConversionError`.
        # Its message is kept, so that the field error built from it says
        # why the value was refused rather than "Invalid value.".
        if isinstance(value, self.origin):
            return value
        try:
            return self.origin.from_other(value)
        except ConversionError:
            raise
        except (TypeError, ValueError) as e:
            raise self.value_error(value, str(e)) from e
