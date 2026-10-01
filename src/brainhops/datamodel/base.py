"""The base class shared by every data model, and its converter."""

__all__ = ["DataModelBase", "DataModelConverter"]

# externals
from collections.abc import Mapping

import typing_extensions as tx
from bagof.converters import ConversionError, Converter, register_converter
from bagof.core.magic import safe_isinstance
from bagof.magic import HIDE_IF_NONE, Magic, fields


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
        this class will be used.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the values in the dictionary.

        A key naming a field that this class fixes (a field that cannot
        be passed to its constructor) is checked instead of used: a
        dictionary that sets it to anything other than `None` or the
        value of this class is refused with a [`ValueError`][].
        """
        fixed = {}
        for field in fields(cls):
            key = field.public_name
            if key not in other:
                continue
            if field.init and field.kw:
                kwargs.setdefault(key, other[key])
            elif not field.init:
                fixed[field] = other[key]
        return _check_fixed(cls(*args, **kwargs), fixed)

    @classmethod
    def from_instance(cls, other: tx.Self, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from an instance of a similar
        class.

        Only attributes of the other instance that match keyword-like
        fields of this class will be used.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the attributes in the instance.

        An attribute naming a field that this class fixes (a field that
        cannot be passed to its constructor) is checked instead of used:
        an instance that sets it to anything other than `None` or the
        value of this class is refused with a [`ValueError`][]. A
        generic `Axis` whose orientation is right-to-left, for example,
        cannot be read as a `LeftToRightAxis`.
        """
        fixed = {}
        for field in fields(cls):
            key, _key = field.public_name, field.name
            if not hasattr(other, _key):
                continue
            if field.init and field.kw:
                kwargs.setdefault(key, getattr(other, _key))
            elif not field.init:
                fixed[field] = getattr(other, _key)
        return _check_fixed(cls(*args, **kwargs), fixed)

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
        """
        if isinstance(other, Mapping):
            return cls.from_dict(other, *args, **kwargs)
        elif isinstance(other, cls):
            return cls.from_instance(other, *args, **kwargs)
        elif isinstance(other, DataModelBase) and issubclass(cls, type(other)):
            return cls.from_instance(other, *args, **kwargs)
        else:
            return cls(other, *args, **kwargs)


def _check_fixed(obj: DataModelBase, fixed: tx.Mapping) -> DataModelBase:
    """Refuse values that contradict the fields `obj`'s class fixes.

    `fixed` maps each field that the class of `obj` does not take in its
    constructor to the value that the source (a dictionary or an
    instance) carried for it. `None` means "not set", and is accepted.
    Any other value is converted the way the field converts, and must
    then equal the value of `obj`.
    """
    for field, value in fixed.items():
        if value is None:
            continue
        expected = getattr(obj, field.name)
        try:
            if callable(field.converter):
                value = field.converter(value)
            same = bool(value == expected)
        except (TypeError, ValueError):
            same = False
        if not same:
            name = type(obj).__name__
            raise ValueError(
                f"{name}.{field.public_name} is always {expected!r}, so "
                f"a value whose {field.public_name} is {fixed[field]!r} "
                f"cannot be read as a {name}."
            )
    return obj


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
        if safe_isinstance(value, self.origin):
            return value
        try:
            return self.origin.from_other(value)
        except ConversionError:
            raise
        except (TypeError, ValueError) as e:
            raise self.value_error(value, str(e)) from e
