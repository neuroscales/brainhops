"""The base class shared by every data model, and its converter."""

__all__ = ["DataModelBase", "DataModelConverter"]

# externals
from collections.abc import Mapping

import typing_extensions as tx
from bagof.converters import Converter, register_converter
from bagof.magic import HIDE_IF_NONE, Magic, fields


class DataModelBase(
    Magic,
    convert=True,
    mapping=HIDE_IF_NONE,
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
        """
        for field in fields(cls):
            key = field.public_name
            if field.init and field.kw and key in other:
                kwargs.setdefault(key, other[key])
        return cls(*args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Self, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from an instance of a similar
        class.

        Only attributes of the other instance that match keyword-like
        fields of this class will be used.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the attributes in the instance.
        """
        for field in fields(cls):
            key, _key = field.public_name, field.name
            if field.init and field.kw and hasattr(other, _key):
                value = getattr(other, _key)
                kwargs.setdefault(key, value)
        return cls(*args, **kwargs)

    @classmethod
    def from_other(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """
        Create an instance of the class from any object that can be
        interpreted as a dictionary, or an instance of a similar class,
        or an arguments to be passed to the constructor.
        """
        if isinstance(other, Mapping):
            return cls.from_dict(other, *args, **kwargs)
        elif isinstance(other, cls):
            return cls.from_instance(other, *args, **kwargs)
        elif issubclass(cls, type(other)):
            return cls.from_instance(other, *args, **kwargs)
        else:
            return cls(other, *args, **kwargs)


@register_converter(DataModelBase)
class DataModelConverter(Converter[DataModelBase, tx.Any]):
    """Converts a value to a [`DataModelBase`][] instance.

    This converter is registered for [`DataModelBase`][], and is used
    automatically wherever a field is typed with [`DataModelBase`][] or
    one of its subclasses and conversion is enabled. A value that is
    already an instance of the target type is returned unchanged. Any
    other value is converted through [`DataModelBase.from_other`][].
    """

    _DEFAULT = DataModelBase

    def _convert(self, value: tx.Any) -> DataModelBase:
        if not isinstance(value, self.type):
            return self.type.from_other(value)
        return value
