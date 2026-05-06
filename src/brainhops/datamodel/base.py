__all__ = ["DataModelBase"]

# stdlib
from collections.abc import Mapping

# externals
import typing_extensions as _tx

# internals
from brainhops._ext.struct import Struct, HIDE_IF_NONE
from brainhops._ext.struct.converters import (
    register as register_converter,
    ObjectConverter,
)


class DataModelBase(
    Struct,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):
    """Base class for all data models."""

    # We use this base class to set options that we want to propagate to
    # all classes in the hierarchy.

    @classmethod
    def from_dict(cls, other: Mapping, *args, **kwargs) -> "DataModelBase":
        """
        Create an instance of the class from a dictionary-like object.

        Only keys in the dictionary that match keyword-like fields of
        this class will be used.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the values in the dictionary.
        """
        for key, value in cls.__struct_fields__.items():
            if value.init and value.kw and key in other:
                kwargs.setdefault(key, other[key])
        return cls(*args, **kwargs)

    @classmethod
    def from_instance(cls, other: "DataModelBase", *args, **kwargs) -> "DataModelBase":
        """
        Create an instance of the class from an instance of a similar
        class.

        Only attibutes of the other instance that match keyword-like
        fields of this class will be used.

        Additional positional and/or keyword arguments can be provided,
        and will take precedence over the attributes in the instance.
        """
        for key, value in cls.__struct_fields__.items():
            if value.init and value.kw and hasattr(other, key):
                kwargs.setdefault(key, getattr(other, key))
        return cls(*args, **kwargs)

    @classmethod
    def from_other(cls, other: _tx.Any, *args, **kwargs) -> "DataModelBase":
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
class DataModelConverter(ObjectConverter[DataModelBase]):

    _DEFAULT = DataModelBase

    def _convert(self, value: _tx.Any) -> DataModelBase:
        if not isinstance(value, self.type):
            return self.type.from_other(value)
        return value
