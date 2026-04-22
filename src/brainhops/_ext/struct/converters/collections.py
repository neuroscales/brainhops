__all__ = [
    "IterableConverter",
    "IteratorConverter",
    "SequenceConverter",
    "MutableSequenceConverter",
    "AbstractSetConverter",
    "AbstractMutableSetConverter",
    "MappingConverter",
    "MutableMappingConverter",
    "TupleConverter",
    "ListConverter",
    "FrozenSetConverter",
    "SetConverter",
    "DictConverter",
]
# stdlib
from collections import abc
from inspect import isabstract

# externals
import typing_extensions as _tx

# internals
from .abc import HintConverter, _register
from .base import ObjectConverter
from .utils import ConversionError


# ----------------------------------------------------------------------
#
#   ABC collections
#
# ----------------------------------------------------------------------


ITERABLE = _tx.TypeVar("ITERABLE", bound=abc.Iterable)
ITERATOR = _tx.TypeVar("ITERATOR", bound=abc.Iterator)
SEQUENCE = _tx.TypeVar("SEQUENCE", bound=abc.Sequence)
MSEQUENCE = _tx.TypeVar("MSEQUENCE", bound=abc.MutableSequence)
ABCMSET = _tx.TypeVar("ABCMSET", bound=abc.MutableSet)
ABCSET = _tx.TypeVar("ABCSET", bound=abc.Set)
MAPPING = _tx.TypeVar("MAPPING", bound=abc.Mapping)
MMAPPING = _tx.TypeVar("MMAPPING", bound=abc.MutableMapping)


@_register(abc.Iterable, _tx.Iterable)
class IterableConverter(ObjectConverter[ITERABLE]):
    
    _DEFAULT = abc.Iterable

    def _init_check(self):
        return issubclass(self.origin, self._DEFAULT)

    def _convert(self, value: _tx.Any) -> ITERABLE:
        origin, args = self.origin, self.args
        if not isinstance(value, origin):
            raise ConversionError
        if not args:
            return value
        converter = HintConverter(args[0])
        return (converter(v) for v in value)
    

@_register(abc.Iterator, _tx.Iterator)
class IteratorConverter(IterableConverter[ITERATOR]):
    
    _DEFAULT = abc.Iterator

    def _convert(self, value: _tx.Any) -> ITERATOR:
        origin, args = self.origin, self.args
        if not isinstance(value, origin):
            value = iter(value)
        if not args:
            return value
        converter = HintConverter(args[0])
        return iter(converter(v) for v in value)
    

@_register(abc.Sequence, _tx.Sequence)
class SequenceConverter(IterableConverter[SEQUENCE]):

    _DEFAULT = abc.Sequence
    _FALLBACK = tuple

    def _convert(self, value: _tx.Any) -> SEQUENCE:
        origin, args = self.origin, self.args
        if not isinstance(value, origin):
            if isabstract(origin):
                origin = self._FALLBACK
            value = origin(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return type(value)(converter(v) for v in value)
    

@_register(abc.MutableSequence, _tx.MutableSequence)
class MutableSequenceConverter(SequenceConverter[MSEQUENCE]):
    
    _DEFAULT = abc.MutableSequence
    _FALLBACK = list


@_register(abc.Set, _tx.Set)
class AbstractSetConverter(SequenceConverter[ABCSET]):
    
    _DEFAULT = abc.Set
    _FALLBACK = frozenset
    

@_register(abc.MutableSet, _tx.MutableSet)
class AbstractMutableSetConverter(
    AbstractSetConverter[ABCMSET], 
    MutableSequenceConverter[ABCMSET]
):
    
    _DEFAULT = abc.MutableSet
    _FALLBACK = set


@_register(abc.Mapping, _tx.Mapping)
class MappingConverter(IterableConverter[MAPPING]):
    
    _DEFAULT = abc.Mapping
    _FALLBACK = dict
    
    def _convert(self, value: _tx.Any) -> MAPPING:
        origin, args = self.origin, self.args
        if not isinstance(value, origin):
            if isabstract(origin):
                origin = self._FALLBACK
            value = dict(value)
        if not args:
            return value
        key_converter = HintConverter(args[0])
        value_converter = HintConverter(args[1])
        return type(value)(
            (key_converter(k), value_converter(v)) 
            for k, v in value.items()
        )


@_register(abc.MutableMapping, _tx.MutableMapping)
class MutableMappingConverter(MappingConverter[MMAPPING]):
    
    _DEFAULT = abc.MutableMapping
    _FALLBACK = dict


# ----------------------------------------------------------------------
#
#   Builtin collections
#
# ----------------------------------------------------------------------


TUPLE = _tx.TypeVar("TUPLE", bound=tuple)
LIST = _tx.TypeVar("LIST", bound=list)
SET = _tx.TypeVar("SET", bound=set)
FROZENSET = _tx.TypeVar("FROZENSET", bound=frozenset)
DICT = _tx.TypeVar("DICT", bound=dict)


@_register(tuple, _tx.Tuple)
class TupleConverter(SequenceConverter[TUPLE]):

    _DEFAULT = tuple
    
    def _convert(self, value: _tx.Any) -> TUPLE:
        origin, args = self.origin, self.args
        if not isinstance(value, origin):
            if isabstract(origin):
                origin = self._FALLBACK
            value = origin(value)
            
        if not args or args == (...,):
            return value
        
        # Expand ellipsis if present
        if ... in args:
            args = list(args)
            pre = args[:args.index(...)]
            post = args[args.index(...)+1:]
            if not pre:
                args = post[:1] * max(0, len(value) - len(post)) + post
            else:
                args = pre + pre[-1:] * max(0, len(value) - len(pre)) + post

        # Check length
        if len(args) != len(value):
            raise ConversionError(
                f"Value {value} does not have the same length as the "
                f"Tuple type {self.type}"
            )
        
        # Convert each element
        converters = [HintConverter(arg) for arg in args]
        return type(value)(
            converter(v) for converter, v in zip(converters, value)
        )
    

@_register(list, _tx.List)
class ListConverter(MutableSequenceConverter[LIST]):

    _DEFAULT = list
    

@_register(frozenset, _tx.FrozenSet)
class FrozenSetConverter(AbstractSetConverter[FROZENSET]):
    
    _DEFAULT = frozenset


@_register(set, _tx.Set)
class SetConverter(AbstractMutableSetConverter[SET]):
    
    _DEFAULT = set


@_register(dict, _tx.Dict)
class DictConverter(MutableMappingConverter[DICT]):
    
    _DEFAULT = dict
