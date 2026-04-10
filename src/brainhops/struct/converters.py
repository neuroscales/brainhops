__all__ = ["HintConverter", "ConversionError"]

import types as _t
import typing_extensions as _tx
import numbers
import fractions
from collections import abc
from types import NoneType

T = _tx.TypeVar("T")


class ConversionError(ValueError):
    ...


class HintConverter(_tx.Generic[T]):

    __slots__ = ("type",)

    _CONVERTERS = {}

    def __new__(cls, type: _tx.Type[T]) -> _tx.Self:
        if cls is not HintConverter:
            return super().__new__(cls)
        origin = _get_origin(type)
        if origin is _tx.Annotated:
            type = _tx.get_args(type)[0]
            origin = _get_origin(type)
        cls = cls._CONVERTERS.get(origin, cls)
        return super().__new__(cls)

    def __init__(self, type: _tx.Type[T]) -> None:
        self.type = type
        if not self._init_check(type):
            raise TypeError(self._init_error_message(type))
    
    def _init_check(self, type: _tx.Any) -> bool:
        return True
    
    def _init_error_message(self, type: _tx.Any) -> str:
        return f"Invalid type for converter: {type}"

    def _convert(self, value: _tx.Any) -> T:
        return self.type(value)

    def _convert_error_message(self, value: _tx.Any) -> str:
        return f"Value {value} cannot be converted to type {self.type}"

    def __call__(self, value: _tx.Any) -> T:
        try:
            return self._convert(value)
        except Exception as e:
            raise ConversionError(self._convert_error_message(value)) from e


def _get_origin(type: _tx.Any) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    return origin


@_tx.overload
def _register(
    *origins: _tx.Tuple[_tx.Any], 
    converter: None = None
) -> _tx.Callable[[_tx.Type[HintConverter]], _tx.Type[HintConverter]]: ...


@_tx.overload
def _register(
    *origins: _tx.Tuple[_tx.Any], 
    converter: _tx.Type[HintConverter]
) -> _tx.Type[HintConverter]: ...


def _register(*origins, converter=None):
    if converter is None:
        return lambda cls: _register(*origins, converter=cls)
    for origin in origins:
        HintConverter._CONVERTERS[origin] = converter
    return converter


# ----------------------------------------------------------------------
#
#   Meta
#
# ----------------------------------------------------------------------


@_register(_tx.Any)
class AnyConverter(HintConverter[_tx.Any]):

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) is _tx.Any

    def _convert(self, value: _tx.Any) -> _tx.Any:
        return value
    

@_register(_tx.Union, _t.UnionType)
class UnionConverter(HintConverter[T]):

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) in (_tx.Union, _t.UnionType)

    def _convert_error_message(self, value: _tx.Any) -> str:
        return (
            f"Value {value} cannot be converted to any of the types "
            f"in {self.type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        for arg in _tx.get_args(self.type):
            converter = HintConverter(arg)
            try:
                return converter(value)
            except Exception as e:
                continue
        raise e
    

@_register(_tx.Optional)
class OptionalConverter(HintConverter[T]):
        
    def _init_check(self, type):
        return _get_origin(type) is _tx.Optional

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"OptionalConverter can only be used with Optional types,  "
            f"got {type}"
        )
        
    def _convert(self, value: _tx.Any) -> T:
        if value is None:
            return None
        converter = HintConverter(_tx.get_args(self.type)[0])
        return converter(value)


@_register(None, NoneType)
class NoneConverter(HintConverter[None]):

    def __init__(self, type: _tx.Union[None, NoneType] = None) -> None:
        super().__init__(type)
    
    def _init_check(self, type):
        return type is None or type is NoneType

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"NoneConverter can only be used with None or NoneType, "
            f"got {type}"
        )

    def _convert(self, value: _tx.Any) -> None:
        if value is not None:
            raise ConversionError(f"Value {value} is not None")
        return None


# ----------------------------------------------------------------------
#
#   Builtin bases
#
# ----------------------------------------------------------------------


@_register(object)
class ObjectConverter(HintConverter[object]):

    def _init_check(self, type):
        return type is object

    def _convert(self, value: _tx.Any) -> object:
        if not isinstance(value, object):
            raise ConversionError(f"Value {value} is not an object")


@_register(type, _tx.Type)
class TypeConverter(HintConverter[type]):

    def _init_check(self, type_):
        return _get_origin(type_) in (type, _tx.Type)

    def _convert(self, value: _tx.Any) -> type:
        if not isinstance(value, type):
            raise ConversionError(f"Value {value} is not a type")
        if _tx.get_args(self.type):
            if not issubclass(value, _tx.get_args(self.type)[0]):
                raise ConversionError(
                    f"Value {value} is not a subclass of "
                    f"{_tx.get_args(self.type)[0]}"
                )
        return value


# ----------------------------------------------------------------------
#
#   Numbers
#
# ----------------------------------------------------------------------


@_register(numbers.Number)
class NumberConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = numbers.Number) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) is numbers.Number
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, numbers.Number):
            for type in (int, fractions.Fraction, float, complex, bool):
                try:
                    return type(value)
                except Exception as e:
                    ...
        raise e
    

@_register(numbers.Complex)
class ComplexConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = numbers.Complex) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) is numbers.Complex
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, numbers.Complex):
            for type in (int, fractions.Fraction, float, complex, bool):
                try:
                    return type(value)
                except Exception as e:
                    ...
        raise e
    

@_register(numbers.Real)
class RealConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = numbers.Real) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) is numbers.Real
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, numbers.Real):
            for type in (int, fractions.Fraction, float, bool):
                try:
                    return type(value)
                except Exception as e:
                    ...
        raise e
    

@_register(numbers.Rational)
class RationalConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = numbers.Rational) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) is numbers.Rational
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, numbers.Rational):
            for type in (int, fractions.Fraction, bool):
                try:
                    return type(value)
                except Exception as e:
                    ...
        raise e
    

@_register(numbers.Integral)
class IntegralConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = numbers.Integral) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) is numbers.Integral
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, numbers.Integral):
            for type in (int, bool):
                try:
                    return type(value)
                except Exception as e:
                    ...
        raise e


# ----------------------------------------------------------------------
#
#   Builtin collections
#
# ----------------------------------------------------------------------


@_register(list, _tx.List)
class ListConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = list) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (list, _tx.List)
    
    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"ListConverter can only be used with List types, got {type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        args = _tx.get_args(self.type)
        if not args:
            if not isinstance(value, list):
                value = list(value)
            return value
        converter = HintConverter(args[0])
        return [converter(v) for v in value]


@_register(tuple, _tx.Tuple)
class TupleConverter(HintConverter[T]):

    def __init__(self, type: _tx.Type[T] = tuple) -> None:
        super().__init__(type)

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) in (tuple, _tx.Tuple)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"TupleConverter can only be used with Tuple types, "
            f"got {type}"
        )
    
    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, tuple):
            value = tuple(value)
        args = _tx.get_args(self.type)
        if not args or args == (...,):
            return value
        if ... in args:
            args = list(args)
            pre = args[:args.index(...)]
            post = args[args.index(...)+1:]
            if not pre:
                args = post[:1] * max(0, len(value) - len(post)) + post
            else:
                args = pre + pre[-1:] * max(0, len(value) - len(pre)) + post
        if len(args) != len(value):
            raise ConversionError(
                f"Value {value} does not have the same length as the "
                f"Tuple type {self.type}"
            )
        converters = [HintConverter(arg) for arg in args]
        return tuple(
            converter(v) for converter, v in zip(converters, value)
        )


@_register(set, _tx.Set)
class SetConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = set) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (set, _tx.Set)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"SetConverter can only be used with Set types, got {type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        args = _tx.get_args(self.type)
        if not args:
            if not isinstance(value, set):
                 value = set(value)
            return value
        converter = HintConverter(args[0])
        return {converter(v) for v in value}
    

@_register(frozenset, _tx.FrozenSet)
class FrozenSetConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = frozenset) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (frozenset, _tx.FrozenSet)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"FrozenSetConverter can only be used with FrozenSet types, "
            f"got {type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        args = _tx.get_args(self.type)
        if not args:
            if not isinstance(value, frozenset):
                value = frozenset(value)
            return value
        converter = HintConverter(args[0])
        return frozenset(converter(v) for v in value)


@_register(dict, _tx.Dict)
class DictConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = dict) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (dict, _tx.Dict)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"DictConverter can only be used with Dict types, got {type}"
        )

    def _convert(self, value: _tx.Any) -> T:
        args = _tx.get_args(self.type)
        if not args:
            if not isinstance(value, dict):
                value = dict(value)
            return value
        key_converter = HintConverter(args[0])
        value_converter = HintConverter(args[1])
        return {
            key_converter(k): value_converter(v) for k, v in value.items()
        }
    

# ----------------------------------------------------------------------
#
#   ABC collections
#
# ----------------------------------------------------------------------


@_register(abc.Sequence, _tx.Sequence)
class AbcSequenceConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.Sequence) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Sequence, _tx.Sequence)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.Sequence):
            value = tuple(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return type(value)([converter(v) for v in value])
    

@_register(abc.MutableSequence, _tx.MutableSequence)
class AbcMutableSequenceConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.MutableSequence) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableSequence, _tx.MutableSequence)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.MutableSequence):
            value = list(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return type(value)([converter(v) for v in value])
    

@_register(abc.MutableSet, _tx.MutableSet)
class AbcMutableSetConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.MutableSet) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableSet, _tx.MutableSet)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.MutableSet):
            value = set(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return type(value)(converter(v) for v in value)


@_register(abc.Set)
class AbcSetConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.Set) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Set,)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.Set):
            value = frozenset(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return type(value)(converter(v) for v in value)


@_register(abc.Mapping, _tx.Mapping)
class AbcMappingConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.Mapping) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Mapping, _tx.Mapping)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.Mapping):
            value = dict(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        key_converter = HintConverter(args[0])
        value_converter = HintConverter(args[1])
        return type(value)({
            key_converter(k): value_converter(v) for k, v in value.items()
        })
    

@_register(abc.MutableMapping, _tx.MutableMapping)
class AbcMutableMappingConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.MutableMapping) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableMapping, _tx.MutableMapping)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.MutableMapping):
            value = dict(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        key_converter = HintConverter(args[0])
        value_converter = HintConverter(args[1])
        return type(value)({
            key_converter(k): value_converter(v) for k, v in value.items()
        })
    

@_register(abc.Iterable, _tx.Iterable)
class AbcIterableConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.Iterable) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Iterable, _tx.Iterable)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.Iterable):
            value = iter(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return (converter(v) for v in value)
    

@_register(abc.Iterator, _tx.Iterator)
class AbcIteratorConverter(HintConverter[T]):
    
    def __init__(self, type: _tx.Type[T] = abc.Iterator) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Iterator, _tx.Iterator)

    def _convert(self, value: _tx.Any) -> T:
        if not isinstance(value, abc.Iterator):
            value = iter(value)
        args = _tx.get_args(self.type)
        if not args:
            return value
        converter = HintConverter(args[0])
        return (converter(v) for v in value)