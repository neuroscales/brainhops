__all__ = ["HintConverter", "ConversionError", "register"]

import types as _t
from collections import abc
from types import NoneType

import typing_extensions as _tx

T = _tx.TypeVar("T")


class ValidationError(ValueError):
    pass


class HintValidator(_tx.Generic[T]):

    __slots__ = ("type",)

    _VALIDATORS = {}

    def __new__(cls, type: _tx.Type[T]) -> _tx.Self:
        if cls is not HintValidator:
            return super().__new__(cls)
        origin = _get_origin(type)
        if origin is _tx.Annotated:
            type = _tx.get_args(type)[0]
            origin = _get_origin(type)
        cls = cls._VALIDATORS.get(origin, cls)
        return super().__new__(cls)

    def __init__(self, type: _tx.Type[T]) -> None:
        self.type = type
        if not self._init_check(type):
            raise TypeError(self._init_error_message(type))

    def _init_check(self, type: _tx.Any) -> bool:
        return True

    def _init_error_message(self, type: _tx.Any) -> str:
        return f"Invalid type for validator: {type}"

    def _validate(self, value: _tx.Any) -> T:
        if not isinstance(value, self.type):
            raise ValidationError
        return value

    def _validate_error_message(self, value: _tx.Any) -> str:
        return f"Value {value} is not of type {self.type}"

    def __call__(self, value: _tx.Any) -> T:
        try:
            return self._validate(value)
        except Exception as e:
            raise ValidationError(self._validate_error_message(value)) from e


def _get_origin(type: _tx.Any) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    return origin


@_tx.overload
def register(
    *origins: _tx.Tuple[_tx.Any],
    validator: None = None
) -> _tx.Callable[[_tx.Type[HintValidator]], _tx.Type[HintValidator]]: ...


@_tx.overload
def register(
    *origins: _tx.Tuple[_tx.Any],
    validator: _tx.Type[HintValidator]
) -> _tx.Type[HintValidator]: ...


def register(*origins, validator=None):
    if validator is None:
        return lambda cls: register(*origins, validator=cls)
    for origin in origins:
        HintValidator._VALIDATORS[origin] = validator
    return validator


# ----------------------------------------------------------------------
#
#   Meta
#
# ----------------------------------------------------------------------


@register(_tx.Any)
class AnyValidator(HintValidator[_tx.Any]):

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) is _tx.Any

    def _validate(self, value: _tx.Any) -> _tx.Any:
        return value


@register(_tx.Union, _t.UnionType)
class UnionValidator(HintValidator[T]):

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) in (_tx.Union, _t.UnionType)

    def _validate_error_message(self, value: _tx.Any) -> str:
        return (
            f"Value {value} is not of any of the types "
            f"in {self.type}"
        )

    def _validate(self, value: _tx.Any) -> T:
        for arg in _tx.get_args(self.type):
            validator = HintValidator(arg)
            try:
                return validator(value)
            except Exception as e:
                continue
        raise e


@register(_tx.Optional)
class OptionalValidator(HintValidator[T]):

    def _init_check(self, type):
        return _get_origin(type) is _tx.Optional

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"OptionalValidator can only be used with Optional types,  "
            f"got {type}"
        )

    def _validate(self, value: _tx.Any) -> T:
        if value is None:
            return value
        validator = HintValidator(_tx.get_args(self.type)[0])
        return validator(value)


@register(None, NoneType)
class NoneValidator(HintValidator[None]):

    def __init__(self, type: _tx.Union[None, NoneType] = None) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return type is None or type is NoneType

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"NoneValidator can only be used with None or NoneType, "
            f"got {type}"
        )

    def _validate(self, value: _tx.Any) -> bool:
        if value is not None:
            raise ValidationError
        return value


# ----------------------------------------------------------------------
#
#   Builtin bases
#
# ----------------------------------------------------------------------


@register(type, _tx.Type)
class TypeValidator(HintValidator[T]):

    def _init_check(self, type_):
        return _get_origin(type_) in (type, _tx.Type)

    def _validate(self, value: _tx.Any) -> T:
        if not isinstance(value, type):
            raise ValidationError(f"Value {value} is not a type")
        if _tx.get_args(self.type):
            if not issubclass(value, _tx.get_args(self.type)[0]):
                raise ValidationError(
                    f"Value {value} is not a subclass of "
                    f"{_tx.get_args(self.type)[0]}"
                )
        return value


@register(object)
class ObjectValidator(HintValidator[object]):

    def _init_check(self, type):
        return issubclass(type, object)

    def _validate(self, value: _tx.Any) -> object:
        if not isinstance(value, self.type):
            raise ValueError(
                f"Value {value} is not an instance of {self.type}"
            )
        return value

# ----------------------------------------------------------------------
#
#   Builtin collections
#
# ----------------------------------------------------------------------


@register(list, _tx.List)
class ListValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = list) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (list, _tx.List)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"ListValidator can only be used with List types, got {type}"
        )

    def _validate(self, value: _tx.Any) -> T:
        if not isinstance(value, list):
            raise ValidationError
        args = _tx.get_args(self.type)
        if args:
            validator = HintValidator(args[0])
            return [validator(v) for v in value]
        return value


@register(tuple, _tx.Tuple)
class TupleValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = tuple) -> None:
        super().__init__(type)

    def _init_check(self, type: _tx.Any) -> bool:
        return _get_origin(type) in (tuple, _tx.Tuple)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"TupleValidator can only be used with Tuple types, "
            f"got {type}"
        )

    def _validate(self, value: _tx.Any) -> T:
        if not isinstance(value, tuple):
            raise ValidationError
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
            raise ValidationError(
                f"Value {value} does not have the same length as the "
                f"Tuple type {self.type}"
            )
        converters = [HintValidator(arg) for arg in args]
        return tuple(validator(v) for validator, v in zip(converters, value))


@register(set, _tx.Set)
class SetValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = set) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (set, _tx.Set)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"SetValidator can only be used with Set types, got {type}"
        )

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, set):
            raise ValidationError
        args = _tx.get_args(self.type)
        if args:
            validator = HintValidator(args[0])
            return set(validator(v) for v in value)
        return value


@register(frozenset, _tx.FrozenSet)
class FrozenSetValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = frozenset) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (frozenset, _tx.FrozenSet)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"FrozenSetValidator can only be used with FrozenSet types, "
            f"got {type}"
        )

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, frozenset):
            raise ValidationError
        args = _tx.get_args(self.type)
        if args:
            validator = HintValidator(args[0])
            return frozenset(validator(v) for v in value)
        return value


@register(dict, _tx.Dict)
class DictValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = dict) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (dict, _tx.Dict)

    def _init_error_message(self, type: _tx.Any) -> str:
        return (
            f"DictValidator can only be used with Dict types, got {type}"
        )

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, dict):
            raise ValidationError
        args = _tx.get_args(self.type)
        if args:
            key_validator = HintValidator(args[0])
            value_validator = HintValidator(args[1])
            return {
                key_validator(k): value_validator(v) for k, v in value.items()
            }
        return value


# ----------------------------------------------------------------------
#
#   ABC collections
#
# ----------------------------------------------------------------------


@register(abc.Sequence, _tx.Sequence)
class AbcSequenceValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.Sequence) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Sequence, _tx.Sequence)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.Sequence):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return type(value)(validator(v) for v in value)


@register(abc.MutableSequence, _tx.MutableSequence)
class AbcMutableSequenceValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.MutableSequence) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableSequence, _tx.MutableSequence)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.MutableSequence):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return type(value)(validator(v) for v in value)


@register(abc.MutableSet, _tx.MutableSet)
class AbcMutableSetValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.MutableSet) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableSet, _tx.MutableSet)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.MutableSet):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return type(value)(validator(v) for v in value)


@register(abc.Set)
class AbcSetValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.Set) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Set,)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.Set):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return type(value)(validator(v) for v in value)


@register(abc.Mapping, _tx.Mapping)
class AbcMappingValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.Mapping) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Mapping, _tx.Mapping)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.Mapping):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        key_validator = HintValidator(args[0])
        value_validator = HintValidator(args[1])
        return type(value)({
            key_validator(k): value_validator(v) for k, v in value.items()
        })


@register(abc.MutableMapping, _tx.MutableMapping)
class AbcMutableMappingValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.MutableMapping) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.MutableMapping, _tx.MutableMapping)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.MutableMapping):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        key_validator = HintValidator(args[0])
        value_validator = HintValidator(args[1])
        return type(value)({
            key_validator(k): value_validator(v) for k, v in value.items()
        })


@register(abc.Iterable, _tx.Iterable)
class AbcIterableValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.Iterable) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Iterable, _tx.Iterable)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.Iterable):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return list(validator(v) for v in value)


@register(abc.Iterator, _tx.Iterator)
class AbcIteratorValidator(HintValidator[T]):

    def __init__(self, type: _tx.Type[T] = abc.Iterator) -> None:
        super().__init__(type)

    def _init_check(self, type):
        return _get_origin(type) in (abc.Iterator, _tx.Iterator)

    def _validate(self, value: _tx.Any) -> bool:
        if not isinstance(value, abc.Iterator):
            raise ValidationError
        args = _tx.get_args(self.type)
        if not args:
            return value
        validator = HintValidator(args[0])
        return list(validator(v) for v in value)
