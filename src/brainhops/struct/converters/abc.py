__all__ = ["HintConverter"]

from abc import ABC, abstractmethod
from collections import abc
import typing_extensions as _tx

from .constants import MISSING
from .utils import ConversionError
from .utils import _get_origin, _issubclass, _isinstance, _HINT_TO_TYPE


T = _tx.TypeVar("T")


class HintConverter(_tx.Generic[T], ABC):

    __slots__ = ("type",)

    _CONVERTERS = {}
    _DEFAULT = MISSING

    @_tx.overload
    def __new__(cls) -> _tx.Self: ...

    @_tx.overload
    def __new__(cls, hint: _tx.Type[T]) -> _tx.Self: ...

    @_tx.overload
    def __init__(cls) -> None: ...

    @_tx.overload
    def __init__(cls, hint: _tx.Type[T]) -> None: ...

    def __new__(cls, hint=MISSING):
        if cls is HintConverter:
            if isinstance(hint, HintConverter):
                return hint
            else:
                cls = _get_converter(hint)
        return super().__new__(cls)

    def __init__(self, type=MISSING):
        if type is MISSING:
            type = self._DEFAULT
        if type is MISSING:
            raise TypeError("Type hint is required for this converter")
        self.type = type
        if not self._init_check():
            raise TypeError(self._init_error_message())

    @property
    def origin(self) -> _tx.Type:
        origin = _get_origin(self.type)
        origin = _HINT_TO_TYPE.get(origin, origin)
        return origin
    
    @property
    def args(self) -> tuple:
        return _tx.get_args(self.type)
    
    def _init_check(self) -> bool:
        return True

    def _init_error_message(self) -> str:
        return f"Invalid type for converter: {self.type}"

    def __call__(self, value: _tx.Any) -> T:
        try:
            return self._convert(value)
        except Exception as e:
            raise ConversionError(self._convert_error_message(value)) from e
        
    @abstractmethod
    def _convert(self, value: _tx.Any) -> T:
        return NotImplemented

    def _convert_error_message(self, value: _tx.Any) -> str:
        return f"Value {value} cannot be converted to type {self.type}"


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


def _get_converter(hint: _tx.Any) -> type:
    """Get a converter class.

    Parameters
    ----------
    hint : Any
        The target type or type hint.

    Returns
    -------
    adaptor : type
        The adaptor class.
    """
    CONVERTERS = HintConverter._CONVERTERS

    # Unwrap annotated types
    hint = _get_origin(hint, unfold=_tx.Annotated)

    # Find it directly
    if hint in CONVERTERS:
        return CONVERTERS[hint]

    # Find the best matching adaptor (via issubclass)
    NOHINT = object()
    best_hint, best_adaptor = NOHINT, NOHINT
    for adaptor_hint, adaptor in CONVERTERS.items():
        if _issubclass(hint, adaptor_hint):
            if best_hint is NOHINT:
                best_hint = adaptor_hint
                best_adaptor = adaptor
            elif _issubclass(adaptor_hint, best_hint):
                best_hint = adaptor_hint
                best_adaptor = adaptor
        elif _issubclass(
            _get_origin(hint),
            _get_origin(adaptor_hint)
        ):
            if best_hint is NOHINT:
                best_hint = adaptor_hint
                best_adaptor = adaptor
            elif _issubclass(
                _get_origin(adaptor_hint),
                _get_origin(best_hint)
            ):
                best_hint = adaptor_hint
                best_adaptor = adaptor

    if best_adaptor is not NOHINT:
        CONVERTERS[hint] = best_adaptor
        return best_adaptor

    # Find the best matching adaptor (via isinstance)
    for adaptor_hint, adaptor in CONVERTERS.items():
        origin = _get_origin(adaptor_hint, unfold="all")
        if _isinstance(hint, origin):
            if best_hint is NOHINT:
                best_hint = origin
                best_adaptor = adaptor
            elif _issubclass(origin, best_hint):
                best_hint = origin
                best_adaptor = adaptor

    if best_adaptor is not NOHINT:
        CONVERTERS[hint] = best_adaptor
        return best_adaptor

    raise KeyError(f"No adaptor registered for {hint}")
