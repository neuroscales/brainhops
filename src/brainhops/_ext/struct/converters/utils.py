from collections import abc

import typing_extensions as _tx

T = _tx.TypeVar("T")


class ConversionError(ValueError):
    ...


_HINT_TO_TYPE = {
    _tx.Iterable: abc.Iterable,
    _tx.Iterator: abc.Iterator,
    _tx.Sequence: abc.Sequence,
    _tx.MutableSequence: abc.MutableSequence,
    _tx.Set: abc.MutableSet,
    _tx.FrozenSet: abc.Set,
    _tx.Mapping: abc.Mapping,
    _tx.MutableMapping: abc.MutableMapping,
    _tx.List: list,
    _tx.Tuple: tuple,
    _tx.Dict: dict,
}


def _get_origin(type: _tx.Any, unfold: _tx.Any = None) -> _tx.Any:
    origin = _tx.get_origin(type)
    if origin is None:
        return type
    if unfold == "all":
        if _tx.get_args(type):
            return _get_origin(_tx.get_args(type)[0], unfold=unfold)
    if unfold:
        if not isinstance(unfold, (list, tuple, set)):
            unfold = (unfold,)
        if origin in unfold:
            return _get_origin(_tx.get_args(type)[0], unfold=unfold)
    return origin


def _issubclass(cls: _tx.Any, base: _tx.Any) -> bool:
    # Robust version of issubclass
    if not isinstance(cls, type) or not isinstance(base, type):
        return False
    cls = _get_origin(cls, unfold=_tx.Annotated)
    base = _get_origin(base, unfold=_tx.Annotated)
    try:
        return issubclass(cls, base)
    except TypeError:
        return False


def _isinstance(obj: _tx.Any, type: _tx.Any) -> bool:
    if type is _tx.Any:
        return True
    origin = _get_origin(type)
    origin = _HINT_TO_TYPE.get(origin, origin)
    try:
        return isinstance(obj, origin)
    except TypeError:
        return False
