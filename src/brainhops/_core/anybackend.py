# dependencies
import typing_extensions as tx

# core
from .backends import ArrayProtocol, array_backend

# typing
_ArrayFunc: tx.TypeAlias = tx.Callable[[ArrayProtocol, tx.Any], ArrayProtocol]


def _on_backend(func: str) -> _ArrayFunc:
    """Return a function that calls the backend's implementation of `func`."""

    def wrapper(array: ArrayProtocol, *args, **kwargs) -> ArrayProtocol:
        xp = array_backend(array)
        backend_func = getattr(xp, func)
        return backend_func(array, *args, **kwargs)

    return wrapper


def squeeze(
    array: ArrayProtocol, axis: tx.Union[int, tx.Sequence[int], None]
) -> ArrayProtocol:
    """Squeeze singleton axes of an array."""
    ndim = array.ndim
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        axis = tuple(i for i in range(ndim) if array.shape[i] == 1)
    non_singleton = [i for i in axis if array.shape[i] != 1]
    if non_singleton:
        raise ValueError(f"Cannot squeeze non singleting axes {non_singleton}")
    index = tuple(0 if i == axis else slice(None) for i in range(ndim))
    return array[index]


moveaxis: _ArrayFunc = _on_backend("moveaxis")
