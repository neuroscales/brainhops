__all__ = [
    "DTypeProtocol",
    "DTypeLike",
    "dtype",
    "ArrayProtocol",
    "ArrayLike",
    "NDArray",
    "ndarray",
]
# externals
import typing_extensions as _tx

# internals
from .numpy_typing import ArrayProtocol, DTypeProtocol, DTypeLike, dtype

# optionals
if _tx.TYPE_CHECKING:
    import dask.array as da
else:
    try:
        import dask.array as da
    except ImportError:
        da = None


SHAPE = _tx.TypeVar("SHAPE", bound=tuple)
DTYPE = _tx.TypeVar("DTYPE", bound=type)


if da:

    class Array(da.Array, _tx.Generic[SHAPE, DTYPE]): ...

    NDArray = Array[_tx.Tuple[_tx.Any, ...], dtype[DTYPE]]

else:

    class Array(_tx.Generic[SHAPE, DTYPE]): ...

    NDArray = Array[_tx.Tuple[_tx.Any, ...], dtype[DTYPE]]
