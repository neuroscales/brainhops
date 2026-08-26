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
from .numpy_typing import ArrayProtocol, DTypeLike, DTypeProtocol, dtype

# optionals
if _tx.TYPE_CHECKING:
    import cupy as cp
    import cupy.typing as cpt
else:
    try:
        import cupy as cp
    except ImportError:
        cp = None

    try:
        import cupy.typing as cpt
    except ImportError:
        cpt = None


SHAPE = _tx.TypeVar("SHAPE", bound=tuple)
DTYPE = _tx.TypeVar("DTYPE", bound=type)


if cpt:

    ndarray = cp.ndarray
    NDArray = cpt.NDArray

elif cp:

    class ndarray(cp.ndarray, _tx.Generic[SHAPE, DTYPE]): ...

    NDArray = ndarray[_tx.Tuple[_tx.Any, ...], dtype[DTYPE]]

else:

    class ndarray(_tx.Generic[SHAPE, DTYPE]): ...

    NDArray = ndarray[_tx.Tuple[_tx.Any, ...], dtype[DTYPE]]
