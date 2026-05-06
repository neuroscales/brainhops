__all__ = [
    "DTypeProtocol",
    "DTypeLike",
    "dtype",
    "ArrayProtocol",
    "ArrayLike",
    "NDArray",
    "ndarray",
]
# stdlib
from numbers import Number

# externals
import typing_extensions as _tx

# optionals
if _tx.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
else:
    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        import numpy.typing as npt
    except ImportError:
        npt = None


SHAPE = _tx.TypeVar("SHAPE", bound=tuple)
DTYPE = _tx.TypeVar("DTYPE", bound=type)


@_tx.runtime_checkable
class ArrayProtocol(_tx.Protocol):

    def __array__(self, dtype=None): ...


@_tx.runtime_checkable
class DTypeProtocol(_tx.Protocol):

    dtype: DTYPE


_ArrayLike = _tx.Union[Number, _tx.Sequence, ArrayProtocol]


if npt:

    dtype = np.dtype
    ndarray = np.ndarray
    NDArray = npt.NDArray
    ArrayLike = npt.ArrayLike
    DTypeLike = npt.DTypeLike

elif np:

    ArrayLike = _ArrayLike
    DTypeLike = _tx.Union[np.dtype, type, str, DTypeProtocol]

    class dtype(np.dtype, _tx.Generic[DTYPE]):
        pass


    class ndarray(np.ndarray, _tx.Generic[SHAPE, DTYPE]):
        pass


    NDArray = ndarray[_tx.Tuple[int, ...], dtype[DTYPE]]

else:

    DTypeLike = _tx.Union[type, str, DTypeProtocol]

    class dtype(_tx.Generic[DTYPE]):
        pass


    class ndarray(_tx.Generic[SHAPE, DTYPE]):
        pass


    NDArray = ndarray[_tx.Tuple[int, ...], dtype[DTYPE]]
