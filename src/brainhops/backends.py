# stdlib
from contextlib import contextmanager
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
from brainhops._core.dependencies import cp, cpndi, da, dkndi, np, npndi

_BACKEND = "dask"
_PRIORITY = ("dask", "cupy", "numpy")


def best_backend(*backends) -> ModuleType:
    """Return the array backend with highest priority from a list."""
    references = map(get_array_backend, _PRIORITY)
    backends = tuple(map(get_array_backend, backends))
    for ref in references:
        if ref in backends:
            return ref
    raise ValueError(f"No supported backends found in {backends}")


@contextmanager
def backend(
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> tx.Generator[str, None, None]:
    """Context manager to temporarily set the array backend"""
    global _BACKEND
    old_backend = _BACKEND
    if backend is not None:
        set_backend(backend)
    try:
        yield _BACKEND
    finally:
        _BACKEND = old_backend


def get_backend() -> str:
    """Get the current array backend"""
    return _BACKEND


def set_backend(backend: str) -> None:
    """Set the current array backend"""
    global _BACKEND
    if backend not in ["numpy", "cupy", "dask"]:
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "numpy" and np is None:
        raise ImportError("NumPy is not available")
    if backend == "cupy" and cp is None:
        raise ImportError("CuPy is not available")
    if backend == "dask" and da is None:
        raise ImportError("Dask is not available")
    _BACKEND = backend


def to_backend(
    x: ArrayProtocol, backend: tx.Optional[tx.Union[str, ModuleType]] = None
) -> ArrayProtocol:
    backend = get_array_backend(backend)
    return backend.asarray(x)


def get_array_backend(
    x: tx.Optional[tx.Union[ArrayProtocol, ModuleType, str]] = None,
) -> ModuleType:
    """Determine the array package for a given array

    One of: numpy, cupy, dask.array
    """
    if x is None:
        x = get_backend()

    # Guess from backend name
    if isinstance(x, str):
        return {"numpy": np, "cupy": cp, "dask": da}[x]

    # Guess from module type
    if isinstance(x, ModuleType):
        # Already an array module
        if x is np:
            return np
        if x is cp:
            return cp
        if x is da:
            return da

        # Guess from image module?
        if x is npndi:
            return np
        if x is cpndi:
            return cp
        if x is dkndi:
            return da

        raise TypeError(f"Unknown module: {x}")

    # Guess from array type
    if np and isinstance(x, np.ndarray):
        return np
    if cp and isinstance(x, cp.ndarray):
        return cp
    if da and isinstance(x, da.Array):
        return da

    return get_array_backend()


def get_ndimage_backend(
    x: tx.Optional[tx.Union[ArrayProtocol, ModuleType, str]] = None,
) -> ModuleType:
    """Determine the ndimage package for a given array

    One of: scipy.ndimage, cupyx.scipy.ndimage, dask_image.ndinterp
    """
    if x is None:
        x = get_backend()

    # Guess from backend name
    if isinstance(x, str):
        return {"numpy": npndi, "cupy": cpndi, "dask": dkndi}[x]

    # Guess from module type
    if isinstance(x, ModuleType):
        # Already a image module?
        if x is npndi:
            return npndi
        if x is cpndi:
            return cpndi
        if x is dkndi:
            return dkndi

        # Guess from array module
        if x is np:
            return npndi
        if x is cp:
            return cpndi
        if x is da:
            return dkndi

        raise TypeError(f"Unknown module: {x}")

    # Guess from array type
    if cp and cpndi and isinstance(x, cp.ndarray):
        return cpndi
    if np and npndi and isinstance(x, np.ndarray):
        return npndi
    if da and isinstance(x, da.Array):
        if dkndi:
            return dkndi
        if npndi:
            return npndi

    return get_ndimage_backend()
