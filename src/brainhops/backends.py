# stdlib
from contextlib import contextmanager
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
from brainhops._core.dependencies import cp, cpndi, da, dkndi, np, npndi

#: The array package of each backend, with the ndimage package it needs.
#: A backend needs both: one that can hold an array but not interpolate it
#: is not a backend brainhops can use.
_MODULES = {
    "numpy": (np, npndi),
    "cupy": (cp, cpndi),
    "dask": (da, dkndi),
}

#: The distribution that supplies each backend's ndimage package, named in
#: the error raised when it is missing.
_NDIMAGE_PACKAGE = {
    "numpy": "scipy",
    "cupy": "cupy",
    "dask": "dask-image",
}

_PRIORITY = ("dask", "cupy", "numpy")


def available_backends() -> tx.Tuple[str, ...]:
    """The backends that can be selected, most preferred first.

    A backend appears only when both of its packages are installed. dask
    without `dask-image` is therefore not available: it could interpolate
    only by materializing the array it was handed, which for a lazily read
    volume is the very allocation dask is used to avoid. It is deactivated
    rather than quietly served by scipy.
    """
    return tuple(
        name
        for name in _PRIORITY
        if all(module is not None for module in _MODULES[name])
    )


#: The backend used when none is selected: the most preferred one that is
#: actually available.
_BACKEND = (available_backends() or ("numpy",))[0]


def best_backend(*backends) -> ModuleType:
    """Return the array backend with highest priority from a list."""
    references = map(get_array_backend, available_backends())
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
    """Set the current array backend

    Raises
    ------
    ValueError
        If `backend` does not name a backend.
    ImportError
        If the backend's array package, or the ndimage package it needs, is
        not installed. Both are required -- see
        [available_backends][brainhops.backends.available_backends].
    """
    global _BACKEND
    if backend not in _MODULES:
        raise ValueError(f"Unsupported backend: {backend}")
    array, image = _MODULES[backend]
    if array is None:
        raise ImportError(f"The {backend} backend is not installed")
    if image is None:
        raise ImportError(
            f"The {backend} backend needs {_NDIMAGE_PACKAGE[backend]}, "
            "which is not installed, so it cannot interpolate"
        )
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


def _ndimage_of(name: str) -> ModuleType:
    """The ndimage package of a backend, by name.

    A missing package is reported, never substituted. Serving the dask
    backend with scipy would materialize the array it was handed, which for
    a lazily read volume is the allocation dask is used to avoid, so a dask
    backend without `dask-image` raises here and is absent from
    [available_backends][brainhops.backends.available_backends].
    """
    array, image = _MODULES[name]
    if array is None:
        raise ImportError(f"The {name} backend is not installed")
    if image is None:
        raise ImportError(
            f"The {name} backend needs {_NDIMAGE_PACKAGE[name]}, which is "
            "not installed, so it cannot interpolate"
        )
    return image


def get_ndimage_backend(
    x: tx.Optional[tx.Union[ArrayProtocol, ModuleType, str]] = None,
) -> ModuleType:
    """Determine the ndimage package for a given array

    One of: scipy.ndimage, cupyx.scipy.ndimage, dask_image.ndinterp

    Raises
    ------
    ImportError
        If the backend the array belongs to has no ndimage package
        installed. It is never stood in for by another backend's.
    """
    if x is None:
        x = get_backend()

    # Guess from backend name
    if isinstance(x, str):
        return _ndimage_of(x)

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
            return _ndimage_of("numpy")
        if x is cp:
            return _ndimage_of("cupy")
        if x is da:
            return _ndimage_of("dask")

        raise TypeError(f"Unknown module: {x}")

    # Guess from array type
    if cp and isinstance(x, cp.ndarray):
        return _ndimage_of("cupy")
    if np and isinstance(x, np.ndarray):
        return _ndimage_of("numpy")
    if da and isinstance(x, da.Array):
        return _ndimage_of("dask")

    return get_ndimage_backend()
