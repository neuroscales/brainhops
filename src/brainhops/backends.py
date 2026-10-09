from contextlib import contextmanager
from types import ModuleType

import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

from brainhops._core.dependencies import cp, cpndi, da, dkndi, np, npndi

# A backend needs both its array package and its ndimage package.
_MODULES = {
    "numpy": (np, npndi),
    "cupy": (cp, cpndi),
    "dask": (da, dkndi),
}

# Distribution that provides each ndimage package, for error messages.
_NDIMAGE_PACKAGE = {
    "numpy": "scipy",
    "cupy": "cupy",
    "dask": "dask",
}

_PRIORITY = ("dask", "cupy", "numpy")


def available_backends() -> tx.Tuple[str, ...]:
    """Return the names of the installed backends, most preferred first.

    A backend is installed when both its array and ndimage packages are.
    Dask uses [`brainhops._core.dask_ndimage`][], so it needs nothing else.
    """
    return tuple(
        name
        for name in _PRIORITY
        if all(module is not None for module in _MODULES[name])
    )


_BACKEND = (available_backends() or ("numpy",))[0]


def best_backend(*backends) -> ModuleType:
    """Return the array module of the most preferred given backend.

    Each argument is anything accepted by [`get_array_backend`][].

    Raises
    ------
    ValueError
        If none of the given backends is installed.
    """
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
    """Select a backend, given by name or module, inside a `with` block.

    The previous backend is restored on exit. `None` keeps the current one.

    Yields
    ------
    str
        The name of the backend in use.
    """
    global _BACKEND
    old_backend = _BACKEND
    if backend is not None:
        set_backend(backend)
    try:
        yield _BACKEND
    finally:
        _BACKEND = old_backend


def get_backend() -> str:
    """Return the name of the current backend."""
    return _BACKEND


def backend_name(
    backend: tx.Union[str, ModuleType, ArrayProtocol],
) -> str:
    """Return the name of the backend of a name, module or array.

    Raises
    ------
    ValueError
        If the argument does not designate a supported backend.
    """
    if isinstance(backend, str):
        if backend not in _MODULES:
            raise ValueError(f"Unsupported backend: {backend}")
        return backend
    module = get_array_backend(backend)
    for name, (array, _image) in _MODULES.items():
        if module is array:
            return name
    raise ValueError(f"Unsupported backend: {backend}")


def set_backend(backend: tx.Union[str, ModuleType]) -> None:
    """Set the current backend, given by name or array module.

    Passing the result of [`get_array_backend`][] selects the backend of an
    array. The installed backends are listed by [`available_backends`][].

    Raises
    ------
    ValueError
        If the argument does not designate a supported backend.
    ImportError
        If the array or ndimage package of the backend is not installed.
    """
    global _BACKEND
    backend = backend_name(backend)
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
    """Return the array package of an array, module or backend name.

    The result is `numpy`, `cupy` or `dask.array`. `None`, or an object of
    no known backend, gives the package of the current backend.

    Raises
    ------
    TypeError
        If the argument is a module that belongs to no known backend.
    """
    if x is None:
        x = get_backend()

    if isinstance(x, str):
        return {"numpy": np, "cupy": cp, "dask": da}[x]

    if isinstance(x, ModuleType):
        if x is np:
            return np
        if x is cp:
            return cp
        if x is da:
            return da

        if x is npndi:
            return np
        if x is cpndi:
            return cp
        if x is dkndi:
            return da

        raise TypeError(f"Unknown module: {x}")

    if np and isinstance(x, np.ndarray):
        return np
    if cp and isinstance(x, cp.ndarray):
        return cp
    if da and isinstance(x, da.Array):
        return da

    return get_array_backend()


def may_share_memory(x: ArrayProtocol, y: ArrayProtocol) -> tx.Optional[bool]:
    """Return whether two arrays might share memory.

    Numpy and cupy arrays use the conservative bounds check of their
    backend, so `False` means certainly disjoint. Host and device arrays
    never share memory. Dask arrays are immutable and writes rebind their
    own graph, so distinct dask arrays never do. `None` is returned for an
    array-like of unknown backend, such as a lazy file proxy.
    """
    if x is y:
        return True
    backends = []
    for array in (x, y):
        if np is not None and isinstance(array, np.ndarray):
            backends.append(np)
        elif cp is not None and isinstance(array, cp.ndarray):
            backends.append(cp)
        elif da is not None and isinstance(array, da.Array):
            backends.append(da)
        else:
            return None
    if da in backends:
        # A computed result that views a source numpy array is ignored.
        return False
    if backends[0] is not backends[1]:
        return False
    return bool(backends[0].may_share_memory(x, y))


def copy_array(x: ArrayProtocol) -> ArrayProtocol:
    """Return a copy of an array, on the same backend, sharing no memory.

    A dask copy is a new object over the same graph, which suffices because
    dask arrays are immutable.
    """
    if hasattr(x, "copy"):
        return x.copy()
    return np.array(x, copy=True)


def _ndimage_of(name: str) -> ModuleType:
    """Return the ndimage package of a backend name, or raise.

    A missing package is never replaced: scipy would load a lazy dask
    volume into memory.
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
    """Return the ndimage package of an array, module or backend name.

    The result is `scipy.ndimage`, `cupyx.scipy.ndimage` or
    [`brainhops._core.dask_ndimage`][]. `None`, or an object of no known
    backend, gives the package of the current backend.

    Raises
    ------
    ImportError
        If the ndimage package of the backend is not installed.
    TypeError
        If the argument is a module that belongs to no known backend.
    """
    if x is None:
        x = get_backend()

    if isinstance(x, str):
        return _ndimage_of(x)

    if isinstance(x, ModuleType):
        if x is npndi:
            return npndi
        if x is cpndi:
            return cpndi
        if x is dkndi:
            return dkndi

        if x is np:
            return _ndimage_of("numpy")
        if x is cp:
            return _ndimage_of("cupy")
        if x is da:
            return _ndimage_of("dask")

        raise TypeError(f"Unknown module: {x}")

    if cp and isinstance(x, cp.ndarray):
        return _ndimage_of("cupy")
    if np and isinstance(x, np.ndarray):
        return _ndimage_of("numpy")
    if da and isinstance(x, da.Array):
        return _ndimage_of("dask")

    return get_ndimage_backend()
