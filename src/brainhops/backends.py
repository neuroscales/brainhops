"""Array backends: the packages that hold the values of arrays.

brainhops holds its arrays with NumPy, CuPy or Dask. The backend setting
decides which package a conversion produces and in which package new
arrays are created. The design is described in the memo
`docs/design/array-backends.md`.

A setting is one of the following names, or `None`.

| Setting       | In memory or on a GPU | At once or lazily        |
|---------------|-----------------------|--------------------------|
| `None`        | as the array is       | as the array is          |
| `"auto"`      | in memory             | chosen by size           |
| `"auto+cupy"` | on a GPU              | chosen by size           |
| `"numpy"`     | in memory             | at once                  |
| `"cupy"`      | on a GPU              | at once                  |
| `"dask"`      | in memory             | lazily                   |
| `"dask+cupy"` | on a GPU              | lazily, in CuPy chunks   |

`None`, the default, keeps every array that is already loaded in its
backend, and resolves a lazy array of a file with `"auto"`. `"auto"`
gives a NumPy array to an array of at most [`get_auto_threshold`][]
bytes, and a Dask array to a larger one when Dask is installed. A module
(`numpy`, `cupy` or `dask.array`) is accepted as the name of its backend.
"""

# stdlib
import math
from contextlib import contextmanager
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
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

_SETTINGS = ("auto", "auto+cupy", "numpy", "cupy", "dask", "dask+cupy")
"""The names of the backend settings, besides `None`."""

_CREATED = {
    None: "numpy",
    "auto": "numpy",
    "auto+cupy": "cupy",
    "numpy": "numpy",
    "cupy": "cupy",
    "dask": "dask",
    "dask+cupy": "dask",
}
"""The backend in which a setting creates new arrays."""

Setting = tx.Optional[tx.Union[str, ModuleType]]
"""A backend setting: one of the names of `_SETTINGS`, a module, or `None`."""

_SETTING: tx.Optional[str] = None
"""The current backend setting."""

_THRESHOLD: int = 64 * 2**20
"""The largest number of bytes that `"auto"` keeps at once, 64 MiB."""


class _Current:
    """The default of an argument that means the current setting."""

    def __repr__(self) -> str:
        return "<current setting>"


_CURRENT = _Current()


def available_backends() -> tx.Tuple[str, ...]:
    """Return the names of the installed backends, most preferred first.

    A backend counts as installed when both its array package and its
    ndimage package are installed. The ndimage package of dask is
    [`brainhops._core.dask_ndimage`][], which ships with brainhops, so dask
    needs nothing else.
    """
    return tuple(
        name
        for name in _PRIORITY
        if all(module is not None for module in _MODULES[name])
    )


def best_backend(*backends) -> ModuleType:
    """Return the array module of the most preferred of the given backends.

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


# ----------------------------------------------------------------------
#   SETTINGS
# ----------------------------------------------------------------------


def _setting_name(setting: Setting) -> tx.Optional[str]:
    """Return the name of a setting given by name, by module or as `None`.

    Raises
    ------
    ValueError
        If the argument is not a setting.
    """
    if setting is None:
        return None
    if isinstance(setting, str):
        if setting not in _SETTINGS:
            raise ValueError(
                f"Unsupported backend: {setting!r}. The settings are None, "
                + ", ".join(repr(name) for name in _SETTINGS)
                + "."
            )
        return setting
    if isinstance(setting, ModuleType):
        return backend_name(setting)
    raise ValueError(f"Unsupported backend: {setting!r}")


def _require(setting: tx.Optional[str]) -> None:
    """Check that the packages that a setting needs are installed.

    `"auto"` needs nothing besides NumPy, because it falls back on NumPy
    when Dask is missing.

    Raises
    ------
    ImportError
        If a package that the setting needs is not installed.
    """
    needed = {
        "auto+cupy": ("cupy",),
        "numpy": ("numpy",),
        "cupy": ("cupy",),
        "dask": ("dask",),
        "dask+cupy": ("dask", "cupy"),
    }.get(setting, ())
    for name in needed:
        array, image = _MODULES[name]
        if array is None:
            raise ImportError(f"The {name} backend is not installed")
        if image is None:
            raise ImportError(
                f"The {name} backend needs {_NDIMAGE_PACKAGE[name]}, "
                "which is not installed, so it cannot interpolate"
            )


def get_backend() -> tx.Optional[str]:
    """Return the current backend setting, a name or `None`."""
    return _SETTING


def set_backend(backend: Setting) -> None:
    """Set the current backend setting, given by name, by module or as `None`.

    Passing the result of [`get_array_backend`][] selects the backend of an
    array. The installed backends are listed by [`available_backends`][].

    Raises
    ------
    ValueError
        If the argument is not a setting.
    ImportError
        If a package that the setting needs is not installed.
    """
    global _SETTING
    name = _setting_name(backend)
    _require(name)
    _SETTING = name


def get_auto_threshold() -> int:
    """Return the number of bytes up to which `"auto"` keeps arrays at once."""
    return _THRESHOLD


def set_auto_threshold(nbytes: int) -> None:
    """Set the number of bytes up to which `"auto"` keeps arrays at once.

    Raises
    ------
    ValueError
        If the number is negative.
    """
    global _THRESHOLD
    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError(f"The threshold must not be negative: {nbytes}")
    _THRESHOLD = nbytes


@contextmanager
def backend(
    backend: tx.Union[Setting, _Current] = _CURRENT,
    threshold: tx.Optional[int] = None,
) -> tx.Generator[tx.Optional[str], None, None]:
    """Select a backend setting inside a `with` block.

    The previous setting and threshold are restored on exit. Without an
    argument, the current setting is kept, whereas `backend(None)` selects
    the setting `None`.

    Parameters
    ----------
    backend : str or module, optional
        The setting to select. By default, the current setting is kept.
    threshold : int, optional
        The number of bytes up to which `"auto"` keeps arrays at once. By
        default, the current threshold is kept.

    Yields
    ------
    str or None
        The setting in use.
    """
    global _SETTING, _THRESHOLD
    old_setting, old_threshold = _SETTING, _THRESHOLD
    try:
        if not isinstance(backend, _Current):
            set_backend(backend)
        if threshold is not None:
            set_auto_threshold(threshold)
        yield _SETTING
    finally:
        _SETTING, _THRESHOLD = old_setting, old_threshold


def backend_name(
    backend: tx.Union[str, ModuleType, ArrayProtocol],
) -> str:
    """Return the name of the package of a name, module or array.

    The name is `"numpy"`, `"cupy"` or `"dask"`.

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


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------


def _is_loaded(x: tx.Any) -> bool:
    """Tell whether an object is a NumPy, CuPy or Dask array."""
    return (
        (np is not None and isinstance(x, np.ndarray))
        or (cp is not None and isinstance(x, cp.ndarray))
        or (da is not None and isinstance(x, da.Array))
    )


def _on_gpu(x: tx.Any) -> bool:
    """Tell whether an array is a CuPy array or a Dask array of CuPy chunks."""
    if cp is None:
        return False
    if isinstance(x, cp.ndarray):
        return True
    return (
        da is not None
        and isinstance(x, da.Array)
        and isinstance(getattr(x, "_meta", None), cp.ndarray)
    )


def _nbytes(x: tx.Any) -> int:
    """Return the number of bytes that the values of an array take."""
    shape = tuple(int(n) for n in x.shape)
    return math.prod(shape) * np.dtype(x.dtype).itemsize


def _current(setting: tx.Union[Setting, _Current]) -> tx.Optional[str]:
    """Return the name of a setting, or the current one by default."""
    if isinstance(setting, _Current):
        return _SETTING
    return _setting_name(setting)


def is_lazy(
    nbytes: int, backend: tx.Union[Setting, _Current] = _CURRENT
) -> bool:
    """Tell whether a setting reads a lazy array of a given size lazily.

    Readers that can give either a loaded array or a lazy one ask this
    question about the array of a file that they have not read yet.

    Parameters
    ----------
    nbytes : int
        The number of bytes that the values take.
    backend : str or module, optional
        The setting. By default, the current setting is used.

    Returns
    -------
    bool
        Whether the array is resolved to a Dask array.
    """
    name = _current(backend)
    if name in (None, "auto", "auto+cupy"):
        return da is not None and nbytes > _THRESHOLD
    return name in ("dask", "dask+cupy")


def resolve_backend(
    x: tx.Any, backend: tx.Union[Setting, _Current] = _CURRENT
) -> str:
    """Return the backend to which a setting converts an array.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    backend : str or module, optional
        The setting. By default, the current setting is used.

    Returns
    -------
    str
        One of `"numpy"`, `"cupy"`, `"dask"` and `"dask+cupy"`.
    """
    name = _current(backend)
    if name is None:
        if _is_loaded(x):
            if da is not None and isinstance(x, da.Array):
                return "dask+cupy" if _on_gpu(x) else "dask"
            return "cupy" if _on_gpu(x) else "numpy"
        name = "auto"
    if name in ("auto", "auto+cupy"):
        gpu = name == "auto+cupy"
        if da is not None and _nbytes(x) > _THRESHOLD:
            return "dask+cupy" if gpu else "dask"
        return "cupy" if gpu else "numpy"
    return name


def _to_numpy(x: tx.Any) -> tx.Any:
    """Return the values of an array as a NumPy array."""
    if da is not None and isinstance(x, da.Array):
        x = x.compute()
    if cp is not None and isinstance(x, cp.ndarray):
        return x.get()
    return np.asarray(x)


def _to_cupy(x: tx.Any) -> tx.Any:
    """Return the values of an array as a CuPy array."""
    if da is not None and isinstance(x, da.Array):
        x = x.compute()
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(np.asarray(x))


def _to_dask(x: tx.Any, gpu: bool) -> tx.Any:
    """Return an array as a Dask array, of CuPy chunks when `gpu` is true."""
    if isinstance(x, da.Array):
        lazy = x
    elif cp is not None and isinstance(x, cp.ndarray):
        if gpu:
            return da.from_array(x, chunks="auto", asarray=False)
        lazy = da.from_array(x.get(), chunks="auto")
    elif isinstance(x, np.ndarray):
        lazy = da.from_array(x, chunks="auto")
    elif hasattr(x, "to_dask"):
        lazy = x.to_dask()
    else:
        meta = np.empty((0,) * len(x.shape), np.dtype(x.dtype))
        lazy = da.from_array(
            x, chunks="auto", asarray=True, fancy=False, meta=meta
        )
    if gpu and not _on_gpu(lazy):
        meta = cp.empty((0,) * lazy.ndim, lazy.dtype)
        return lazy.map_blocks(cp.asarray, meta=meta)
    if not gpu and _on_gpu(lazy):
        meta = np.empty((0,) * lazy.ndim, lazy.dtype)
        return lazy.map_blocks(cp.asnumpy, meta=meta)
    return lazy


def to_backend(
    x: tx.Any, backend: tx.Union[Setting, _Current] = _CURRENT
) -> tx.Any:
    """Convert an array, or a lazy array of a file, as a setting says.

    Under `None`, a loaded array is returned as it is, and a lazy array is
    resolved with `"auto"`. Converting to NumPy or CuPy computes a Dask
    array and reads a lazy array. Converting to Dask reads nothing.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    backend : str or module, optional
        The setting. By default, the current setting is used.

    Returns
    -------
    array
        The converted array.

    Raises
    ------
    ImportError
        If a package that the setting needs is not installed.
    """
    target = resolve_backend(x, backend)
    _require(target)
    if target == "numpy":
        return _to_numpy(x)
    if target == "cupy":
        return _to_cupy(x)
    return _to_dask(x, gpu=target == "dask+cupy")


# ----------------------------------------------------------------------
#   PACKAGES
# ----------------------------------------------------------------------


def get_array_backend(
    x: tx.Optional[tx.Union[ArrayProtocol, ModuleType, str]] = None,
) -> ModuleType:
    """Return the array package of an array, module or setting.

    The result is `numpy`, `cupy` or `dask.array`. `None`, or an object of
    no known backend, gives the package in which the current setting
    creates new arrays: NumPy under `None` and `"auto"`, CuPy under
    `"cupy"` and `"auto+cupy"`, and Dask under `"dask"` and `"dask+cupy"`.

    Raises
    ------
    TypeError
        If the argument is a module that belongs to no known backend.
    ValueError
        If the argument is a string that names no setting.
    """
    if x is None:
        x = _CREATED[_SETTING]

    if isinstance(x, str):
        if x not in _MODULES:
            x = _CREATED[_setting_name(x)]
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
    own graph, so two distinct dask arrays never share memory. `None` is
    returned for an array-like of unknown backend, such as a lazy file
    proxy.
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
        # A dask array whose computed result would view a source numpy
        # array is not counted as sharing memory with it.
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

    A missing package is never replaced by another one, because scipy, for
    example, would load a lazy dask volume into memory.
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
    """Return the ndimage package of an array, module or setting.

    The result is `scipy.ndimage`, `cupyx.scipy.ndimage` or
    [`brainhops._core.dask_ndimage`][]. `None`, or an object of no known
    backend, gives the package of the backend in which the current setting
    creates new arrays.

    Raises
    ------
    ImportError
        If the ndimage package of the backend is not installed.
    TypeError
        If the argument is a module that belongs to no known backend.
    ValueError
        If the argument is a string that names no setting.
    """
    if x is None:
        x = _CREATED[_SETTING]

    if isinstance(x, str):
        if x not in _MODULES:
            x = _CREATED[_setting_name(x)]
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
