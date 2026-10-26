"""Computational backends.

`brainhops` supports multiple computational backends, which allow users
to make optimal user of their computational infrastructure (threads,
processes, GPUs, computing clusters, etc.).

Backends can be set globally with [`set_backend`][] or locally with the
[`backend`][] context manager. The setting decides in which backend a
computation runs and where its result lives, for example when an image
is resliced or when the lazy array of a file is summed. Reading the
`data` of an object does not convert it: `data` returns the array that
the object stores, and the `to_*` and `as_*` methods convert it on
request. Under the default setting, `"native"`, a computation runs where
its operands live, and operands that live in different backends are
promoted to one backend (see [`promote_backends`][]):

- if one of the operands is a Dask array, the computation is lazy;
- if one of the operands is on a GPU, the computation runs on the GPU;
- otherwise, the computation runs with NumPy.

The lazy array of a file counts as the array that `"auto"` would make
of it.

The following backend settings are supported:

| Setting       | On CPU or GPU memory   | Eagerly or lazily        |
|---------------|------------------------|--------------------------|
| `"native"`    | as the array is        | as the array is          |
| `"auto"`      | CPU memory             | depends on array size    |
| `"auto+cupy"` | GPU memory             | depends on array size    |
| `"numpy"`     | CPU memory             | eagerly                  |
| `"cupy"`      | GPU memory             | eagerly                  |
| `"dask"`      | CPU memory             | lazily, in NumPy chunks  |
| `"dask+cupy"` | GPU memory             | lazily, in CuPy chunks   |

When dask is installed and the `"auto"` setting is selected, arrays that
are larger than [`get_auto_threshold`][] bytes are promoted to Dask arrays,
while smaller arrays are kept as NumPy arrays.

The `"auto+cupy"` setting behaves similarly to `"auto"`, but returns
either a CuPy array, or a Dask array with CuPy chunks.

A setting can be set by a module (`numpy`, `cupy` or `dask.array`)
instead of a name.
"""

# stdlib
import math
from contextlib import contextmanager
from numbers import Integral
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# internals
from brainhops._core import partial
from brainhops._core.dependencies import cp, cpndi, da, dkndi, np, npndi
from brainhops._core.typing import ArrayLike

# typing
if tx.TYPE_CHECKING:
    from cupy import ndarray as cparray
    from cupy.cuda import Device as CudaDevice
    from dask.array import Array as daarray
    from numpy import ndarray as nparray

BackendName: tx.TypeAlias = tx.Literal["numpy", "cupy", "dask"]
"""
The name of a backend supported by `brainhops`.
"""

BackendSetting: tx.TypeAlias = tx.Literal[
    "auto",
    "auto+cupy",
    "numpy",
    "cupy",
    "dask",
    "dask+cupy",
    "native",
]
"""
A backend setting supported by `brainhops`:

- `"native"` uses the current package of each array, without any conversion.
- `"numpy"` performs all computations in memory with NumPy, on the CPU.
- `"cupy"` performs all computations in memory, with CuPy, on the GPU.
- `"dask"` performs all computations lazily with Dask, on the CPU.
- `"dask+cupy"` performs all computations lazily with Dask, on the GPU.
- `"auto"` selects either `"numpy"` or `"dask"` based on array size.
- `"auto+cupy"` selects either `"cupy"` or `"dask+cupy"` based on array size.
"""

BackendLike: tx.TypeAlias = tx.Union[BackendSetting, ModuleType]
"""
A value that defines a backend: the name of a setting, or a backend module.
"""

DeviceLike: tx.TypeAlias = tx.Union[str, int, "CudaDevice"]
"""
A value that defines a device:
the name of a device, an integer, a CudaDevice, or None.
"""

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

# When computing on arrays spread across multiple backends, move them
# all on the same backend with this order of preference.
_PRIORITY = ("dask", "cupy", "numpy")

_BACKEND_MODULES = BackendName.__args__
"""The name of the backend modules."""

_BACKEND_SETTINGS = BackendSetting.__args__
"""The names of the backend settings."""

_CREATED = {
    "native": "numpy",
    "auto": "numpy",
    "auto+cupy": "cupy",
    "numpy": "numpy",
    "cupy": "cupy",
    "dask": "dask",
    "dask+cupy": "dask",
}
"""The backend in which a setting creates new arrays."""

_SETTING: BackendSetting = "native"
"""The current backend setting."""

_THRESHOLD: int = 64 * 2**20
"""The largest number of bytes that `"auto"` keeps at once, 64 MiB."""


def available_backends() -> tx.Tuple[BackendName, ...]:
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


# ----------------------------------------------------------------------
#   SETTINGS
# ----------------------------------------------------------------------


def get_backend() -> BackendSetting:
    """Return the current backend setting."""
    return _SETTING


def set_backend(backend: tx.Optional[BackendLike] = None) -> None:
    """Set the current backend setting, given by name or by module.

    Passing the result of [`get_array_backend`][] selects the backend of an
    array. The installed backends are listed by [`available_backends`][].

    Raises
    ------
    ValueError
        If the argument is not a setting.
    ImportError
        If a package that the setting needs is not installed.
    """
    if backend is not None:
        global _SETTING
        name = _setting_name(backend)
        _require(name)
        _SETTING = name


def get_auto_threshold() -> int:
    """Return the number of bytes up to which `"auto"` keeps arrays at once."""
    return _THRESHOLD


def set_auto_threshold(nbytes: tx.Optional[int] = None) -> None:
    """Set the number of bytes up to which `"auto"` keeps arrays at once.

    Raises
    ------
    ValueError
        If the number is negative.
    """
    if nbytes is not None:
        global _THRESHOLD
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"The threshold must not be negative: {nbytes}")
        _THRESHOLD = nbytes


@contextmanager
def backend(
    backend: tx.Optional[BackendLike] = None,
    threshold: tx.Optional[int] = None,
) -> tx.Generator[BackendSetting, None, None]:
    """Select a backend setting inside a `with` block.

    The previous setting and threshold are restored on exit. Without an
    argument, or with `None`, the current setting is kept, whereas
    `backend("native")` selects the setting that converts nothing.

    !!! example
        ```pycon
        with backend("cupy"):
            x = np.arange(3)
            y = to_backend(x)
        ```

    Parameters
    ----------
    backend : {"native", "numpy", "cupy", "dask[+cupy]", "auto[+cupy]"} | module, optional
        The setting to select. By default, the current setting is kept.
    threshold : int, optional
        The number of bytes up to which `"auto"` keeps arrays at once.
        By default, the current threshold is kept.

    Yields
    ------
    {"native", "numpy", "cupy", "dask[+cupy]", "auto[+cupy]"}
        The setting in use.
    """  # noqa: E501
    global _SETTING, _THRESHOLD
    old_setting, old_threshold = _SETTING, _THRESHOLD
    try:
        set_backend(backend)
        set_auto_threshold(threshold)
        yield _SETTING
    finally:
        _SETTING, _THRESHOLD = old_setting, old_threshold


def _backend_name(
    backend: tx.Union[str, ModuleType, ArrayProtocol],
) -> BackendName:
    """Return the name of the package of a name, module or array.

    Returns
    -------
    {"numpy", "cupy", "dask"}

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
    for name, (array, image) in _MODULES.items():
        if module is array:
            return name
        if module is image:
            return name
    raise ValueError(f"Unsupported backend: {backend}")


def _setting_name(setting: tx.Optional[BackendLike]) -> BackendSetting:
    """Return the name of a setting given by name, by module.

    Raises
    ------
    ValueError
        If the argument is not a setting.
    """
    if setting is None:
        return _SETTING
    if isinstance(setting, str):
        if setting not in _BACKEND_SETTINGS:
            raise ValueError(
                f"Unsupported backend: {setting!r}. The settings are "
                + ", ".join(repr(name) for name in _BACKEND_SETTINGS)
                + "."
            )
        return setting
    if isinstance(setting, ModuleType):
        return _backend_name(setting)
    raise ValueError(f"Unsupported backend: {setting!r}")


def _require(setting: tx.Optional[BackendSetting]) -> None:
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


# ----------------------------------------------------------------------
#   CONVERSIONS
# ----------------------------------------------------------------------


def resolve_backend(
    x: ArrayProtocol, backend: tx.Optional[BackendLike] = None
) -> BackendSetting:
    """Return the backend to which a setting converts an array.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    backend : str | module, optional
        The setting. By default, the current setting is used.

    Returns
    -------
    backend : {"numpy", "cupy", "dask", "dask+cupy"}
    """
    name = _backend_or_current(backend)
    if name == "native":
        if _is_loaded(x):
            if da and isinstance(x, da.Array):
                return "dask+cupy" if _on_gpu(x) else "dask"
            return "cupy" if _on_gpu(x) else "numpy"
        name = "auto"
    if name in ("auto", "auto+cupy"):
        gpu = name == "auto+cupy"
        if da is not None and _nbytes(x) > _THRESHOLD:
            return "dask+cupy" if gpu else "dask"
        return "cupy" if gpu else "numpy"
    return name


def to_numpy(x: ArrayLike) -> "nparray":
    """
    Return the values of an array as a NumPy array.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.

    Returns
    -------
    numpy.ndarray
        The values of the array, on the CPU.
    """
    if _is_numpy(x):
        return x
    if _is_dask(x):
        x = x.compute()
    if _is_cupy(x):
        return x.get()
    return np.asarray(x)


def to_cupy(
    x: ArrayProtocol, device: tx.Union[int, "CudaDevice", None] = None
) -> "cparray":
    """Return the values of an array as a CuPy array.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    device : int, optional
        The device to use. If not specified, the current device is used.

    Returns
    -------
    cupy.ndarray
        The values of the array, on the GPU.
    """
    _require("cupy")
    if _is_cupy(x) and _compatible_device(x, device):
        return x
    if not _is_cupy(x):
        x = to_numpy(x)
    if device is not None:
        device = _normalize_device(device)[1]
        with device:
            x = cp.asarray(x)
    else:
        x = cp.asarray(x)
    return x


def to_dask(
    x: ArrayProtocol, device: tx.Union[str, int, "CudaDevice", None] = "cpu"
) -> "daarray":
    """
    Return an array as a Dask array on the requested device.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    device : {"cpu", "cuda", "cuda:{index}"} or int or None, optional
        The device to use for the chunks.
    """
    to_gpu, device = _normalize_device(device)
    _require("dask+cupy" if to_gpu else "dask")

    if _is_dask(x):
        lazy = x

    elif _is_cupy(x):
        if to_gpu:
            # Wrap cupy array
            x = to_cupy(x, device=device)
            return da.from_array(x, chunks="auto", asarray=False)

        # Wrap numpy array
        lazy = da.from_array(x.get(), chunks="auto")

    elif _is_numpy(x):
        # Wrap numpy array
        lazy = da.from_array(x, chunks="auto")

    elif hasattr(x, "to_dask"):
        # Get dask view
        lazy = x.to_dask()

    else:
        # Wrap any array-like object
        lazy = da.from_array(
            x, chunks="auto", asarray=True, fancy=False, meta=_meta(x)
        )

    current_device = get_device(lazy)
    target_device = (
        device if device is not None else ("cuda" if to_gpu else "cpu")
    )
    if _compatible_device(current_device, target_device):
        return lazy

    if to_gpu:
        device = device if device is not None else current_device
        meta = _meta(lazy, device=device)
        return lazy.map_blocks(partial(to_cupy, device=device), meta=meta)

    return lazy.map_blocks(to_numpy, meta=_meta(lazy))


def to_device(
    x: ArrayProtocol, device: tx.Union[str, int, None] = "cpu"
) -> ArrayProtocol:
    """Copy an array to a GPU or to memory, keeping its package kind.

    A Dask array stays a Dask array, with CuPy or NumPy chunks, and an
    array in memory becomes a CuPy or a NumPy array. An array that is
    already on that device, or that is not an array, is returned as it
    is.
    """
    if not _is_loaded(x):
        return x
    if _compatible_device(x, device):
        return x
    to_gpu, device = _normalize_device(device)
    if da is not None and isinstance(x, da.Array):
        target = device if to_gpu else "cpu"
        return to_dask(x, device=target)
    return to_cupy(x, device=device) if to_gpu else to_numpy(x)


def to_backend(
    x: ArrayProtocol,
    backend: tx.Optional[BackendLike] = None,
    *,
    device: tx.Union[str, int, None] = None,
) -> ArrayProtocol:
    """Convert an array, or a lazy array of a file, as a setting says.

    Under `"native"`, a loaded array is returned as it is, and a lazy array is
    resolved with `"auto"`. Converting to NumPy or CuPy computes a Dask
    array and reads a lazy array. Converting to Dask reads nothing.

    Parameters
    ----------
    x : array-like
        A NumPy, CuPy or Dask array, or a lazy array of a file.
    backend : str | module, optional
        The setting. By default, the current setting is used.
    device : str | int, optional
        The device for converted arrays. The backend supplies its default.

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
        return to_numpy(x)
    if target == "cupy":
        return to_cupy(x, device=device)
    if device is None:
        device = "cuda" if target == "dask+cupy" else "cpu"
    return to_dask(x, device=device)


def promote_backends(*arrays_or_backends: tx.Any) -> tx.Optional[str]:
    """Return the setting under which a computation on arrays runs.

    A computation, such as the reslice of an image, decides once where
    its result lives, so that it reads its inputs and creates its arrays
    there directly rather than moving them later. Under an explicit
    setting, the result follows that setting, which is returned. Under
    the setting `"native"`, the result is lazy when one of the arrays is a
    Dask array, and on a GPU when one of them is on a GPU.
    A lazy array of a file counts as the array that `"auto"` would make
    of it.

    Parameters
    ----------
    *arrays_or_backends : Any
        The voxel arrays of the inputs, or lazy arrays of files. `None`
        is ignored.

    Returns
    -------
    str or None
        One of `"numpy"`, `"cupy"`, `"dask"` and `"dask+cupy"`, the
        current setting when it is not `"native"`, or `None` when there is
        no array.
    """
    if _SETTING != "native":
        return _SETTING
    names = [
        x if isinstance(x, str) else resolve_backend(x, "native")
        for x in arrays_or_backends
        if x is not None
    ]
    if not names:
        return None
    lazy = any(name.startswith("dask") for name in names)
    gpu = any(name.endswith("cupy") for name in names)
    if lazy:
        return "dask+cupy" if gpu else "dask"
    return "cupy" if gpu else "numpy"


def promote_dtypes(*arrays_or_dtypes) -> np.dtype:
    """
    Return the dtype that a computation on arrays promotes to.
    """
    root, *arrays_or_dtypes = arrays_or_dtypes
    if hasattr(root, "dtype"):
        root = root.dtype
    root = np.dtype(root)
    for x in arrays_or_dtypes:
        if hasattr(x, "dtype"):
            x = x.dtype
        x = np.dtype(x)
        root = np.promote_types(root, x)
    return root


def to_promoted_backend(
    *arrays: tx.Unpack[tx.Tuple[ArrayLike, ...]],
) -> tx.Tuple[ArrayLike, ...]:
    """Move arrays to the backend in which a computation combines them.

    When computing on multiple arrays that live on different backends,
    this function moves them all to compatible backends.

    When one of the arrays is on a GPU, the others are copied to it:
    a NumPy array becomes a CuPy array, and a Dask array of NumPy chunks
    becomes a Dask array of CuPy chunks.

    A NumPy array that meets a Dask array in memory is returned as it is,
    because Dask accepts NumPy operands.

    An argument that is not an array, such as `None` or a number, is
    returned as it is.

    Parameters
    ----------
    *arrays : Any
        The arrays that the computation combines.

    Returns
    -------
    tuple
        The arrays, in the same order.
    """
    if not any(_on_gpu(x) for x in arrays if _is_loaded(x)):
        return arrays
    return tuple(to_device(x, "cuda") for x in arrays)


def to_backend_of(x: ArrayProtocol, reference: ArrayProtocol) -> ArrayProtocol:
    """Move an array to the backend of a reference array.

    Parameters
    ----------
    x : Any
        The array to move.
        An argument that is not an array is returned as it is.
    reference : Any
        The array whose device `x` takes.

    Returns
    -------
    Any
        The moved array.
    """
    if not _is_loaded(x) or not _is_loaded(reference):
        return x
    device = get_device(reference)
    if device is None:
        device = "cpu"
    return to_device(x, device=device)


def _is_numpy(x: ArrayProtocol) -> bool:
    """Tell whether an object is a NumPy array."""
    return np and isinstance(x, np.ndarray)


def _is_cupy(x: ArrayProtocol) -> bool:
    """Tell whether an object is a CuPy array."""
    return cp and isinstance(x, cp.ndarray)


def _is_dask(x: ArrayProtocol) -> bool:
    """Tell whether an object is a Dask array."""
    return da and isinstance(x, da.Array)


def _is_loaded(x: ArrayProtocol) -> bool:
    """Tell whether an object is a NumPy, CuPy or Dask array."""
    return _is_numpy(x) or _is_cupy(x) or _is_dask(x)


def _nbytes(x: ArrayProtocol) -> int:
    """Return the number of bytes that the values of an array take."""
    shape = tuple(int(n) for n in x.shape)
    return math.prod(shape) * np.dtype(x.dtype).itemsize


def _backend_or_current(
    setting: tx.Optional[BackendLike] = None,
) -> BackendSetting:
    """Return the name of a setting, or the current one by default."""
    if setting is None:
        return _SETTING
    return _setting_name(setting)


def _is_lazy(
    array_or_nbytes: tx.Union[int, ArrayProtocol],
    backend: tx.Optional[BackendLike] = None,
) -> bool:
    """Tell whether a setting reads a lazy array of a given size lazily.

    Readers that can give either a loaded array or a lazy one ask this
    question about the array of a file that they have not read yet.

    Parameters
    ----------
    array_or_nbytes : int | array-like
        The number of bytes that the values take.
        If an array, `nbytes` is computed from its shape and dtype.
    backend : str or module, optional
        The setting. By default, the current setting is used.

    Returns
    -------
    bool
        Whether the array is resolved to a Dask array.
    """
    name = _backend_or_current(backend)
    if not isinstance(array_or_nbytes, Integral):
        array_or_nbytes = _nbytes(array_or_nbytes)
    if name in ("native", "auto", "auto+cupy"):
        return da is not None and array_or_nbytes > _THRESHOLD
    return name in ("dask", "dask+cupy")


def _creation_options(like: tx.Any = None) -> tx.Dict[str, tx.Any]:
    """Return the options that create Dask arrays of CuPy chunks.

    Dask creates NumPy chunks unless it is given a CuPy array as
    `like`. When the new array goes with a Dask array of CuPy chunks,
    given as `like`, or, without `like`, under the setting
    `"dask+cupy"`, the functions that create arrays take these options
    so that their chunks are created on the GPU. Of the creation
    functions of Dask, only `arange` takes them, so `eye` and `zeros`
    still create NumPy chunks, which are copied to the GPU right after.

    Parameters
    ----------
    like : array, optional
        The array with which the new array is combined.

    Returns
    -------
    dict
        The keyword arguments of `dask.array.arange`, or nothing.
    """
    if like is None:
        gpu = _SETTING == "dask+cupy"
    else:
        gpu = da is not None and isinstance(like, da.Array) and _on_gpu(like)
    return {"like": cp.empty(0)} if gpu else {}


def _meta(
    x: ArrayProtocol, device: tx.Union[str, int, None] = None
) -> ArrayProtocol:
    to_gpu, device = _normalize_device(device)
    if to_gpu:
        with cp.cuda.Device(device):
            meta = cp.empty((0,) * x.ndim, x.dtype)
    else:
        meta = np.empty((0,) * x.ndim, x.dtype)
    return meta


# ----------------------------------------------------------------------
#   PACKAGES
# ----------------------------------------------------------------------


def get_array_backend(
    x: tx.Optional[tx.Union[ArrayProtocol, ModuleType, str]] = None,
) -> ModuleType:
    """Return the array package of an array, module or setting.

    The result is `numpy`, `cupy` or `dask.array`. `None`, or an object of
    no known backend, gives the package in which the current setting
    creates new arrays: NumPy under `"native"` and `"auto"`, CuPy under
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
        if _is_numpy(array):
            backends.append(np)
        elif _is_cupy(array):
            backends.append(cp)
        elif _is_dask(array):
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

    if _is_cupy(x):
        return _ndimage_of("cupy")
    if _is_numpy(x):
        return _ndimage_of("numpy")
    if _is_dask(x):
        return _ndimage_of("dask")

    return get_ndimage_backend()


# ----------------------------------------------------------------------
#   CUPY
# ----------------------------------------------------------------------


def _on_gpu(x: ArrayLike) -> bool:
    """Tell whether an array is a CuPy array or a Dask array of CuPy chunks."""
    return get_device(x) is not None


def get_device(x: ArrayLike) -> tx.Optional["CudaDevice"]:
    """Return the device of a CuPy array or a Dask array of CuPy chunks."""
    if cp is None:
        return None
    if _is_cupy(x):
        return x.device
    if _is_dask(x):
        meta = getattr(x, "_meta", None)
        if _is_cupy(meta):
            return meta.device
    return None


def _normalize_device(
    device: tx.Union["CudaDevice", str, int, bool, None],
) -> tx.Tuple[bool, tx.Optional["CudaDevice"]]:
    """Return whether a device uses CUDA and its explicit CUDA device."""
    if device == "cpu" or device is False:
        return False, None
    if device is None or device is True or device == "cuda":
        return True, None
    if isinstance(device, str):
        prefix, separator, index = device.partition(":")
        if prefix != "cuda" or not separator or not index.isdigit():
            raise ValueError(f"Invalid device: {device!r}")
        device = int(index)
    _require("cupy")
    return True, cp.cuda.Device(device)


def _compatible_device(
    array_device: tx.Union[ArrayLike, tx.Optional["CudaDevice"]],
    target_device: tx.Union["CudaDevice", str, int, None],
) -> bool:
    """Tell whether an array or device is already on the target device."""
    if array_device is not None and not (
        cp is not None and isinstance(array_device, cp.cuda.Device)
    ):
        array_device = get_device(array_device)
    to_gpu, target_device = _normalize_device(target_device)
    if array_device is None:
        return not to_gpu
    if not to_gpu:
        return False
    return target_device is None or array_device == target_device
