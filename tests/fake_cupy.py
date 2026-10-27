"""A strict stand-in for CuPy, which runs on the CPU.

The tests that need a GPU are skipped where CuPy is not installed, which
leaves untested the code that meets CuPy arrays. This module gives a
backend with the rules of CuPy that matter to that code, so that the code
can be tested without a GPU:

- an array is not a NumPy array, and NumPy refuses to convert it, as
  `np.asarray(cupy_array)` raises with CuPy;
- the functions and the operators refuse NumPy arrays and the other
  array-likes as operands, as CuPy refuses to move values to the GPU
  without being asked;
- an operand of a type that it does not know is given the computation
  through `NotImplemented`, as CuPy does;
- `asarray` converts anything that NumPy converts.

The values are held in NumPy arrays. [`install`][] makes the module the
CuPy of the backends of brainhops for the duration of a test.
"""

# stdlib
import types
from numbers import Number

# dependencies
import numpy as np
import pytest
import scipy.ndimage as _scipy_ndimage
from numpy.lib.mixins import NDArrayOperatorsMixin


class ndarray(NDArrayOperatorsMixin):  # noqa: N801
    """An array of the fake CuPy, which holds its values in NumPy."""

    __slots__ = ("_values",)

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    # --- description ------------------------------------------------------

    shape = property(lambda self: self._values.shape)
    dtype = property(lambda self: self._values.dtype)
    ndim = property(lambda self: self._values.ndim)
    size = property(lambda self: self._values.size)
    nbytes = property(lambda self: self._values.nbytes)
    itemsize = property(lambda self: self._values.itemsize)
    strides = property(lambda self: self._values.strides)
    flags = property(lambda self: self._values.flags)
    T = property(lambda self: ndarray(self._values.T))

    @property
    def device(self) -> "Device":
        """The device of the array, which is always the first one."""
        return Device(0)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"fake_cupy.ndarray({self._values!r})"

    # --- conversions ------------------------------------------------------

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )

    def get(self) -> np.ndarray:
        """Return the values as a NumPy array."""
        return np.array(self._values)

    def item(self) -> object:
        return self._values.item()

    def __bool__(self) -> bool:
        return bool(self._values)

    def __int__(self) -> int:
        return int(self._values)

    def __float__(self) -> float:
        return float(self._values)

    def __complex__(self) -> complex:
        return complex(self._values)

    def __index__(self) -> int:
        return self._values.__index__()

    # --- indexing ---------------------------------------------------------

    def __getitem__(self, index: object) -> object:
        return _wrap(self._values[_unwrap(_check(index))])

    def __setitem__(self, index: object, value: object) -> None:
        self._values[_unwrap(_check(index))] = _unwrap(_check(value))

    def __iter__(self) -> object:
        return (_wrap(v) for v in self._values)

    # --- methods ----------------------------------------------------------

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        attribute = getattr(self._values, name)
        if not callable(attribute):
            return _wrap(attribute)

        def method(*args: object, **kwargs: object) -> object:
            args, kwargs = _unwrap(_check((args, kwargs)))
            return _wrap(attribute(*args, **kwargs))

        return method

    # --- NumPy protocols --------------------------------------------------

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        operands = inputs + tuple(kwargs.get("out", ()))
        if any(_foreign(x) for x in operands):
            return NotImplemented
        inputs, kwargs = _unwrap(_check((inputs, kwargs)))
        return _wrap(getattr(ufunc, method)(*inputs, **kwargs))

    def __array_function__(
        self,
        func: object,
        types: object,
        args: object,
        kwargs: object,
    ) -> object:
        if not all(issubclass(t, (ndarray, np.ndarray)) for t in types):
            return NotImplemented
        args, kwargs = _unwrap(_check((args, kwargs)))
        return _wrap(func(*args, **kwargs))


class Device:
    """A device of the fake CuPy, usable as a context manager."""

    def __init__(self, id: int = 0) -> None:  # noqa: A002
        self.id = int(getattr(id, "id", id))

    def __enter__(self) -> "Device":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


def _foreign(x: object) -> bool:
    # An operand of a type that the fake does not know, which is given
    # the computation, as CuPy gives it to a type that defines
    # `__array_ufunc__`.
    return (
        not isinstance(x, (ndarray, np.ndarray, np.generic, Number))
        and hasattr(type(x), "__array_ufunc__")
        and type(x).__array_ufunc__ is not None
    )


def _check(values: object) -> object:
    """Refuse the arrays that CuPy refuses, among nested arguments."""
    if isinstance(values, ndarray):
        return values
    if isinstance(values, np.ndarray) and values.ndim > 0:
        raise TypeError(f"Unsupported type {type(values)}")
    if isinstance(values, (list, tuple)):
        for value in values:
            _check(value)
        return values
    if isinstance(values, dict):
        for value in values.values():
            _check(value)
        return values
    if (
        hasattr(values, "shape")
        and hasattr(values, "dtype")
        and not isinstance(values, (np.ndarray, np.generic, np.dtype, type))
    ):
        raise TypeError(f"Unsupported type {type(values)}")
    return values


def _unwrap(values: object) -> object:
    if isinstance(values, ndarray):
        return values._values
    if isinstance(values, tuple) and hasattr(values, "_fields"):
        return type(values)(*(_unwrap(v) for v in values))
    if isinstance(values, (list, tuple)):
        return type(values)(_unwrap(v) for v in values)
    if isinstance(values, dict):
        return {k: _unwrap(v) for k, v in values.items()}
    return values


def _wrap(values: object) -> object:
    # CuPy gives the result of a reduction as an array of no dimension.
    if isinstance(values, np.generic):
        return ndarray(np.asarray(values))
    if isinstance(values, np.ndarray):
        return ndarray(values)
    if isinstance(values, tuple) and hasattr(values, "_fields"):
        return type(values)(*(_wrap(v) for v in values))
    if isinstance(values, (list, tuple)):
        return type(values)(_wrap(v) for v in values)
    return values


def _strict(function: object) -> object:
    """Wrap a NumPy function so that it takes and gives fake arrays."""

    def wrapper(*args: object, **kwargs: object) -> object:
        args, kwargs = _unwrap(_check((args, kwargs)))
        return _wrap(function(*args, **kwargs))

    wrapper.__name__ = getattr(function, "__name__", "function")
    return wrapper


def _module(name: str, source: types.ModuleType, extra: dict) -> object:
    module = types.ModuleType(name)

    def __getattr__(attr: str) -> object:  # noqa: N807
        value = getattr(source, attr)
        if callable(value) and not isinstance(value, type):
            return _strict(value)
        return value

    module.__getattr__ = __getattr__
    for key, value in extra.items():
        setattr(module, key, value)
    return module


def asarray(x: object, dtype: object = None, **kwargs: object) -> ndarray:
    """Convert anything that NumPy converts, as `cupy.asarray` does."""
    if isinstance(x, ndarray):
        return x if dtype is None else ndarray(x._values.astype(dtype))
    return ndarray(np.array(x, dtype=dtype))


def asnumpy(x: object) -> np.ndarray:
    return x.get() if isinstance(x, ndarray) else np.asarray(x)


cupy = _module(
    "fake_cupy",
    np,
    {
        "ndarray": ndarray,
        "asarray": asarray,
        "array": asarray,
        "asnumpy": asnumpy,
        "cuda": types.SimpleNamespace(Device=Device),
    },
)
"""The fake `cupy` module."""

ndimage = _module("fake_cupyx_scipy_ndimage", _scipy_ndimage, {})
"""The fake `cupyx.scipy.ndimage` module."""


def install(monkeypatch: pytest.MonkeyPatch) -> object:
    """Make the fake the CuPy of the backends of brainhops.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The fixture that undoes the change after the test.

    Returns
    -------
    module
        The fake `cupy` module.
    """
    from brainhops._core import backends

    monkeypatch.setattr(backends, "cp", cupy)
    monkeypatch.setattr(backends, "cpndi", ndimage)
    monkeypatch.setitem(backends._MODULES, "cupy", (cupy, ndimage))
    return cupy
