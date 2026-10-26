"""Tests of the backend settings of brainhops.backends."""

# dependencies
import numpy as np
import pytest

# core
from brainhops._core import backends
from brainhops._core.backends import (
    array_backend,
    backend,
    get_auto_threshold,
    get_backend,
    is_lazy,
    ndimage_backend,
    resolve_backend,
    set_backend,
    to_backend,
)
from brainhops._core.proxy import StoredProxy

da = pytest.importorskip("dask.array")


class _Proxy(StoredProxy):
    """A lazy array over an array in memory, which counts its reads."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.reads = 0

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def _read(self, index: tuple) -> np.ndarray:
        self.reads += 1
        return self.array[index]


SMALL = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def test_the_default_setting_is_native() -> None:
    # The setting no longer depends on whether Dask is installed.
    assert get_backend() == "native"
    assert array_backend() is np
    assert ndimage_backend().__name__ == "scipy.ndimage"


def test_none_keeps_loaded_arrays() -> None:
    # Under None, a loaded array keeps its backend.
    lazy = da.from_array(SMALL)
    assert to_backend(SMALL) is SMALL
    assert to_backend(lazy) is lazy
    assert resolve_backend(lazy) == "dask"


def test_none_resolves_proxies_by_size() -> None:
    # Under None, a lazy array is resolved with "auto": a small one is
    # read with NumPy, and a large one becomes a Dask array.
    proxy = _Proxy(SMALL)
    values = to_backend(proxy)
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, SMALL)
    with backend(threshold=8):
        lazy = to_backend(_Proxy(SMALL))
    assert isinstance(lazy, da.Array)
    np.testing.assert_array_equal(lazy.compute(), SMALL)


def test_auto_decides_by_size_for_loaded_arrays() -> None:
    # "auto" converts loaded arrays too: a large NumPy array becomes a
    # Dask array, and a small Dask array is computed.
    with backend("auto", threshold=8):
        assert isinstance(to_backend(SMALL), da.Array)
    with backend("auto"):
        small = to_backend(da.from_array(SMALL))
    assert isinstance(small, np.ndarray)
    np.testing.assert_array_equal(small, SMALL)


def test_explicit_settings_convert_every_array() -> None:
    # "numpy" and "dask" convert loaded arrays and lazy arrays alike.
    lazy = da.from_array(SMALL)
    assert isinstance(to_backend(lazy, "numpy"), np.ndarray)
    assert isinstance(to_backend(SMALL, "dask"), da.Array)
    proxy = _Proxy(SMALL)
    converted = to_backend(proxy, "dask")
    assert isinstance(converted, da.Array)
    assert proxy.reads == 0
    np.testing.assert_array_equal(converted.compute(), SMALL)
    assert isinstance(to_backend(SMALL, np), np.ndarray)
    assert isinstance(to_backend(SMALL, da), da.Array)


def test_none_differs_from_the_current_setting() -> None:
    # Without an argument, to_backend follows the current setting, and an
    # explicit "native" keeps a loaded array whatever the current setting.
    with backend("dask"):
        assert isinstance(to_backend(SMALL), da.Array)
        assert isinstance(to_backend(SMALL, None), da.Array)
        assert to_backend(SMALL, "native") is SMALL
        assert array_backend() is da


def test_the_context_manager_restores_the_setting() -> None:
    # The setting and the threshold are restored on exit, and a call
    # without an argument, or with None, keeps the current setting.
    threshold = get_auto_threshold()
    with backend("dask", threshold=1):
        assert get_backend() == "dask"
        with backend():
            assert get_backend() == "dask"
        with backend(None):
            assert get_backend() == "dask"
        with backend("native"):
            assert get_backend() == "native"
        assert get_auto_threshold() == 1
    assert get_backend() == "native"
    assert get_auto_threshold() == threshold


def test_set_backend_accepts_names_modules_and_none() -> None:
    # A setting is given by name or by module, and None keeps the
    # current setting.
    previous = get_backend()
    try:
        set_backend(da)
        assert get_backend() == "dask"
        set_backend(None)
        assert get_backend() == "dask"
        set_backend("native")
        assert get_backend() == "native"
    finally:
        set_backend(previous)


def test_unknown_settings_are_refused() -> None:
    # A string that names no setting is refused.
    with pytest.raises(ValueError):
        set_backend("torch")
    with pytest.raises(ValueError):
        to_backend(SMALL, "gpu")
    with pytest.raises(ValueError):
        array_backend("torch")


@pytest.mark.skipif(backends.cp is not None, reason="CuPy is installed")
def test_gpu_settings_need_cupy() -> None:
    # The settings that put arrays on a GPU need CuPy.
    for name in ("cupy", "auto+cupy", "dask+cupy"):
        with pytest.raises(ImportError):
            set_backend(name)


def test_is_lazy() -> None:
    # A reader asks whether an array of a given size stays lazy.
    assert not is_lazy(10)
    assert is_lazy(get_auto_threshold() + 1)
    assert is_lazy(10, "dask")
    assert not is_lazy(10**12, "numpy")
    with backend(threshold=4):
        assert is_lazy(10)
