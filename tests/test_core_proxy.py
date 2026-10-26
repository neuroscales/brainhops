"""Tests of the lazy arrays of brainhops._core.proxy."""

import pickle

import numpy as np
import pytest

from brainhops._core.proxy import (
    ArrayProxyProtocol,
    IndexedProxy,
    NewAxis,
    PermutedProxy,
    ScaledProxy,
    StoredProxy,
    _normalize_index,
)


class _Recording(StoredProxy):
    """A proxy over an array in memory that records the boxes it reads."""

    def __init__(self, array: np.ndarray, chunks: tuple = None) -> None:
        self.array = array
        self.reads = []
        self._chunks = chunks

    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def chunks(self) -> tuple:
        return self._chunks

    def _read(self, index: tuple) -> np.ndarray:
        assert len(index) == self.array.ndim
        for item, n in zip(index, self.array.shape):
            if isinstance(item, slice):
                assert isinstance(item.start, int)
                assert isinstance(item.stop, int)
                assert 0 <= item.start < item.stop <= n
                assert item.step >= 1
            else:
                assert 0 <= item < n
        self.reads.append(index)
        return self.array[index]


def _array(shape: tuple = (5, 6, 7)) -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)


def test_satisfies_the_protocol() -> None:
    # A proxy, a NumPy array and a view all satisfy the protocol.
    proxy = _Recording(_array())
    assert isinstance(proxy, ArrayProxyProtocol)
    assert isinstance(proxy[1:], ArrayProxyProtocol)
    assert isinstance(np.zeros(3), ArrayProxyProtocol)


def test_indexing_reads_nothing() -> None:
    # Indexing and transposing return proxies without reading, and the
    # attributes of the result describe the selection.
    proxy = _Recording(_array())
    view = proxy[1:4, None, ::-2].T[0]
    assert isinstance(view, IndexedProxy)
    assert proxy.reads == []
    expected = _array()[1:4, None, ::-2].T[0]
    assert view.shape == expected.shape
    assert view.ndim == expected.ndim
    assert view.size == expected.size
    assert len(view) == len(expected)


@pytest.mark.parametrize(
    "indices",
    [
        [(Ellipsis,)],
        [(0,)],
        [(-1, slice(None), 2)],
        [(slice(None, None, -1),)],
        [(slice(4, 0, -2), slice(1, None, 3))],
        [(None, Ellipsis, None)],
        [(Ellipsis, 3)],
        [([4, 0, 2],)],
        [(slice(None), np.array([5, -1, 0]))],
        [(slice(1, 4),), (slice(None, None, -1), 2)],
        [([3, 1],), (slice(None), [5, 0, 5])],
        [(None,), (0,), (slice(None, None, 2),)],
        [(None,), (slice(0, 0),)],
        [(None,), ([0, 0, -1],)],
        [(None,), ([0, 0, 0],), ([1, 2],)],
        [(None,), ([0, 0, 0],), (slice(1, None),), (0,)],
        [(slice(2, 2),)],
        [(np.int64(2), np.array(1))],
        [(slice(-100, 100),)],
    ],
)
def test_views_match_numpy(indices: list) -> None:
    # A chain of indexes reads what NumPy gives for the same chain.
    array = _array()
    proxy = _Recording(array)
    expected = array
    for index in indices:
        proxy = proxy[index]
        expected = expected[index]
    assert proxy.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(proxy), expected)


def test_random_chains_match_numpy() -> None:
    # Random chains of indexes and transpositions read what NumPy gives.
    rng = np.random.default_rng(0)
    array = _array((4, 5, 6))
    for _ in range(300):
        proxy = _Recording(array)
        expected = array
        for _ in range(rng.integers(1, 4)):
            if rng.random() < 0.2:
                perm = tuple(rng.permutation(expected.ndim))
                proxy = proxy.transpose(perm)
                expected = expected.transpose(perm)
                continue
            index = []
            for n in expected.shape:
                choice = rng.integers(4)
                if choice == 0 and n:
                    index.append(int(rng.integers(-n, n)))
                elif choice == 1:
                    start, stop = rng.integers(-n - 1, n + 2, size=2)
                    step = int(rng.choice([-3, -2, -1, 1, 2, 3]))
                    index.append(slice(int(start), int(stop), step))
                elif choice == 2:
                    index.append(None)
                    index.append(slice(None))
                else:
                    index.append(slice(None))
            first = index[0] if index else None
            n = expected.shape[0] if expected.ndim else 0
            if first == slice(None) and n and rng.random() < 0.3:
                index[0] = rng.integers(-n, n, size=rng.integers(1, 4))
            index = tuple(index)
            proxy = proxy[index]
            expected = expected[index]
        assert proxy.shape == expected.shape
        np.testing.assert_array_equal(np.asarray(proxy), expected)


def test_reads_the_bounding_box() -> None:
    # A view reads the smallest box with a positive step that holds its
    # values, once.
    proxy = _Recording(_array())
    np.asarray(proxy[4:0:-2, 3, [5, 1]])
    assert proxy.reads == [(slice(2, 5, 2), 3, slice(1, 6, 1))]


def test_empty_selection_reads_nothing() -> None:
    # An empty view is answered without reading.
    proxy = _Recording(_array())
    values = np.asarray(proxy[2:2])
    assert values.shape == (0, 6, 7)
    assert values.dtype == np.float32
    assert proxy.reads == []


def test_points_are_refused() -> None:
    # Several arrays and boolean masks select points, which a proxy
    # refuses to describe.
    proxy = _Recording(_array())
    with pytest.raises(IndexError):
        proxy[[0, 1], [0, 1]]
    with pytest.raises(IndexError):
        proxy[np.ones(5, bool)]
    with pytest.raises(IndexError):
        proxy[np.zeros((2, 2), int)]
    with pytest.raises(IndexError):
        proxy[0, 0, 0, 0]
    with pytest.raises(IndexError):
        proxy[5]
    with pytest.raises(TypeError):
        proxy[0.5]


def test_array_conversion_options() -> None:
    # `dtype` converts the values, and `copy=False` is refused because
    # values cannot be read without making them.
    proxy = _Recording(_array())
    assert np.asarray(proxy, dtype=np.int64).dtype == np.int64
    assert np.array(proxy, copy=True).shape == proxy.shape
    with pytest.raises(ValueError):
        proxy.__array__(copy=False)


def test_chunks_follow_the_view() -> None:
    # The preferred blocks are carried through a view.
    proxy = _Recording(_array((8, 6, 4)), chunks=(4, 3, 4))
    assert proxy.chunks == (4, 3, 4)
    assert proxy[::2, None, 1:2].chunks == (2, 1, 1, 4)
    assert proxy.T.chunks == (4, 3, 4)
    assert _Recording(_array()).chunks is None


def test_to_dask_reads_each_block_once() -> None:
    # Dask reads the proxy block by block, without probing it.
    da = pytest.importorskip("dask.array")
    array = _array((8, 6, 4))
    proxy = _Recording(array, chunks=(4, 3, 4))
    lazy = proxy.to_dask()
    assert isinstance(lazy, da.Array)
    assert proxy.reads == []
    np.testing.assert_array_equal(lazy.compute(scheduler="sync"), array)
    np.testing.assert_array_equal(
        proxy[1:7].to_dask().compute(scheduler="sync"), array[1:7]
    )


def test_views_pickle() -> None:
    # An indexed proxy keeps its selection when it is pickled, and so
    # does a scaled proxy keep its factors.
    proxy = _Recording(_array())
    scaled = ScaledProxy(proxy, np.arange(7.0), 1.0)[1:, None, [0, 2]].T
    copied = pickle.loads(pickle.dumps(scaled))
    np.testing.assert_array_equal(np.asarray(copied), np.asarray(scaled))
    view = proxy[1:, [0, 2]].T
    copied = pickle.loads(pickle.dumps(view))
    np.testing.assert_array_equal(np.asarray(copied), np.asarray(view))


def test_transpose_arguments() -> None:
    # `transpose` accepts integers or one sequence, and refuses axes that
    # are not a permutation.
    proxy = _Recording(_array())
    assert proxy.transpose(2, 0, 1).shape == (7, 5, 6)
    assert proxy.transpose([2, 0, 1]).shape == (7, 5, 6)
    assert proxy.transpose(-1, 0, 1).shape == (7, 5, 6)
    with pytest.raises(ValueError):
        proxy.transpose(0, 0, 1)
    with pytest.raises(ValueError):
        proxy.transpose(0, 1, 3)


def test_layers_stay_flat() -> None:
    # An index of an indexed proxy holds the same stored proxy, a
    # transpose holds a permuted proxy of it, and permutations compose.
    proxy = _Recording(_array())
    view = proxy[1:][::2, None][:, :, [0, 2]]
    assert isinstance(view, IndexedProxy)
    assert view.base is proxy
    transposed = view.T.T.transpose(3, 1, 2, 0)
    assert isinstance(transposed.base, PermutedProxy)
    assert transposed.base.base is proxy
    assert proxy.T.T is proxy
    assert isinstance(proxy.T, PermutedProxy)
    assert proxy.transpose(2, 0, 1).transpose(1, 2, 0) is proxy


def test_a_whole_index_gives_the_stored_proxy() -> None:
    # An index that selects every value in order returns the stored
    # proxy itself, so that a writer can copy its bytes.
    proxy = _Recording(_array())
    assert proxy[...] is proxy
    assert proxy[:, ::1, -100:100] is proxy
    assert proxy[None][0] is proxy
    assert proxy[::-1][::-1] is proxy
    assert proxy[1:] is not proxy


def test_new_axes_are_kept_in_the_selection() -> None:
    # A new axis is an entry of the selection, which a later index can
    # keep, remove or repeat.
    proxy = _Recording(_array())
    view = proxy[None, 1:3]
    assert view.selection[0] == NewAxis()
    assert view[[0, 0, 0]].shape == (3, 2, 6, 7)
    assert view[1:].shape == (0, 2, 6, 7)


def test_numpy_moves_the_axis_of_an_array() -> None:
    # NumPy moves the axis of an array to the front when the array and
    # the integers of the index are not next to each other, even when an
    # ellipsis between them stands for no axis.
    array = _array((3, 4, 2, 2))
    proxy = _Recording(array)
    for index in [
        (0, slice(None), [1, 0]),
        (0, None, [1, 0]),
        ([1, 0], slice(None), 0),
        (slice(2, 1), -2, Ellipsis, [1], -1),
        (slice(2, 1), -2, [1], -1),
        (0, Ellipsis, [1, 1]),
    ]:
        expected = array[index]
        assert proxy[index].shape == expected.shape, index
        np.testing.assert_array_equal(np.asarray(proxy[index]), expected)
    # The axis of the array would be the second one of the result.
    assert _normalize_index((0, slice(None), [1]), (3, 4, 2))[1] == 1
    assert _normalize_index((0, [1], slice(None)), (3, 4, 2))[1] is None


def test_scaled_proxies_scale_what_they_read() -> None:
    # A scaled proxy indexes and transposes its factors with its values,
    # keeps an axis of length one of a factor of length one, and reads
    # only the box that the index selects.
    rng = np.random.default_rng(0)
    array = _array((5, 6, 7, 4)).astype(np.int16)
    slope = rng.random((1, 1, 1, 4)).astype(np.float32) + 0.5
    inter = rng.random((6, 1, 1)).astype(np.float32)
    proxy = _Recording(array)
    scaled = ScaledProxy(proxy, slope, inter)
    expected = (array * slope + inter).astype(np.float32)
    assert scaled.dtype == np.float32
    for index in [
        (Ellipsis,),
        (0, slice(None), [1, 1, 2], slice(None, None, -2)),
        (None, 2, slice(1, 4), Ellipsis, [3]),
        (slice(4, 0, -2), 3),
    ]:
        part = scaled[index]
        for factor in (part.slope, part.inter):
            assert all(f in (1, n) for f, n in zip(factor.shape, part.shape))
        np.testing.assert_allclose(np.asarray(part), expected[index])
    part = scaled.transpose(3, 1, 0, 2)[1:3]
    np.testing.assert_allclose(
        np.asarray(part), expected.transpose(3, 1, 0, 2)[1:3]
    )
    proxy.reads.clear()
    np.asarray(scaled[1, 2:4])
    assert proxy.reads == [(1, slice(2, 4, 1), slice(0, 7, 1), slice(0, 4, 1))]


def test_the_stored_type_decides_the_scaled_type() -> None:
    # The factors are cast to the type of the values, so a float64
    # factor keeps float32 values in float32, and integers of 16 bits
    # become float32 while integers of 32 bits become float64.
    values = _array((3, 4))
    for stored, expected in [
        (np.float32, np.float32),
        (np.int16, np.float32),
        (np.uint8, np.float32),
        (np.int32, np.float64),
        (np.float64, np.float64),
    ]:
        proxy = _Recording(values.astype(stored))
        slope = np.array([[1.5, 2.0, 2.5, 3.0]], dtype=np.float64)
        scaled = ScaledProxy(proxy, slope, np.float64(0.25))
        assert scaled.dtype == expected
        read = np.asarray(scaled)
        assert read.dtype == expected
        reference = values.astype(stored).astype(expected)
        reference = reference * slope.astype(expected)
        reference = reference + np.asarray(0.25, dtype=expected)
        np.testing.assert_array_equal(read, reference)


def test_scaling_leaves_what_it_reads_unchanged(tmp_path) -> None:  # noqa: ANN001
    # Values that are already in the scaled type may be a view of an
    # array that the proxy keeps, or a read-only memory map, so they are
    # scaled into a new array. Values of another type are converted into
    # a new array, which is scaled in place.
    from brainhops._core.strided import StridedProxy

    values = _array((3, 4))
    kept = values.copy()
    scaled = ScaledProxy(_Recording(values), 2.0, 1.0)
    np.testing.assert_array_equal(np.asarray(scaled), kept * 2 + 1)
    np.testing.assert_array_equal(np.asarray(scaled), kept * 2 + 1)
    np.testing.assert_array_equal(values, kept)
    path = tmp_path / "values.bin"
    path.write_bytes(values.astype("=f4").tobytes())
    mapped = StridedProxy.contiguous(path, "=f4", (3, 4))
    assert not np.asarray(mapped).flags.writeable
    scaled = ScaledProxy(mapped, 2.0, 1.0)
    np.testing.assert_array_equal(np.asarray(scaled), kept * 2 + 1)
    np.testing.assert_array_equal(np.asarray(mapped), kept)
    integers = _Recording(values.astype(np.int16))
    scaled = ScaledProxy(integers, 2.0, 1.0)
    np.testing.assert_array_equal(np.asarray(scaled), kept * 2 + 1)
    np.testing.assert_array_equal(integers.array, kept)
    zero = ScaledProxy(_Recording(np.array(3.0, np.float32)), 2.0, 1.0)
    assert np.asarray(zero).shape == ()
    assert float(np.asarray(zero)) == 7.0


def test_scaled_proxies_refuse_large_factors() -> None:
    # A factor with more dimensions than the values cannot broadcast.
    with pytest.raises(ValueError):
        ScaledProxy(_Recording(_array((2, 3))), np.ones((1, 2, 3)))


def test_the_root_is_below_every_layer() -> None:
    # Each layer keeps the proxy below it as its base, and the root of
    # every layer is the stored proxy that a reader built.
    proxy = _Recording(_array())
    assert proxy.root is proxy
    assert proxy[1:].T.root is proxy
    assert proxy.T.root is proxy
    assert ScaledProxy(proxy, 2.0)[0].T.root is proxy
    scaled = ScaledProxy(proxy, 2.0)
    assert scaled.base is proxy
    assert scaled[1:].base.base is proxy


def test_views_read_through_the_original_proxy() -> None:
    # An indexed proxy reads through the proxy that it was made from, so
    # that a handle opened by a read is shared by every index of it.
    proxy = _Recording(_array())
    np.asarray(proxy[1:][::2].T)
    assert proxy.reads == [(slice(1, 4, 2), slice(0, 6, 1), slice(0, 7, 1))]


def test_to_numpy_reads_the_view() -> None:
    # to_numpy reads the values of a view.
    proxy = _Recording(_array())
    values = proxy[1:3].to_numpy()
    assert isinstance(values, np.ndarray)
    np.testing.assert_array_equal(values, _array()[1:3])


def test_to_backend_follows_the_setting() -> None:
    # Under the default setting, a small proxy is read with NumPy and a
    # large one becomes a Dask array; an explicit setting is obeyed.
    da = pytest.importorskip("dask.array")
    from brainhops._core.backends import backend

    proxy = _Recording(_array(), chunks=(2, 3, 7))
    assert isinstance(proxy.to_backend(), np.ndarray)
    with backend(threshold=8):
        lazy = proxy.to_backend()
    assert isinstance(lazy, da.Array)
    assert lazy.chunksize == (2, 3, 7)
    assert isinstance(proxy.to_backend("dask"), da.Array)
    assert isinstance(proxy.to_backend("numpy"), np.ndarray)


def test_gpu_conversions_need_cupy() -> None:
    # Without CuPy, the conversions to the GPU say what is missing.
    pytest.importorskip("dask.array")
    try:
        import cupy  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("CuPy is installed")
    proxy = _Recording(_array())
    with pytest.raises(ImportError):
        proxy.to_cupy()
    with pytest.raises(ImportError):
        proxy.to_dask(device="cuda")
    with pytest.raises(ImportError):
        proxy.to_backend("dask+cupy")
