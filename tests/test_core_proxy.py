"""Tests of the lazy array base class of brainhops._core.proxy."""

import pickle

import numpy as np
import pytest

from brainhops._core.proxy import ArrayProxy, ArrayProxyProtocol


class _Recording(ArrayProxy):
    """A proxy over an array in memory that records the boxes it reads."""

    def __init__(self, array: np.ndarray, chunks: tuple = None) -> None:
        self.array = array
        self.reads = []
        self._chunks = chunks

    @property
    def base_shape(self) -> tuple:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def base_chunks(self) -> tuple:
        return self._chunks

    def _read(self, index: tuple) -> np.ndarray:
        for item in index:
            if isinstance(item, slice):
                assert 0 <= item.start < item.stop <= self.array.shape[0] * 99
                assert item.step >= 1
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
    # Indexing and transposing return views without reading, and the
    # attributes of a view describe the selection.
    proxy = _Recording(_array())
    view = proxy[1:4, None, ::-2].T[0]
    assert isinstance(view, _Recording)
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
    # A view keeps its selection when it is pickled.
    view = _Recording(_array())[1:, [0, 2]].T
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


def test_is_view() -> None:
    # Only a view that changes the selection or the order of the axes
    # counts as a view.
    proxy = _Recording(_array())
    assert not proxy.is_view
    assert not proxy[...].is_view
    assert not proxy[:, ::1].T.T.is_view
    assert proxy[1:].is_view
    assert proxy[::-1].is_view
    assert proxy[None].is_view
    assert proxy.T.is_view
    assert proxy[[0, 1, 2, 3, 4]].is_view


def test_views_read_through_the_original_proxy() -> None:
    # A view reads through the proxy that it was made from, so that a
    # handle opened by a read is shared by every view.
    proxy = _Recording(_array())
    np.asarray(proxy[1:][::2].T)
    assert proxy.reads == [(slice(1, 4, 2), slice(0, 6, 1), slice(0, 7, 1))]
