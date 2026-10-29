"""Tests of the dask version of the port of torch-interpol.

The functions `pull`, `spline_coeff` and `spline_coeff1d` of
`brainhops._ext.interpol` run lazily when one of their arrays is a dask
array. The tests compare their results with those of the same functions
on NumPy arrays, for every boundary condition and order, with chunks of
every kind, and check that a block of the output reads only the window
of the input that its coordinates need.
"""

import itertools

import fake_cupy
import numpy as np
import pytest

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")

from brainhops._ext.interpol import (  # noqa: E402
    _backend,
    _dask,
    pull,
    spline_coeff,
    spline_coeff1d,
)

BOUNDS = ["zero", "replicate", "dct1", "dct2", "dst1", "dst2", "dft"]

SHAPES = {1: (23,), 2: (13, 11), 3: (7, 9, 6)}
"""Spatial shapes of the inputs, with no two axes of equal length."""

INPUT_CHUNKS = {
    1: [(5,), (1,), (23,)],
    2: [(4, 3), (1, 11), (13, 2)],
    3: [(3, 4, 5), (1, 9, 2), (7, 2, 6)],
}
"""Chunks of the inputs, which do not divide the shape, have length 1,
or cover a whole axis. Most of them are shorter than the halo of the
prefilter."""

GRID_CHUNKS = {1: (7, 1), 2: (3, 4, 2), 3: (2, 3, 2, 3)}
"""Chunks of the grids, whose spatial shape is that of `POINTS`."""

POINTS = {1: (15,), 2: (5, 7), 3: (4, 5, 3)}


def _grid(rng, ndim):  # noqa: ANN001, ANN202
    # Coordinates inside the input, near its edges, and far outside it,
    # in slabs along the first axis of the grid.
    shape = np.asarray(SHAPES[ndim], dtype=float)
    grid = rng.uniform(-2, shape + 1, POINTS[ndim] + (ndim,))
    far = grid[: POINTS[ndim][0] // 3]
    far += rng.choice([-1, 1], far.shape) * rng.uniform(20, 80, far.shape)
    return grid


def _close(actual, expected, atol=1e-12):  # noqa: ANN001, ANN202
    assert isinstance(actual, da.Array)
    computed = actual.compute()
    assert computed.shape == expected.shape
    assert computed.dtype == expected.dtype
    scale = max(1.0, float(np.abs(expected).max(initial=0)))
    np.testing.assert_allclose(computed, expected, rtol=0, atol=atol * scale)


# --- interpolation ------------------------------------------------------


ORDERS = {1: range(8), 2: [0, 2, 3, 5, 7], 3: [1, 3, 6]}
"""The orders that each number of spatial axes is tested with."""


@pytest.mark.parametrize(
    "ndim, order", [(d, o) for d in ORDERS for o in ORDERS[d]]
)
@pytest.mark.parametrize("bound", BOUNDS)
def test_pull_matches_eager(ndim, order, bound) -> None:  # noqa: ANN001
    # The input, the grid or both are dask arrays, with chunks of every
    # kind, and the prefilter is on and off.
    rng = np.random.default_rng(order + 10 * ndim)
    x = rng.standard_normal(SHAPES[ndim])
    grid = _grid(rng, ndim)
    lazy_grid = da.from_array(grid, chunks=GRID_CHUNKS[ndim])
    for prefilter in (False, True):
        expected = pull(x, grid, order, bound, prefilter=prefilter)
        for chunks in INPUT_CHUNKS[ndim]:
            lazy = da.from_array(x, chunks=chunks)
            _close(
                pull(lazy, grid, order, bound, prefilter=prefilter), expected
            )
            _close(
                pull(lazy, lazy_grid, order, bound, prefilter=prefilter),
                expected,
            )
        _close(pull(x, lazy_grid, order, bound, prefilter=prefilter), expected)


@pytest.mark.parametrize("bound", BOUNDS)
def test_pull_local_transformation(bound) -> None:  # noqa: ANN001
    # A small rotation and shift of a larger volume, whose blocks read
    # local windows, and coordinates that the boundary folds onto the
    # far side of the axis.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((21, 19, 12))
    lazy = da.from_array(x, chunks=(6, 7, 5))
    angle = 0.1
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    index = np.stack(np.meshgrid(*map(np.arange, x.shape), indexing="ij"), -1)
    centre = (np.asarray(x.shape) - 1) / 2
    grid = (index - centre) @ rotation.T + centre + [0.3, -1.7, 2.2]
    lazy_grid = da.from_array(grid, chunks=(7, 10, 6, 3))
    for order, prefilter in ((1, False), (3, True), (6, True)):
        expected = pull(x, grid, order, bound, prefilter=prefilter)
        _close(pull(lazy, grid, order, bound, prefilter=prefilter), expected)
        if order < 6:
            # The tasks of a lazy grid compute the coefficients of their
            # windows again, which is slow at high orders.
            out = pull(lazy, lazy_grid, order, bound, prefilter=prefilter)
            _close(out, expected)


@pytest.mark.parametrize("bound", BOUNDS)
def test_lazy_grid_prefilters_windows(bound) -> None:  # noqa: ANN001
    # With a lazy grid, each task computes the coefficients of its own
    # window, extended by the halo of the prefilter. On an axis much
    # longer than the halo, the extended windows are cut inside the
    # axis, and with the periodic boundary they continue past the edges.
    rng = np.random.default_rng(10)
    x = rng.standard_normal((260, 5))
    x *= 1 + 100 * (np.arange(260)[:, None] % 7 == 0)
    lazy = da.from_array(x, chunks=(40, 5))
    index = np.stack(np.meshgrid(np.arange(260), np.arange(5), indexing="ij"))
    grid = np.moveaxis(index, 0, -1) + [1.6, -0.3]
    lazy_grid = da.from_array(grid, chunks=(20, 5, 2))
    for order in (2, 3, 5, 7):
        expected = pull(x, grid, order, bound, prefilter=True)
        for a in (lazy, x):
            out = pull(a, lazy_grid, order, bound, prefilter=True)
            _close(out, expected)


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("bound", ["zero", "replicate", "dst2"])
def test_pull_extrapolate(order, bound) -> None:  # noqa: ANN001
    # The mask of `extrapolate` and the zeros of the zero boundary are
    # those of the whole input, not of the window that a block reads.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((12, 10))
    grid = rng.uniform(-1.5, 12.5, (8, 9, 2))
    lazy = da.from_array(x, chunks=(3, 4))
    for extrapolate in (True, False, "edges"):
        expected = pull(x, grid, order, bound, extrapolate, prefilter=True)
        _close(
            pull(lazy, grid, order, bound, extrapolate, prefilter=True),
            expected,
        )
        _close(
            pull(
                lazy,
                da.from_array(grid, chunks=(3, 2, 2)),
                order,
                bound,
                extrapolate,
                prefilter=True,
            ),
            expected,
        )


@pytest.mark.parametrize(
    "batch_input, batch_grid, input_chunks, grid_chunks",
    [
        ((3,), (), (2,), ()),
        ((3,), (3,), (2,), (1,)),
        ((3,), (1,), (1,), (1,)),
        ((2, 3), (1,), (1, 2), (1,)),
        ((1,), (4,), (1,), (3,)),
        ((), (4,), (), (3,)),
        ((2, 1), (1, 3), (1, 1), (1, 2)),
    ],
)
def test_pull_batch_axes(
    batch_input: tuple,
    batch_grid: tuple,
    input_chunks: tuple,
    grid_chunks: tuple,
) -> None:
    # Chunked batch axes of the input and of the grid, which broadcast
    # against each other.
    rng = np.random.default_rng(len(batch_input) + 3 * len(batch_grid))
    x = rng.standard_normal(batch_input + (9, 8))
    grid = rng.uniform(-3, 11, batch_grid + (5, 6, 2))
    batch = np.broadcast_shapes(batch_input, batch_grid)
    lazy = da.from_array(x, chunks=input_chunks + (4, 3))
    lazy_grid = da.from_array(grid, chunks=grid_chunks + (2, 4, 2))
    for order, bound in ((1, "dft"), (3, "dst1"), (5, "replicate")):
        expected = pull(x, grid, order, bound, prefilter=True)
        assert expected.shape == batch + (5, 6)
        for a, b in ((lazy, grid), (x, lazy_grid), (lazy, lazy_grid)):
            out = pull(a, b, order, bound, prefilter=True)
            _close(out, expected)
        # The batch axes of the output have the chunks of the input, or
        # those of the grid where the input does not have the axis.
        out = pull(lazy, lazy_grid, order, bound)
        assert out.chunks[-2:] == lazy_grid.chunks[-3:-1]


@pytest.mark.parametrize(
    "options",
    [
        {"strategy": "batched"},
        {"strategy": "blocked", "block_size": 7},
    ],
)
def test_pull_options(options) -> None:  # noqa: ANN001
    # The strategy and the block size apply to each block.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 10, 9))
    grid = rng.uniform(-3, 12, (6, 7, 2))
    expected = pull(x, grid, [3, 1], ["dct1", "zero"], prefilter=True)
    lazy = da.from_array(x, chunks=(1, 4, 5))
    out = pull(lazy, grid, [3, 1], ["dct1", "zero"], prefilter=True, **options)
    _close(out, expected)


def test_pull_data_types_and_empty_output() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 9)).astype(np.float32)
    grid = rng.uniform(0, 8, (5, 4, 2)).astype(np.float32)
    lazy = da.from_array(x, chunks=(3, 4))
    out = pull(lazy, grid, 3, "dct2", prefilter=True)
    assert out.dtype == np.float32
    expected = pull(x, grid, 3, "dct2", prefilter=True)
    np.testing.assert_allclose(out.compute(), expected, atol=1e-6)
    labels = da.from_array(np.arange(72).reshape(8, 9), chunks=4)
    assert pull(labels, grid, 1).dtype == np.float64
    assert spline_coeff(labels, 3).dtype == np.float64
    _close(spline_coeff(labels, 1), np.arange(72.0).reshape(8, 9))
    empty = pull(lazy, np.zeros((0, 4, 2)), 3)
    assert isinstance(empty, da.Array)
    assert empty.compute().shape == (0, 4)
    with pytest.raises(ValueError):
        pull(lazy, np.zeros((3, 2)), 1)
    with pytest.raises(ValueError):
        pull(lazy, grid, 1, strategy="fast")


def _input_chunks_read(out, x) -> list:  # noqa: ANN001
    # The chunks of the input, a dask array named like x, on which each
    # block of the output depends in the graph.
    graph = dict(out.__dask_graph__())
    counts = []
    for key in dask.core.flatten(out.__dask_keys__()):
        needed, _ = dask.optimization.cull(graph, [key])
        counts.append(sum(1 for k in needed if k[0] == x.name))
    return counts


def test_block_reads_only_its_window() -> None:
    # A shift of two samples reads, for each block of 8 points, at most
    # two chunks of 8 samples along each axis, rather than the whole
    # input, whether the windows are known when the graph is built or
    # only when the tasks run.
    pytest.importorskip("dask.optimization")
    rng = np.random.default_rng(5)
    shape = (32, 32, 32)
    x = rng.standard_normal(shape)
    lazy = da.from_array(x, chunks=8)
    index = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1)
    grid = index + [2.0, -2.0, 1.5]
    out = pull(lazy, grid, 1, "dct2")
    assert out.numblocks == (4, 4, 4)
    counts = _input_chunks_read(out, lazy)
    assert max(counts) <= 8
    assert min(counts) >= 1
    expected = pull(x, grid, 1, "dct2")
    _close(out, expected)

    # With a lazy grid, each task computes its window itself, and the
    # chunks of the input record when they are computed.
    reads = []

    def record(block, block_info=None):  # noqa: ANN001, ANN202
        reads.append(tuple(block_info[0]["chunk-location"]))
        return block

    recorded = lazy.map_blocks(record, dtype=lazy.dtype)
    lazy_grid = da.from_array(grid, chunks=(8, 8, 8, 3))
    out = pull(recorded, lazy_grid, 1, "dct2")
    for block in itertools.product(range(4), repeat=3):
        reads.clear()
        values = out.blocks[block].compute(scheduler="sync")
        assert 1 <= len(reads) <= 8
        select = tuple(slice(8 * b, 8 * b + 8) for b in block)
        np.testing.assert_allclose(values, expected[select], atol=1e-12)


def test_prefiltered_blocks_read_only_their_halo() -> None:
    # With the prefilter, a block of the output reads the coefficients
    # of its window, each of which reads the chunks within the halo of
    # the prefilter along each axis.
    pytest.importorskip("dask.optimization")
    rng = np.random.default_rng(6)
    x = rng.standard_normal((200, 3))
    lazy = da.from_array(x, chunks=(20, 3))
    coeff = spline_coeff1d(lazy, 3, "dct2", axis=0)
    halo = _dask._halo(3)
    assert halo == 30
    graph = dict(coeff.__dask_graph__())
    for chunk in range(10):
        needed, _ = dask.optimization.cull(graph, [(coeff.name, chunk, 0)])
        read = sorted(k[1] for k in needed if k[0] == lazy.name)
        first = max(0, (20 * chunk - halo) // 20)
        last = min(9, (20 * chunk + 20 + halo - 1) // 20)
        assert read == list(range(first, last + 1))


# --- prefilter ----------------------------------------------------------


@pytest.mark.parametrize("bound", BOUNDS)
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7])
def test_spline_coeff1d_matches_eager(bound, order) -> None:  # noqa: ANN001
    # Chunks of length 1, chunks shorter than the halo, chunks that do
    # not divide the axis, and axes shorter than the halo, on which the
    # window of every chunk is the whole axis.
    halo = _dask._halo(order)
    rng = np.random.default_rng(order)
    for n in (1, 2, 5, halo, 2 * halo + 1, 3 * halo + 7):
        x = rng.standard_normal((n, 3))
        # A large value at every seventh sample makes the coefficients
        # depend strongly on the samples near each cut of a window.
        x *= 1 + 100 * (np.arange(n)[:, None] % 7 == 0)
        expected = spline_coeff1d(x, order, bound, axis=0)
        for length in (1, 7, max(1, halo // 3), n):
            if length == 1 and n > halo:
                # Chunks of length 1 are only tested on short axes,
                # where they are fast.
                continue
            lazy = da.from_array(x, chunks=(length, 2))
            out = spline_coeff1d(lazy, order, bound, axis=0)
            assert out.chunks == lazy.chunks
            _close(out, expected)
        out = spline_coeff1d(lazy.T, order, bound, axis=-1)
        _close(out, expected.T)


def test_halo_is_needed(monkeypatch) -> None:  # noqa: ANN001
    # A window shorter than the horizon of the prefilter changes the
    # coefficients by more than the tolerance of the tests, which
    # therefore check the halo.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, 2))
    lazy = da.from_array(x, chunks=(10, 2))
    expected = spline_coeff1d(x, 7, "dct2", axis=0)
    _close(spline_coeff1d(lazy, 7, "dct2", axis=0), expected)
    halo = _dask._halo
    monkeypatch.setattr(_dask, "_halo", lambda order: halo(order) // 3)
    out = spline_coeff1d(lazy, 7, "dct2", axis=0).compute()
    assert np.abs(out - expected).max() > 1e-8


@pytest.mark.parametrize("bound", BOUNDS)
def test_spline_coeff_matches_eager(bound) -> None:  # noqa: ANN001
    # Several axes, with a batch axis and one order per axis.
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 40, 9, 70))
    lazy = da.from_array(x, chunks=(1, 6, 4, 25))
    orders = [5, 1, 3]
    expected = spline_coeff(x, orders, bound)
    out = spline_coeff(lazy, orders, bound)
    assert out.chunks == lazy.chunks
    _close(out, expected)
    _close(spline_coeff(lazy, 7, bound, ndim=2), spline_coeff(x, 7, bound, 2))


# --- CuPy ---------------------------------------------------------------


@pytest.fixture
def cupy(monkeypatch):  # noqa: ANN001, ANN201
    """Make a strict fake of CuPy, which runs on the CPU, the CuPy."""
    monkeypatch.setattr(_backend, "_cupy", fake_cupy.cupy)
    return fake_cupy.cupy


@pytest.mark.parametrize("bound", ["zero", "dct1", "dft"])
def test_cupy_blocks_stay_on_device(cupy, bound) -> None:  # noqa: ANN001
    # The fake refuses every implicit conversion to NumPy, so that a
    # block that moves to the host would raise, and the blocks of the
    # results are fake CuPy arrays, which dask knows from their meta.
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 20, 17))
    grid = rng.uniform(-5, 22, (2, 6, 7, 2))
    expected = pull(x, grid, 3, bound, prefilter=True)

    def device(array, chunks):  # noqa: ANN001, ANN202
        meta = cupy.empty((0,) * array.ndim, dtype=array.dtype)
        return da.from_array(
            cupy.asarray(array), chunks=chunks, asarray=False, meta=meta
        )

    lazy = device(x, (1, 7, 5))
    coeff = spline_coeff(lazy, 3, bound, ndim=2)
    assert isinstance(coeff._meta, fake_cupy.ndarray)
    computed = coeff.compute(scheduler="sync")
    assert isinstance(computed, fake_cupy.ndarray)
    np.testing.assert_allclose(
        computed.get(), spline_coeff(x, 3, bound, ndim=2), atol=1e-12
    )
    for g in (cupy.asarray(grid), device(grid, (1, 3, 4, 2)), grid):
        out = pull(lazy, g, 3, bound, prefilter=True)
        assert isinstance(out._meta, fake_cupy.ndarray)
        computed = out.compute(scheduler="sync")
        assert isinstance(computed, fake_cupy.ndarray)
        np.testing.assert_allclose(computed.get(), expected, atol=1e-12)
    out = pull(cupy.asarray(x), device(grid, (1, 3, 4, 2)), 3, bound)
    assert isinstance(out.compute(scheduler="sync"), fake_cupy.ndarray)
