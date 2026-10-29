"""Tests of spline sampling and prefiltering of dask arrays.

The wrappers of `brainhops._core.bsplines` hand dask arrays to
`brainhops._ext.interpol`, which samples each block of coordinates from
the window of the input that its stencils read. The tests compare the
lazy results with those of NumPy, for points on both sides of every
axis, and check that the coefficients of a chunk only read the chunks
around it.
"""

import numpy as np
import pytest

dask = pytest.importorskip("dask")
da = pytest.importorskip("dask.array")
pytest.importorskip("dask.optimization")

from brainhops._core.bsplines import (  # noqa: E402
    coeff2value,
    pull,
    pull_field,
    value2coeff,
)

BOUNDS = [
    "nearest",
    "reflect",
    "mirror",
    "grid-wrap",
    "constant",
    "antimirror",
    "antireflect",
    2.5,
]
DEGREES = [0, 1, 2, 3, 4, 5]
SHAPE = (7, 8, 9)
"""Small spatial shape with no two axes of equal length."""


def _input() -> np.ndarray:
    """Return a batch of two inputs, so that the batch loop is exercised."""
    return np.random.default_rng(1).standard_normal((2, *SHAPE))


def _coords() -> np.ndarray:
    """Return coordinates in four slabs along the first axis.

    The first slab lies before the start of every axis, the second past the
    end of every axis, the third inside the grid, and the last anywhere.
    """
    rng = np.random.default_rng(2)
    high = np.asarray(SHAPE, dtype=float)
    coords = np.empty((4, 5, 6, 3))
    coords[0] = rng.uniform(-30, -5, size=(5, 6, 3))
    coords[1] = high + rng.uniform(5, 30, size=(5, 6, 3))
    coords[2] = rng.uniform(1, 4, size=(5, 6, 3))
    coords[3] = rng.uniform(-30, 40, size=(5, 6, 3))
    return coords


@pytest.mark.parametrize("coeff", [False, True])
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("bound", BOUNDS, ids=str)
@pytest.mark.parametrize(
    "chunks", [(2, 7, 8, 9), (1, 3, 4, 4)], ids=["one-chunk", "chunked"]
)
def test_dask_samples_outside_the_grid_as_numpy_does(
    bound,  # noqa: ANN001
    degree: int,
    coeff: bool,
    chunks: tuple,
) -> None:
    values, coords = _input(), _coords()
    expected = pull(values, coords, degree, bound, coeff)
    lazy = pull(
        da.from_array(values, chunks=chunks),
        da.from_array(coords, chunks=(1, 5, 6, 3)),
        degree,
        bound,
        coeff,
    )
    assert isinstance(lazy, da.Array)
    np.testing.assert_allclose(
        np.asarray(lazy.compute()), expected, rtol=1e-7, atol=1e-12
    )


@pytest.mark.parametrize("bound", ["nearest", "reflect", "constant"])
def test_dask_input_with_numpy_coordinates(bound: str) -> None:
    """A dask input with numpy coordinates gives the same result."""
    values, coords = _input(), _coords()
    expected = pull(values, coords, 1, bound, False)
    lazy = pull(
        da.from_array(values, chunks=(1, 3, 4, 4)), coords, 1, bound, False
    )
    np.testing.assert_allclose(np.asarray(lazy.compute()), expected)


def test_dask_sampling_is_deterministic() -> None:
    """Repeated samplings agree bit for bit while other arrays come and go."""
    field = np.random.default_rng(3).standard_normal((*SHAPE, 3))
    coords = _coords()
    lazy = da.from_array(field, chunks=(4, 4, 5, 3))
    first = np.asarray(pull_field(lazy, coords, 1, "nearest", False).compute())
    for size in (10, 1000, 100000):
        np.empty(size).fill(np.nan)
        again = pull_field(lazy, coords, 1, "nearest", False).compute()
        np.testing.assert_array_equal(np.asarray(again), first)
    np.testing.assert_allclose(
        first, pull_field(field, coords, 1, "nearest", False)
    )


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
@pytest.mark.parametrize("bound", ["reflect", "mirror", "grid-wrap"])
@pytest.mark.parametrize("size", [7, 20])
def test_the_window_prefilter_is_exact(
    degree: int, bound: str, size: int
) -> None:
    """The window prefilter equals the spline of the fully extended signal.

    The reference prefilters, with SciPy, an extension long enough that
    its ends have no weight.
    """
    from scipy.ndimage import map_coordinates, spline_filter1d

    rng = np.random.default_rng(9)
    values = rng.standard_normal(size)
    points = rng.uniform(-30, size + 30, size=200)
    pad = 600
    index = np.arange(-pad, size + pad)
    if bound == "grid-wrap":
        index = np.mod(index, size)
    else:
        period = 2 * size - (2 if bound == "mirror" else 0)
        index = np.mod(index, period)
        index = np.where(
            index >= size,
            period - (1 if bound == "reflect" else 0) - index,
            index,
        )
    coeff = spline_filter1d(values[index], degree, mode="mirror")
    exact = map_coordinates(
        coeff, [points + pad], order=degree, prefilter=False, mode="mirror"
    )
    lazy = pull(
        da.from_array(values, chunks=4), points[:, None], degree, bound, False
    )
    np.testing.assert_allclose(np.asarray(lazy.compute()), exact, atol=1e-10)


LARGE = (300, 400, 500)
"""A volume much larger than the window of a block."""


def test_points_outside_a_constant_boundary_take_the_fill_value() -> None:
    points = np.full((2, 2, 2, 3), -100.0)
    volume = da.random.default_rng(8).random(LARGE, chunks=100)
    out = pull(volume, points, 3, 2.5, False).compute()
    np.testing.assert_allclose(np.asarray(out), 2.5)


def test_a_point_that_is_not_finite_maps_to_nan() -> None:
    """A non-finite point maps to NaN, and the other points are unaffected."""
    values, coords = _input(), _coords()
    coords[2, 0, 0] = [np.nan, 1.0, 2.0]
    coords[2, 0, 1] = [1.0, np.inf, 2.0]
    lazy = pull(
        da.from_array(values, chunks=(1, 3, 4, 4)), coords, 3, "nearest", False
    )
    lazy = np.asarray(lazy.compute())
    assert np.isnan(lazy[:, 2, 0, :2]).all()
    finite = np.isfinite(coords).all(-1)
    expected = pull(values, coords, 3, "nearest", False)
    np.testing.assert_allclose(
        lazy[:, finite], expected[:, finite], rtol=1e-7, atol=1e-12
    )


def test_numpy_coordinates_are_sampled_in_blocks() -> None:
    """Numpy coordinates are cut into blocks, which dask computes."""
    axis = np.arange(64.0) + 100.0
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    volume = da.random.default_rng(10).random(LARGE, chunks=100)
    with dask.config.set({"array.chunk-size": "256KiB"}):
        lazy = pull(volume, points, 3, "nearest", False)
        assert lazy.numblocks != (1, 1, 1)
        out = lazy.compute()
    np.testing.assert_allclose(
        np.asarray(out),
        pull(
            np.asarray(volume[90:180, 90:180, 90:180]),
            points - 90,
            3,
            "nearest",
            False,
        ),
        atol=1e-12,
    )


# ----------------------------------------------------------------------
#   CONVERTING VALUES TO SPLINE COEFFICIENTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("degree", [2, 3, 5])
@pytest.mark.parametrize(
    "bound",
    ["nearest", "reflect", "mirror", "grid-wrap", "antireflect", 2.5],
    ids=str,
)
@pytest.mark.parametrize("chunks", [(4, 5, 6), (2, 3, 3)], ids=["4", "2"])
def test_dask_coefficients_are_numpys(
    bound,  # noqa: ANN001
    degree: int,
    chunks: tuple,
) -> None:
    """Dask coefficients keep the chunks and equal those of NumPy.

    The check includes chunks shorter than the halo of the prefilter.
    """
    values = np.random.default_rng(11).standard_normal((2, 12, 14, 16))
    expected = value2coeff(values, degree, bound, ndim=3)
    lazy = value2coeff(
        da.from_array(values, chunks=(1, *chunks)), degree, bound, ndim=3
    )
    assert lazy.chunks == da.from_array(values, chunks=(1, *chunks)).chunks
    np.testing.assert_allclose(np.asarray(lazy), expected, atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 5])
@pytest.mark.parametrize(
    "bound", ["nearest", "reflect", "mirror", "grid-wrap", "constant", 2.5]
)
def test_dask_coefficients_round_trip(degree: int, bound: object) -> None:
    values = np.random.default_rng(12).standard_normal((12, 14, 16))
    lazy = da.from_array(values, chunks=5)
    coeff = value2coeff(lazy, degree, bound)
    back = coeff2value(coeff, degree, bound)
    assert isinstance(back, da.Array)
    np.testing.assert_allclose(np.asarray(back), values, atol=1e-9)


def test_a_coefficient_chunk_reads_its_neighbours_only() -> None:
    """A coefficient chunk depends only on the input chunks within its halo."""
    # The horizon of the prefilter at degree 3 is shorter than a chunk.
    size = 40
    lazy = da.from_array(np.zeros((6 * size,) * 3), chunks=size)
    coeff = value2coeff(lazy, 3, "mirror")
    assert coeff.chunks == lazy.chunks
    block = coeff.blocks[2, 2, 2]
    graph = dict(block.__dask_graph__())
    keys = list(dask.core.flatten(block.__dask_keys__()))
    needed, _ = dask.optimization.cull(graph, keys)
    inputs = {k for k in needed if isinstance(k, tuple) and k[0] == lazy.name}
    assert inputs == {
        (lazy.name, i, j, k)
        for i in (1, 2, 3)
        for j in (1, 2, 3)
        for k in (1, 2, 3)
    }


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("lazy", [False, True], ids=["numpy", "dask"])
def test_degrees_zero_and_one_are_their_own_coefficients(
    degree: int, lazy: bool
) -> None:
    values = np.random.default_rng(13).standard_normal((5, 6, 7))
    array = da.from_array(values, chunks=3) if lazy else values
    for convert in (value2coeff, coeff2value):
        out = convert(array, degree, "mirror")
        assert out is not array
        np.testing.assert_array_equal(np.asarray(out), values)
        assert convert(array, degree, "mirror", inplace=True) is array


def test_the_dask_backend_needs_only_dask() -> None:
    """The dask backend is available whenever dask is, without dask_image."""
    import sys

    from brainhops._core.backends import backends, ndimage_backend

    assert "dask" in backends()
    with pytest.raises(TypeError):
        ndimage_backend(da.zeros(3))
    assert "dask_image" not in sys.modules
