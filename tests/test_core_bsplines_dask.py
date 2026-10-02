"""
Sampling a dask array outside of its grid.

`dask_image.ndinterp.map_coordinates` crops the input around the
coordinates of each chunk and clips that crop to the grid. When every
coordinate of a chunk lies beyond the start of an axis the crop is empty,
and `scipy` then reads outside of a zero-length array: the result changed
from run to run, with `PYTHONHASHSEED` and with earlier allocations. A crop
clipped at the end of an axis kept a single sample, which is wrong for the
boundary conditions that fold back into the grid.

The sampler now crops the input itself and keeps a whole axis whenever a
stencil reaches past its edge. These tests pin it to the numpy backend,
which samples the whole array, for points on both sides of every axis.
"""

import itertools

import numpy as np
import pytest

da = pytest.importorskip("dask.array")
pytest.importorskip("dask_image")

from brainhops._core.bsplines import (  # noqa: E402
    _crop_for,
    pull,
    pull_field,
)

BOUNDS = ["nearest", "reflect", "mirror", "grid-wrap", "wrap", "constant", 2.5]
ORDERS = [0, 1, 2, 3]
SHAPE = (7, 8, 9)
"""The spatial shape of the input: small, and no two axes alike."""


def _input() -> np.ndarray:
    """A `(2, *SHAPE)` batch of values, so the batch loop is exercised."""
    return np.random.default_rng(1).standard_normal((2, *SHAPE))


def _coords() -> np.ndarray:
    """
    A `(4, 5, 6, 3)` coordinates field, chunked along its first axis.

    The first slab lies before the start of every axis -- the case that
    gave an empty crop -- the second past the end of every axis, the
    third inside the grid, and the last anywhere, inside and out.
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
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("bound", BOUNDS, ids=str)
@pytest.mark.parametrize(
    "chunks", [(2, 7, 8, 9), (1, 3, 4, 4)], ids=["one-chunk", "chunked"]
)
def test_dask_samples_outside_the_grid_as_numpy_does(
    bound,  # noqa: ANN001
    order: int,
    coeff: bool,
    chunks: tuple,
) -> None:
    values, coords = _input(), _coords()
    expected = pull(values, coords, order, bound, coeff)
    lazy = pull(
        da.from_array(values, chunks=chunks),
        da.from_array(coords, chunks=(1, 5, 6, 3)),
        order,
        bound,
        coeff,
    )
    assert isinstance(lazy, da.Array)
    np.testing.assert_allclose(np.asarray(lazy.compute()), expected)


@pytest.mark.parametrize("bound", ["nearest", "reflect", "constant"])
def test_dask_input_with_numpy_coordinates(bound: str) -> None:
    """A dask input sampled at numpy coordinates takes the same path."""
    values, coords = _input(), _coords()
    expected = pull(values, coords, 1, bound, False)
    lazy = pull(
        da.from_array(values, chunks=(1, 3, 4, 4)), coords, 1, bound, False
    )
    np.testing.assert_allclose(np.asarray(lazy.compute()), expected)


def test_dask_sampling_is_deterministic() -> None:
    """
    Repeated samplings agree, bit for bit, while other arrays are made and
    dropped in between -- the allocations that used to show through.
    """
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


def test_a_crop_never_reaches_past_the_grid() -> None:
    """
    An axis is cropped only when every stencil lies inside the grid, so a
    crop is never empty and the boundary condition is only ever applied
    at the real edge of the grid.
    """
    shape = (10,)
    for lo, hi in itertools.product([-20.0, -0.5, 0.0, 3.0], [3.5, 9.0, 25.0]):
        coords = np.array([[lo, hi]])
        for order in ORDERS:
            (crop,) = _crop_for(coords, shape, order, whole=False)
            assert 0 <= crop.start < crop.stop <= shape[0]
            if lo < 2 or hi > 7:
                assert (crop.start, crop.stop) == (0, shape[0])


def test_a_crop_keeps_whole_axes_for_a_prefilter_and_non_finite_points() -> (
    None
):
    coords = np.array([[4.0, 5.0], [np.nan, 5.0]])
    crops = _crop_for(coords, (10, 10), 1, whole=False)
    assert (crops[0].start, crops[0].stop) == (3, 7)
    assert (crops[1].start, crops[1].stop) == (0, 10)
    crops = _crop_for(coords, (10, 10), 3, whole=True)
    assert all((c.start, c.stop) == (0, 10) for c in crops)
