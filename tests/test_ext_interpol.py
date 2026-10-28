"""Tests of the NumPy and CuPy port of torch-interpol.

The interpolation is compared with `scipy.ndimage.map_coordinates` for
every boundary that SciPy implements, and with independent references
for the orders and the boundaries that SciPy does not have. The
prefilter is compared with `scipy.ndimage.spline_filter`, and checked to
give coefficients that interpolate the values exactly.
"""

import fractions

import fake_cupy
import numpy as np
import pytest
import scipy.interpolate
import scipy.ndimage as ndi

from brainhops._ext import interpol
from brainhops._ext.interpol import (
    _backend,
    pull,
    spline_coeff,
    spline_coeff1d,
)

SCIPY_MODES = {
    "grid-constant": "grid-constant",
    "zero": "grid-constant",
    "nearest": "nearest",
    "mirror": "mirror",
    "reflect": "reflect",
    "grid-wrap": "grid-wrap",
}
"""Names of the port, and the SciPy modes that they match."""

ALL_BOUNDS = [
    "zero",
    "nearest",
    "mirror",
    "reflect",
    "grid-wrap",
    "dst1",
    "dst2",
]

SHAPES = {1: (11,), 2: (7, 9), 3: (6, 7, 5)}
POINTS = {1: (40,), 2: (5, 8), 3: (4, 5, 2)}

OPTIONS = {
    "batched": {"strategy": "batched"},
    "blocked": {"strategy": "blocked"},
    "blocks-of-1": {"strategy": "blocked", "block_size": 1},
    "blocks-of-7": {"strategy": "blocked", "block_size": 7},
    "one-block": {"strategy": "blocked", "block_size": 10**9},
}
"""Strategies of `pull`, and block sizes, that the comparisons run with.

A block of 1 value holds one point of one batch item. A block of 7
values leaves a smaller last block in most tests, and splits the batch
when it has more than 7 items. A block of 10**9 values is larger than
every problem of the tests.
"""


@pytest.fixture(params=sorted(OPTIONS))
def options(request):  # noqa: ANN001, ANN201
    """Return the keyword arguments that choose the strategy of `pull`."""
    return OPTIONS[request.param]


def _scipy_pull(x, grid, order, mode, prefilter):  # noqa: ANN001, ANN202
    # Interpolate every batch item with SciPy, coordinates last.
    ndim = grid.shape[-1]
    coords = np.moveaxis(grid, -1, 0)
    items = x.reshape((-1,) + x.shape[-ndim:])
    out = [
        ndi.map_coordinates(
            item, coords, order=order, mode=mode, prefilter=prefilter
        )
        for item in items
    ]
    return np.stack(out).reshape(x.shape[:-ndim] + grid.shape[:-1])


# --- interpolation ------------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("bound", sorted(SCIPY_MODES))
def test_pull_matches_scipy(ndim, order, bound, options) -> None:  # noqa: ANN001
    rng = np.random.default_rng(order + 10 * ndim)
    shape = SHAPES[ndim]
    x = rng.standard_normal(shape)
    # The coordinates reach four voxels beyond each edge, so that both
    # the samples inside the array and those outside it are compared.
    low = np.full(ndim, -4.0)
    high = np.asarray(shape) + 3.0
    grid = rng.uniform(low, high, POINTS[ndim] + (ndim,))
    mode = SCIPY_MODES[bound]
    expected = _scipy_pull(x, grid, order, mode, False)
    np.testing.assert_allclose(
        pull(x, grid, order, bound, **options), expected, atol=1e-12
    )
    # SciPy prefilters correctly for these modes only, and its
    # coefficients near the edges are wrong for the others (#250).
    if mode in ("mirror", "reflect", "grid-wrap"):
        expected = _scipy_pull(x, grid, order, mode, True)
    else:
        coeff = spline_coeff(x, order, bound)
        expected = _scipy_pull(coeff, grid, order, mode, False)
    np.testing.assert_allclose(
        pull(x, grid, order, bound, prefilter=True, **options),
        expected,
        atol=1e-10,
    )


@pytest.mark.parametrize("bound", sorted(SCIPY_MODES))
def test_order0_ties_inside(bound, options) -> None:  # noqa: ANN001
    # A coordinate halfway between two samples takes the upper one, as
    # in SciPy.
    x = np.arange(6.0) ** 2
    grid = np.arange(0.0, 5.0, 0.5)[:, None]
    expected = _scipy_pull(x, grid, 0, SCIPY_MODES[bound], False)
    np.testing.assert_array_equal(pull(x, grid, 0, bound, **options), expected)


def test_order0_ties_outside() -> None:
    # Outside the array, the tie is broken before the boundary condition
    # is applied, whereas SciPy reflects the coordinate first. The two
    # differ only for the symmetric boundaries, at exact ties.
    x = np.arange(6.0)
    grid = np.asarray([[-1.5], [-0.5], [6.5]])
    np.testing.assert_array_equal(pull(x, grid, 0, "reflect"), [0, 0, 4])
    np.testing.assert_array_equal(pull(x, grid, 0, "mirror"), [1, 0, 3])
    np.testing.assert_array_equal(pull(x, grid, 0, "nearest"), [0, 0, 5])


def _extended(x, bound, lo, hi):  # noqa: ANN001, ANN202
    # Build the extension of a vector by its definition, from index lo to
    # index hi excluded.
    n = len(x)
    if bound == "zero":
        return np.asarray([x[i] if 0 <= i < n else 0.0 for i in range(lo, hi)])
    if bound == "nearest":
        return np.asarray([x[min(max(i, 0), n - 1)] for i in range(lo, hi)])
    if bound == "mirror":
        base, first = np.concatenate([x, x[-2:0:-1]]), 0
    elif bound == "reflect":
        base, first = np.concatenate([x, x[::-1]]), 0
    elif bound == "grid-wrap":
        base, first = x, 0
    elif bound == "dst1":
        base, first = np.concatenate([[0.0], x, [0.0], -x[::-1]]), -1
    elif bound == "dst2":
        base, first = np.concatenate([x, -x[::-1]]), 0
    period = len(base)
    return np.asarray([base[(i - first) % period] for i in range(lo, hi)])


@pytest.mark.parametrize("bound", ALL_BOUNDS)
@pytest.mark.parametrize("n", [1, 2, 4, 5])
def test_bound_extensions(bound, n, options) -> None:  # noqa: ANN001
    # Order 0 at integer coordinates reads the extended array.
    x = np.arange(1.0, n + 1.0)
    grid = np.arange(-13.0, n + 13.0)[:, None]
    if n == 1 and bound == "mirror":
        expected = np.ones(len(grid))
    else:
        expected = _extended(x, bound, -13, n + 13)
    np.testing.assert_array_equal(pull(x, grid, 0, bound, **options), expected)


@pytest.mark.parametrize("order", range(8))
def test_weights_match_bspline(order) -> None:  # noqa: ANN001
    # The weights are the values of the cardinal B-spline, which sum to
    # one for any coordinate.
    knots = np.arange(order + 2) - (order + 1) / 2
    spline = scipy.interpolate.BSpline.basis_element(knots, False)
    coords = np.random.default_rng(order).uniform(-3, 3, 200)
    total = 0
    for tap in range(order + 1):
        first = np.floor(coords - (order - 1) / 2)
        t = coords - first
        weight = interpol._splines._tap_weight(np, order, tap, t)
        distance = t - tap
        expected = np.nan_to_num(spline(distance))
        np.testing.assert_allclose(weight, expected, atol=1e-12)
        total = total + weight
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


@pytest.mark.parametrize("order", [6, 7])
@pytest.mark.parametrize("bound", ALL_BOUNDS)
def test_high_orders_match_definition(order, bound, options) -> None:  # noqa: ANN001
    # SciPy stops at order 5, so orders 6 and 7 are compared with the
    # definition: a sum of B-splines centred on the extended samples.
    rng = np.random.default_rng(order)
    n = 9
    x = rng.standard_normal(n)
    coords = rng.uniform(-5, n + 4, 50)
    knots = np.arange(order + 2) - (order + 1) / 2
    spline = scipy.interpolate.BSpline.basis_element(knots, False)
    ext = _extended(x, bound, -20, n + 20)
    j = np.arange(-20, n + 20)
    weights = np.nan_to_num(spline(coords[:, None] - j[None, :]))
    expected = weights @ ext
    np.testing.assert_allclose(
        pull(x, coords[:, None], order, bound, **options),
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_per_axis_order_and_bound(ndim, options) -> None:  # noqa: ANN001
    # On a separable grid, the interpolation is a sequence of 1-D
    # interpolations, which SciPy computes axis by axis.
    rng = np.random.default_rng(ndim)
    shape = SHAPES[ndim]
    x = rng.standard_normal(shape)
    orders = [3, 1, 5][:ndim]
    bounds = ["mirror", "grid-wrap", "nearest"][:ndim]
    axes = [rng.uniform(-3, n + 2, n + 2) for n in shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    expected = x
    for d in range(ndim):
        moved = np.moveaxis(expected, d, -1)
        out = _scipy_pull(moved, axes[d][:, None], orders[d], bounds[d], False)
        expected = np.moveaxis(out, -1, d)
    np.testing.assert_allclose(
        pull(x, grid, orders, bounds, **options), expected, atol=1e-12
    )
    # With the prefilter, the reference prefilters each axis with its own
    # order and boundary.
    # SciPy does not prefilter the nearest boundary exactly (#250), so
    # that axis is prefiltered with the port.
    coeff = x
    for d in range(ndim):
        if orders[d] < 2:
            continue
        if bounds[d] == "nearest":
            coeff = spline_coeff1d(coeff, orders[d], bounds[d], axis=d)
        else:
            coeff = ndi.spline_filter1d(
                coeff, orders[d], axis=d, mode=bounds[d]
            )
    np.testing.assert_allclose(
        pull(x, grid, orders, bounds, prefilter=True, **options),
        pull(coeff, grid, orders, bounds, **options),
        atol=1e-12,
    )


def test_batch_axes(options) -> None:  # noqa: ANN001
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 7, 9))
    grid = rng.uniform(-2, 9, (4, 5, 2))
    out = pull(x, grid, 3, "reflect", prefilter=True, **options)
    assert out.shape == (2, 3, 4, 5)
    expected = _scipy_pull(x, grid, 3, "reflect", True)
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize(
    "batch_input, batch_grid",
    [
        ((3,), (3,)),
        ((3,), (1,)),
        ((2, 3), (1,)),
        ((1,), (3,)),
        ((), (3,)),
        ((2, 1), (1, 3)),
    ],
)
def test_grid_batch_axes(batch_input, batch_grid, options) -> None:  # noqa: ANN001
    # The batch axes of the input and of the grid broadcast against each
    # other, and each pair of items is compared with one call to SciPy.
    rng = np.random.default_rng(len(batch_input) + 3 * len(batch_grid))
    x = rng.standard_normal(batch_input + (7, 9))
    grid = rng.uniform(-3, 10, batch_grid + (4, 5, 2))
    batch = np.broadcast_shapes(batch_input, batch_grid)
    out = pull(x, grid, 3, "mirror", prefilter=True, **options)
    assert out.shape == batch + (4, 5)
    xb = np.broadcast_to(x, batch + (7, 9))
    gb = np.broadcast_to(grid, batch + (4, 5, 2))
    for index in np.ndindex(*batch):
        expected = ndi.map_coordinates(
            xb[index], np.moveaxis(gb[index], -1, 0), order=3, mode="mirror"
        )
        np.testing.assert_allclose(out[index], expected, atol=1e-12)


@pytest.mark.parametrize(
    "input_shape, grid_shape, out_shape",
    [
        ((0, 5, 5), (3, 4, 2), (0, 3, 4)),
        ((1, 5, 5), (0, 3, 4, 2), (0, 3, 4)),
        ((2, 5, 5), (0, 4, 2), (2, 0, 4)),
    ],
)
@pytest.mark.parametrize("prefilter", [False, True])
def test_empty_output(
    input_shape: tuple,
    grid_shape: tuple,
    out_shape: tuple,
    prefilter: bool,
    options: dict,
) -> None:
    # An empty batch or an empty grid gives an empty output instead of
    # an error.
    x = np.ones(input_shape)
    grid = np.zeros(grid_shape)
    out = pull(x, grid, 3, prefilter=prefilter, **options)
    assert out.shape == out_shape


def test_grid_needs_spatial_axes() -> None:
    # The grid has as many spatial axes as coordinates, as in
    # torch-interpol, so scattered points need singleton axes.
    x = np.zeros((4, 4))
    with pytest.raises(ValueError):
        pull(x, np.zeros((3, 2)), 1)
    # Batch axes that do not broadcast are refused.
    with pytest.raises(ValueError):
        pull(np.zeros((5, 4, 4)), np.zeros((2, 3, 1, 2)), 1)


@pytest.mark.parametrize(
    "perm", [(0, 1, 2, 3), (3, 0, 1, 2), (0, 3, 1, 2), (2, 0, 3, 1)]
)
def test_memory_layouts(perm, options) -> None:  # noqa: ANN001
    # Inputs whose axes were moved are read in place, whatever the
    # position of the spatial axis in memory.
    rng = np.random.default_rng(2)
    x = np.transpose(rng.standard_normal((4, 5, 6, 7)), perm)
    grid = rng.uniform(-2, 8, (9, 1))
    expected = _scipy_pull(x, grid, 3, "reflect", False)
    np.testing.assert_allclose(
        pull(x, grid, 3, "reflect", **options), expected, atol=1e-12
    )


def test_extrapolate(options) -> None:  # noqa: ANN001
    x = np.arange(1.0, 6.0)
    grid = np.asarray([[-0.6], [-0.4], [-0.01], [2.0], [4.01], [4.4], [4.6]])
    full = pull(x, grid, 1, "nearest", **options)
    np.testing.assert_array_equal(
        pull(x, grid, 1, "nearest", extrapolate=False, **options),
        full * [0, 0, 1, 1, 1, 0, 0],
    )
    np.testing.assert_array_equal(
        pull(x, grid, 1, "nearest", extrapolate="edges", **options),
        full * [0, 1, 1, 1, 1, 1, 0],
    )


def test_data_types() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 7)).astype(np.float32)
    grid = rng.uniform(0, 6, (5, 1, 2))
    out = pull(x, grid, 3, "reflect", prefilter=True)
    assert out.dtype == np.float32
    expected = _scipy_pull(x.astype(np.float64), grid, 3, "reflect", True)
    np.testing.assert_allclose(out, expected, atol=1e-5)
    labels = np.arange(42).reshape(6, 7)
    assert pull(labels, grid, 1).dtype == np.float64
    assert spline_coeff(labels, 3).dtype == np.float64


@pytest.mark.parametrize("shape", [(), (5,), (3, 4), (2, 3, 4), (1, 7, 1)])
@pytest.mark.parametrize("size", [1, 2, 5, 7, 100])
def test_batch_blocks_cover_the_batch(shape, size) -> None:  # noqa: ANN001
    # Every batch item belongs to exactly one block, and no block holds
    # more items than the block size.
    counts = np.zeros(shape, dtype=int)
    for select in interpol._pull._batch_blocks(shape, size):
        assert counts[select].size <= size
        counts[select] += 1
    np.testing.assert_array_equal(counts, 1)


def test_strategy_errors() -> None:
    x = np.zeros((4, 4))
    grid = np.zeros((3, 1, 2))
    with pytest.raises(ValueError):
        pull(x, grid, 1, strategy="fast")
    with pytest.raises(ValueError):
        pull(x, grid, 1, block_size=0)


def test_names_and_errors() -> None:
    assert interpol._bounds._normalize_bound("grid-mirror") == "dct2"
    assert interpol._bounds._normalize_bound("antireflect") == "dst2"
    for name in ("wrap", "constant", "unknown"):
        with pytest.raises(ValueError):
            interpol._bounds._normalize_bound(name)
    x = np.zeros((4, 4))
    grid = np.zeros((3, 1, 2))
    with pytest.raises(ValueError):
        pull(x, grid, 8)
    with pytest.raises(ValueError):
        pull(x, grid, [1, 2, 3])
    with pytest.raises(ValueError):
        pull(x, grid, 1, extrapolate="sometimes")
    np.testing.assert_array_equal(
        pull(x, grid, "cubic", "zeros"), pull(x, grid, 3, "grid-constant")
    )


# --- prefilter ----------------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["mirror", "reflect", "grid-wrap"])
def test_spline_coeff_matches_scipy(ndim, order, mode) -> None:  # noqa: ANN001
    x = np.random.default_rng(order).standard_normal(SHAPES[ndim])
    expected = ndi.spline_filter(x, order, mode=mode)
    np.testing.assert_allclose(
        spline_coeff(x, order, mode), expected, atol=1e-10
    )


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7])
def test_poles_are_roots(order) -> None:  # noqa: ANN001
    # The values of the spline at the integers match SciPy, and the
    # hard-coded poles are roots of the polynomial that they define, to
    # the rounding of a float64. The distance to the root is measured
    # by one step of Newton's method in exact arithmetic.
    knots = np.arange(order + 2) - (order + 1) / 2
    spline = scipy.interpolate.BSpline.basis_element(knots, False)
    values = interpol._coeff._stencil(order)
    h = order // 2
    np.testing.assert_allclose(
        [float(b) for b in values], spline(np.arange(-h, h + 1)), atol=1e-15
    )
    poles = interpol._coeff._poles(order)
    assert len(poles) == h
    assert list(poles) == sorted(poles)
    gain = 1.0
    for z in poles:
        assert -1 < z < 0
        exact = fractions.Fraction(z)
        powers = range(2 * h, -1, -1)
        value = sum(b * exact**p for b, p in zip(values, powers))
        slope = sum(p * b * exact ** (p - 1) for b, p in zip(values, powers))
        assert abs(float(value / slope / exact)) < 2e-16
        gain *= (1 - z) * (1 - 1 / z)
    assert interpol._coeff._gain(order) == pytest.approx(gain, rel=1e-15)


def _sampled(c, order, bound):  # noqa: ANN001, ANN202
    # Sample at the integers the spline of the coefficients c along the
    # first axis, by its definition: a sum of B-splines centred on the
    # samples of the extended coefficients.
    n = len(c)
    h = order // 2
    knots = np.arange(order + 2) - (order + 1) / 2
    spline = scipy.interpolate.BSpline.basis_element(knots, False)
    j = np.arange(-h, n + h)
    weights = np.nan_to_num(spline(np.arange(n)[:, None] - j[None, :]))
    ext = np.stack(
        [_extended(column, bound, -h, n + h) for column in c.T], axis=1
    )
    return weights @ ext


@pytest.mark.parametrize("bound", ALL_BOUNDS)
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 10, 30, 200])
def test_spline_coeff_is_exact(bound, order, n) -> None:  # noqa: ANN001
    # Sampling the coefficients at the integer coordinates gives back
    # the values, to rounding errors.
    x = np.random.default_rng(n).standard_normal((n, 3))
    coeff = spline_coeff1d(x, order, bound, axis=0)
    np.testing.assert_allclose(_sampled(coeff, order, bound), x, atol=1e-14)
    grid = np.arange(n, dtype=float)[:, None]
    values = pull(np.moveaxis(coeff, 0, -1), grid, order, bound)
    np.testing.assert_allclose(values, x.T, atol=1e-11)


@pytest.mark.parametrize("bound", ["nearest", "constant", "reflect"])
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_spline_coeff_matches_brainhops(bound, order) -> None:  # noqa: ANN001
    # brainhops solves for the coefficients of these boundaries.
    from brainhops._core.bsplines import value2coeff

    x = np.random.default_rng(order).standard_normal((12, 30))
    name = "zero" if bound == "constant" else bound
    np.testing.assert_allclose(
        spline_coeff(x, order, name),
        value2coeff(x, order, bound),
        atol=1e-10,
    )


def test_spline_coeff_does_not_modify_input() -> None:
    x = np.random.default_rng(4).standard_normal((5, 6))
    copy = x.copy()
    spline_coeff(x, 3, "zero")
    spline_coeff(np.moveaxis(x, 0, 1), 3, "dst1", ndim=1)
    np.testing.assert_array_equal(x, copy)


# --- CuPy ---------------------------------------------------------------


@pytest.fixture
def cupy(monkeypatch):  # noqa: ANN001, ANN201
    """Make a strict fake of CuPy, which runs on the CPU, the CuPy."""
    monkeypatch.setattr(_backend, "_cupy", fake_cupy.cupy)
    return fake_cupy.cupy


def test_auto_strategy(cupy, monkeypatch) -> None:  # noqa: ANN001
    # The automatic strategy makes a single block for CuPy arrays, and
    # blocks of the given size for NumPy arrays. Each block of the
    # output is computed by one call to `_accumulate`.
    blocks = []
    accumulate = interpol._pull._accumulate

    def counting(xp, taps, out, gather, pool):  # noqa: ANN001, ANN202
        blocks.append(out.shape)
        return accumulate(xp, taps, out, gather, pool)

    monkeypatch.setattr(interpol._pull, "_accumulate", counting)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 6, 7))
    grid = rng.uniform(-2, 8, (4, 5, 2))
    assert interpol._pull._select_strategy(cupy, "auto") == "batched"
    assert interpol._pull._select_strategy(np, "auto") == "blocked"
    pull(cupy.asarray(x), cupy.asarray(grid), 3, block_size=1)
    assert len(blocks) == 1
    blocks.clear()
    pull(x, grid, 3, block_size=1)
    assert len(blocks) == 3 * 20


CUPY_OPTIONS = {
    "auto": {},
    "batched": {"strategy": "batched"},
    "blocked": {"strategy": "blocked", "block_size": 5},
}
"""Strategies of `pull` that the tests with the fake CuPy run with."""


@pytest.mark.parametrize("options", sorted(CUPY_OPTIONS))
@pytest.mark.parametrize("bound", ALL_BOUNDS)
@pytest.mark.parametrize("order", [0, 1, 3, 5])
def test_cupy_arrays_stay_on_device(cupy, bound, order, options) -> None:  # noqa: ANN001
    # The fake refuses every implicit conversion to NumPy and every NumPy
    # operand, so that a step that moves data to the host would raise.
    # A block size of 5 also splits the inner batch axis of the moved
    # input below, which is gathered by indexing.
    options = CUPY_OPTIONS[options]
    rng = np.random.default_rng(order)
    x = rng.standard_normal((2, 6, 7, 5))
    grid = rng.uniform(-3, 8, (4, 3, 2, 3))
    for extrapolate in (True, False):
        out = pull(
            cupy.asarray(x),
            cupy.asarray(grid),
            order,
            bound,
            extrapolate=extrapolate,
            prefilter=True,
            **options,
        )
        assert isinstance(out, fake_cupy.ndarray)
        reference = pull(x, grid, order, bound, extrapolate, prefilter=True)
        np.testing.assert_allclose(out.get(), reference, atol=1e-12)
    # A NumPy grid is uploaded explicitly to the device of the input.
    grid1 = rng.uniform(-3, 8, (5, 1))
    moved = np.moveaxis(x, 1, -1)
    out = pull(
        cupy.asarray(moved), grid1, order, bound, prefilter=True, **options
    )
    assert isinstance(out, fake_cupy.ndarray)
    expected = pull(moved, grid1, order, bound, prefilter=True)
    np.testing.assert_allclose(out.get(), expected, atol=1e-12)
    # A grid with its own batch axes.
    grid2 = rng.uniform(-3, 8, (2, 1, 4, 3, 2, 3))
    out = pull(
        cupy.asarray(x[:, None]), cupy.asarray(grid2), order, bound, **options
    )
    assert isinstance(out, fake_cupy.ndarray)
    expected = pull(x[:, None], grid2, order, bound)
    np.testing.assert_allclose(out.get(), expected, atol=1e-12)
    coeff = spline_coeff(cupy.asarray(x), order, bound, ndim=3)
    assert isinstance(coeff, fake_cupy.ndarray)
    np.testing.assert_allclose(
        coeff.get(), spline_coeff(x, order, bound, ndim=3), atol=1e-12
    )


@pytest.mark.parametrize("bound", ALL_BOUNDS)
@pytest.mark.parametrize("order", [2, 4, 7])
def test_cupy_spline_coeff_stays_on_device(cupy, bound, order) -> None:  # noqa: ANN001
    # Every way of computing the coefficients runs on the device: the
    # recursive filter with and without the truncation of its initial
    # values, the correction near the edges, and the direct solve for
    # short axes.
    rng = np.random.default_rng(order)
    for n in (1, 2, 3, 5, 30, 100):
        x = rng.standard_normal((n, 4))
        coeff = spline_coeff1d(cupy.asarray(x), order, bound, axis=0)
        assert isinstance(coeff, fake_cupy.ndarray)
        np.testing.assert_allclose(
            coeff.get(), spline_coeff1d(x, order, bound, axis=0), atol=1e-14
        )
