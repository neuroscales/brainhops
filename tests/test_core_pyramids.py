"""Tests for the arithmetic that builds the coarser levels of a pyramid.

The coarse voxel `i` lies at the fine voxel `s * i + t`, with the actual
scale `s` and the shift `t` of the alignment: `f` and `(f - 1) / 2` for
`"exact"`, `n / ceil(n / f)` and `(s - 1) / 2` for `"edges"`, and
`(n - 1) / ceil((n - 1) / f)` and 0 for `"centers"`.
"""

import math

import numpy as np
import pytest
import typing_extensions as tx

from brainhops._core.pyramids import (
    ALIGNMENTS,
    BLOCK_MODES,
    FIELD_MODES,
    MODES,
    check_mode,
    default_factors,
    downsample,
    level_map,
    rescale_vectors,
)


@pytest.mark.parametrize("align", ALIGNMENTS)
@pytest.mark.parametrize("mode", ["mean", "binomial", "gaussian"])
@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("length", [7, 8, 9, 13])
def test_a_line_stays_exact_at_the_placement_of_its_level(
    align: str, mode: str, factor: int, length: int
) -> None:
    # A line is averaged, or pulled, into its value where the level puts
    # the coarse voxel, up to the edges with linear edges.
    values = 3.0 * np.arange(length) - 2.0

    (level,) = downsample(values, [(factor,)], mode, "linear", align)

    count, scale, shift = level_map(align, length, factor)
    positions = scale * np.arange(count) + shift
    assert level.shape == (count,)
    np.testing.assert_allclose(level, 3.0 * positions - 2.0, atol=1e-9)


@pytest.mark.parametrize(
    "align, expected",
    [
        ("exact", [(3, 2.0, 0.5), (4, 2.0, 0.5), (4, 2.0, 0.5)]),
        ("edges", [(4, 1.75, 0.375), (4, 2.0, 0.5), (5, 1.8, 0.4)]),
        ("centers", [(4, 2.0, 0.0), (5, 1.75, 0.0), (5, 2.0, 0.0)]),
    ],
)
def test_the_map_of_each_alignment(align: str, expected: list) -> None:
    maps = [level_map(align, length, 2) for length in (7, 8, 9)]

    np.testing.assert_allclose(maps, expected)


@pytest.mark.parametrize("align", ALIGNMENTS)
def test_the_maps_compose_from_level_to_level(align: str) -> None:
    # A level computed from the level before it has the shape and the
    # values of a level computed from the finest one.
    values = 3.0 * np.arange(37) - 2.0

    chained = downsample(values, [(2,), (6,)], "mean", "linear", align)
    direct = downsample(values, [(6,)], "mean", "linear", align)

    np.testing.assert_allclose(chained[1], direct[0], atol=1e-9)


def test_the_mean_of_a_window_of_two_voxels() -> None:
    values = np.arange(16.0).reshape(4, 4)

    (level,) = downsample(values, [(2, 2)], "mean")

    np.testing.assert_allclose(level, [[2.5, 4.5], [10.5, 12.5]])


def test_exact_crops_an_axis_to_a_multiple_of_the_factor() -> None:
    values = np.arange(7.0)

    (level,) = downsample(values, [(2,)], "mean")

    np.testing.assert_allclose(level, [0.5, 2.5, 4.5])


def test_the_binomial_filter_of_a_factor_of_two() -> None:
    # Voxel 2 is the last tap of the new voxel 0, which covers the
    # voxels -1 to 2, and the second tap of the new voxel 1, which
    # covers 1 to 4.
    values = np.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0])

    (level,) = downsample(values, [(2,)], "binomial")

    np.testing.assert_allclose(level, [1.0, 3.0, 0.0])


def test_the_binomial_filter_cancels_the_highest_frequency() -> None:
    values = np.tile([1.0, -1.0], 8)

    (level,) = downsample(values, [(2,)], "binomial")

    np.testing.assert_allclose(level[1:-1], 0.0, atol=1e-15)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_the_gaussian_filter_follows_the_fwhm_principle(factor: int) -> None:
    # The filter brings the FWHM of a voxel, 1, to the FWHM `f`, so its
    # standard deviation is `sqrt(f ** 2 - 1) / (2 * sqrt(2 * log(2)))`.
    # An impulse far from the edges gives each coarse voxel the weight
    # of the filter at its distance, so the logarithm of the values is a
    # parabola of the squared distance whose slope gives the deviation.
    position = 10 * factor
    values = np.zeros(20 * factor)
    values[position] = 1.0

    (level,) = downsample(values, [(factor,)], "gaussian")

    reached = np.flatnonzero(level > 0)
    centres = factor * reached + (factor - 1) / 2
    distances = (position - centres) ** 2
    assert len(set(distances)) > 1
    slope = np.polyfit(distances, np.log(level[reached]), 1)[0]
    sigma = math.sqrt(factor**2 - 1) / (2 * math.sqrt(2 * math.log(2)))
    np.testing.assert_allclose(np.sqrt(-1 / (2 * slope)), sigma)


_LABELS = np.array(
    [
        [1, 1, 2, 2],
        [1, 3, 2, 2],
        [5, 5, 7, 6],
        [5, 6, 6, 7],
    ],
    dtype="uint16",
)


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("nearest", [[1, 2], [5, 7]]),
        ("mode", [[1, 2], [5, 7]]),
        ("max", [[3, 2], [6, 7]]),
        ("min", [[1, 2], [5, 6]]),
    ],
)
def test_the_modes_that_keep_values(mode: str, expected: list) -> None:
    # The window [[7, 6], [6, 7]] ties, and the mode keeps the value of
    # its first voxel.
    (level,) = downsample(_LABELS, [(2, 2)], mode)

    assert level.dtype == np.uint16
    np.testing.assert_array_equal(level, expected)


@pytest.mark.parametrize("align", ALIGNMENTS)
def test_nearest_keeps_the_labels(align: str) -> None:
    labels = np.random.default_rng(1).integers(0, 5, (13, 12)) * 100

    (level,) = downsample(labels, [(3, 3)], "nearest", align=align)

    assert set(np.unique(level)) <= set(np.unique(labels))


def test_nearest_picks_the_voxels_of_the_finest_level() -> None:
    # The voxel `(f - 1) // 2` of each window of the finest level, also
    # for a level whose factors are multiples of the level before it.
    values = np.arange(32.0)

    levels = downsample(values, [(2,), (4,), (8,)], "nearest")

    for level, factor in zip(levels, (2, 4, 8)):
        np.testing.assert_array_equal(
            level, values[(factor - 1) // 2 :: factor]
        )


def test_the_block_modes_need_the_exact_alignment() -> None:
    for mode in BLOCK_MODES:
        assert check_mode(mode, align="exact") == mode
        for align in ("edges", "centers"):
            with pytest.raises(ValueError, match="exact"):
                check_mode(mode, align=align)


def test_an_averaged_integer_level_keeps_its_type() -> None:
    values = np.arange(64, dtype="uint8").reshape(8, 8)

    (level,) = downsample(values, [(2, 2)], "mean")

    assert level.dtype == np.uint8
    assert level.shape == (4, 4)


def test_an_axis_of_factor_one_is_left_as_it_is() -> None:
    values = np.random.default_rng(2).random((8, 5))

    (level,) = downsample(values, [(2, 1)], "mean")

    assert level.shape == (4, 5)
    np.testing.assert_allclose(level, (values[0::2] + values[1::2]) / 2)


def test_the_label_modes_are_refused_for_a_field() -> None:
    for mode in MODES:
        if mode in FIELD_MODES:
            assert check_mode(mode, field=True) == mode
        else:
            with pytest.raises(ValueError, match="field"):
                check_mode(mode, field=True)
    with pytest.raises(ValueError, match="modes"):
        check_mode("median")


@pytest.mark.parametrize(
    "kind, expected",
    [
        (None, [[4.0, 6.0]]),
        ("displacements", [[2.0, 1.5]]),
        ("coordinates", [[1.75, 1.0]]),
    ],
)
def test_the_vectors_in_the_grid_of_the_field_are_rescaled(
    kind: object, expected: list
) -> None:
    values = np.array([[4.0, 6.0]])

    rescaled = rescale_vectors(values, [2.0, 4.0], [0.5, 2.0], kind)

    np.testing.assert_allclose(rescaled, expected)
    back = rescale_vectors(rescaled, [2.0, 4.0], [0.5, 2.0], kind, True)
    np.testing.assert_allclose(back, values)


def test_an_axis_at_the_minimum_size_freezes() -> None:
    # An axis of 40 voxels is halved to 20, and an axis of 32 or fewer
    # voxels stays as it is.
    factors = default_factors([256, 40, 32], [1.0] * 3, [2, 2, 2])

    assert factors == [(2, 2, 1), (4, 2, 1), (8, 2, 1)]


def test_coarse_axes_wait_until_the_voxels_are_roughly_isotropic() -> None:
    # Voxels of 1 x 1 x 5 are halved in x and y until their size is
    # close enough to 5, and then the three axes are halved together.
    factors = default_factors(
        [1024, 1024, 256], [1.0, 1.0, 5.0], [2, 2, 2], nscales=5
    )

    assert factors == [(2, 2, 1), (4, 4, 1), (8, 8, 2), (16, 16, 4)]


def test_a_frozen_axis_is_never_downsampled() -> None:
    factors = default_factors([256, 256, 100], [1.0] * 3, [2, 2, 1])

    assert all(level[2] == 1 for level in factors)


def test_the_number_of_levels_is_bounded_by_the_rules() -> None:
    assert len(default_factors([1024], [1.0], [2], nscales=3)) == 2
    assert default_factors([1024], [1.0], [2], nscales=1) == []
    # An axis of 64 voxels is halved to 32, and then freezes.
    assert len(default_factors([64], [1.0], [2], nscales=4)) == 1
    with pytest.raises(ValueError, match="only"):
        default_factors([64], [1.0], [2], nscales=4, strict=True)
    with pytest.raises(ValueError):
        default_factors([1024], [1.0], [2], nscales=0)


@pytest.mark.parametrize(
    "align, mode, expected",
    [
        (
            "edges",
            "mean",
            [
                1.471590909090909,
                2.6363636363636367,
                1.2613636363636362,
                2.4375,
            ],
        ),
        ("edges", "binomial", [1.375, 2.5625, 1.3515625, 2.3828125]),
        (
            "edges",
            "gaussian",
            [
                1.3619646772506666,
                2.524277749306318,
                1.3844031738187659,
                2.3557595422596362,
            ],
        ),
        ("edges", "nearest", [1.0, 2.0, 1.0, 4.0]),
        ("centers", "mean", [0.3, 2.55, 2.5, 1.1, 2.6]),
        ("centers", "binomial", [0.5, 2.46875, 2.375, 1.28125, 2.4375]),
        (
            "centers",
            "gaussian",
            [
                0.4433467155441159,
                2.509800111032883,
                2.482547159675649,
                1.2118732467391746,
                2.437155814190656,
            ],
        ),
        ("centers", "nearest", [0.0, 2.0, 4.0, 1.0, 2.0]),
    ],
)
def test_the_pull_keeps_the_values_of_the_first_implementation(
    align: str, mode: str, expected: list
) -> None:
    # The values that the first implementation of the pull, which
    # interpolated with index lists, gave before the interpolation of
    # `brainhops._ext.interpol` replaced it.
    da = pytest.importorskip("dask.array")
    values = (np.arange(11.0) ** 2) % 7

    (level,) = downsample(values, [(3,)], mode, align=align)
    (lazy,) = downsample(
        da.from_array(values, chunks=4), [(3,)], mode, align=align
    )

    np.testing.assert_allclose(level, expected, rtol=1e-12)
    np.testing.assert_allclose(lazy.compute(), expected, rtol=1e-12)


@pytest.mark.parametrize("align", ALIGNMENTS)
@pytest.mark.parametrize("mode", ["mean", "binomial", "gaussian", "nearest"])
def test_the_passes_of_the_axes_are_separable(mode: str, align: str) -> None:
    # One pass per axis, in any order, gives the level of both axes.
    values = np.random.default_rng(4).random((23, 19))

    (both,) = downsample(values, [(2, 3)], mode, align=align)
    (rows,) = downsample(values, [(2, 1)], mode, align=align)
    (first,) = downsample(rows, [(1, 3)], mode, align=align)
    (columns,) = downsample(values, [(1, 3)], mode, align=align)
    (second,) = downsample(columns, [(2, 1)], mode, align=align)

    np.testing.assert_allclose(first, both, atol=1e-12)
    np.testing.assert_allclose(second, both, atol=1e-12)


def _tasks(array: tx.Any) -> int:
    return len(dict(array.__dask_graph__()))


@pytest.mark.parametrize("align", ALIGNMENTS)
@pytest.mark.parametrize("mode", MODES)
def test_a_dask_level_has_one_task_per_chunk(mode: str, align: str) -> None:
    # Each chunk of a level is one task that reads the chunks of the
    # finer level that its region overlaps, so a level adds as many
    # tasks to the graph as it has chunks, and its chunks have the shape
    # of the chunks of the finest level.
    da = pytest.importorskip("dask.array")
    if mode in BLOCK_MODES and align != "exact":
        pytest.skip("The block modes need the exact alignment.")
    values = np.random.default_rng(5).integers(0, 4, (64, 48, 40)).astype("f8")
    lazy = da.from_array(values, chunks=(16, 16, 16))
    factors = [(2, 2, 2), (4, 4, 2), (8, 8, 4)]

    levels = downsample(lazy, factors, mode, align=align)
    expected = downsample(values, factors, mode, align=align)

    finer = lazy
    for level, reference in zip(levels, expected):
        assert isinstance(level, da.Array)
        assert all(max(c) <= 16 for c in level.chunks)
        added = _tasks(level) - _tasks(finer)
        assert added == level.npartitions
        np.testing.assert_allclose(level.compute(), reference, atol=1e-12)
        finer = level if mode != "nearest" else lazy


@pytest.mark.parametrize("align", ALIGNMENTS)
@pytest.mark.parametrize("mode", MODES)
def test_each_mode_of_a_dask_array_equals_the_numpy_one(
    mode: str, align: str
) -> None:
    da = pytest.importorskip("dask.array")
    if mode in BLOCK_MODES and align != "exact":
        pytest.skip("The block modes need the exact alignment.")
    values = np.random.default_rng(3).integers(0, 4, (21, 18)).astype("f8")
    factors = [(2, 2), (6, 4), (6, 8)]

    expected = downsample(values, factors, mode, align=align)
    lazy = downsample(
        da.from_array(values, chunks=(7, 5)), factors, mode, align=align
    )

    for level, reference in zip(lazy, expected):
        assert isinstance(level, da.Array)
        np.testing.assert_allclose(level.compute(), reference, atol=1e-12)


@pytest.mark.parametrize("edge", ["repeat", "linear"])
@pytest.mark.parametrize("align", ALIGNMENTS)
@pytest.mark.parametrize("mode", ["binomial", "gaussian"])
def test_the_chunks_at_the_edges_extend_the_array_as_numpy_does(
    mode: str, align: str, edge: str
) -> None:
    # A chunk at an edge reads the voxels that the extension of the
    # edge needs, so that it extends the array as the whole array is
    # extended, even with chunks shorter than the filter.
    da = pytest.importorskip("dask.array")
    values = np.random.default_rng(6).random((29, 26))

    (expected,) = downsample(values, [(4, 3)], mode, edge, align)
    (lazy,) = downsample(
        da.from_array(values, chunks=3), [(4, 3)], mode, edge, align
    )

    np.testing.assert_allclose(lazy.compute(), expected, atol=1e-12)


def test_a_level_keeps_the_type_of_a_float32_array() -> None:
    values = np.random.default_rng(7).random((16, 16)).astype("float32")

    for align in ALIGNMENTS:
        for mode in ("mean", "binomial", "gaussian", "nearest"):
            (level,) = downsample(values, [(2, 2)], mode, align=align)
            assert level.dtype == np.float32
