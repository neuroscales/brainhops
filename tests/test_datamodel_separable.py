"""Tests for the separable reslice (issue #11).

These cover the dependency detector, the component partition, the step
ordering, the per-class execution, and the end-to-end reslice, which must
match the monolithic pull within the interpolation tolerance and be
bit-identical when nothing is separable.
"""

import numpy as np
import pytest

import brainhops.backends as backends
from brainhops._core.bsplines import pull, spline_matrix
from brainhops.backends import backend
from brainhops.datamodel import _xforms_separable as sep
from brainhops.datamodel import hierarchy
from brainhops.datamodel.axes import A, Axis, R, S, SpatialAxis, TimeAxis
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CompositionError,
    DisplacementField,
    Permutation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

# ----------------------------------------------------------------------
#   FIXTURES AND HELPERS
# ----------------------------------------------------------------------


def _sp(name: str) -> SpatialAxis:
    return SpatialAxis(name=name, unit=None)


def _time(name: str = "t", discrete: object = None) -> Axis:
    return Axis(name=name, type="time", unit=None, discrete=discrete)


def _components(els: list, n_grid: int) -> object:
    """The (grid axes, data axes) of every group of a chain of elements."""
    deps, stage_dims = sep._build_deps(els, n_grid)
    comps = sep._components(deps, n_grid, stage_dims)
    if comps is None:
        return None
    return sorted((tuple(c["G"]), tuple(c["D"])) for c in comps)


# ----------------------------------------------------------------------
#   DETECTOR
# ----------------------------------------------------------------------


def test_diagonal_affine_splits_into_singletons() -> None:
    matrix = np.zeros((4, 5))
    matrix[range(4), range(4)] = [2.0, 3.0, 0.5, 1.0]
    els = [Affine(matrix=matrix)]
    assert _components(els, 4) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_rotation_block_and_identity_axis_form_two_groups() -> None:
    # A rotation about a non-cardinal axis, so every spatial axis couples
    # to every other. A rotation about a single axis would leave one axis
    # free and split the block.
    def _rot(axis: int, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        rots = {
            0: [[1, 0, 0], [0, c, -s], [0, s, c]],
            1: [[c, 0, s], [0, 1, 0], [-s, 0, c]],
            2: [[c, -s, 0], [s, c, 0], [0, 0, 1]],
        }
        return np.asarray(rots[axis], dtype=float)

    rotation = _rot(2, 0.5) @ _rot(1, 0.5) @ _rot(0, 0.5)
    matrix = np.zeros((4, 5))
    matrix[:3, :3] = rotation
    matrix[3, 3] = 1.0
    els = [Affine(matrix=matrix)]
    # The rotation couples x and y, z stands alone, and the last axis is
    # its own group.
    assert _components(els, 4) == [((0, 1, 2), (0, 1, 2)), ((3,), (3,))]


def test_a_single_shear_entry_merges_two_axes() -> None:
    matrix = np.eye(4, 5)
    matrix[0, 1] = 0.5
    els = [Affine(matrix=matrix)]
    # Output axis 0 now reads input axes 0 and 1, so the two merge.
    assert _components(els, 4) == [
        ((0, 1), (0, 1)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_permutation_swaps_axes_and_relabels_groups() -> None:
    # Output i takes input permutation[i]. Swapping axis 0 and axis 3
    # keeps every axis in its own group, but pairs a grid axis with a
    # different data axis.
    els = [Permutation(permutation=np.asarray([3, 1, 2, 0]))]
    assert _components(els, 4) == [
        ((0,), (3,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (0,)),
    ]


def test_subspace_field_couples_its_axes_and_passes_the_rest() -> None:
    warp = np.zeros((5, 5, 5, 3))
    inner = DisplacementField(field=warp, order=1, bound="reflect")
    sub = SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()]),
    )
    assert _components([sub], 4) == [((0, 1, 2), (0, 1, 2)), ((3,), (3,))]


def test_raw_field_is_a_single_group() -> None:
    warp = np.zeros((4, 4, 4, 4, 4))
    field = DisplacementField(field=warp, order=1, bound="reflect")
    assert _components([field], 4) == [((0, 1, 2, 3), (0, 1, 2, 3))]


def test_closure_through_permutation_after_subspace() -> None:
    warp = np.zeros((5, 5, 5, 3))
    inner = DisplacementField(field=warp, order=1, bound="reflect")
    sub = SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()]),
    )
    # A permutation that moves the coupled spatial block to axes 1, 2, 3
    # and the time axis to axis 0. The closure carries the coupling
    # through the permutation, so the group's grid axes are the spatial
    # axes and its data axes are 1, 2, 3.
    perm = Permutation(permutation=np.asarray([3, 0, 1, 2]))
    comps = _components([sub, perm], 4)
    assert comps == [((0, 1, 2), (1, 2, 3)), ((3,), (0,))]


# ----------------------------------------------------------------------
#   DISCRETE AXES
# ----------------------------------------------------------------------


def _diagonal_seq(
    scale: object,
    shift: object,
    grid_system: object,
    data_system: object,
    shape: tuple,
) -> Sequence:
    matrix = np.zeros((len(shape), len(shape) + 1))
    matrix[range(len(shape)), range(len(shape))] = scale
    matrix[:, -1] = shift
    grid = CartesianField(shape=shape, input=grid_system, output=grid_system)
    affine = Affine(matrix=matrix, input=grid_system, output=data_system)
    return Sequence(transformations=[grid, affine])


def test_discrete_axis_with_non_integer_scale_raises() -> None:
    system = CoordinateSystem(axes=[_sp("x"), _time(discrete=True)])
    data = np.arange(12.0).reshape(4, 3)
    seq = _diagonal_seq([1.0, 1.5], [0.0, 0.0], system, system, (4, 3))
    with pytest.raises(CompositionError):
        sep.pull_separable(
            data,
            seq,
            order=1,
            bound="reflect",
            coeff=False,
        )


def test_discrete_axis_with_integer_shift_is_a_gather() -> None:
    system = CoordinateSystem(axes=[_sp("x"), _time(discrete=True)])
    data = np.arange(12.0).reshape(4, 3)
    seq = _diagonal_seq([1.0, 1.0], [0.0, 1.0], system, system, (4, 3))
    with backend("numpy"):
        got = sep.pull_separable(
            data,
            seq,
            order=1,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            seq.compute().field,
            order=1,
            bound="reflect",
            coeff=False,
        )
    assert np.array_equal(got, ref)


def test_discrete_axis_order_zero_is_a_gather() -> None:
    # An order-0 integer shift along a discrete axis is exact and allowed:
    # it moves whole samples without reading any value between them. The
    # maintainer chose to allow this case.
    system = CoordinateSystem(axes=[_sp("x"), _time(discrete=True)])
    data = np.arange(12.0).reshape(4, 3)
    seq = _diagonal_seq([1.0, 1.0], [0.0, 1.0], system, system, (4, 3))
    with backend("numpy"):
        got = sep.pull_separable(
            data,
            seq,
            order=0,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=0, bound="reflect", coeff=False
        )
    assert np.array_equal(got, ref)


def test_discrete_channel_untouched_at_order_zero_matches_monolithic() -> None:
    # A label map with a discrete channel axis, resliced by a diagonal scale
    # on the spatial axes at order 0. The channel axis is untouched, so the
    # reslice succeeds and matches the monolithic pull.
    system = CoordinateSystem(
        axes=[_sp("x"), _sp("y"), _time("c", discrete=True)]
    )
    data = np.arange(4 * 5 * 3, dtype=float).reshape(4, 5, 3)
    seq = _diagonal_seq(
        [1.5, 2.0, 1.0], [0.0, 0.0, 0.0], system, system, (4, 5, 3)
    )
    with backend("numpy"):
        got = sep.pull_separable(
            data,
            seq,
            order=0,
            bound="nearest",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=0, bound="nearest", coeff=False
        )
    assert np.allclose(got, ref)


def test_interpolating_across_discrete_channel_still_raises() -> None:
    # A non-integer scale along the discrete channel axis would resample
    # across it, which is refused.
    system = CoordinateSystem(
        axes=[_sp("x"), _sp("y"), _time("c", discrete=True)]
    )
    data = np.arange(4 * 5 * 3, dtype=float).reshape(4, 5, 3)
    seq = _diagonal_seq(
        [1.5, 2.0, 1.5], [0.0, 0.0, 0.0], system, system, (4, 5, 3)
    )
    with pytest.raises(CompositionError):
        sep.pull_separable(
            data,
            seq,
            order=1,
            bound="nearest",
            coeff=False,
        )


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("coeff", [False, True])
def test_discrete_channel_whole_sample_map_gathers_at_high_order(
    order: int, coeff: bool
) -> None:
    # A discrete channel axis moved by a whole-sample shift is an exact
    # gather at any order, even with reflect boundaries or spline
    # coefficients where a continuous axis would take the weight-matrix
    # path. The step must succeed and must not blend samples across the
    # discrete axis. This restores the behaviour before the class-(b)
    # routing change and is the correct no-interpolation-across-discrete
    # semantics: the monolithic pull would wrongly mix the channels here.
    system = CoordinateSystem(
        axes=[_sp("x"), _sp("y"), _sp("z"), _time("c", discrete=True)]
    )
    shape = (4, 5, 6, 3)
    rng = np.random.default_rng(0)
    data = rng.normal(size=shape)

    def _seq(channel_shift: float) -> Sequence:
        matrix = np.eye(4, 5)
        matrix[3, -1] = channel_shift
        grid = CartesianField(shape=shape, input=system, output=system)
        affine = Affine(matrix=matrix, input=system, output=system)
        return Sequence(transformations=[grid, affine])

    with backend("numpy"):
        base = sep.pull_separable(
            data,
            _seq(0.0),
            order=order,
            bound="reflect",
            coeff=coeff,
        )
        shifted = sep.pull_separable(
            data,
            _seq(1.0),
            order=order,
            bound="reflect",
            coeff=coeff,
        )
    # The channel shift only reindexes whole channels, so every output
    # channel of the shifted reslice equals one channel of the unshifted
    # reslice exactly. A blend across the discrete axis would leave no
    # output channel equal to any single input channel.
    assert shifted.shape == shape
    for c in range(shape[-1]):
        assert any(
            np.array_equal(shifted[..., c], base[..., other])
            for other in range(shape[-1])
        )


def test_discrete_channel_genuine_scale_raises_at_high_order() -> None:
    # A genuine resampling of the discrete channel axis is refused at every
    # order, so the whole-sample gather is not mistaken for permission to
    # interpolate across the axis.
    system = CoordinateSystem(
        axes=[_sp("x"), _sp("y"), _sp("z"), _time("c", discrete=True)]
    )
    shape = (4, 5, 6, 3)
    data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    matrix = np.eye(4, 5)
    matrix[3, 3] = 1.5
    grid = CartesianField(shape=shape, input=system, output=system)
    affine = Affine(matrix=matrix, input=system, output=system)
    seq = Sequence(transformations=[grid, affine])
    with pytest.raises(CompositionError):
        sep.pull_separable(
            data,
            seq,
            order=3,
            bound="reflect",
            coeff=False,
        )


# ----------------------------------------------------------------------
#   ORDERING
# ----------------------------------------------------------------------


def _step(kind: str, k: int, n_in: int, n_out: int) -> dict:
    return {"kind": kind, "k": k, "n_in": n_in, "n_out": n_out}


def test_ordering_runs_shrinking_before_expanding() -> None:
    shrink = _step("b", 1, 100, 10)
    grow = _step("b", 1, 10, 100)
    same = _step("a", 1, 10, 10)
    order = sep._order_steps(
        [grow, same, shrink], (0,), (0,), order=1, coeff=False
    )
    assert order[0] is shrink
    assert order[1] is same
    assert order[2] is grow


def test_ordering_puts_views_before_other_unit_steps() -> None:
    view = _step("a", 1, 10, 10)
    matmul = _step("b", 1, 10, 10)
    order = sep._order_steps([matmul, view], (0,), (0,), order=1, coeff=False)
    assert order[0] is view
    assert order[1] is matmul


def test_ordering_smaller_expansion_first() -> None:
    small = _step("b", 1, 10, 20)
    large = _step("b", 1, 10, 100)
    order = sep._order_steps([large, small], (0,), (0,), order=1, coeff=False)
    assert order[0] is small
    assert order[1] is large


# ----------------------------------------------------------------------
#   ONE-DIMENSIONAL AFFINE VS SPLINE MATRIX
# ----------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("bound", ["reflect", "nearest", "mirror", 2.5])
@pytest.mark.parametrize("coeff", [False, True])
def test_spline_matrix_reproduces_pull_1d(
    order: int, bound: object, coeff: bool
) -> None:
    with backend("numpy"):
        arr = np.linspace(-1.0, 1.0, 12)
        # A scale and shift that sends some samples outside the grid.
        coords = np.arange(9) * 1.3 - 1.5
        weights = np.asarray(spline_matrix(12, coords, order, bound, coeff))
        got = weights @ arr
        if not isinstance(bound, str):
            got = got + float(bound) * (1.0 - weights.sum(1))
        ref = pull(arr, coords[:, None], order, bound, coeff)
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   NUMERICAL EQUIVALENCE OF THE SEPARABLE RESLICE
# ----------------------------------------------------------------------


def _voxel_system(n: int) -> CoordinateSystem:
    axes = [_sp(name) for name in "xyz"[:n]]
    return CoordinateSystem(name="voxel", axes=axes)


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("bound", ["reflect", "nearest", "mirror", 2.5])
@pytest.mark.parametrize("coeff", [False, True])
def test_separable_matches_monolithic_diagonal(
    order: int, bound: object, coeff: bool
) -> None:
    system = _voxel_system(3)
    with backend("numpy"):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(6, 7, 5))
        matrix = np.zeros((3, 4))
        matrix[range(3), range(3)] = [1.3, 0.7, 1.0]
        matrix[:, -1] = [0.4, -0.6, 0.2]
        grid = CartesianField(shape=(6, 7, 5), input=system, output=system)
        affine = Affine(matrix=matrix, input=system, output=system)
        seq = Sequence(transformations=[grid, affine])
        got = sep.pull_separable(
            data,
            seq,
            order=order,
            bound=bound,
            coeff=coeff,
        )
        ref = pull(
            data, seq.compute().field, order=order, bound=bound, coeff=coeff
        )
    assert np.allclose(got, ref)


def test_scale_of_exactly_one_is_a_view_and_near_one_interpolates() -> None:
    system = _voxel_system(1)
    with backend("numpy"):
        data = np.arange(10.0)
        # A unit scale with an integer shift is a gather, exact for any
        # order. A scale a hair away from one is a genuine resampling and
        # falls to the weight-matrix path, still matching the monolithic
        # pull.
        for scale, shift in ((1.0, 2.0), (0.9999999, 0.0)):
            matrix = np.asarray([[scale, shift]])
            grid = CartesianField(shape=(10,), input=system, output=system)
            affine = Affine(matrix=matrix, input=system, output=system)
            seq = Sequence(transformations=[grid, affine])
            got = sep.pull_separable(
                data,
                seq,
                order=3,
                bound="reflect",
                coeff=False,
            )
            ref = pull(
                data,
                seq.compute().field,
                order=3,
                bound="reflect",
                coeff=False,
            )
            assert np.allclose(got, ref)


def _pull_pair_cast(
    column: np.ndarray,
    scale0: float,
    shift0: float,
    order: int,
    bound: object,
) -> tuple:
    # The separable and monolithic reslice of a two-axis image, for
    # comparing the integer and boolean cast. The first axis carries the
    # column and is scaled and shifted, so it takes the interpolating
    # separable path and the assembled result is cast at the end. The second
    # axis is a plain gather, which keeps the image separable into two
    # groups so the monolithic fallback is not taken.
    system = _voxel_system(2)
    data = np.repeat(column[:, None], 3, axis=1)
    shape = data.shape
    matrix = np.zeros((2, 3))
    matrix[0, 0], matrix[1, 1] = scale0, 1.0
    matrix[0, 2], matrix[1, 2] = shift0, 0.0
    grid = CartesianField(shape=shape, input=system, output=system)
    affine = Affine(matrix=matrix, input=system, output=system)
    seq = Sequence(transformations=[grid, affine])
    with backend("numpy"):
        got = sep.pull_separable(
            data,
            seq,
            order=order,
            bound=bound,
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=order, bound=bound, coeff=False
        )
    return got, ref


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
def test_integer_output_clips_overshoot_and_rounds_half_away(
    dtype: object,
) -> None:
    # A cubic reslice of a sharp step overshoots the dtype range. The
    # monolithic pull clips the overshoot to the range and rounds half away
    # from zero, so the separable reslice must produce the same integers
    # rather than wrap the overshoot or round a tie to even.
    info = np.iinfo(dtype)
    column = (np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]) * info.max).astype(
        dtype
    )
    got, ref = _pull_pair_cast(column, -0.8, 8.0, 3, "mirror")
    assert got.dtype == np.dtype(dtype)
    assert np.array_equal(got, ref)


def test_integer_downscale_rounds_ties_half_away_from_zero() -> None:
    # A half-integer sample position lands a linear interpolation exactly on
    # a tie. The monolithic pull rounds the tie away from zero, so a ramp
    # must match it rather than round to even.
    column = np.arange(6, dtype=np.uint8)
    got, ref = _pull_pair_cast(column, 0.5, 0.0, 1, "reflect")
    assert got.dtype == np.uint8
    assert np.array_equal(got, ref)


def test_boolean_output_truncates_toward_zero() -> None:
    # A boolean reslice truncates toward zero, so a fractional value below
    # one becomes False. The monolithic pull does the same, and the
    # separable reslice must not round such a value up to True.
    column = np.array([False, True, False, True, False, True])
    got, ref = _pull_pair_cast(column, 0.3, 0.0, 1, "reflect")
    assert got.dtype == np.bool_
    assert np.array_equal(got, ref)


def _classify_single(
    scale: float,
    shift: float,
    order: int = 3,
    bound: object = "mirror",
    coeff: bool = False,
) -> str:
    els = [Affine(matrix=np.asarray([[scale, shift]]))]
    deps, stage_dims = sep._build_deps(els, 1)
    comps = sep._components(deps, 1, stage_dims)
    step = sep._classify(
        comps[0],
        els,
        (10,),
        (10,),
        order=order,
        bound=bound,
        coeff=coeff,
        grid_system=None,
        data_system=None,
    )
    return step["kind"]


def test_exact_unit_scale_is_a_gather_and_near_unit_is_a_matrix() -> None:
    # A scale of exactly one, or a flip, with an integer shift is a gather,
    # applied without interpolation. A scale a hair away from one, or a
    # genuine resampling, falls to the weight-matrix path.
    assert _classify_single(1.0, 2.0) == "a"
    assert _classify_single(-1.0, 3.0) == "a"
    assert _classify_single(0.9999999, 0.0) == "b"
    assert _classify_single(1.3, 0.0) == "b"


def test_unit_scale_with_coeff_at_high_order_is_a_matrix() -> None:
    # With spline coefficients at order two or above, the monolithic pull
    # returns the reconstruction of the coefficients, not the raw
    # coefficient a gather would return. Such a step takes the weight
    # matrix instead, at order one or below it stays a gather.
    assert _classify_single(1.0, 2.0, order=3, coeff=True) == "b"
    assert _classify_single(1.0, 2.0, order=1, coeff=True) == "a"
    assert _classify_single(1.0, 2.0, order=0, coeff=True) == "a"


def test_unit_scale_with_reflect_above_order_one_is_a_matrix() -> None:
    # Scipy's reflect prefilter is not exactly interpolating above order
    # one, so a reflect gather would diverge from the monolithic pull. Such
    # a step takes the weight matrix, at order one or below it stays a
    # gather.
    assert _classify_single(1.0, 2.0, order=3, bound="reflect") == "b"
    assert _classify_single(1.0, 2.0, order=5, bound="reflect") == "b"
    assert _classify_single(1.0, 2.0, order=1, bound="reflect") == "a"


def test_restrict_handles_matrix_less_affine() -> None:
    # An affine with no matrix is the identity, the reading `_dependency`
    # gives it. `_restrict` must handle it without indexing into a missing
    # matrix.
    matrix = np.zeros((2, 3))
    matrix[0, 0], matrix[1, 1] = 2.0, 3.0
    els = [Affine(matrix=matrix), Affine(matrix=None)]
    deps, stage_dims = sep._build_deps(els, 2)
    comps = sep._components(deps, 2, stage_dims)
    assert comps is not None
    for comp in comps:
        sub = sep._restrict(comp, els, (6, 7))
        # The grid and the diagonal affine; the matrix-less affine is
        # dropped as the identity.
        assert len(sub) == 2


def test_constant_boundary_near_all_corners_of_a_warp() -> None:
    # A coupled three-dimensional warp inside a four-dimensional image,
    # with a large displacement that samples outside the grid near every
    # corner, a non-zero constant fill, and an order above one. The spatial
    # warp is a class-(c) batched pull, and the constant fill near the
    # padded corners must match the monolithic pull.
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    vox4 = CoordinateSystem(name="voxel4", axes=[xax, yax, zax, _time()])
    world4 = CoordinateSystem(
        name="world4", axes=[R, A, S, TimeAxis(name="t")]
    )
    vox3 = CoordinateSystem(name="vox3", axes=[xax, yax, zax])
    with backend("numpy"):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(6, 6, 6, 4))
        warp = rng.normal(size=(6, 6, 6, 3)) * 3.0
        dfield = DisplacementField(
            field=warp, input=vox3, output=vox3, order=3, bound=7.0
        )
        aff4 = Affine(
            matrix=np.diag([1.0, 1.0, 1.0, 1.0, 1.0])[:-1],
            input=vox4,
            output=world4,
        )
        preferred = aff4 @ dfield
        img = SingleScaleImage(data=data, transformations=[preferred])
        target = Affine(
            matrix=np.diag([1.0, 1.0, 1.0, 2.0, 1.0])[:-1],
            input=vox4,
            output=world4,
        )
        geometry = Geometry(
            (
                CartesianField(shape=(6, 6, 6, 2), input=vox4, output=vox4),
                target,
            )
        )
        transformation = (
            img.transformation.inverse()
            @ geometry.transformation
            @ geometry.grid
        )
        got = sep.pull_separable(
            data,
            transformation,
            order=3,
            bound=7.0,
            coeff=False,
        )
        ref = pull(
            data,
            transformation.compute().field,
            order=3,
            bound=7.0,
            coeff=False,
        )
    assert np.allclose(got, ref)


def test_class_a_gather_with_coeff_matches_monolithic() -> None:
    # A flip is class-(a) eligible, but with spline coefficients at order
    # three the monolithic pull returns the reconstruction of the
    # coefficients, not the raw coefficient a gather would return. The step
    # must route to the weight matrix so the result still matches.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(6, 7))
        # Axis 0 is a flip; axis 1 is a genuine rescale, so the reslice
        # factors into two groups.
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            order=3,
            bound="mirror",
            coeff=True,
        )
        ref = pull(
            data, seq.compute().field, order=3, bound="mirror", coeff=True
        )
    assert np.allclose(got, ref)


@pytest.mark.parametrize("order", [3, 5])
def test_class_a_reflect_gather_matches_monolithic(order: int) -> None:
    # A flip is class-(a) eligible, but scipy's reflect prefilter is not
    # exactly interpolating above order one, so the gather would diverge
    # from the monolithic pull. The step routes to the weight matrix.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(6, 7))
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            order=order,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            seq.compute().field,
            order=order,
            bound="reflect",
            coeff=False,
        )
    assert np.allclose(got, ref)


def test_output_dtype_float32_is_preserved() -> None:
    # The separable pipeline runs in a floating working dtype, then casts
    # back to the input dtype once at the end, matching the monolithic pull.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(6, 7)).astype(np.float32)
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            order=3,
            bound="mirror",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=3, bound="mirror", coeff=False
        )
    assert got.dtype == np.float32
    assert ref.dtype == np.float32
    assert np.allclose(got, ref)


def test_output_dtype_uint8_order_zero_matches_monolithic() -> None:
    # An integer input is lifted to float for the pipeline and rounded back
    # once at the end, the way the monolithic pull rounds an integer output.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(5)
        data = rng.integers(0, 255, size=(6, 7)).astype(np.uint8)
        seq = _diagonal_seq([-1.0, 1.5], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            order=0,
            bound="nearest",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=0, bound="nearest", coeff=False
        )
    assert got.dtype == np.uint8
    assert ref.dtype == np.uint8
    assert np.array_equal(got, ref)


def test_class_a_only_pipeline_preserves_dtype() -> None:
    # A pipeline of only gather steps keeps the input dtype, the same dtype
    # the monolithic pull returns.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(9)
        data = rng.normal(size=(6, 7)).astype(np.float32)
        # Two flips, each a class-(a) gather.
        seq = _diagonal_seq([-1.0, -1.0], [5.0, 6.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            order=1,
            bound="mirror",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=1, bound="mirror", coeff=False
        )
    assert got.dtype == np.float32
    assert ref.dtype == np.float32
    assert np.allclose(got, ref)


def test_subspace_warp_without_axes_does_not_silently_pass_through() -> None:
    # An interpolating subspace transform that names no axes is malformed.
    # It must not be read as pass-through and return the data unwarped. The
    # separable path falls back to the monolithic pull, which raises.
    system = _voxel_system(3)
    with backend("numpy"):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(5, 5, 5))
        warp = rng.normal(size=(5, 5, 5, 3))
        inner = DisplacementField(field=warp, order=1, bound="reflect")
        sub = SubspaceTransformation(transformation=inner)
        grid = CartesianField(shape=(5, 5, 5), input=system, output=system)
        seq = Sequence(transformations=[grid, sub])
        with pytest.raises(Exception) as sep_error:
            sep.pull_separable(
                data,
                seq,
                order=1,
                bound="reflect",
                coeff=False,
            )
        with pytest.raises(Exception) as mono_error:
            pull(
                data,
                seq.compute().field,
                order=1,
                bound="reflect",
                coeff=False,
            )
    assert type(sep_error.value) is type(mono_error.value)


# ----------------------------------------------------------------------
#   FALLBACK
# ----------------------------------------------------------------------


def test_cras_to_fras_bridge_factors_into_singletons() -> None:
    # A source grid in cRAS order and a target grid in fRAS order meet
    # through a permutation and flip bridge. The detector must track each
    # axis through the relabelling permutation, so the chain factors into
    # per-axis groups rather than one coupled group.
    from brainhops.datamodel.systems import (
        CRASCoordinateSystem,
        FRASCoordinateSystem,
        RASCoordinateSystem,
    )

    with backend("numpy"):
        data = np.arange(6 * 6 * 6, dtype=float).reshape(6, 6, 6)
        src = Affine(
            matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
            input=CRASCoordinateSystem(),
            output=RASCoordinateSystem(),
        )
        img = SingleScaleImage(data=data, transformations=[src])
        target = Affine(
            matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
            input=FRASCoordinateSystem(),
            output=RASCoordinateSystem(),
        )
        geometry = Geometry(
            (
                CartesianField(
                    shape=(6, 6, 6),
                    input=FRASCoordinateSystem(),
                    output=FRASCoordinateSystem(),
                ),
                target,
            )
        )
        transformation = (
            img.transformation.inverse()
            @ geometry.transformation
            @ geometry.grid
        )
        part = transformation.compute(mode=hierarchy.AffineTransformation)
        els = list(part.transformations[1:])
        comps = _components(els, 3)
        # Every group is a single grid axis and a single data axis, and
        # every axis is covered.
        assert comps is not None
        assert len(comps) == 3
        for grid_axes, data_axes in comps:
            assert len(grid_axes) == 1 and len(data_axes) == 1

        got = sep.pull_separable(
            data,
            transformation,
            order=1,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            transformation.compute().field,
            order=1,
            bound="reflect",
            coeff=False,
        )
    assert np.allclose(got, ref)


def test_coarser_grid_and_sub_geometry_reslice_match_monolithic() -> None:
    from brainhops.datamodel.systems import (
        RASCoordinateSystem,
        VoxelCoordinateSystem,
    )

    with backend("numpy"):
        data = np.arange(6 * 6 * 6, dtype=float).reshape(6, 6, 6)
        src = Affine(
            matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
            input=VoxelCoordinateSystem(),
            output=RASCoordinateSystem(),
        )
        img = SingleScaleImage(data=data, transformations=[src])
        for geometry in (
            Geometry(
                (
                    CartesianField(
                        shape=(3, 3, 3),
                        input=VoxelCoordinateSystem(),
                        output=VoxelCoordinateSystem(),
                    ),
                    Affine(
                        matrix=np.diag([4.0, 4.0, 4.0, 1.0])[:-1],
                        input=VoxelCoordinateSystem(),
                        output=RASCoordinateSystem(),
                    ),
                )
            ),
            img.geometry[:, 1:5, :],
        ):
            transformation = (
                img.transformation.inverse()
                @ geometry.transformation
                @ geometry.grid
            )
            got = sep.pull_separable(
                data,
                transformation,
                order=1,
                bound="reflect",
                coeff=False,
            )
            ref = pull(
                data,
                transformation.compute().field,
                order=1,
                bound="reflect",
                coeff=False,
            )
            assert np.allclose(got, ref)


def test_three_d_image_through_raw_warp_is_the_fallback() -> None:
    system = _voxel_system(3)
    with backend("numpy"):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(5, 5, 5))
        warp = rng.normal(size=(5, 5, 5, 3)) * 0.8
        field = DisplacementField(
            field=warp, input=system, output=system, order=3, bound="reflect"
        )
        grid = CartesianField(shape=(5, 5, 5), input=system, output=system)
        seq = Sequence(transformations=[grid, field])
        got = sep.pull_separable(
            data,
            seq,
            order=3,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, order=3, bound="reflect", coeff=False
        )
    # A single coupled group falls back to the monolithic pull, which is
    # bit-identical.
    assert np.array_equal(got, ref)


# ----------------------------------------------------------------------
#   ISSUE #11 DEMONSTRATION
# ----------------------------------------------------------------------


def _demonstration_image() -> tuple:
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    vox4 = CoordinateSystem(name="voxel4", axes=[xax, yax, zax, _time()])
    world4 = CoordinateSystem(
        name="world4", axes=[R, A, S, TimeAxis(name="t")]
    )
    vox3 = CoordinateSystem(name="vox3", axes=[xax, yax, zax])
    rng = np.random.default_rng(0)
    warp = rng.normal(size=(6, 6, 6, 3)) * 0.7
    dfield = DisplacementField(
        field=warp, input=vox3, output=vox3, order=3, bound="reflect"
    )
    aff4 = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.5, 1.0])[:-1],
        input=vox4,
        output=world4,
    )
    preferred = aff4 @ dfield
    data = np.arange(6 * 6 * 6 * 5, dtype=float).reshape(6, 6, 6, 5)
    return (
        SingleScaleImage(data=data, transformations=[preferred]),
        vox4,
        world4,
    )


def test_issue_11_spatial_warp_and_time_affine() -> None:
    img, vox4, world4 = _demonstration_image()
    data = np.asarray(img.data)
    # Reslice onto fewer time points, with the time axis rescaled.
    target = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 3.0, 1.0])[:-1],
        input=vox4,
        output=world4,
    )
    geometry = Geometry(
        (CartesianField(shape=(6, 6, 6, 3), input=vox4, output=vox4), target)
    )
    transformation = (
        img.transformation.inverse() @ geometry.transformation @ geometry.grid
    )

    with backend("numpy"):
        ref = pull(
            data,
            transformation.compute().field,
            order=1,
            bound="reflect",
            coeff=False,
        )

    shapes = []
    original = backends.npndi.map_coordinates

    def spy(inp: object, coords: object, **kwargs: object) -> object:
        shapes.append(coords.shape)
        return original(inp, coords, **kwargs)

    backends.npndi.map_coordinates = spy
    try:
        with backend("numpy"):
            got = sep.pull_separable(
                data,
                transformation,
                order=1,
                bound="reflect",
                coeff=False,
            )
    finally:
        backends.npndi.map_coordinates = original

    assert np.allclose(got, ref)
    # The coupled spatial warp runs as a three-dimensional pull, and the
    # four-dimensional pull never happens.
    ndims = {shape[0] for shape in shapes}
    assert 4 not in ndims
    assert 3 in ndims


def test_issue_11_flip_and_scale_do_not_interpolate_the_data() -> None:
    img, vox4, world4 = _demonstration_image()
    # Drop the warp so the whole transformation is a diagonal affine: a
    # y-flip and a time rescale onto fewer time points.
    src = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0, 1.0])[:-1],
        input=vox4,
        output=world4,
    )
    data = np.asarray(img.data)
    img = SingleScaleImage(data=data, transformations=[src])
    flip = np.diag([2.0, -2.0, 2.0, 2.0, 1.0])[:-1]
    flip[1, -1] = 2.0 * (6 - 1)
    target = Affine(matrix=flip, input=vox4, output=world4)
    geometry = Geometry(
        (CartesianField(shape=(6, 6, 6, 3), input=vox4, output=vox4), target)
    )
    transformation = (
        img.transformation.inverse() @ geometry.transformation @ geometry.grid
    )

    with backend("numpy"):
        ref = pull(
            data,
            transformation.compute().field,
            order=1,
            bound="reflect",
            coeff=False,
        )

    inputs = []
    original = backends.npndi.map_coordinates

    def spy(inp: object, coords: object, **kwargs: object) -> object:
        inputs.append(np.asarray(inp).shape)
        return original(inp, coords, **kwargs)

    backends.npndi.map_coordinates = spy
    try:
        with backend("numpy"):
            got = sep.pull_separable(
                data,
                transformation,
                order=1,
                bound="reflect",
                coeff=False,
            )
    finally:
        backends.npndi.map_coordinates = original

    assert np.allclose(got, ref)
    # No call interpolates the data itself. Every call builds a weight
    # matrix by resampling the rows of a small identity, which are
    # one-dimensional, so the flip and the scale move samples without
    # sampling between them. A pull of the data would be three- or
    # four-dimensional.
    for shape in inputs:
        assert len(shape) == 1


# ----------------------------------------------------------------------
#   LARGE-AXIS ROUTING (C1)
# ----------------------------------------------------------------------


def test_large_one_dimensional_axis_routes_to_batched_pull() -> None:
    # A one-dimensional interpolating step builds a weight matrix of
    # `n_out * n_in` elements, which is enormous for a long axis. Above the
    # threshold the step falls back to the batched pull the coupled class
    # uses, and below it the weight matrix is used. Both paths match the
    # monolithic pull.
    system = _voxel_system(2)
    threshold = sep._MAX_WEIGHT_MATRIX_ELEMENTS

    def _build(n: int) -> tuple:
        data = np.arange(n * 3, dtype=float).reshape(n, 3)
        matrix = np.zeros((2, 3))
        # Axis 0 is a genuine rescale, so it takes the interpolating class-(b)
        # path. Axis 1 is a flip, a class-(a) gather, which never pulls.
        matrix[0, 0], matrix[1, 1] = 1.3, -1.0
        matrix[1, 2] = 2.0
        grid = CartesianField(shape=(n, 3), input=system, output=system)
        affine = Affine(matrix=matrix, input=system, output=system)
        return data, Sequence(transformations=[grid, affine])

    calls = {"pull_axes": 0}
    original = sep.pull_axes

    def spy(*args: object, **kwargs: object) -> object:
        calls["pull_axes"] += 1
        return original(*args, **kwargs)

    small_n = 100
    assert small_n * small_n <= threshold
    sep.pull_axes = spy
    try:
        with backend("numpy"):
            data, seq = _build(small_n)
            small = sep.pull_separable(
                data, seq, order=3, bound="mirror", coeff=False
            )
            small_ref = pull(
                data, seq.compute().field, order=3, bound="mirror", coeff=False
            )
    finally:
        sep.pull_axes = original
    # The small axis uses the weight matrix, so no batched pull runs.
    assert calls["pull_axes"] == 0
    assert np.allclose(small, small_ref)

    large_n = int(threshold**0.5) + 200
    assert large_n * large_n > threshold
    calls["pull_axes"] = 0
    sep.pull_axes = spy
    try:
        with backend("numpy"):
            data, seq = _build(large_n)
            large = sep.pull_separable(
                data, seq, order=3, bound="mirror", coeff=False
            )
            large_ref = pull(
                data, seq.compute().field, order=3, bound="mirror", coeff=False
            )
    finally:
        sep.pull_axes = original
    # The large axis exceeds the threshold, so the class-(b) step routes to
    # the batched pull.
    assert calls["pull_axes"] >= 1
    assert np.allclose(large, large_ref)


# ----------------------------------------------------------------------
#   SUBSPACE INNER DEPENDENCY (C3 + C4)
# ----------------------------------------------------------------------


_SUB_SYSTEM = CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()])


def _subspace(inner: object, axes: list) -> SubspaceTransformation:
    return SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray(axes),
        output_axes=np.asarray(axes),
        input=_SUB_SYSTEM,
    )


def test_subspace_wrapping_diagonal_scaling_splits_into_singletons() -> None:
    # A subspace wrapping a diagonal scaling mixes no axes. Its dependency is
    # read from the inner scaling, so each axis is its own group rather than
    # one coupled block.
    inner = Scaling(scale=np.asarray([2.0, 3.0, 0.5]))
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], 4) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_subspace_wrapping_diagonal_affine_splits_into_singletons() -> None:
    inner = Affine(matrix=np.diag([2.0, 3.0, 0.5, 1.0])[:-1])
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], 4) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_nested_subspace_couples_only_the_truly_warped_axis() -> None:
    # A subspace whose inner is itself a subspace over a sub-subset, nested
    # twice, down to a one-axis warp. Only the truly warped axis couples; the
    # rest pass through as singletons.
    warp = DisplacementField(field=np.zeros((5, 1)), order=1, bound="reflect")
    level1 = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0]),
        output_axes=np.asarray([0]),
    )
    level0 = SubspaceTransformation(
        transformation=level1,
        input_axes=np.asarray([0, 1]),
        output_axes=np.asarray([0, 1]),
    )
    sub = _subspace(level0, [0, 1, 2])
    assert _components([sub], 4) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_subspace_inner_warp_on_a_subset_couples_only_that_subset() -> None:
    # A warp that touches only axes 0 and 1, inside a subspace over axes
    # 0, 1, 2. Axes 0 and 1 couple, axis 2 passes through.
    warp = DisplacementField(
        field=np.zeros((5, 5, 2)), order=1, bound="reflect"
    )
    inner = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1]),
        output_axes=np.asarray([0, 1]),
    )
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], 4) == [
        ((0, 1), (0, 1)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_subspace_wrapping_raw_field_couples_all_its_axes() -> None:
    # A raw field inner mixes every axis it names, so the acted-on axes stay
    # one coupled block. This is the safe over-approximation.
    warp = DisplacementField(
        field=np.zeros((5, 5, 5, 3)), order=1, bound="reflect"
    )
    sub = _subspace(warp, [0, 1, 2])
    assert _components([sub], 4) == [((0, 1, 2), (0, 1, 2)), ((3,), (3,))]


@pytest.mark.parametrize(
    "inner",
    [
        Scaling(scale=np.asarray([2.0, 3.0, 0.5])),
        SubspaceTransformation(
            transformation=DisplacementField(
                field=np.zeros((5, 5, 2)), order=1, bound="reflect"
            ),
            input_axes=np.asarray([0, 1]),
            output_axes=np.asarray([0, 1]),
        ),
    ],
)
def test_reverting_subspace_recursion_to_all_ones_over_couples(
    inner: object, monkeypatch: object
) -> None:
    # Without the recursion the subspace couples every acted-on axis. This
    # confirms the recursion is what produces the finer partition, and that
    # the fallback it replaces is an over-approximation, never a false split.
    sub = _subspace(inner, [0, 1, 2])
    fine = _components([sub], 4)
    monkeypatch.setattr(
        sep,
        "_transform_dependency",
        lambda transform, i, o: np.ones((o, i), dtype=bool),
    )
    coarse = _components([sub], 4)
    assert coarse == [((0, 1, 2), (0, 1, 2)), ((3,), (3,))]
    assert fine != coarse


def test_subspace_wrapping_diagonal_reslice_matches_monolithic() -> None:
    # The finer partition from a subspace wrapping a diagonal must still equal
    # the monolithic reslice, so the split is never a false one.
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    vox4 = CoordinateSystem(name="voxel4", axes=[xax, yax, zax, _time()])
    vox3 = CoordinateSystem(name="vox3", axes=[xax, yax, zax])
    with backend("numpy"):
        rng = np.random.default_rng(21)
        data = rng.normal(size=(6, 7, 5, 4))
        inner = Affine(
            matrix=np.diag([1.3, 0.7, 1.5, 1.0])[:-1],
            input=vox3,
            output=vox3,
        )
        sub = SubspaceTransformation(
            transformation=inner,
            input_axes=np.asarray([0, 1, 2]),
            output_axes=np.asarray([0, 1, 2]),
            input=vox4,
            output=vox4,
        )
        grid = CartesianField(shape=(6, 7, 5, 4), input=vox4, output=vox4)
        seq = Sequence(transformations=[grid, sub])
        got = sep.pull_separable(
            data, seq, order=3, bound="reflect", coeff=False
        )
        ref = pull(
            data, seq.compute().field, order=3, bound="reflect", coeff=False
        )
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   RESTRICTION KEEPS THE CHEAPER TYPE (C5)
# ----------------------------------------------------------------------


def test_restrict_keeps_scaling_and_translation_types() -> None:
    # A restricted scaling stays a scaling and a restricted translation stays
    # a translation, rather than being widened to a general affine sub-block.
    els = [
        Scaling(scale=np.asarray([2.0, 3.0])),
        Translation(translation=np.asarray([1.0, -2.0])),
    ]
    deps, stage_dims = sep._build_deps(els, 2)
    comps = sep._components(deps, 2, stage_dims)
    assert comps is not None
    for comp in comps:
        sub = sep._restrict(comp, els, (6, 7))
        types = [type(t).__name__ for t in sub[1:]]
        assert "Affine" not in types
        assert set(types) == {"Scaling", "Translation"}


def test_restrict_keeps_permutation_type() -> None:
    els = [Permutation(permutation=np.asarray([1, 0, 2]))]
    deps, stage_dims = sep._build_deps(els, 3)
    comps = sep._components(deps, 3, stage_dims)
    assert comps is not None
    for comp in comps:
        sub = sep._restrict(comp, els, (5, 5, 5))
        assert [type(t).__name__ for t in sub[1:]] == ["Permutation"]


def test_scaling_translation_restricted_steps_match_monolithic() -> None:
    # A scale-and-translation pipeline whose restricted steps keep their
    # cheaper types still reproduces the monolithic reslice.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(33)
        data = rng.normal(size=(6, 7))
        els = [
            Scaling(scale=np.asarray([1.3, 0.7])),
            Translation(translation=np.asarray([0.4, -0.6])),
        ]
        deps, stage_dims = sep._build_deps(els, 2)
        comps = sep._components(deps, 2, stage_dims)
        grid = CartesianField(shape=(6, 7))
        ref = pull(
            data,
            Sequence(transformations=[grid, *els]).compute().field,
            order=3,
            bound="mirror",
            coeff=False,
        )
        for comp in comps:
            sub = sep._restrict(comp, els, (6, 7))
            assert set(type(t).__name__ for t in sub[1:]) == {
                "Scaling",
                "Translation",
            }
        steps = [
            sep._classify(
                comp,
                els,
                (6, 7),
                data.shape,
                3,
                "mirror",
                False,
                system,
                system,
            )
            for comp in comps
        ]
        labels = [("data", i) for i in range(data.ndim)]
        arr = data
        for step in steps:
            arr, labels = step["run"](arr, labels)
        perm = [labels.index(("grid", g)) for g in range(2)]
        got = np.transpose(arr, perm)
    assert np.allclose(got, ref)


def test_permutation_reslice_matches_monolithic() -> None:
    # A permutation survives the affine reduction as its own element, so the
    # type-preserving restriction runs end to end and matches the monolithic
    # reslice.
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    system = CoordinateSystem(name="vox", axes=[xax, yax, zax])
    with backend("numpy"):
        rng = np.random.default_rng(35)
        data = rng.normal(size=(4, 5, 6))
        perm = Permutation(
            permutation=np.asarray([2, 0, 1]), input=system, output=system
        )
        grid = CartesianField(shape=(6, 4, 5), input=system, output=system)
        seq = Sequence(transformations=[grid, perm])
        got = sep.pull_separable(
            data, seq, order=1, bound="reflect", coeff=False
        )
        ref = pull(
            data, seq.compute().field, order=1, bound="reflect", coeff=False
        )
    assert np.allclose(got, ref)
