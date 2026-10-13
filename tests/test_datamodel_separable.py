"""Tests for the separable reslice (#11).

The axis groups are read from the factored normal form of
`compute(factor=True)` and executed by `separable`. The tests cover the
groups, the order of the steps, the strategy of each step (gather, weight
matrix or pull) and equality with the monolithic pull, which is exact when
nothing is separable.
"""

import numpy as np
import pytest

import brainhops.backends as backends
from brainhops._core.bsplines import pull, spline_matrix
from brainhops.backends import backend
from brainhops.datamodel import kinds
from brainhops.datamodel._transformations.compute import factor as fac
from brainhops.datamodel._transformations.compute import separable as sep
from brainhops.datamodel.axes import A, Axis, R, S, SpaceAxis, TimeAxis
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    DisplacementField,
    Permutation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)
from brainhops.errors import CompositionError

# ----------------------------------------------------------------------
#   FIXTURES AND HELPERS
# ----------------------------------------------------------------------


def _sp(name: str) -> SpaceAxis:
    return SpaceAxis(name=name, unit="index")


def _time(name: str = "t", discrete: object = None) -> Axis:
    return Axis(name=name, type="time", unit="index", discrete=discrete)


def _components(els: list, shape: tuple) -> object:
    """Return (grid axes, data axes) per group, or None if unfactored."""
    seq = Sequence(transformations=[CartesianField(shape=shape), *els])
    read = sep._groups(seq.compute(mode=kinds.Affine, factor=True), len(shape))
    if read is None:
        return None
    return sorted(
        (tuple(sorted(g["G"])), tuple(sorted(g["D"]))) for g in read[1]
    )


# ----------------------------------------------------------------------
#   DETECTOR
# ----------------------------------------------------------------------


def test_diagonal_affine_splits_into_singletons() -> None:
    matrix = np.zeros((4, 5))
    matrix[range(4), range(4)] = [2.0, 3.0, 0.5, 1.0]
    els = [Affine(matrix=matrix)]
    assert _components(els, (4, 5, 6, 3)) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_rotation_block_and_identity_axis_form_two_groups() -> None:
    # A non-cardinal rotation couples all spatial axes.
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
    # The spatial block is coupled and the last axis is its own group.
    assert _components(els, (4, 5, 6, 3)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]


def test_a_single_shear_entry_merges_two_axes() -> None:
    matrix = np.eye(4, 5)
    matrix[0, 1] = 0.5
    els = [Affine(matrix=matrix)]
    # The shear makes output axis 0 read inputs 0 and 1, so they merge.
    assert _components(els, (4, 5, 6, 3)) == [
        ((0, 1), (0, 1)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_permutation_swaps_axes_and_relabels_groups() -> None:
    # Swapping 0 and 3 pairs a grid axis with a different data axis.
    els = [Permutation(permutation=np.asarray([3, 1, 2, 0]))]
    assert _components(els, (4, 5, 6, 3)) == [
        ((0,), (3,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (0,)),
    ]


def test_subspace_field_couples_its_axes_and_passes_the_rest() -> None:
    warp = np.zeros((5, 5, 5, 3))
    inner = DisplacementField(field=warp, degree=1, bound="reflect")
    sub = SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()]),
    )
    assert _components([sub], (5, 5, 5, 3)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]


def test_raw_field_is_a_single_group() -> None:
    # A raw field couples everything into one unfactored group.
    warp = np.zeros((4, 4, 4, 4, 4))
    field = DisplacementField(field=warp, degree=1, bound="reflect")
    assert _components([field], (4, 4, 4, 4)) is None


def test_closure_through_permutation_after_subspace() -> None:
    warp = np.zeros((5, 5, 5, 3))
    inner = DisplacementField(field=warp, degree=1, bound="reflect")
    sub = SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()]),
    )
    # The permutation moves the spatial block to data axes 1 to 3.
    perm = Permutation(permutation=np.asarray([3, 0, 1, 2]))
    comps = _components([sub, perm], (5, 5, 5, 3))
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
            degree=1,
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
            degree=1,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            seq.compute().field,
            degree=1,
            bound="reflect",
            coeff=False,
        )
    assert np.array_equal(got, ref)


def test_discrete_axis_degree_zero_is_a_gather() -> None:
    # A degree-0 integer shift on a discrete axis takes whole samples.
    system = CoordinateSystem(axes=[_sp("x"), _time(discrete=True)])
    data = np.arange(12.0).reshape(4, 3)
    seq = _diagonal_seq([1.0, 1.0], [0.0, 1.0], system, system, (4, 3))
    with backend("numpy"):
        got = sep.pull_separable(
            data,
            seq,
            degree=0,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=0, bound="reflect", coeff=False
        )
    assert np.array_equal(got, ref)


def test_discrete_channel_untouched_at_degree_zero_matches_monolithic() -> (
    None
):
    # At degree 0, a diagonal spatial scale leaves the channel untouched.
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
            degree=0,
            bound="nearest",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=0, bound="nearest", coeff=False
        )
    assert np.allclose(got, ref)


def test_interpolating_across_discrete_channel_still_raises() -> None:
    # A non-integer scale would resample across the discrete channel.
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
            degree=1,
            bound="nearest",
            coeff=False,
        )


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("coeff", [False, True])
def test_discrete_channel_whole_sample_map_gathers_at_high_degree(
    degree: int, coeff: bool
) -> None:
    # A whole-sample shift of a discrete channel is an exact gather at any
    # degree, where the monolithic pull would blend the channels.
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
            degree=degree,
            bound="reflect",
            coeff=coeff,
        )
        shifted = sep.pull_separable(
            data,
            _seq(1.0),
            degree=degree,
            bound="reflect",
            coeff=coeff,
        )
    # Each shifted channel equals exactly one input channel; a blend would not.
    assert shifted.shape == shape
    for c in range(shape[-1]):
        assert any(
            np.array_equal(shifted[..., c], base[..., other])
            for other in range(shape[-1])
        )


def test_discrete_channel_genuine_scale_raises_at_high_degree() -> None:
    # Resampling a discrete channel is refused at every degree.
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
            degree=3,
            bound="reflect",
            coeff=False,
        )


# ----------------------------------------------------------------------
#   ORDERING
# ----------------------------------------------------------------------


def _step(kind: str, k: int, n_in: int, n_out: int) -> dict:
    return {"kind": kind, "k": k, "n_in": n_in, "n_out": n_out}


def test_ordering_runs_shrinking_before_expanding() -> None:
    shrink = _step("matrix", 1, 100, 10)
    grow = _step("matrix", 1, 10, 100)
    same = _step("gather", 1, 10, 10)
    order = sep._order_steps([grow, same, shrink], degree=1, coeff=False)
    assert order[0] is shrink
    assert order[1] is same
    assert order[2] is grow


def test_ordering_puts_views_before_other_unit_steps() -> None:
    view = _step("gather", 1, 10, 10)
    matmul = _step("matrix", 1, 10, 10)
    order = sep._order_steps([matmul, view], degree=1, coeff=False)
    assert order[0] is view
    assert order[1] is matmul


def test_ordering_smaller_expansion_first() -> None:
    small = _step("matrix", 1, 10, 20)
    large = _step("matrix", 1, 10, 100)
    order = sep._order_steps([large, small], degree=1, coeff=False)
    assert order[0] is small
    assert order[1] is large


# ----------------------------------------------------------------------
#   ONE-DIMENSIONAL AFFINE VS SPLINE MATRIX
# ----------------------------------------------------------------------


@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("bound", ["reflect", "nearest", "mirror", 2.5])
@pytest.mark.parametrize("coeff", [False, True])
def test_spline_matrix_reproduces_pull_1d(
    degree: int, bound: object, coeff: bool
) -> None:
    with backend("numpy"):
        arr = np.linspace(-1.0, 1.0, 12)
        # Scale and shift send some samples outside the grid.
        coords = np.arange(9) * 1.3 - 1.5
        weights = np.asarray(spline_matrix(12, coords, degree, bound, coeff))
        got = weights @ arr
        if not isinstance(bound, str):
            got = got + float(bound) * (1.0 - weights.sum(1))
        ref = pull(arr, coords[:, None], degree, bound, coeff)
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   NUMERICAL EQUIVALENCE OF THE SEPARABLE RESLICE
# ----------------------------------------------------------------------


def _voxel_system(n: int) -> CoordinateSystem:
    axes = [_sp(name) for name in "xyz"[:n]]
    return CoordinateSystem(name="voxel", axes=axes)


@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("bound", ["reflect", "nearest", "mirror", 2.5])
@pytest.mark.parametrize("coeff", [False, True])
def test_separable_matches_monolithic_diagonal(
    degree: int, bound: object, coeff: bool
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
            degree=degree,
            bound=bound,
            coeff=coeff,
        )
        ref = pull(
            data, seq.compute().field, degree=degree, bound=bound, coeff=coeff
        )
    assert np.allclose(got, ref)


def test_scale_of_exactly_one_is_a_view_and_near_one_interpolates() -> None:
    system = _voxel_system(1)
    with backend("numpy"):
        data = np.arange(10.0)
        # The first case is a gather; the second is a genuine resampling
        # through the weight matrix.
        for scale, shift in ((1.0, 2.0), (0.9999999, 0.0)):
            matrix = np.asarray([[scale, shift]])
            grid = CartesianField(shape=(10,), input=system, output=system)
            affine = Affine(matrix=matrix, input=system, output=system)
            seq = Sequence(transformations=[grid, affine])
            got = sep.pull_separable(
                data,
                seq,
                degree=3,
                bound="reflect",
                coeff=False,
            )
            ref = pull(
                data,
                seq.compute().field,
                degree=3,
                bound="reflect",
                coeff=False,
            )
            assert np.allclose(got, ref)


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_separable_matches_monolithic_through_an_inner_less_reindex(
    degree: int,
) -> None:
    # A subspace with no inner transformation that maps axes to other
    # axes is a reindex (#110).
    system = _voxel_system(3)
    with backend("numpy"):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(7, 6, 5))
        grid = CartesianField(shape=(6, 7, 5), input=system, output=system)
        swap = SubspaceTransformation(
            transformation=None,
            input_axes=np.asarray([0, 1]),
            output_axes=np.asarray([1, 0]),
            input=system,
            output=system,
        )
        matrix = np.zeros((3, 4))
        matrix[range(3), range(3)] = [0.9, 1.2, 1.0]
        matrix[:, -1] = [0.3, -0.4, 1.0]
        affine = Affine(matrix=matrix, input=system, output=system)
        seq = Sequence(transformations=[grid, swap, affine])
        opt = dict(degree=degree, bound="reflect", coeff=False)
        assert sep._plan(data.shape, seq, **opt) is not None
        got = sep.pull_separable(data, seq, **opt)
        ref = pull(data, seq.compute().field, **opt)
    assert got.shape == (6, 7, 5)
    assert np.allclose(got, ref)


def _pull_pair_cast(
    column: np.ndarray,
    scale0: float,
    shift0: float,
    degree: int,
    bound: object,
) -> tuple:
    # Axis 0 interpolates while axis 1 is a gather, so two groups remain.
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
            degree=degree,
            bound=bound,
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=degree, bound=bound, coeff=False
        )
    return got, ref


@pytest.mark.parametrize("dtype", [np.uint8, np.int16])
def test_integer_output_clips_overshoot_and_rounds_half_away(
    dtype: object,
) -> None:
    # A cubic step overshoots; both paths clip and round half away from zero.
    info = np.iinfo(dtype)
    column = (np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0]) * info.max).astype(
        dtype
    )
    got, ref = _pull_pair_cast(column, -0.8, 8.0, 3, "mirror")
    assert got.dtype == np.dtype(dtype)
    assert np.array_equal(got, ref)


def test_integer_downscale_rounds_ties_half_away_from_zero() -> None:
    # A half-integer position makes the linear interpolation a tie.
    column = np.arange(6, dtype=np.uint8)
    got, ref = _pull_pair_cast(column, 0.5, 0.0, 1, "reflect")
    assert got.dtype == np.uint8
    assert np.array_equal(got, ref)


def test_boolean_output_truncates_toward_zero() -> None:
    # Boolean output truncates toward zero.
    column = np.array([False, True, False, True, False, True])
    got, ref = _pull_pair_cast(column, 0.3, 0.0, 1, "reflect")
    assert got.dtype == np.bool_
    assert np.array_equal(got, ref)


def _plan(seq: Sequence, data_shape: tuple, **opt: object) -> dict:
    """Return the planned strategy of each group, keyed by its grid axes."""
    steps = sep._plan(data_shape, seq, **opt)
    return {tuple(step["G"]): step["kind"] for step in steps}


def _classify_single(
    scale: float,
    shift: float,
    degree: int = 3,
    bound: object = "mirror",
    coeff: bool = False,
) -> str:
    # Axis 0 is rescaled and shifted, axis 1 is untouched.
    matrix = np.asarray([[scale, 0.0, shift], [0.0, 1.0, 0.0]])
    grid = CartesianField(shape=(10, 3))
    seq = Sequence(transformations=[grid, Affine(matrix=matrix)])
    plan = _plan(seq, (10, 3), degree=degree, bound=bound, coeff=coeff)
    return plan[(0,)]


def test_exact_unit_scale_is_a_gather_and_near_unit_is_a_matrix() -> None:
    # A unit scale or flip with an integer shift is a gather.
    assert _classify_single(1.0, 2.0) == "gather"
    assert _classify_single(-1.0, 3.0) == "gather"
    assert _classify_single(0.9999999, 0.0) == "matrix"
    assert _classify_single(1.3, 0.0) == "matrix"


def test_unit_scale_with_coeff_at_high_degree_is_a_matrix() -> None:
    # Above degree 1, coefficients need a reconstruction, so the step
    # uses the weight matrix.
    assert _classify_single(1.0, 2.0, degree=3, coeff=True) == "matrix"
    assert _classify_single(1.0, 2.0, degree=1, coeff=True) == "gather"
    assert _classify_single(1.0, 2.0, degree=0, coeff=True) == "gather"


def test_unit_scale_with_reflect_above_degree_one_is_a_matrix() -> None:
    # The reflect prefilter is inexact above degree 1, so the step uses
    # the weight matrix.
    assert _classify_single(1.0, 2.0, degree=3, bound="reflect") == "matrix"
    assert _classify_single(1.0, 2.0, degree=5, bound="reflect") == "matrix"
    assert _classify_single(1.0, 2.0, degree=1, bound="reflect") == "gather"


def test_coupled_and_interpolating_groups_are_pulled() -> None:
    # The shear couples axes 0 and 1; a one-axis warp is pulled too.
    shear = np.eye(3, 4)
    shear[0, 1] = 0.5
    grid = CartesianField(shape=(4, 5, 6))
    opt = dict(degree=1, bound="reflect", coeff=False)
    seq = Sequence(transformations=[grid, Affine(matrix=shear)])
    assert _plan(seq, (4, 5, 6), **opt) == {(0, 1): "pull", (2,): "gather"}
    warp = SubspaceTransformation(
        transformation=_warp((6,), 3),
        input_axes=np.asarray([2]),
        output_axes=np.asarray([2]),
    )
    seq = Sequence(transformations=[grid, warp])
    assert _plan(seq, (4, 5, 6), **opt) == {
        (0,): "gather",
        (1,): "gather",
        (2,): "pull",
    }


def test_matrix_less_affine_is_the_identity_in_a_group() -> None:
    # A matrix-less affine is the identity and contributes nothing.
    matrix = np.zeros((2, 3))
    matrix[0, 0], matrix[1, 1] = 2.0, 3.0
    els = [Affine(matrix=matrix), Affine(matrix=None)]
    assert _components(els, (6, 7)) == [((0,), (0,)), ((1,), (1,))]
    seq = Sequence(transformations=[CartesianField(shape=(6, 7)), *els])
    opt = dict(degree=1, bound="reflect", coeff=False)
    assert _plan(seq, (6, 7), **opt) == {(0,): "matrix", (1,): "matrix"}


def test_constant_boundary_near_all_corners_of_a_warp() -> None:
    # A coupled warp that samples outside near every corner, with a
    # nonzero constant fill above degree 1, must match the monolithic pull.
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    vox4 = CoordinateSystem(name="voxel4", axes=[xax, yax, zax, _time()])
    world4 = CoordinateSystem(
        name="world4", axes=[R(), A(), S(), TimeAxis(name="t")]
    )
    vox3 = CoordinateSystem(name="vox3", axes=[xax, yax, zax])
    with backend("numpy"):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(6, 6, 6, 4))
        warp = rng.normal(size=(6, 6, 6, 3)) * 3.0
        dfield = DisplacementField(
            field=warp, input=vox3, output=vox3, degree=3, bound=7.0
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
            degree=3,
            bound=7.0,
            coeff=False,
        )
        ref = pull(
            data,
            transformation.compute().field,
            degree=3,
            bound=7.0,
            coeff=False,
        )
    assert np.allclose(got, ref)


def test_class_a_gather_with_coeff_matches_monolithic() -> None:
    # A flip with coefficients at degree 3 needs the weight matrix.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(6, 7))
        # Axis 0 is flipped and axis 1 genuinely rescaled.
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            degree=3,
            bound="mirror",
            coeff=True,
        )
        ref = pull(
            data, seq.compute().field, degree=3, bound="mirror", coeff=True
        )
    assert np.allclose(got, ref)


@pytest.mark.parametrize("degree", [3, 5])
def test_class_a_reflect_gather_matches_monolithic(degree: int) -> None:
    # A flip with the reflect prefilter above degree 1 needs a matrix.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(6, 7))
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            degree=degree,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            seq.compute().field,
            degree=degree,
            bound="reflect",
            coeff=False,
        )
    assert np.allclose(got, ref)


def test_output_dtype_float32_is_preserved() -> None:
    # The pipeline works in float and casts back once at the end.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(6, 7)).astype(np.float32)
        seq = _diagonal_seq([-1.0, 1.3], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            degree=3,
            bound="mirror",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=3, bound="mirror", coeff=False
        )
    assert got.dtype == np.float32
    assert ref.dtype == np.float32
    assert np.allclose(got, ref)


def test_output_dtype_uint8_degree_zero_matches_monolithic() -> None:
    # Integer input is rounded back once at the end.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(5)
        data = rng.integers(0, 255, size=(6, 7)).astype(np.uint8)
        seq = _diagonal_seq([-1.0, 1.5], [5.0, 0.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            degree=0,
            bound="nearest",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=0, bound="nearest", coeff=False
        )
    assert got.dtype == np.uint8
    assert ref.dtype == np.uint8
    assert np.array_equal(got, ref)


def test_class_a_only_pipeline_preserves_dtype() -> None:
    # A gather-only pipeline keeps the input dtype.
    system = _voxel_system(2)
    with backend("numpy"):
        rng = np.random.default_rng(9)
        data = rng.normal(size=(6, 7)).astype(np.float32)
        # The two flips are both gathers.
        seq = _diagonal_seq([-1.0, -1.0], [5.0, 6.0], system, system, (6, 7))
        got = sep.pull_separable(
            data,
            seq,
            degree=1,
            bound="mirror",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=1, bound="mirror", coeff=False
        )
    assert got.dtype == np.float32
    assert ref.dtype == np.float32
    assert np.allclose(got, ref)


def test_subspace_warp_without_axes_does_not_silently_pass_through() -> None:
    # A malformed interpolating subspace falls back to the monolithic pull.
    system = _voxel_system(3)
    with backend("numpy"):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(5, 5, 5))
        warp = rng.normal(size=(5, 5, 5, 3))
        inner = DisplacementField(field=warp, degree=1, bound="reflect")
        sub = SubspaceTransformation(transformation=inner)
        grid = CartesianField(shape=(5, 5, 5), input=system, output=system)
        seq = Sequence(transformations=[grid, sub])
        with pytest.raises(Exception) as sep_error:
            sep.pull_separable(
                data,
                seq,
                degree=1,
                bound="reflect",
                coeff=False,
            )
        with pytest.raises(Exception) as mono_error:
            pull(
                data,
                seq.compute().field,
                degree=1,
                bound="reflect",
                coeff=False,
            )
    assert type(sep_error.value) is type(mono_error.value)


# ----------------------------------------------------------------------
#   FALLBACK
# ----------------------------------------------------------------------


def test_cras_to_fras_bridge_factors_into_singletons() -> None:
    # The bridge relabels the axes but must still factor per axis.
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
        nf = transformation.compute(mode=kinds.Affine, factor=True)
        _, groups = sep._groups(nf, 3)
        # Each group holds one grid axis and one data axis, and the groups
        # cover every axis.
        assert len(groups) == 3
        for group in groups:
            assert len(group["G"]) == 1 and len(group["D"]) == 1

        got = sep.pull_separable(
            data,
            transformation,
            degree=1,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data,
            transformation.compute().field,
            degree=1,
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
                degree=1,
                bound="reflect",
                coeff=False,
            )
            ref = pull(
                data,
                transformation.compute().field,
                degree=1,
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
            field=warp, input=system, output=system, degree=3, bound="reflect"
        )
        grid = CartesianField(shape=(5, 5, 5), input=system, output=system)
        seq = Sequence(transformations=[grid, field])
        got = sep.pull_separable(
            data,
            seq,
            degree=3,
            bound="reflect",
            coeff=False,
        )
        ref = pull(
            data, seq.compute().field, degree=3, bound="reflect", coeff=False
        )
    # A single coupled group falls back to the monolithic pull exactly.
    assert np.array_equal(got, ref)


# ----------------------------------------------------------------------
#   ISSUE #11 DEMONSTRATION
# ----------------------------------------------------------------------


def _demonstration_image() -> tuple:
    xax, yax, zax = _sp("x"), _sp("y"), _sp("z")
    vox4 = CoordinateSystem(name="voxel4", axes=[xax, yax, zax, _time()])
    world4 = CoordinateSystem(
        name="world4", axes=[R(), A(), S(), TimeAxis(name="t")]
    )
    vox3 = CoordinateSystem(name="vox3", axes=[xax, yax, zax])
    rng = np.random.default_rng(0)
    warp = rng.normal(size=(6, 6, 6, 3)) * 0.7
    dfield = DisplacementField(
        field=warp, input=vox3, output=vox3, degree=3, bound="reflect"
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
    # Reslice onto fewer time points, rescaling the time axis.
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
            degree=1,
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
                degree=1,
                bound="reflect",
                coeff=False,
            )
    finally:
        backends.npndi.map_coordinates = original

    assert np.allclose(got, ref)
    # The coupled warp runs as a 3-D pull, never a 4-D one.
    ndims = {shape[0] for shape in shapes}
    assert 4 not in ndims
    assert 3 in ndims


def test_issue_11_flip_and_scale_do_not_interpolate_the_data() -> None:
    img, vox4, world4 = _demonstration_image()
    # There is no warp, only a diagonal affine with a y flip and fewer
    # time points.
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
            degree=1,
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
                degree=1,
                bound="reflect",
                coeff=False,
            )
    finally:
        backends.npndi.map_coordinates = original

    assert np.allclose(got, ref)
    # Only 1-D identity rows are resampled, to build the weight matrices.
    for shape in inputs:
        assert len(shape) == 1


# ----------------------------------------------------------------------
#   LARGE-AXIS ROUTING (C1)
# ----------------------------------------------------------------------


def test_large_one_dimensional_axis_routes_to_batched_pull() -> None:
    # The weight matrix of a long axis is huge, so above a threshold the
    # step falls back to a batched pull.
    system = _voxel_system(2)
    threshold = sep._MAX_WEIGHT_MATRIX_ELEMENTS

    def _build(n: int) -> tuple:
        data = np.arange(n * 3, dtype=float).reshape(n, 3)
        matrix = np.zeros((2, 3))
        # Axis 0 is genuinely rescaled and axis 1 flipped.
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
                data, seq, degree=3, bound="mirror", coeff=False
            )
            small_ref = pull(
                data,
                seq.compute().field,
                degree=3,
                bound="mirror",
                coeff=False,
            )
    finally:
        sep.pull_axes = original
    # A small axis uses the weight matrix.
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
                data, seq, degree=3, bound="mirror", coeff=False
            )
            large_ref = pull(
                data,
                seq.compute().field,
                degree=3,
                bound="mirror",
                coeff=False,
            )
    finally:
        sep.pull_axes = original
    # A large axis uses the batched pull.
    assert calls["pull_axes"] >= 1
    assert np.allclose(large, large_ref)


# ----------------------------------------------------------------------
#   SUBSPACE INNER DEPENDENCY (C3 + C4)
# ----------------------------------------------------------------------


_SUB_SYSTEM = CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z"), _time()])
_SUB_SYSTEM3 = CoordinateSystem(axes=[_sp("x"), _sp("y"), _sp("z")])
_SUB_SYSTEM2 = CoordinateSystem(axes=[_sp("x"), _sp("y")])


def _subspace(
    inner: object,
    axes: list,
    out_axes: object = None,
    sysin: object = _SUB_SYSTEM,
    sysout: object = None,
) -> SubspaceTransformation:
    out_axes = axes if out_axes is None else out_axes
    return SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray(axes),
        output_axes=np.asarray(out_axes),
        input=sysin,
        output=sysout,
    )


def _warp(
    shape_axes: tuple, seed: int, scale: float = 0.5
) -> DisplacementField:
    # Nonzero displacements on every axis expose a wrong axis mapping.
    k = len(shape_axes)
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(*shape_axes, k)) * scale
    return DisplacementField(field=field, degree=1, bound="reflect")


def _reslice_equal(
    sub: object,
    shape: tuple,
    seed: int,
    grid_system: object = None,
    degree: int = 3,
    bound: object = "reflect",
) -> None:
    # Compare with the monolithic pull on random data.
    with backend("numpy"):
        rng = np.random.default_rng(seed + 1000)
        data = rng.normal(size=shape)
        if grid_system is not None:
            grid = CartesianField(
                shape=shape, input=grid_system, output=grid_system
            )
        else:
            grid = CartesianField(shape=shape)
        seq = Sequence(transformations=[grid, sub])
        got = sep.pull_separable(
            data, seq, degree=degree, bound=bound, coeff=False
        )
        ref = pull(
            data, seq.compute().field, degree=degree, bound=bound, coeff=False
        )
    assert np.allclose(got, ref)


# A non-interpolating inner transform gives a finer partition.


# --- Non-interpolating inner: the finer partition is kept and reslices. ---


def test_subspace_wrapping_diagonal_scaling_splits_into_singletons() -> None:
    # A diagonal scaling mixes nothing, so every axis is its own group.
    inner = Scaling(
        scale=np.asarray([1.3, 0.7, 0.5]),
        input=_SUB_SYSTEM3,
        output=_SUB_SYSTEM3,
    )
    sub = _subspace(inner, [0, 1, 2], sysout=_SUB_SYSTEM)
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 1, grid_system=_SUB_SYSTEM)


def test_subspace_wrapping_diagonal_affine_splits_into_singletons() -> None:
    inner = Affine(
        matrix=np.diag([1.3, 0.7, 1.5, 1.0])[:-1],
        input=_SUB_SYSTEM3,
        output=_SUB_SYSTEM3,
    )
    sub = _subspace(inner, [0, 1, 2], sysout=_SUB_SYSTEM)
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 2, grid_system=_SUB_SYSTEM)


def test_nested_non_interpolating_subspace_splits_into_singletons() -> None:
    # A nested diagonal affine also gives singletons.
    innermost = Affine(
        matrix=np.diag([1.4, 0.6, 1.0])[:-1],
        input=_SUB_SYSTEM2,
        output=_SUB_SYSTEM2,
    )
    level1 = _subspace(
        innermost, [0, 1], sysin=_SUB_SYSTEM3, sysout=_SUB_SYSTEM3
    )
    sub = _subspace(level1, [0, 1, 2], sysout=_SUB_SYSTEM)
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 3, grid_system=_SUB_SYSTEM)


def test_subspace_shear_on_a_subblock_couples_only_the_sheared_axes() -> None:
    # The shear couples axes 0 and 1, and axis 2 passes through.
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.5
    inner = Affine(matrix=matrix, input=_SUB_SYSTEM3, output=_SUB_SYSTEM3)
    sub = _subspace(inner, [0, 1, 2], sysout=_SUB_SYSTEM)
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1), (0, 1)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 4, grid_system=_SUB_SYSTEM)


def test_subspace_rotation_on_a_subblock_couples_only_the_rotated_axes() -> (
    None
):
    # The z rotation couples axes 0 and 1, and axis 2 passes through.
    c, s = np.cos(0.4), np.sin(0.4)
    matrix = np.zeros((3, 4))
    matrix[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    inner = Affine(matrix=matrix, input=_SUB_SYSTEM3, output=_SUB_SYSTEM3)
    sub = _subspace(inner, [0, 1, 2], sysout=_SUB_SYSTEM)
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1), (0, 1)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 5, grid_system=_SUB_SYSTEM)


def test_reverting_subspace_recursion_to_all_ones_over_couples(
    monkeypatch: object,
) -> None:
    # Treating every subspace as one block over-couples but never splits.
    inner = Scaling(scale=np.asarray([2.0, 3.0, 0.5]))
    sub = _subspace(inner, [0, 1, 2])
    fine = _components([sub], (5, 6, 7, 4))
    assert fine == [((0,), (0,)), ((1,), (1,)), ((2,), (2,)), ((3,), (3,))]
    original = fac._read_pattern

    def all_ones(element: object, ni: int, no: int, *args: object) -> object:
        if element is inner:
            return np.ones((no, ni), dtype=bool)
        return original(element, ni, no, *args)

    monkeypatch.setattr(fac, "_read_pattern", all_ones)
    coarse = _components([sub], (5, 6, 7, 4))
    assert coarse == [((0, 1, 2), (0, 1, 2)), ((3,), (3,))]
    assert fine != coarse


# An interpolating inner transform keeps its axes in one block.


# --- Interpolating inner: the acted-on axes stay one coupled block. ---


def test_subspace_wrapping_raw_field_couples_all_its_axes() -> None:
    # A raw field mixes all its axes.
    warp = _warp((5, 6, 7), 6)
    sub = _subspace(warp, [0, 1, 2])
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 6)


def test_nested_subspace_over_interpolating_inner_couples_all_its_axes() -> (
    None
):
    # Recursing into the interpolating inner would misindex the axes;
    # the equality check fails if it does.
    warp = _warp((5,), 7)
    inner = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0]),
        output_axes=np.asarray([0]),
    )
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 7)


@pytest.mark.parametrize(
    "warp_axes, seed",
    [((1, 2), 11), ((0, 1), 12), ((0, 2), 13)],
)
def test_subspace_over_interpolating_subset_warp_reslices(
    warp_axes: tuple, seed: int
) -> None:
    # A warp on any two-axis subset stays one coupled block.
    shape = (5, 6, 7, 4)
    field_shape = tuple(shape[a] for a in warp_axes)
    warp = _warp(field_shape, seed)
    inner = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray(warp_axes),
        output_axes=np.asarray(warp_axes),
    )
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, shape, seed)


def test_permuting_subspace_over_interpolating_inner_reslices() -> None:
    # A subspace that permutes its axes stays one block.
    warp = _warp((5, 6, 7), 14)
    sub = _subspace(warp, [0, 1, 2], out_axes=[2, 0, 1])
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 14)


def test_subspace_over_sequence_with_interpolating_member_reslices() -> None:
    # A sequence interpolates when a member does.
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.4
    shear = Affine(matrix=matrix)
    warpz = _warp((7,), 15)
    subwarp = SubspaceTransformation(
        transformation=warpz,
        input_axes=np.asarray([2]),
        output_axes=np.asarray([2]),
    )
    inner = Sequence(transformations=[shear, subwarp])
    sub = _subspace(inner, [0, 1, 2])
    assert _components([sub], (5, 6, 7, 4)) == [
        ((0, 1, 2), (0, 1, 2)),
        ((3,), (3,)),
    ]
    _reslice_equal(sub, (5, 6, 7, 4), 15)


def test_subspace_wrapping_diagonal_reslice_matches_monolithic() -> None:
    # The finer partition reslices as the monolithic pull does.
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
            data, seq, degree=3, bound="reflect", coeff=False
        )
        ref = pull(
            data, seq.compute().field, degree=3, bound="reflect", coeff=False
        )
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   RESTRICTED TYPES RESLICE (C5)
# ----------------------------------------------------------------------


def test_scaling_translation_reslice_matches_monolithic() -> None:
    # Restricted pieces keep their cheaper types.
    with backend("numpy"):
        rng = np.random.default_rng(33)
        data = rng.normal(size=(6, 7))
        els = [
            Scaling(scale=np.asarray([1.3, 0.7])),
            Translation(translation=np.asarray([0.4, -0.6])),
        ]
        seq = Sequence(transformations=[CartesianField(shape=(6, 7)), *els])
        opt = dict(degree=3, bound="mirror", coeff=False)
        assert _plan(seq, data.shape, **opt) == {
            (0,): "matrix",
            (1,): "matrix",
        }
        got = sep.pull_separable(data, seq, **opt)
        ref = pull(data, seq.compute().field, **opt)
    assert np.allclose(got, ref)


def test_permutation_reslice_matches_monolithic() -> None:
    # A permutation survives the affine reduction as its own element.
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
            data, seq, degree=1, bound="reflect", coeff=False
        )
        ref = pull(
            data, seq.compute().field, degree=1, bound="reflect", coeff=False
        )
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   CHAINS THE MONOLITHIC READING DID NOT FACTOR
# ----------------------------------------------------------------------


def test_widened_intermediate_stage_reslices_with_a_two_dimensional_pull() -> (
    None
):
    # The middle stages have four axes, but x, y and t form two groups.
    embed = np.zeros((4, 4))
    embed[0, 0], embed[1, 1], embed[2, 2] = 1.3, 0.7, 1.0
    embed[2, 3], embed[3, 3] = 1.0, 0.5
    rng = np.random.default_rng(41)
    warp = DisplacementField(
        field=rng.normal(size=(4, 5, 2, 3)) * 0.3, degree=1, bound="reflect"
    )
    sub = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 3]),
        output_axes=np.asarray([0, 1, 3]),
    )
    grid = CartesianField(shape=(4, 5, 3))
    seq = Sequence(
        transformations=[
            grid,
            Affine(matrix=embed),
            sub,
            Affine(matrix=np.eye(3, 5)),
        ]
    )
    opt = dict(degree=1, bound="nearest", coeff=False)
    assert _plan(seq, (6, 5, 4), **opt) == {(0, 1): "pull", (2,): "gather"}
    with backend("numpy"):
        data = rng.normal(size=(6, 5, 4))
        got = sep.pull_separable(data, seq, **opt)
        ref = pull(data, seq.compute().field, **opt)
    assert np.allclose(got, ref)


def test_endpointless_subspace_beside_a_permutation_reslices() -> None:
    # The monolithic pull cannot build these coordinates, so compare by hand.
    grid = CartesianField(shape=(4, 5, 6))
    scale = SubspaceTransformation(
        transformation=Scaling(scale=np.asarray([0.7])),
        input_axes=np.asarray([1]),
        output_axes=np.asarray([1]),
    )
    perm = Permutation(permutation=np.asarray([2, 0, 1]))
    seq = Sequence(transformations=[grid, scale, perm])
    opt = dict(degree=3, bound="mirror", coeff=False)
    with backend("numpy"):
        rng = np.random.default_rng(43)
        data = rng.normal(size=(6, 4, 5))
        coords = np.asarray(grid.field, dtype=float)
        coords[..., 1] *= 0.7
        ref = pull(data, coords[..., [2, 0, 1]], **opt)
        got = sep.pull_separable(data, seq, **opt)
    assert np.allclose(got, ref)


# ----------------------------------------------------------------------
#   COPY
# ----------------------------------------------------------------------


def _own_grid_image(dtype: object = float) -> SingleScaleImage:
    data = np.arange(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
    src = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
        input=_voxel_system(3),
        output=_voxel_system(3),
    )
    return SingleScaleImage(data=data, transformations=[src])


def _cras_to_fras() -> tuple:
    # Map a cRAS image onto an fRAS grid by a flip and a swap, which
    # only need gathers.
    from brainhops.datamodel.systems import (
        CRASCoordinateSystem,
        FRASCoordinateSystem,
        RASCoordinateSystem,
    )

    data = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    src = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:-1],
        input=CRASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    # Output voxel (a, b, c) reads input voxel (3 - b, a, c).
    target = Affine(
        matrix=np.asarray(
            [[0.0, -2.0, 0.0, 6.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]
        ),
        input=FRASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    geometry = Geometry(
        (
            CartesianField(
                shape=(5, 4, 6),
                input=FRASCoordinateSystem(),
                output=FRASCoordinateSystem(),
            ),
            target,
        )
    )
    return SingleScaleImage(data=data, transformations=[src]), geometry


def _assert_fresh_copy(resliced: np.ndarray, source: np.ndarray) -> None:
    before = source.copy()
    assert not np.shares_memory(resliced, source)
    resliced[...] = -1
    assert np.array_equal(source, before)


def test_own_grid_reslice_without_copy_is_a_view() -> None:
    # A reslice onto its own grid only gathers and returns a view.
    with backend("numpy"):
        img = _own_grid_image()
        resliced = img.reslice(copy=False)
    assert np.shares_memory(resliced.data, img.data)
    assert np.array_equal(resliced.data, img.data)


def test_own_grid_reslice_with_copy_is_fresh() -> None:
    with backend("numpy"):
        img = _own_grid_image()
        resliced = img.reslice(copy=True)
    assert np.array_equal(resliced.data, img.data)
    _assert_fresh_copy(resliced.data, img.data)


def test_gather_only_reslice_without_copy_is_a_view() -> None:
    with backend("numpy"):
        img, geometry = _cras_to_fras()
        resliced = img.reslice(geometry, copy=False)
        ref = img.reslice(geometry, copy=True)
    assert np.shares_memory(resliced.data, img.data)
    assert np.array_equal(resliced.data, ref.data)


def test_gather_only_reslice_with_copy_is_fresh() -> None:
    with backend("numpy"):
        img, geometry = _cras_to_fras()
        resliced = img.reslice(geometry, copy=True)
    # The flip and the swap change the result.
    expected = img.data[::-1].transpose(1, 0, 2)
    assert np.array_equal(resliced.data, expected)
    _assert_fresh_copy(resliced.data, img.data)


def test_interpolating_reslice_with_copy_is_correct_and_not_recopied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pipeline already allocates, so copy=True must not copy again.
    def _no_copy(arr: object) -> object:
        raise AssertionError("an already fresh result was copied")

    monkeypatch.setattr(sep, "copy_array", _no_copy)
    system = _voxel_system(3)
    matrix = np.eye(4)[:-1]
    matrix[:, -1] = 0.5
    grid = CartesianField(shape=(3, 4, 5), input=system, output=system)
    affine = Affine(matrix=matrix, input=system, output=system)
    seq = Sequence(transformations=[grid, affine])
    opt = dict(degree=1, bound="reflect", coeff=False)
    with backend("numpy"):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        got = sep.pull_separable(data, seq, copy=True, **opt)
        ref = pull(data, seq.compute().field, **opt)
    assert np.allclose(got, ref)
    _assert_fresh_copy(got, data)


def test_reslice_through_a_split_sequence_inner_is_correct() -> None:
    # Regression: an inner sequence split across groups was lost.
    inner = Sequence(
        [
            Scaling(scale=np.array([0.5, 0.75])),
            Translation(translation=np.array([1.0, -0.5])),
        ]
    )
    sub = SubspaceTransformation(
        transformation=inner,
        input_axes=np.array([0, 1]),
        output_axes=np.array([0, 1]),
    )
    seq = Sequence([CartesianField(shape=(4, 5, 6)), sub])
    opt = dict(degree=1, bound="reflect", coeff=False)
    with backend("numpy"):
        data = np.random.default_rng(0).normal(size=(4, 5, 6))
        got = sep.pull_separable(data, seq, **opt)
        ref = pull(data, seq.compute().field, **opt)
    assert np.allclose(got, ref)


def test_backend_reports_memory_sharing() -> None:
    data = np.arange(6.0)
    assert backends.may_share_memory(data, data[::-1]) is True
    assert backends.may_share_memory(data, data.copy()) is False
    copied = backends.copy_array(data[::-1])
    assert not np.shares_memory(copied, data)
    # An array-like of unknown backend cannot be inspected.
    assert backends.may_share_memory(data, [0.0, 1.0]) is None
    da = pytest.importorskip("dask.array")
    lazy = da.from_array(data, chunks=2)
    # Dask arrays are immutable, so distinct ones never alias.
    assert backends.may_share_memory(lazy[::-1], lazy) is False
    assert backends.may_share_memory(lazy[::-1], data) is False
    assert backends.may_share_memory(lazy, lazy) is True


def test_copy_on_the_dask_backend_is_lazy_and_not_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A lazy dask result is returned as is under copy=True.
    da = pytest.importorskip("dask.array")

    def _no_copy(arr: object) -> object:
        raise AssertionError("a dask result was copied")

    monkeypatch.setattr(sep, "copy_array", _no_copy)
    with backend("dask"):
        img = _own_grid_image()
        source = np.asarray(img.data).copy()
        lazy = da.from_array(source, chunks=2)
        img = SingleScaleImage(data=lazy, transformations=img.transformations)
        resliced = img.reslice(copy=True).data
    assert isinstance(resliced, da.Array)
    assert resliced is not lazy
    assert np.array_equal(resliced.compute(), source)


# ----------------------------------------------------------------------
#   A PRODUCT OF SUBSPACES (A SPACE-AND-TIME GEOMETRY)
# ----------------------------------------------------------------------


def _space_and_time(angle: float, dz: float, dt: float) -> Sequence:
    # Build an affine over x, y and z, followed by a scaling and a
    # translation over t.
    full = CoordinateSystem().expand(4)
    c, s = np.cos(angle), np.sin(angle)
    spatial = Affine(
        matrix=np.array([[c, -s, 0, 1.0], [s, c, 0, -1.0], [0, 0, dz, 0]])
    )
    temporal = Sequence(
        transformations=[Scaling(scale=[dt]), Translation(translation=[0.5])]
    )
    return Sequence(
        transformations=[
            SubspaceTransformation(
                transformation=spatial,
                input_axes=[0, 1, 2],
                output_axes=[0, 1, 2],
                input=full,
                output=full,
            ),
            SubspaceTransformation(
                transformation=temporal,
                input_axes=[3],
                output_axes=[3],
                input=full,
                output=full,
            ),
        ]
    )


def test_a_subspace_product_factors_into_space_and_time() -> None:
    # The rotation couples x and y, z is rescaled and t shifted by frames.
    shape = (6, 7, 5, 4)
    source = _space_and_time(0.3, 2.0, 2.0)
    target = _space_and_time(0.0, 1.0, 2.0)
    seq = source.inverse() @ target @ CartesianField(shape=shape)
    opt = dict(degree=1, bound="reflect", coeff=False)
    assert _plan(seq, shape, **opt) == {
        (0, 1): "pull",
        (2,): "matrix",
        (3,): "gather",
    }


def test_a_subspace_product_against_its_inverse_is_the_grid() -> None:
    # The same geometry on both sides cancels, leaving the grid.
    shape = (6, 7, 5, 4)
    product = _space_and_time(0.3, 2.0, 2.0)
    grid = CartesianField(shape=shape)
    seq = product.inverse() @ product @ grid
    nf = seq.compute(mode=kinds.Affine, factor=True)
    assert isinstance(nf, CartesianField) and nf.shape == shape
    data = np.random.default_rng(0).random(shape)
    out = sep.pull_separable(data, seq, degree=1, bound="reflect", coeff=False)
    assert np.shares_memory(out, data)
    assert np.array_equal(out, data)
