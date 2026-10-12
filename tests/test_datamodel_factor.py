"""Tests of the factoring pass, `compute(factor=True)`.

The pass rewrites a chain that preserves the number of dimensions into an
optional grid, one subspace factor per group of coupled axes, and an
optional trailing permutation.
"""

import numpy as np
import pytest

import brainhops._ext.invfield as invfield
from brainhops.datamodel._transformations import sequence as seqmod
from brainhops.datamodel._transformations.compute import factor as fac
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)

# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _sub(inner: Transformation, axes: list) -> SubspaceTransformation:
    idx = np.asarray(axes, dtype=int)
    return SubspaceTransformation(
        transformation=inner, input_axes=idx, output_axes=idx
    )


def _field(x: Transformation) -> np.ndarray:
    """Return the coordinate field computed by a transform."""
    return np.asarray(x.compute().field)


def _engine_factored(seq: Sequence) -> Transformation:
    """Factor as the compute loop does, composing inside each group."""
    return fac.factor_sequence(seq, compute=True, simplify="analytic")


def _factors(result: Transformation) -> list:
    return [
        t
        for t in (result.transformations or [])
        if isinstance(t, SubspaceTransformation)
    ]


def _factor_axes(result: Transformation) -> list:
    return sorted(
        tuple(int(a) for a in t.input_axes) for t in _factors(result)
    )


# ----------------------------------------------------------------------
#   NORMAL FORM STRUCTURE
# ----------------------------------------------------------------------


def test_diagonal_chain_splits_into_per_axis_factors() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Translation(translation=np.array([1.0, 0.0, -1.0])),
        ]
    )
    nf = seq.compute(factor=True)
    assert isinstance(nf.transformations[0], CartesianField)
    assert _factor_axes(nf) == [(0,), (1,), (2,)]


def test_reorder_and_merge_into_two_factors() -> None:
    grid = CartesianField(shape=(4, 5))
    seq = Sequence(
        [
            grid,
            _sub(Scaling(scale=np.array([2.0])), [0]),
            _sub(Scaling(scale=np.array([5.0])), [1]),
            _sub(Translation(translation=np.array([3.0])), [0]),
        ]
    )
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0,), (1,)]


def test_pure_permutation_reduces_to_grid_and_permutation() -> None:
    grid = CartesianField(shape=(3, 4))
    seq = Sequence([grid, Permutation(permutation=np.array([1, 0]))])
    nf = seq.compute(factor=True)
    kinds = [type(t).__name__ for t in nf.transformations]
    assert kinds == ["CartesianField", "Permutation"]
    assert _factors(nf) == []


def test_single_group_returned_unchanged_same_objects() -> None:
    # A fully coupled affine forms a single group, so the sequence is returned
    # unchanged, with the same objects.
    grid = CartesianField(shape=(4, 5, 6))
    dense = Affine(
        matrix=np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 0.0],
                [7.0, 8.0, 10.0, 0.0],
            ]
        )
    )
    seq = Sequence([grid, dense])
    nf = seq.compute(factor=True)
    assert list(nf.transformations) == list(seq.transformations)
    assert nf.transformations[1] is dense


# ----------------------------------------------------------------------
#   EQUIVALENCE (factor o unfactor == compute)
# ----------------------------------------------------------------------


def test_factor_unfactor_equivalence_diagonal() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Translation(translation=np.array([1.0, 0.0, -1.0])),
        ]
    )
    nf = seq.compute(factor=True)
    recomposed = _field(Sequence(list(nf.transformations)))
    assert np.allclose(recomposed, _field(seq))


def test_factor_unfactor_equivalence_block_and_diagonal() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    block = Affine(
        matrix=np.array(
            [
                [1.0, 0.5, 0.0, 2.0],
                [0.3, 1.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 1.0],
            ]
        )
    )
    seq = Sequence([grid, block])
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1), (2,)]
    assert np.allclose(_field(Sequence(list(nf.transformations))), _field(seq))


def test_pass_through_axis_inside_a_group() -> None:
    # Axis 2 passes through the first subspace but is coupled to axis 1 by the
    # second, so the first piece must stay a subspace.
    grid = CartesianField(shape=(3, 4, 5, 6))
    a = _sub(
        Linear(matrix=np.array([[1.0, 0.5], [0.2, 1.0]])),
        [0, 1],
    )
    b = _sub(
        Linear(matrix=np.array([[1.0, 0.3], [0.4, 1.0]])),
        [1, 2],
    )
    c = _sub(Scaling(scale=np.array([2.0])), [3])
    seq = Sequence([grid, a, b, c])
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1, 2), (3,)]
    assert np.allclose(_field(Sequence(list(nf.transformations))), _field(seq))


def test_subspace_piece_meeting_an_affine_piece_in_a_group() -> None:
    # The subspace piece meets the affine piece within one group.
    grid = CartesianField(shape=(3, 4, 5, 2))
    lin = np.array([[1.0, 0.5], [0.1, 1.0]])
    couple = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.3, 0.0, 0.0],
            [0.0, 0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0],
        ]
    )
    seq = Sequence(
        [grid, _sub(Linear(matrix=lin), [0, 1]), Affine(matrix=couple)]
    )
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1, 2), (3,)]
    x = np.array(grid.field, dtype=float)
    x[..., :2] = x[..., :2] @ lin.T
    truth = x @ couple[:, :-1].T + couple[:, -1]
    assert np.allclose(_field(Sequence(list(nf.transformations))), truth)


def test_reordered_subspace_axes_recompose() -> None:
    # A subspace that names its axes out of order acts in that order.
    grid = CartesianField(shape=(3, 4, 5))
    swirl = SubspaceTransformation(
        transformation=Affine(
            matrix=np.array([[2.0, 0.5, 1.0], [0.0, 3.0, -1.0]])
        ),
        input_axes=np.array([1, 0]),
        output_axes=np.array([1, 0]),
    )
    seq = Sequence([grid, swirl, _sub(Scaling(scale=np.array([4.0])), [2])])
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1), (2,)]
    assert np.allclose(_field(Sequence(list(nf.transformations))), _field(seq))


def test_trailing_permutation_scatters_work_axes() -> None:
    # The trailing permutation scatters the grid axes to the data axes. It is
    # applied by hand, since the default `compute()` cannot embed the subspace
    # next to it.
    grid = CartesianField(shape=(3, 4, 5, 2))
    block = Affine(
        matrix=np.array(
            [
                [1.0, 0.4, 0.0, 0.0, 1.0],
                [0.2, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.5, 0.0],
            ]
        )
    )
    seq = Sequence(
        [grid, Permutation(permutation=np.array([2, 3, 0, 1])), block]
    )
    nf = seq.compute(factor=True)
    *body, perm = nf.transformations
    assert isinstance(perm, Permutation)
    assert _factor_axes(nf) == [(0,), (1,), (2, 3)]
    scattered = _field(Sequence(body))[..., np.asarray(perm.permutation)]
    assert np.allclose(scattered, _field(seq))


def test_permutation_recomposes_to_correct_coordinates() -> None:
    # The ground truth applies each permutation to the grid directly.
    grid = CartesianField(shape=(3, 4, 5))
    p1 = Permutation(permutation=np.array([1, 0, 2]))
    p2 = Permutation(permutation=np.array([0, 2, 1]))
    seq = Sequence([grid, p1, p2, p1, p2])
    nf = seq.compute(factor=True)
    ground = np.asarray(grid.field)
    for p in (p1, p2, p1, p2):
        ground = ground[..., np.asarray(p.permutation)]
    assert np.allclose(_field(Sequence(list(nf.transformations))), ground)


# ----------------------------------------------------------------------
#   IDEMPOTENCE / IDENTITY PRESERVATION
# ----------------------------------------------------------------------


def test_normal_form_is_idempotent_same_objects() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Translation(translation=np.array([1.0, 2.0, 3.0])),
        ]
    )
    nf = seq.compute(factor=True)
    again = Sequence(list(nf.transformations)).compute(factor=True)
    assert len(again.transformations) == len(nf.transformations)
    assert all(
        a is b for a, b in zip(again.transformations, nf.transformations)
    )


def test_reversed_order_gives_the_same_partition() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    a = _sub(Scaling(scale=np.array([2.0, 3.0])), [0, 1])
    b = _sub(Scaling(scale=np.array([5.0])), [2])
    forward = Sequence([grid, a, b]).compute(factor=True)
    reverse = Sequence([grid, b, a]).compute(factor=True)
    assert _factor_axes(forward) == _factor_axes(reverse)


# ----------------------------------------------------------------------
#   RESTRICTION KEEPS THE CHEAPER TYPE
# ----------------------------------------------------------------------


def _pieces(els: list, ndim: int) -> list:
    """Return the restricted pieces of every group, before composition."""
    stages = fac._build_stages(els, ndim, fac.PatternCache())
    groups = fac._partition(stages, ndim)
    return [fac._restrict_group(group, stages) for group in groups]


def test_restrict_keeps_scaling_and_translation_types() -> None:
    # A restricted scaling stays a scaling, and a restricted translation stays
    # a translation.
    els = [
        Scaling(scale=np.asarray([2.0, 3.0])),
        Translation(translation=np.asarray([1.0, -2.0])),
    ]
    pieces = _pieces(els, 2)
    assert len(pieces) == 2
    for sub in pieces:
        assert [type(t).__name__ for t in sub] == ["Scaling", "Translation"]


def test_restrict_keeps_permutation_type() -> None:
    els = [Permutation(permutation=np.asarray([1, 0, 2]))]
    pieces = _pieces(els, 3)
    assert len(pieces) == 3
    for sub in pieces:
        assert [type(t).__name__ for t in sub] == ["Permutation"]


def test_restrict_drops_matrix_less_affine() -> None:
    # An affine without a matrix is the identity and is dropped from every
    # group.
    matrix = np.zeros((2, 3))
    matrix[0, 0], matrix[1, 1] = 2.0, 3.0
    els = [Affine(matrix=matrix), Affine(matrix=None)]
    pieces = _pieces(els, 2)
    assert len(pieces) == 2
    for sub in pieces:
        assert [type(t).__name__ for t in sub] == ["Affine"]


# ----------------------------------------------------------------------
#   A WIDER INTERMEDIATE STAGE
# ----------------------------------------------------------------------


def _widened_chain() -> Sequence:
    # A 3-D chain (x, y, t) that embeds a constant z, warps (x, y, z) together
    # and projects z away.
    embed = np.zeros((4, 4))
    embed[0, 0], embed[1, 1], embed[2, 2] = 1.3, 0.7, 1.0
    embed[2, 3], embed[3, 3] = 1.0, 0.5
    rng = np.random.default_rng(0)
    warp = DisplacementField(
        field=rng.normal(size=(4, 5, 2, 3)) * 0.3, degree=1, bound="reflect"
    )
    sub = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 3]),
        output_axes=np.asarray([0, 1, 3]),
    )
    project = np.eye(3, 5)
    return Sequence(
        [
            CartesianField(shape=(4, 5, 3)),
            Affine(matrix=embed),
            sub,
            Affine(matrix=project),
        ]
    )


def test_widened_intermediate_stage_factors() -> None:
    # The warp couples x and y through the embedded z, and t passes through.
    seq = _widened_chain()
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1), (2,)]
    assert np.allclose(_field(Sequence(list(nf.transformations))), _field(seq))


def test_intermediate_axis_that_reaches_no_end_is_left_unfactored() -> None:
    # An axis created from a constant and then dropped reaches no end, so no
    # per-axis step can apply it.
    embed = np.zeros((3, 3))
    embed[0, 0], embed[1, 1], embed[2, 2] = 2.0, 3.0, 1.0
    seq = Sequence(
        [
            CartesianField(shape=(4, 5)),
            Affine(matrix=embed),
            Affine(matrix=np.eye(2, 4)),
        ]
    )
    nf = seq.compute(mode="translation", factor=True)
    assert list(nf.transformations) == list(seq.transformations)


# ----------------------------------------------------------------------
#   CANCELLATION WITHOUT MATERIALIZING A FIELD
# ----------------------------------------------------------------------


def test_subspace_inverse_pair_cancels_zero_inversions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}
    real = invfield.inverse

    def spy(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(invfield, "inverse", spy)

    warp = DisplacementField(field=np.random.randn(4, 5, 6, 3) * 0.05)
    sub_warp = _sub(warp, [0, 1, 2])
    sub_warp_inv = sub_warp.inverse()
    scale = _sub(Scaling(scale=np.array([2.0])), [3])
    seq = Sequence(
        [CartesianField(shape=(4, 5, 6, 7)), sub_warp, scale, sub_warp_inv]
    )
    nf = seq.compute(factor=True)
    # The warp group cancels and is dropped, without inverting the field.
    assert _factor_axes(nf) == [(3,)]
    assert calls["n"] == 0


def test_subspace_cancel_is_a_cost_free_pair_simplifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cancelling a subspace with its inverse is a pair simplifier, consulted
    # before any composer.
    from brainhops.datamodel._transformations.compute.compose import compose
    from brainhops.datamodel._transformations.compute.simplify import (
        get_simplifiers,
        simplify,
    )

    calls = {"n": 0}
    real = invfield.inverse

    def spy(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(invfield, "inverse", spy)

    warp = DisplacementField(field=np.random.randn(4, 5, 3) * 0.05)
    sub = _sub(warp, [0, 1])
    inv = sub.inverse()
    assert get_simplifiers(SubspaceTransformation, SubspaceTransformation)
    # In application order: `sub`, then `inv`.
    assert isinstance(simplify(sub, inv), Identity)
    # In matrix order, `compose(inv, sub)` applies `sub` first.
    assert isinstance(compose(inv, sub), Identity)
    assert calls["n"] == 0
    # A pair that does not cancel is declined, without an error.
    other = _sub(Scaling(scale=np.array([2.0, 3.0])), [0, 1])
    assert simplify(sub, other) is None


# ----------------------------------------------------------------------
#   SIMPLIFY x FACTOR
# ----------------------------------------------------------------------


def test_numeric_downcasts_diagonal_block_to_scaling() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    diag = Affine(
        matrix=np.array(
            [
                [2.0, 0.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ]
        )
    )
    nf = Sequence([grid, diag]).compute(simplify="numeric", factor=True)
    inners = {type(t.transformation).__name__ for t in _factors(nf)}
    # Axes 1 and 2 are downcast to scalings; axis 0 keeps its translation.
    assert "Scaling" in inners
    assert np.allclose(
        _field(Sequence(list(nf.transformations))),
        _field(Sequence([grid, diag])),
    )


# ----------------------------------------------------------------------
#   MODE x FACTOR
# ----------------------------------------------------------------------


def test_single_axis_chain_trivial_under_mode() -> None:
    grid = CartesianField(shape=(6,))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0])),
            Translation(translation=np.array([1.0])),
        ]
    )
    # A single-axis chain is trivial and returned unchanged in any mode.
    nf = seq.compute(mode="translation", factor=True)
    assert list(nf.transformations) == list(seq.transformations)


def test_field_and_affine_groups_stay_separate() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    warp = DisplacementField(field=np.random.randn(4, 5, 2) * 0.05)
    seq = Sequence(
        [
            grid,
            _sub(warp, [0, 1]),
            _sub(Scaling(scale=np.array([2.0])), [2]),
            _sub(Translation(translation=np.array([1.0])), [2]),
        ]
    )
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0, 1), (2,)]
    field_group = [t for t in _factors(nf) if list(t.input_axes) == [0, 1]]
    assert len(field_group) == 1


# ----------------------------------------------------------------------
#   TERMINATION
# ----------------------------------------------------------------------


def test_adversarial_shear_pairs_converge() -> None:
    m = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shear = Linear(matrix=m)
    unshear = Linear(matrix=np.linalg.inv(m))
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence([grid, shear, unshear, shear, unshear])
    result = seq.compute(factor=True)
    assert np.allclose(_field(result), np.asarray(grid.field))


def test_cap_raises_on_non_identity_preserving_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pass that never reaches a fixed point must hit the iteration cap and
    # raise.
    from bagof.magic import replace

    def broken(
        seq: object,
        compute: object = False,
        simplify: object = None,
        cache: object = None,
    ) -> object:
        leaves = list(seq.transformations or [])
        if not leaves:
            return seq
        fresh = [
            replace(t, output=getattr(t, "_output", None)) for t in leaves
        ]
        return replace(seq, transformations=fresh)

    monkeypatch.setattr(seqmod, "factor_sequence", broken)
    grid = CartesianField(shape=(4, 5))
    seq = Sequence([grid, Scaling(scale=np.array([2.0, 3.0]))])
    with pytest.raises(RuntimeError, match="did not converge"):
        seq.compute(factor=True)


# ----------------------------------------------------------------------
#   REGRESSIONS
# ----------------------------------------------------------------------


def _sequence_inner_chain() -> Sequence:
    # A subspace whose inner transform is a sequence is split across two
    # groups.
    inner = Sequence(
        [
            Scaling(scale=np.array([0.5, 0.75])),
            Translation(translation=np.array([1.0, -0.5])),
        ]
    )
    return Sequence([CartesianField(shape=(4, 5, 6)), _sub(inner, [0, 1])])


def test_sequence_inner_split_across_groups_is_kept() -> None:
    # Regression: the pieces were dropped, leaving only the grid.
    seq = _sequence_inner_chain()
    result = seq.compute(factor=True)
    assert _factor_axes(result) == [(0,), (1,)]
    assert np.allclose(_field(result), _field(_sequence_inner_chain()))


def test_unrestrictable_element_leaves_the_chain_unfactored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from brainhops.errors import RestrictionError

    def refuse(*args: object) -> None:
        raise RestrictionError("cannot")

    monkeypatch.setattr(fac, "restrict", refuse)
    seq = Sequence(
        [CartesianField(shape=(4, 5)), Scaling(scale=np.array([2.0, 3.0]))]
    )
    assert _engine_factored(seq) is seq


def test_subspace_without_inner_is_read_as_the_identity() -> None:
    # Regression: this used to crash the dependency reader.
    grid = CartesianField(shape=(4, 5, 6))
    scale = Scaling(scale=np.array([0.5, 0.75, 0.9]))
    seq = Sequence([grid, _sub(None, [1]), scale])
    result = _engine_factored(seq)
    assert _factor_axes(result) == [(0,), (1,), (2,)]
    assert np.allclose(_field(result), _field(Sequence([grid, scale])))


# A subspace without an inner transform over different axes is a reindex
# (#110). Every element carries systems, so that the default `compute()`
# can embed the subspaces.
_XYZ = CoordinateSystem(
    name="voxel", axes=[SpaceAxis(name=n, unit="index") for n in "xyz"]
)
_SHAPE = (4, 5, 6)


def _reindex(in_axes: list, out_axes: list) -> SubspaceTransformation:
    return SubspaceTransformation(
        transformation=None,
        input_axes=np.asarray(in_axes, dtype=int),
        output_axes=np.asarray(out_axes, dtype=int),
        input=_XYZ,
        output=_XYZ,
    )


def _grid_xyz() -> CartesianField:
    return CartesianField(shape=_SHAPE, input=_XYZ, output=_XYZ)


def _coords() -> np.ndarray:
    axes = [np.arange(n, dtype=float) for n in _SHAPE]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _normal_form_field(nf: Sequence) -> np.ndarray:
    *body, last = nf.transformations
    if not isinstance(last, Permutation):
        return _field(nf)
    return _field(Sequence(body))[..., np.asarray(last.permutation)]


def test_subspace_without_inner_over_different_axes_is_a_reindex() -> None:
    # Regression: the chain was left unfactored when the reindex was read as
    # the identity.
    scale = SubspaceTransformation(
        transformation=Scaling(scale=np.array([1.5])),
        input_axes=np.array([2]),
        output_axes=np.array([2]),
        input=_XYZ,
        output=_XYZ,
    )
    seq = Sequence([_grid_xyz(), _reindex([0, 1], [1, 0]), scale])
    assert _engine_factored(seq) is not seq
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(2,)]
    assert list(nf.transformations[-1].permutation) == [1, 0, 2]
    ground = _coords()[..., [1, 0, 2]] * [1.0, 1.0, 1.5]
    assert np.allclose(_normal_form_field(nf), ground)
    assert np.allclose(_normal_form_field(nf), _field(seq))


@pytest.mark.parametrize("swap_first", [True, False])
def test_reindex_swap_next_to_a_scaling(swap_first: bool) -> None:
    # A swap of axes 0 and 1 gives three single-axis groups and a trailing
    # permutation.
    s = np.array([2.0, 3.0, 4.0])
    scaling = Scaling(scale=s, input=_XYZ, output=_XYZ)
    swap = _reindex([0, 1], [1, 0])
    els = [swap, scaling] if swap_first else [scaling, swap]
    seq = Sequence([_grid_xyz(), *els])
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0,), (1,), (2,)]
    assert isinstance(nf.transformations[-1], Permutation)
    assert list(nf.transformations[-1].permutation) == [1, 0, 2]
    x = _coords()
    ground = x[..., [1, 0, 2]] * s if swap_first else (x * s)[..., [1, 0, 2]]
    assert np.allclose(_normal_form_field(nf), ground)
    assert np.allclose(_normal_form_field(nf), _field(seq))


def test_reindex_split_across_groups() -> None:
    # The reindex swaps axes 0 and 2, and the affine couples axes 0 and 1.
    matrix = np.array(
        [[1.0, 0.5, 0.0, 0.0], [0.3, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]]
    )
    affine = Affine(matrix=matrix, input=_XYZ, output=_XYZ)
    seq = Sequence([_grid_xyz(), _reindex([0, 2], [2, 0]), affine])
    nf = seq.compute(factor=True)
    assert _factor_axes(nf) == [(0,), (1, 2)]
    ground = _coords()[..., [2, 1, 0]] @ matrix[:, :3].T + matrix[:, 3]
    assert np.allclose(_normal_form_field(nf), ground)
    assert np.allclose(_normal_form_field(nf), _field(seq))


def _systemless_subspace_then_linear() -> list:
    # The chain is a subspace without systems, which states no axis count,
    # followed by an affine.
    linear = np.array(
        [
            [0.8, 0.4, 0.0, 0.3],
            [0.1, 0.9, 0.0, -0.6],
            [0.0, 0.0, 1.3, 0.0],
            [-0.15, -0.15, 0.0, 0.7],
        ]
    )
    inner = Sequence(
        [
            Scaling(scale=np.array([1.7, 0.75, 0.5])),
            Translation(translation=np.array([-2.0, 0.0, 1.0])),
        ]
    )
    return [
        SubspaceTransformation(
            transformation=inner,
            input_axes=np.array([0, 2, 3]),
            output_axes=np.array([0, 2, 3]),
        ),
        Linear(matrix=linear),
    ]


@pytest.mark.parametrize("form", ["reindex", "permutation"])
def test_group_of_systemless_subspace_and_affine_pieces_is_factored(
    form: str,
) -> None:
    # Regression: factoring raised here although the default `compute()`
    # succeeds. Both forms of the head are the permutation [0, 3, 1, 2].
    if form == "reindex":
        head = SubspaceTransformation(
            transformation=None,
            input_axes=np.array([1, 2, 3]),
            output_axes=np.array([2, 3, 1]),
        )
    else:
        head = Permutation(permutation=np.array([0, 3, 1, 2]))
    grid = CartesianField(shape=(3, 3, 5, 4))
    seq = Sequence([grid, head, *_systemless_subspace_then_linear()])
    assert _engine_factored(seq) is not seq
    assert np.allclose(_field(seq.compute(factor=True)), _field(seq))


# ----------------------------------------------------------------------
#   FALLBACKS (unfactored)
# ----------------------------------------------------------------------


def test_uncomposable_affine_then_subspace_is_left_unfactored() -> None:
    # The mirrored order cannot be composed and is left unfactored.
    forward = Affine(
        matrix=np.array(
            [
                [1.1, 0.0, 0.1, 0.2],
                [-0.1, 1.2, -0.5, -0.6],
                [0.1, 0.2, 0.6, -0.8],
            ]
        )
    )
    nested = SubspaceTransformation(
        transformation=Scaling(scale=np.array([1.8])),
        input_axes=np.array([0]),
        output_axes=np.array([0]),
    )
    seq = Sequence(
        [
            CartesianField(shape=(4, 3, 5, 4)),
            SubspaceTransformation(
                transformation=None,
                input_axes=np.array([1, 3]),
                output_axes=np.array([3, 1]),
            ),
            SubspaceTransformation(
                transformation=forward.inverse(),
                input_axes=np.array([1, 2, 3]),
                output_axes=np.array([1, 2, 3]),
            ),
            SubspaceTransformation(
                transformation=nested,
                input_axes=np.array([1, 3]),
                output_axes=np.array([1, 3]),
            ),
            _sub(Scaling(scale=np.array([1.5, 1.8, 1.65, 1.7])), [0, 1, 2, 3]),
        ]
    )
    assert _engine_factored(seq) is seq
    assert np.allclose(_field(seq.compute(factor=True)), _field(seq))


def test_rectangular_chain_left_unfactored() -> None:
    # A tall affine creates an axis, so the chain is left unfactored.
    grid = CartesianField(shape=(4, 5))
    tall = Affine(
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    )
    seq = Sequence([grid, tall])
    nf = seq.compute(factor=True)
    assert list(nf.transformations) == list(seq.transformations)


def test_projection_chain_left_unfactored() -> None:
    # A projection has no affine reading, so it is kept rather than dropped.
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Projection(dropped=[2]),
        ]
    )
    nf = seq.compute(factor=True)
    assert any(isinstance(t, Projection) for t in nf.transformations)


def test_default_compute_unchanged_by_factor_flag() -> None:
    grid = CartesianField(shape=(4, 5, 6))
    seq = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Translation(translation=np.array([1.0, 0.0, -1.0])),
        ]
    )
    plain = seq.compute()
    plain_again = seq.compute(factor=False)
    assert np.allclose(np.asarray(plain.field), np.asarray(plain_again.field))


# ----------------------------------------------------------------------
#   LEAF DELEGATION
# ----------------------------------------------------------------------


def test_leaf_factor_delegates_to_sequence() -> None:
    diag = Affine(
        matrix=np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ]
        )
    )
    grid = CartesianField(shape=(4, 5, 6))
    nf = Sequence([grid, diag]).compute(factor=True)
    assert _factor_axes(nf) == [(0,), (1,), (2,)]


def _diagonal() -> Affine:
    return Affine(
        matrix=np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
            ]
        )
    )


def test_leaf_factor_method_splits_a_diagonal_affine() -> None:
    # A leaf factors as a chain of one element, with one factor per axis.
    leaf = _diagonal()
    nf = leaf.factor()
    assert _factor_axes(nf) == [(0,), (1,), (2,)]
    grid = CartesianField(shape=(4, 5, 6))
    assert np.allclose(
        _field(Sequence([grid, nf])), _field(Sequence([grid, leaf]))
    )


def test_leaf_factor_method_returns_self_when_it_does_not_factor() -> None:
    # A shear couples both axes into one group, so there is nothing to
    # split and the chain of one is dropped rather than handed back.
    shear = Affine(
        matrix=np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert shear.factor() is shear


# ----------------------------------------------------------------------
#   Sequence.factor()
# ----------------------------------------------------------------------


def _scaled_and_swapped() -> Sequence:
    return Sequence(
        [
            CartesianField(shape=(4, 5, 6)),
            Scaling(scale=np.array([2.0, 3.0, 1.0])),
            Permutation(permutation=np.array([1, 0, 2])),
        ]
    )


def test_factor_method_is_structural_by_default() -> None:
    # Nothing is composed, so each factor holds the sub-chain of its group.
    seq = _scaled_and_swapped()
    nf = seq.factor()
    assert _factor_axes(nf) == [(0,), (1,), (2,)]
    assert all(isinstance(f.transformation, Sequence) for f in _factors(nf))
    assert isinstance(nf.transformations[0], CartesianField)
    assert isinstance(nf.transformations[-1], Permutation)
    assert np.allclose(_field(nf), _field(seq))


def test_factor_method_with_compute_folds_each_group_into_one() -> None:
    seq = _scaled_and_swapped()
    nf = seq.factor(compute=True)
    # The unit scale on the last axis leaves an identity, whose factor is
    # dropped once the group is composed.
    assert _factor_axes(nf) == [(0,), (1,)]
    assert not any(
        isinstance(f.transformation, Sequence) for f in _factors(nf)
    )
    assert np.allclose(_field(nf), _field(seq))


@pytest.mark.parametrize("compute", [False, True])
def test_factor_method_is_one_idempotent_pass(compute: bool) -> None:
    nf = _scaled_and_swapped().factor(compute=compute)
    assert nf.factor(compute=compute) is nf


def test_factor_method_returns_self_when_the_chain_does_not_factor() -> None:
    # A tall affine creates an axis, so the chain is left unfactored.
    tall = Affine(
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    )
    seq = Sequence([CartesianField(shape=(4, 5)), tall])
    assert seq.factor() is seq
