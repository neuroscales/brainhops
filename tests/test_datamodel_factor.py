"""Tests for the factoring pass (``compute(factor=True)``, PR-D).

The factor pass rewrites a dimension-preserving chain into its axis-group
normal form ``[grid?, F_1..F_m, Pi_perm?]``: one axis-preserving subspace
factor per group of coupled axes, plus a trailing reindex permutation. The
tests cover the normal form's structure, its idempotence and
identity-preservation, that it recomposes to the same coordinates as the
monolithic compute, that a lazy inverse cancels inside a group without being
materialized, mode/simplify interaction, termination, and the fallbacks that
leave a chain unfactored.
"""

import numpy as np
import pytest

import brainhops._ext.invfield as invfield
from brainhops.datamodel._transformations import sequence as seqmod
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
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
    """The coordinate field a transform computes, as a numpy array."""
    return np.asarray(x.compute().field)


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
    # A fully coupled affine is one group covering everything -> nothing is
    # separable, so the sequence is returned unchanged, same objects.
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


def test_permutation_recomposes_to_correct_coordinates() -> None:
    # The default `compute()` of a multi-permutation chain is unreliable (a
    # pre-existing bug in the Permutation-o-Permutation composer indexes
    # backwards), so the factored NF is checked against ground truth computed
    # by applying each permutation to the grid coordinates directly.
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
    # The warp group cancels to identity and drops; only the scale factor
    # survives -- and the field was never inverted.
    assert _factor_axes(nf) == [(3,)]
    assert calls["n"] == 0


def test_analytic_subspace_cancel_composer_direct() -> None:
    from brainhops.datamodel._transformations.compose import compose

    warp = DisplacementField(field=np.random.randn(4, 5, 3) * 0.05)
    sub = _sub(warp, [0, 1])
    inv = sub.inverse()
    # compose(inv, sub) applies sub first, then inv -> identity, for free.
    result = compose(inv, sub, analytic_only=True)
    assert isinstance(result, Identity)


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
    # Axes 1, 2 downcast to pure scalings; axis 0 keeps its translation.
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
    # A single-axis chain is one group covering everything -> returned
    # unchanged (trivial), regardless of mode.
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
    # A factor pass that never reaches a fixpoint (returns fresh leaf objects
    # every iteration) must trip the iteration cap and raise, rather than loop
    # forever.
    from bagof.magic import replace

    def broken(
        seq: object,
        mode: object,
        table: object,
        dep_cache: object,
        dep_keepalive: object,
    ) -> object:
        leaves = list(seq.transformations or [])
        if not leaves:
            return seq
        fresh = [
            replace(t, output=getattr(t, "_output", None)) for t in leaves
        ]
        return replace(seq, transformations=fresh)

    monkeypatch.setattr(seqmod, "_factor", broken)
    grid = CartesianField(shape=(4, 5))
    seq = Sequence([grid, Scaling(scale=np.array([2.0, 3.0]))])
    with pytest.raises(RuntimeError, match="did not converge"):
        seq.compute(factor=True)


# ----------------------------------------------------------------------
#   FALLBACKS (unfactored)
# ----------------------------------------------------------------------


def test_rectangular_chain_left_unfactored() -> None:
    # A tall (2 -> 3) affine creates an axis; PR-D leaves it unfactored.
    grid = CartesianField(shape=(4, 5))
    tall = Affine(
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    )
    seq = Sequence([grid, tall])
    nf = seq.compute(factor=True)
    assert list(nf.transformations) == list(seq.transformations)


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
