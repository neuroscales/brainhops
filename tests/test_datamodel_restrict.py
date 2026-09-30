"""Tests for the dispatched `restrict` / `embed` operations.

`restrict(t, rows, cols, ni, no)` cuts a transformation down to a decoupled
block of its axes, keeping the cheaper type, keeping a lazy inverse lazy,
and returning the same object when the block covers all of it. `embed` puts
a transformation over some axes back into a wider space. Each registered
rule is exercised once, plus the refusal of an unsupported type.
"""

import numpy as np
import pytest

from brainhops.datamodel._transformations.errors import RestrictionError
from brainhops.datamodel._transformations.registries import INVERSE_CACHE
from brainhops.datamodel._transformations.restrict import embed, restrict
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)


def _sub(inner: object, in_axes: list, out_axes: list = None) -> object:
    out_axes = in_axes if out_axes is None else out_axes
    return SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray(in_axes, dtype=int),
        output_axes=np.asarray(out_axes, dtype=int),
    )


def _warp(ndim: int) -> DisplacementField:
    rng = np.random.default_rng(0)
    shape = (3,) * ndim + (ndim,)
    return DisplacementField(field=rng.normal(size=shape), order=1)


def _matrix(t: object) -> np.ndarray:
    return np.asarray(t.matrix)


# ----------------------------------------------------------------------
#   RESTRICT: ONE RULE PER TYPE
# ----------------------------------------------------------------------


def test_identity_restricts_to_nothing() -> None:
    assert restrict(Identity(), [0], [0], 2, 2) is None


def test_scaling_keeps_its_type() -> None:
    piece = restrict(
        Scaling(scale=np.asarray([2.0, 3.0, 4.0])), [0, 2], [0, 2], 3, 3
    )
    assert type(piece) is Scaling
    assert np.array_equal(piece.scale, [2.0, 4.0])
    assert restrict(Scaling(scale=None), [0], [0], 2, 2) is None


def test_translation_keeps_its_type() -> None:
    t = Translation(translation=np.asarray([1.0, -2.0, 3.0]))
    piece = restrict(t, [1], [1], 3, 3)
    assert type(piece) is Translation
    assert np.array_equal(piece.translation, [-2.0])
    assert restrict(Translation(translation=None), [0], [0], 2, 2) is None


def test_permutation_keeps_its_type() -> None:
    t = Permutation(permutation=np.asarray([2, 1, 0]))
    piece = restrict(t, [0, 2], [0, 2], 3, 3)
    assert type(piece) is Permutation
    assert list(piece.permutation) == [1, 0]
    assert restrict(Permutation(permutation=None), [0], [0], 2, 2) is None


def test_permutation_reading_outside_the_block_is_an_affine_block() -> None:
    t = Permutation(permutation=np.asarray([1, 0]))
    piece = restrict(t, [0], [0], 2, 2)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), [[0.0, 0.0]])


@pytest.mark.parametrize("cls", [Affine, Linear, Rotation])
def test_affine_family_is_an_affine_sub_block(cls: type) -> None:
    linear = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if cls is Affine:
        t = Affine(matrix=np.concatenate([linear, [[1.0], [2.0], [3.0]]], 1))
        shift = [[3.0]]
    else:
        t = cls(matrix=linear)
        shift = [[0.0]]
    piece = restrict(t, [2], [2], 3, 3)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), np.concatenate([[[1.0]], shift], 1))
    block = restrict(t, [0, 1], [0, 1], 3, 3)
    assert np.array_equal(_matrix(block)[:, :2], linear[:2, :2])


def test_matrix_less_affine_restricts_to_nothing() -> None:
    assert restrict(Affine(matrix=None), [0], [0], 2, 2) is None


def test_field_is_kept_whole() -> None:
    warp = _warp(2)
    assert restrict(warp, [0, 1], [0, 1], 2, 2) is warp


def test_sequence_covering_the_block_is_the_same_object() -> None:
    seq = Sequence([Scaling(scale=np.asarray([2.0, 3.0])), _warp(2)])
    assert restrict(seq, [0, 1], [0, 1], 2, 2) is seq


def test_sequence_is_restricted_member_by_member() -> None:
    # Regression: a partial block of a sequence with no affine reading used
    # to be dropped as the identity.
    seq = Sequence(
        [
            Scaling(scale=np.asarray([2.0, 3.0])),
            Translation(translation=np.asarray([1.0, -1.0])),
        ]
    )
    piece = restrict(seq, [1], [1], 2, 2)
    assert isinstance(piece, Sequence)
    scale, shift = piece.transformations
    assert type(scale) is Scaling and np.array_equal(scale.scale, [3.0])
    assert type(shift) is Translation
    assert np.array_equal(shift.translation, [-1.0])


def test_sequence_member_coupling_the_block_is_refused() -> None:
    shear = Affine(matrix=np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]]))
    seq = Sequence([Scaling(scale=np.asarray([2.0, 3.0])), shear])
    with pytest.raises(RestrictionError):
        restrict(seq, [1], [1], 2, 2)


def test_partial_inverse_of_a_sequence_stays_lazy() -> None:
    forward = Scaling(scale=np.asarray([2.0, 4.0]))
    seq = Sequence([forward, Translation(translation=np.asarray([1.0, 2.0]))])
    piece = restrict(Inverse(forward=seq), [1], [1], 2, 2)
    shift, scale = piece.transformations
    assert isinstance(shift, Inverse) and isinstance(scale, Inverse)
    assert np.array_equal(scale.forward.scale, [4.0])
    assert not getattr(forward, INVERSE_CACHE, None)


@pytest.mark.parametrize(
    "forward",
    [
        Scaling(scale=np.asarray([2.0, 4.0])),
        Translation(translation=np.asarray([1.0, 2.0])),
        Permutation(permutation=np.asarray([0, 1])),
        Affine(matrix=np.asarray([[2.0, 0.0, 1.0], [0.0, 4.0, 2.0]])),
        Linear(matrix=np.diag([2.0, 4.0])),
        Rotation(matrix=np.eye(2)),
        _warp(2),
    ],
    ids=lambda t: type(t).__name__,
)
def test_inverse_covering_the_block_is_the_same_lazy_object(
    forward: object,
) -> None:
    # Every typed inverse (which is also an instance of the family it
    # inverts) reaches the `Inverse` rule, not its family's.
    inverse = forward.inverse()
    assert isinstance(inverse, Inverse)
    assert restrict(inverse, [0, 1], [0, 1], 2, 2) is inverse
    assert not getattr(forward, INVERSE_CACHE, None)


def test_partial_inverse_is_reinverted_lazily() -> None:
    forward = Scaling(scale=np.asarray([2.0, 4.0]))
    piece = restrict(forward.inverse(), [1], [1], 2, 2)
    assert isinstance(piece, Inverse)
    assert type(piece.forward) is Scaling
    assert np.array_equal(piece.forward.scale, [4.0])
    assert not getattr(forward, INVERSE_CACHE, None)
    assert np.allclose(piece.scale, [0.25])


def test_partial_affine_inverse_swaps_the_block() -> None:
    # The forward maps 2 axes to 3 (its inverse maps 3 to 2): the forward
    # is restricted over the swapped block.
    matrix = np.asarray([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
    forward = Affine(matrix=matrix)
    piece = restrict(forward.inverse(), [1], [1, 2], 3, 2)
    assert isinstance(piece, Inverse)
    assert np.array_equal(_matrix(piece.forward), [[4.0, 0.0], [1.0, 0.0]])
    assert not getattr(forward, INVERSE_CACHE, None)


def test_identity_inverse_restricts_to_nothing() -> None:
    assert restrict(Inverse(forward=None), [0], [0], 2, 2) is None


def test_subspace_covering_its_axes_is_its_inner_object() -> None:
    warp = _warp(2)
    piece = restrict(_sub(warp, [0, 2]), [0, 2], [0, 2], 3, 3)
    assert piece is warp


def test_subspace_restricts_its_inner_keeping_its_type() -> None:
    t = _sub(Scaling(scale=np.asarray([2.0, 3.0])), [0, 2])
    piece = restrict(t, [2], [2], 3, 3)
    assert type(piece) is Scaling
    assert np.array_equal(piece.scale, [3.0])


def test_subspace_block_of_pass_through_axes_is_nothing() -> None:
    t = _sub(Scaling(scale=np.asarray([2.0])), [0])
    assert restrict(t, [1, 2], [1, 2], 3, 3) is None


def test_subspace_with_a_pass_through_axis_embeds_as_an_affine() -> None:
    t = _sub(Scaling(scale=np.asarray([2.0])), [0])
    piece = restrict(t, [0, 1], [0, 1], 3, 3)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_subspace_field_with_a_pass_through_axis_stays_wrapped() -> None:
    warp = _warp(1)
    piece = restrict(_sub(warp, [1]), [0, 1], [0, 1], 3, 3)
    assert isinstance(piece, SubspaceTransformation)
    assert piece.transformation is warp
    assert list(piece.input_axes) == [1] and list(piece.output_axes) == [1]


def test_inner_less_reindex_over_its_whole_group_is_a_local_swap() -> None:
    # `in 0 -> out 1`, `in 1 -> out 0`, axis 2 passes through.
    piece = restrict(_sub(None, [0, 1], [1, 0]), [0, 1], [0, 1], 3, 3)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_inner_less_reindex_one_acted_axis_in_order_is_nothing() -> None:
    # The block holds `in 0 -> out 1` only: an identity piece.
    assert restrict(_sub(None, [0, 1], [1, 0]), [1], [0], 3, 3) is None


def test_inner_less_reindex_partial_block_out_of_order() -> None:
    # `in 2 -> out 0` with the pass-through `in 1 -> out 1`: in the block's
    # local axes, output 0 reads input 1 and output 1 reads input 0.
    piece = restrict(_sub(None, [0, 2], [2, 0]), [0, 1], [1, 2], 3, 3)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_restrict_refuses_an_unsupported_type() -> None:
    with pytest.raises(RestrictionError, match="Cannot restrict"):
        restrict(np.eye(2), [0], [0], 2, 2)


def test_restrict_refuses_a_transform_with_no_affine_reading() -> None:
    # Never read as the identity, which would drop it.
    with pytest.raises(RestrictionError, match="no affine reading"):
        restrict(Projection(dropped=[1]), [0], [0], 2, 1)


# ----------------------------------------------------------------------
#   EMBED
# ----------------------------------------------------------------------


def test_embed_identity_is_a_reindex() -> None:
    piece = embed(None, [1, 0], [0, 1], 3, 3)
    assert type(piece) is Affine
    expected = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    assert np.array_equal(_matrix(piece), expected)


def test_embed_affine_ish_is_a_plain_affine() -> None:
    t = Affine(matrix=np.asarray([[2.0, 5.0]]))
    piece = embed(t, [1], [1], 2, 2)
    assert type(piece) is Affine
    assert np.array_equal(_matrix(piece), [[1.0, 0.0, 0.0], [0.0, 2.0, 5.0]])


def test_embed_lazy_inverse_stays_wrapped() -> None:
    forward = Scaling(scale=np.asarray([2.0]))
    inverse = forward.inverse()
    piece = embed(inverse, [1], [1], 2, 2)
    assert isinstance(piece, SubspaceTransformation)
    assert piece.transformation is inverse
    assert not getattr(forward, INVERSE_CACHE, None)


def test_embed_field_stays_wrapped() -> None:
    warp = _warp(1)
    piece = embed(warp, [0], [0], 2, 2)
    assert isinstance(piece, SubspaceTransformation)
    assert piece.transformation is warp


def test_embed_refuses_an_unsupported_type() -> None:
    with pytest.raises(TypeError, match="Cannot embed"):
        embed(np.eye(2), [0], [0], 2, 2)
