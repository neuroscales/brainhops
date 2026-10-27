"""Tests of the block restrictors and of the axis counts that they read.

A block restrictor cuts a transformation down to a closed block of its
axes, in which the kept outputs read only the kept inputs. Each
registered rule is exercised once.
"""

import numpy as np
import pytest

from brainhops.datamodel._transformations.compute.reduce import (
    restrict_block,
)
from brainhops.datamodel._transformations.compute.simplify import (
    SimplifyTable,
)
from brainhops.datamodel._transformations.compute.utils import axis_counts
from brainhops.datamodel.axes import SpaceAxis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinateField,
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

ANALYTIC = SimplifyTable.from_like("analytic")
NUMERIC = SimplifyTable.from_like("numeric")


def _cut(
    t: object, inputs: list, outputs: list, policy: object = None
) -> object:
    return restrict_block(t, inputs, outputs, policy or ANALYTIC)


def _inverted(forward: object) -> bool:
    """Return whether the forward has already derived its inverse parameter."""
    return "_cache__inverse" in forward.__dict__


def _system(n: int) -> CoordinateSystem:
    return CoordinateSystem(
        axes=[SpaceAxis(name=f"a{i}", unit="index") for i in range(n)]
    )


def _sub(inner: object, in_axes: list, out_axes: list = None) -> object:
    out_axes = in_axes if out_axes is None else out_axes
    return SubspaceTransformation(
        transformation=inner,
        input_axes=np.asarray(in_axes, dtype=int),
        output_axes=np.asarray(out_axes, dtype=int),
        input=CoordinateSystem().expand(3),
        output=CoordinateSystem().expand(3),
    )


def _warp(ndim: int) -> DisplacementField:
    rng = np.random.default_rng(0)
    shape = (3,) * ndim + (ndim,)
    return DisplacementField(field=rng.normal(size=shape), degree=1)


def _matrix(t: object) -> np.ndarray:
    return np.asarray(t.matrix)


# ----------------------------------------------------------------------
#   ONE RULE PER TYPE
# ----------------------------------------------------------------------


def test_identity_restricts_to_the_identity() -> None:
    assert isinstance(_cut(Identity(input=_system(2)), [0], [0]), Identity)


def test_scaling_keeps_its_type() -> None:
    piece = _cut(Scaling(scale=np.asarray([2.0, 3.0, 4.0])), [0, 2], [0, 2])
    assert type(piece) is Scaling
    assert np.array_equal(piece.scale, [2.0, 4.0])


def test_translation_keeps_its_type() -> None:
    t = Translation(translation=np.asarray([1.0, -2.0, 3.0]))
    piece = _cut(t, [1], [1])
    assert type(piece) is Translation
    assert np.array_equal(piece.translation, [-2.0])


def test_permutation_keeps_its_type() -> None:
    t = Permutation(permutation=np.asarray([2, 1, 0]))
    piece = _cut(t, [0, 2], [0, 2])
    assert type(piece) is Permutation
    assert list(piece.permutation) == [1, 0]
    # A block that the permutation passes through is the identity.
    assert isinstance(_cut(t, [1], [1]), Identity)


def test_permutation_reading_outside_the_block_declines() -> None:
    t = Permutation(permutation=np.asarray([1, 0]))
    assert _cut(t, [0], [0]) is None


@pytest.mark.parametrize("cls", [Affine, Linear, Rotation])
def test_matrices_are_cut_into_their_family(cls: type) -> None:
    linear = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if cls is Affine:
        t = Affine(matrix=np.concatenate([linear, [[1.0], [2.0], [3.0]]], 1))
    else:
        t = cls(matrix=linear)
    piece = _cut(t, [0, 1, 2], [1])
    assert type(piece) is (Affine if cls is Affine else Linear)
    assert np.array_equal(_matrix(piece)[:, :3], linear[1:2])


def test_outputs_that_read_no_input_are_a_creation() -> None:
    t = Affine(matrix=np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 5.0]]))
    piece = _cut(t, [], [1], NUMERIC)
    assert isinstance(piece, Projection)
    assert list(piece.created) == [0] and list(piece.at) == [5.0]


def test_matrix_less_affine_restricts_to_the_identity() -> None:
    t = Affine(matrix=None, input=_system(2), output=_system(2))
    assert isinstance(_cut(t, [0], [0]), Identity)


def test_field_declines() -> None:
    assert _cut(_warp(2), [0], [0]) is None


def test_projection_cuts_to_the_identity_or_a_smaller_projection() -> None:
    t = Projection(dropped=[1], created=[0], at=[4.0], input=_system(3))
    assert isinstance(_cut(t, [0, 2], [1, 2]), Identity)
    piece = _cut(t, [0, 1, 2], [0, 1])
    assert list(piece.dropped) == [1, 2] and list(piece.created) == [0]
    assert list(piece.at) == [4.0]


def test_sequence_is_restricted_member_by_member() -> None:
    seq = Sequence(
        [
            Scaling(scale=np.asarray([2.0, 3.0])),
            Translation(translation=np.asarray([1.0, -1.0])),
        ]
    )
    piece = _cut(seq, [1], [1])
    assert isinstance(piece, Sequence)
    scale, shift = piece.transformations
    assert type(scale) is Scaling and np.array_equal(scale.scale, [3.0])
    assert type(shift) is Translation
    assert np.array_equal(shift.translation, [-1.0])


@pytest.mark.parametrize("policy", [ANALYTIC, NUMERIC], ids=["a", "n"])
def test_sequence_reads_its_members_with_the_policy(policy: object) -> None:
    # Only the numeric policy sees that the shear leaves y alone.
    shear = Affine(matrix=np.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]]))
    seq = Sequence([Scaling(scale=np.asarray([2.0, 3.0])), shear])
    piece = _cut(seq, [1], [1], policy)
    if policy is ANALYTIC:
        assert piece is None
    else:
        assert np.allclose(Sequence([piece]).to(Affine).matrix, [[3.0, 0.0]])


# ----------------------------------------------------------------------
#   LAZY INVERSES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", [ANALYTIC, NUMERIC], ids=["a", "n"])
def test_partial_inverse_is_reinverted_lazily(policy: object) -> None:
    forward = Scaling(scale=np.asarray([2.0, 4.0]))
    piece = _cut(forward.inverse(), [1], [1], policy)
    assert isinstance(piece, Inverse)
    assert type(piece.forward) is Scaling
    assert np.array_equal(piece.forward.scale, [4.0])
    assert not _inverted(forward)
    assert np.allclose(piece.scale, [0.25])


def test_partial_inverse_of_a_permutation_stays_lazy() -> None:
    forward = Permutation(permutation=np.asarray([1, 0, 2]))
    piece = _cut(forward.inverse(), [0, 1], [0, 1])
    assert isinstance(piece, Inverse)
    assert list(piece.forward.permutation) == [1, 0]


def test_partial_inverse_of_a_sequence_stays_lazy() -> None:
    forward = Scaling(scale=np.asarray([2.0, 4.0]))
    seq = Sequence([forward, Translation(translation=np.asarray([1.0, 2.0]))])
    piece = _cut(Inverse(forward=seq), [1], [1])
    shift, scale = piece.transformations
    assert isinstance(shift, Inverse) and isinstance(scale, Inverse)
    assert np.array_equal(scale.forward.scale, [4.0])
    assert not _inverted(forward)


def test_partial_affine_inverse_swaps_the_block() -> None:
    # The forward is block diagonal, so the block of its inverse is the
    # inverse of its block, and nothing is computed.
    matrix = np.asarray([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0]])
    forward = Affine(matrix=matrix)
    piece = _cut(forward.inverse(), [1], [1], NUMERIC)
    assert isinstance(piece, Inverse)
    assert np.array_equal(_matrix(piece.forward), [[4.0, 0.0]])
    assert not _inverted(forward)


def test_a_coupled_inverse_is_computed_only_under_numeric() -> None:
    shear = Linear(matrix=np.asarray([[1.0, 1.0], [0.0, 1.0]]))
    assert _cut(shear.inverse(), [0, 1], [0], ANALYTIC) is None
    piece = _cut(shear.inverse(), [0, 1], [0], NUMERIC)
    assert type(piece) is Linear
    assert np.array_equal(_matrix(piece), [[1.0, -1.0]])


def test_identity_inverse_restricts_to_the_identity() -> None:
    assert isinstance(_cut(Inverse(forward=None), [0], [0]), Identity)


# ----------------------------------------------------------------------
#   SUBSPACES
# ----------------------------------------------------------------------


def test_subspace_covering_its_axes_is_its_inner_object() -> None:
    warp = _warp(2)
    piece = _cut(_sub(warp, [0, 2]), [0, 2], [0, 2])
    assert piece is warp


def test_subspace_restricts_its_inner_keeping_its_type() -> None:
    t = _sub(Scaling(scale=np.asarray([2.0, 3.0])), [0, 2])
    piece = _cut(t, [2], [2])
    assert type(piece) is Scaling
    assert np.array_equal(piece.scale, [3.0])


def test_subspace_block_of_pass_through_axes_is_the_identity() -> None:
    t = _sub(Scaling(scale=np.asarray([2.0])), [0])
    assert isinstance(_cut(t, [1, 2], [1, 2]), Identity)


def test_subspace_with_a_pass_through_axis_stays_a_subspace() -> None:
    warp = _warp(1)
    piece = _cut(_sub(warp, [1]), [0, 1], [0, 1])
    assert isinstance(piece, SubspaceTransformation)
    assert piece.transformation is warp
    assert list(piece.input_axes) == [1] and list(piece.output_axes) == [1]


def test_inner_less_reindex_over_its_whole_group_is_a_local_swap() -> None:
    # Input 0 goes to output 1, input 1 to output 0, and axis 2 passes
    # through.
    piece = _cut(_sub(None, [0, 1], [1, 0]), [0, 1], [0, 1])
    assert isinstance(piece, SubspaceTransformation)
    assert piece.transformation is None
    assert list(piece.input_axes) == [0, 1]
    assert list(piece.output_axes) == [1, 0]


def test_inner_less_reindex_one_acted_axis_is_the_identity() -> None:
    # The block holds only input 0 going to output 1.
    assert isinstance(_cut(_sub(None, [0, 1], [1, 0]), [0], [1]), Identity)


def test_restrict_declines_an_unsupported_type() -> None:
    assert restrict_block(np.eye(2), [0], [0], ANALYTIC) is None


def test_restrict_declines_a_projection_without_counts() -> None:
    assert _cut(Projection(dropped=[1]), [0], [0]) is None


# ----------------------------------------------------------------------
#   AXIS COUNTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "t, counts",
    [
        (Affine(matrix=np.zeros((2, 4))), (3, 2)),
        (Linear(matrix=np.zeros((2, 3))), (3, 2)),
        (Rotation(matrix=np.eye(3)), (3, 3)),
        (Scaling(scale=np.ones(3)), (3, 3)),
        (Translation(translation=np.zeros(4)), (4, 4)),
        (Permutation(permutation=np.asarray([1, 0])), (2, 2)),
        (DisplacementField(field=np.zeros((3, 3, 2))), (2, 2)),
        (CoordinateField(field=np.zeros((3, 3, 4))), (2, 4)),
        (CartesianField(shape=(3, 4, 5)), (3, 3)),
        (Scaling(scale=None, input=_system(3), output=_system(3)), (3, 3)),
        (Identity(input=_system(2), output=_system(2)), (2, 2)),
        (Affine(matrix=np.zeros((2, 4))).inverse(), (2, 3)),
        (Inverse(forward=Linear(matrix=np.zeros((2, 3)))), (2, 3)),
        (
            SubspaceTransformation(
                transformation=Scaling(scale=np.ones(1)),
                input_axes=np.asarray([0]),
                output_axes=np.asarray([0]),
                input=_system(4),
                output=_system(4),
            ),
            (4, 4),
        ),
        (
            Sequence(
                [
                    Identity(),
                    Affine(matrix=np.zeros((2, 4))),
                    Scaling(scale=None),
                    Linear(matrix=np.zeros((5, 2))),
                ]
            ),
            (3, 5),
        ),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, tuple) else "",
)
def test_axis_counts_are_read_from_the_transformation(
    t: object, counts: tuple
) -> None:
    assert axis_counts(t) == counts


@pytest.mark.parametrize(
    "t",
    [
        Identity(),
        Affine(matrix=None),
        Scaling(scale=None),
        Inverse(forward=None),
        # A subspace without endpoints spans only up to its highest named axis.
        SubspaceTransformation(
            transformation=Scaling(scale=np.ones(1), input=_system(1)),
            input_axes=np.asarray([0]),
            output_axes=np.asarray([0]),
        ),
        # A member that may change the number of axes by an unknown amount.
        Sequence([Projection(dropped=[1]), Scaling(scale=np.ones(2))]),
    ],
    ids=lambda v: type(v).__name__,
)
def test_axis_counts_unknown_when_nothing_states_them(t: object) -> None:
    ni, _ = axis_counts(t)
    assert ni is None
