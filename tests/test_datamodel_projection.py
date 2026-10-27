"""Tests of how projections read as matrices, compose and cancel."""

import numpy as np
import pytest

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Identity,
    Linear,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
)
from brainhops.errors import ConversionError

# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _xyzt() -> CoordinateSystem:
    return CoordinateSystem(axes=[Axis(name=n) for n in "xyzt"])


def _affine4() -> Affine:
    # A map of (x, y, z, t) in which time is not coupled to space.
    matrix = np.eye(4, 5)
    matrix[:3, :3] = [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]
    matrix[:, -1] = [1.0, 2.0, 3.0, 4.0]
    return Affine(matrix=matrix)


# ----------------------------------------------------------------------
#   SYSTEMS
# ----------------------------------------------------------------------


def test_the_output_system_is_derived_from_a_closed_input() -> None:
    projection = Projection(dropped=[3], input=_xyzt())
    assert list(projection.output.axes.names) == ["x", "y", "z"]


def test_the_input_system_is_derived_from_a_closed_output() -> None:
    xz = CoordinateSystem(axes=[Axis(name="x"), Axis(name="z")])
    embedding = Projection(created=[1], output=xz)
    assert list(embedding.input.axes.names) == ["x"]
    projection = Projection(dropped=[1], output=xz)
    assert projection.input.ndim == 3
    assert list(projection.input.axes.names) == ["x", None, "z"]


def test_no_system_is_derived_from_a_missing_or_open_one() -> None:
    assert Projection(dropped=[3]).output is None
    open_system = CoordinateSystem(axes=[Axis(name="x"), ...])
    assert Projection(dropped=[0], input=open_system).output is None


# ----------------------------------------------------------------------
#   MATRICES
# ----------------------------------------------------------------------


def test_a_projection_reads_as_the_matrix_that_selects_the_kept_axes() -> None:
    projection = Projection(dropped=[1], created=[0], input=_xyzt())
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert np.array_equal(projection.to(Affine).matrix, expected)
    assert np.array_equal(projection.to(Linear).matrix, expected[:, :-1])


def test_a_projection_without_a_number_of_axes_has_no_matrix() -> None:
    with pytest.raises(ConversionError, match="number of axes"):
        Projection(dropped=[3]).to(Affine)


def test_a_projection_is_linear() -> None:
    projection = Projection(dropped=[3])
    assert projection.is_kind("linear")
    assert projection.is_kind("surjective")
    assert not projection.is_kind("injective")
    assert Projection(created=[3]).is_kind("injective")


# ----------------------------------------------------------------------
#   COMPOSITION
# ----------------------------------------------------------------------


def test_a_projection_after_an_affine_composes_into_an_affine() -> None:
    composed = Sequence([_affine4(), Projection(dropped=[3])]).compute()
    assert isinstance(composed, Affine)
    assert np.allclose(composed.matrix, _affine4().matrix[:3])


def test_an_embedding_before_an_affine_composes_into_an_affine() -> None:
    composed = Sequence([Projection(created=[3]), _affine4()]).compute()
    assert isinstance(composed, Affine)
    expected = np.delete(_affine4().matrix, 3, axis=1)
    assert np.allclose(composed.matrix, expected)


def test_a_projection_after_a_subspace_of_an_affine_composes() -> None:
    scaling = Scaling(scale=np.array([2.0, 3.0, 4.0]))
    subspace = SubspaceTransformation(
        scaling,
        input_axes=[0, 1, 2],
        output_axes=[0, 1, 2],
        input=CoordinateSystem().expand(4),
        output=CoordinateSystem().expand(4),
    )
    composed = Sequence([subspace, Projection(dropped=[3])]).compute()
    expected = np.zeros((3, 5))
    expected[[0, 1, 2], [0, 1, 2]] = [2.0, 3.0, 4.0]
    assert np.allclose(composed.to(Affine).matrix, expected)


def test_a_projection_after_a_field_stays_next_to_it() -> None:
    field = DisplacementField(field=np.zeros((2, 2, 2, 3)))
    projection = Projection(dropped=[2])
    composed = Sequence([field, projection]).compute()
    assert isinstance(composed, Sequence)
    assert composed.transformations[-1] is projection


def test_a_projection_that_does_not_fit_its_neighbour_stays_next_to_it() -> (
    None
):
    projection = Projection(dropped=[5])
    composed = Sequence([_affine4(), projection]).compute()
    assert isinstance(composed, Sequence)
    assert composed.transformations[-1] is projection


# ----------------------------------------------------------------------
#   MERGING AND CANCELLATION
# ----------------------------------------------------------------------


def test_an_embedding_followed_by_the_matching_projection_cancels() -> None:
    chain = Sequence([Projection(created=[3]), Projection(dropped=[3])])
    assert isinstance(chain.simplify(), Identity)


def test_a_projection_followed_by_the_matching_embedding_does_not() -> None:
    chain = Sequence([Projection(dropped=[3]), Projection(created=[3])])
    merged = chain.simplify()
    assert isinstance(merged, Projection)
    assert list(merged.dropped) == [3]
    assert list(merged.created) == [3]
    matrix = merged.to(input=_xyzt()).to(Linear).matrix
    assert np.array_equal(matrix, np.diag([1.0, 1.0, 1.0, 0.0]))


def test_two_projections_merge_into_one() -> None:
    # Creating x0 then dropping the third axis of (x0, a, b, c) drops b.
    chain = Sequence([Projection(created=[0]), Projection(dropped=[2])])
    merged = chain.simplify()
    assert list(merged.dropped) == [1]
    assert list(merged.created) == [0]


# ----------------------------------------------------------------------
#   INVERSE
# ----------------------------------------------------------------------


def test_a_projection_that_drops_axes_has_no_inverse() -> None:
    with pytest.raises(ValueError, match="no inverse"):
        Projection(dropped=[3]).inverse()


def test_the_inverse_of_an_embedding_cancels_after_it() -> None:
    embedding = Projection(created=[3])
    chain = Sequence([embedding, embedding.inverse()])
    assert isinstance(chain.simplify(), Identity)
