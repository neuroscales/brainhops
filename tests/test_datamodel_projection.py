"""Tests of how projections read as matrices, compose and cancel."""

import numpy as np
import pytest

from brainhops._core.affines import matmul
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
from brainhops.errors import ConversionError, RestrictionError
from brainhops.io.transformations.zarr._map import (
    OmeMappingError,
    to_ome,
)

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


# ----------------------------------------------------------------------
#   VALUES OF THE CREATED AXES
# ----------------------------------------------------------------------


def _xy() -> CoordinateSystem:
    return CoordinateSystem(axes=[Axis(name=n) for n in "xy"])


def _restricted(affine: Affine, at: float) -> Sequence:
    # The affine of (x, y, t) read from (x, y) to (x, y), with t at `at`.
    return Sequence(
        [Projection(created=[2], at=[at]), affine, Projection(dropped=[2])]
    )


_A1 = Affine(matrix=np.array([[2, 0, 0, 1], [0, 3, 0, 2], [0, 0, 1, 5.0]]))
_A2 = Affine(matrix=np.array([[2, 0, 0.5, 1], [0, 3, 0, 2], [0, 0, 1, 5.0]]))
_A3 = Affine(matrix=np.array([[2, 0, 0, 1], [0, 3, 0, 2], [0, 0.1, 1, 5.0]]))


def test_the_created_axes_are_at_zero_by_default() -> None:
    assert np.array_equal(Projection(created=[2]).at, [0.0])
    matrix = Projection(created=[2], input=_xy()).to(Affine).matrix
    assert np.array_equal(matrix[:, -1], [0.0, 0.0, 0.0])


def test_at_needs_one_value_per_created_axis() -> None:
    with pytest.raises(ValueError, match="one value in `at`"):
        Projection(created=[2], at=[1.0, 2.0])


def test_at_is_the_translation_of_the_created_rows() -> None:
    projection = Projection(created=[1], at=[4.0], input=_xy())
    expected = np.array([[1.0, 0, 0], [0, 0, 4], [0, 1, 0]])
    assert np.array_equal(projection.to(Affine).matrix, expected)
    assert projection.is_kind("affine")
    assert not projection.is_kind("linear")
    with pytest.raises(ConversionError, match="not linear"):
        projection.to(Linear)


def test_a_section_reads_the_value_of_the_created_axis() -> None:
    expected = np.array([[2.0, 0, 3], [0, 3, 2]])
    assert np.allclose(_restricted(_A2, 4.0).compute().matrix, expected)


def test_a_free_axis_that_no_kept_output_reads_disappears() -> None:
    expected = np.array([[2.0, 0, 1], [0, 3, 2]])
    # In A1, nothing reads t. In A3, only the dropped output t' reads it.
    for affine in (_A1, _A3):
        result = _restricted(affine, np.nan).compute()
        assert np.array_equal(result.matrix, expected)


def test_a_free_axis_that_a_kept_output_reads_raises() -> None:
    with pytest.raises(RestrictionError, match="free"):
        _restricted(_A2, np.nan).compute()


def test_a_free_axis_created_then_dropped_cancels() -> None:
    chain = Sequence(
        [Projection(created=[2], at=[np.nan]), Projection(dropped=[2])]
    )
    assert isinstance(chain.compute(), Identity)


def test_a_free_axis_left_in_the_result_raises() -> None:
    with pytest.raises(RestrictionError):
        Projection(created=[2], at=[np.nan]).compute()
    # Without composition, the free axis stays in place and is not checked.
    kept = _restricted(_A1, np.nan).compute(False)
    assert isinstance(kept.transformations[0], Projection)


def test_the_merge_keeps_the_values_of_the_created_axes() -> None:
    chain = Sequence(
        [Projection(dropped=[2]), Projection(created=[2], at=[3.0])]
    )
    merged = chain.simplify()
    assert list(merged.created) == [2]
    assert list(merged.at) == [3.0]
    chain = Sequence(
        [Projection(created=[0], at=[np.nan]), Projection(dropped=[2])]
    )
    merged = chain.simplify()
    assert list(merged.created) == [0] and np.isnan(merged.at[0])


def test_zero_cancels_nan_in_the_affine_product() -> None:
    inner = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, np.nan]])
    outer = np.array([[2.0, 0, 0, 1], [0, 3, 0.5, 2]])
    product = matmul(outer, inner, zero_cancels_nan=True)
    assert np.array_equal(product[0], [2.0, 0, 1])
    assert np.isnan(product[1, -1])
    assert np.array_equal(product[1, :-1], [0.0, 3.0])


def test_the_ome_writer_refuses_a_free_axis() -> None:
    projection = Projection(created=[2], at=[np.nan], input=_xy())
    with pytest.raises(OmeMappingError, match="free"):
        to_ome(projection, [0, 1, 2], 3)


def test_the_ome_writer_refuses_an_element_that_changes_the_axes() -> None:
    # Each element of a sequence is written as a map of the image axes
    # onto as many axes, so an element with another shape would lose
    # rows or columns when its matrix is reordered.
    projection = Projection(created=[2], at=[4.0], input=_xy())
    with pytest.raises(OmeMappingError, match="axes onto"):
        to_ome(projection, [0, 1], 2)
    with pytest.raises(OmeMappingError, match="axes onto"):
        to_ome(Sequence([projection, Affine(matrix=np.eye(3, 4))]), [0, 1], 2)
