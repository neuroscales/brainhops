"""Tests for the content of axes, which says what their entries hold."""

import numpy as np
import pytest

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.content import (
    Content,
    Covector,
    SymmetricMatrix,
    Vector,
)

_J = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 3.0]])
_MRTRIX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def test_content_is_built_from_its_type() -> None:
    assert isinstance(Content(type="vector"), Vector)
    assert isinstance(Content(type="covector"), Covector)
    assert isinstance(Content(type="symmetric-matrix"), SymmetricMatrix)


def test_axis_content_is_read_from_a_mapping() -> None:
    axis = Axis("c", "channel", content={"type": "vector"})

    assert axis.content == Vector()
    assert axis == Axis("c", "channel", content=Vector())
    assert axis != Axis("c", "channel")


def test_an_axis_without_content_merges_with_one_with_content() -> None:

    axis = Axis("c", "channel", content=Vector())

    assert Axis("c").compatible_with(axis)
    assert Axis("c").merge_with(axis).content == Vector()


def test_vector_is_multiplied_by_the_matrix() -> None:
    values = np.random.default_rng(0).random((4, 3, 5))

    out = Vector().apply(values, _J, 1)

    expected = np.einsum("ij,ajb->aib", _J, values)
    assert np.allclose(out, expected)


def test_covector_is_multiplied_by_the_inverse_transpose() -> None:
    values = np.random.default_rng(0).random((4, 3))

    out = Covector().apply(values, _J, 1)

    assert np.allclose(out, values @ np.linalg.inv(_J))
    # The pairing of a covector with a vector does not change.
    vector = np.array([1.0, 2.0, 3.0])
    assert np.allclose(out @ (_J @ vector), values @ vector)


def test_symmetric_matrix_is_transformed_on_both_sides() -> None:
    full = np.random.default_rng(0).random((4, 3, 3))
    full = full + np.swapaxes(full, -1, -2)
    content = SymmetricMatrix(order=_MRTRIX)
    packed = np.stack([full[:, i, j] for i, j in _MRTRIX], -1)

    out = content.apply(packed, _J, 1)

    expected = _J @ full @ _J.T
    assert np.allclose(
        out, np.stack([expected[:, i, j] for i, j in _MRTRIX], -1)
    )


def test_symmetric_matrix_packs_row_by_row_by_default() -> None:
    assert SymmetricMatrix().pairs(6) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )
    assert SymmetricMatrix().frame_ndim(3) == 2


def test_symmetric_matrix_refuses_a_size_that_is_not_a_triangle() -> None:
    with pytest.raises(ValueError):
        SymmetricMatrix().frame_ndim(5)


def test_symmetric_matrix_refuses_an_incomplete_order() -> None:
    content = SymmetricMatrix(order=((0, 0), (1, 1), (0, 1), (0, 1)))

    with pytest.raises(ValueError):
        content.pairs(3)


def test_a_content_of_unknown_type_cannot_be_transformed() -> None:
    with pytest.raises(NotImplementedError):
        Content().apply(np.zeros((2, 3)), np.eye(3), 1)
