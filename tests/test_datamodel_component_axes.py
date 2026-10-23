"""Tests for the component axes, whose type says what their entries hold."""

import itertools

import numpy as np
import pytest

from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    CovectorAxis,
    DisplacementAxis,
    PointAxis,
    TensorAxis,
    VectorAxis,
    is_component_type,
)
from brainhops.datamodel.enums import AxisType
from brainhops.io.transformations.zarr._axes import _to_axis

_J = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 3.0]])
_K = np.linalg.inv(_J)
_MRTRIX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _random(*shape: int) -> np.ndarray:
    return np.random.default_rng(0).random(shape)


# --- types and classes ------------------------------------------------


@pytest.mark.parametrize(
    "kind, cls",
    [
        ("point", PointAxis),
        ("coordinate", CoordinateAxis),
        ("vector", VectorAxis),
        ("displacement", DisplacementAxis),
        ("covector", CovectorAxis),
        ("tensor", TensorAxis),
    ],
)
def test_a_component_axis_is_built_from_its_type(kind: str, cls: type) -> None:
    axis = Axis("c", kind)

    assert isinstance(axis, cls)
    assert axis.type == AxisType(kind)
    assert is_component_type(kind)
    assert isinstance(_to_axis({"name": "c", "type": kind}), cls)


@pytest.mark.parametrize("kind", ["space", "time", "channel", None, "foo"])
def test_other_types_are_not_component_types(kind: str) -> None:
    assert not is_component_type(kind)


def test_a_tensor_axis_is_read_from_a_mapping() -> None:
    axis = Axis(
        "c",
        "tensor",
        indices=["covector", "covector"],
        layout="symmetric",
        order=_MRTRIX,
    )

    assert isinstance(axis, TensorAxis)
    assert axis.indices == ("covector", "covector")
    assert axis.order == _MRTRIX
    assert axis != TensorAxis("c")


# --- vectors, covectors and points ------------------------------------


@pytest.mark.parametrize("cls", [VectorAxis, DisplacementAxis])
def test_a_vector_is_multiplied_by_the_matrix(cls: type) -> None:
    values = _random(4, 3, 5)

    out = cls().apply(values, _J, 1)

    assert np.allclose(out, np.einsum("ij,ajb->aib", _J, values))


def test_a_covector_is_multiplied_by_the_inverse_transpose() -> None:
    values = _random(4, 3)

    out = CovectorAxis().apply(values, _J, 1)

    assert np.allclose(out, values @ _K)
    # The pairing of a covector with a vector does not change.
    vector = np.array([1.0, 2.0, 3.0])
    assert np.allclose(out @ (_J @ vector), values @ vector)


@pytest.mark.parametrize("cls", [PointAxis, CoordinateAxis])
def test_points_cannot_be_transformed_by_a_jacobian(cls: type) -> None:
    with pytest.raises(NotImplementedError):
        cls().apply(np.zeros((2, 3)), np.eye(3), 1)


# --- tensors ----------------------------------------------------------


def test_a_diffusion_tensor_is_transformed_on_both_sides() -> None:
    full = _random(4, 3, 3)
    full = full + np.swapaxes(full, -1, -2)
    axis = TensorAxis(layout="symmetric", order=_MRTRIX)
    packed = np.stack([full[:, i, j] for i, j in _MRTRIX], -1)

    out = axis.apply(packed, _J, 1)

    expected = _J @ full @ _J.T
    assert np.allclose(
        out, np.stack([expected[:, i, j] for i, j in _MRTRIX], -1)
    )


def test_a_hessian_is_transformed_by_the_inverse_on_both_sides() -> None:
    full = _random(4, 3, 3)
    full = full + np.swapaxes(full, -1, -2)
    axis = TensorAxis(indices=("covector", "covector"), layout="symmetric")
    pairs = axis.entries(3)
    packed = np.stack([full[:, i, j] for i, j in pairs], -1)

    out = axis.apply(packed, _J, 1)

    expected = _K.T @ full @ _K
    assert np.allclose(
        out, np.stack([expected[:, i, j] for i, j in pairs], -1)
    )


def test_a_linear_map_is_conjugated_by_the_matrix() -> None:
    full = _random(4, 3, 3)
    axis = TensorAxis(indices=("vector", "covector"))

    out = axis.apply(full.reshape(4, 9), _J, 1)

    assert np.allclose(out.reshape(4, 3, 3), _J @ full @ _K)


def test_a_tensor_of_rank_three_is_transformed_on_each_index() -> None:
    full = _random(2, 3, 3, 3)
    axis = TensorAxis(indices=("vector", "vector", "covector"))

    out = axis.apply(full.reshape(2, 27), _J, 1)

    expected = np.einsum("ai,bj,ck,nijk->nabc", _J, _J, _K.T, full)
    assert np.allclose(out.reshape(2, 3, 3, 3), expected)


def test_a_symmetric_tensor_of_rank_three_is_packed_again() -> None:
    full = _random(2, 3, 3, 3)
    full = sum(
        np.transpose(full, (0, *perm))
        for perm in itertools.permutations((1, 2, 3))
    )
    axis = TensorAxis(indices=("vector",) * 3, layout="symmetric")
    entries = axis.entries(3)
    packed = np.stack([full[(slice(None), *e)] for e in entries], -1)

    out = axis.apply(packed, _J, 1)

    expected = np.einsum("ai,bj,ck,nijk->nabc", _J, _J, _J, full)
    assert packed.shape == (2, 10)
    assert np.allclose(
        out, np.stack([expected[(slice(None), *e)] for e in entries], -1)
    )


def test_a_tensor_takes_a_batch_of_matrices() -> None:
    full = _random(5, 3, 3)
    matrices = np.stack([_J * (1 + k) for k in range(5)])
    axis = TensorAxis()

    out = axis.apply(full.reshape(5, 9), matrices, 1)

    expected = matrices @ full @ np.swapaxes(matrices, -1, -2)
    assert np.allclose(out.reshape(5, 3, 3), expected)


def test_a_tensor_is_transformed_lazily_with_dask() -> None:
    da = pytest.importorskip("dask.array")
    full = _random(4, 3, 3)
    axis = TensorAxis(indices=("covector", "vector"))

    out = axis.apply(da.from_array(full.reshape(4, 9), chunks=2), _J, 1)

    assert isinstance(out, da.Array)
    expected = _K.T @ full @ _J.T
    assert np.allclose(np.asarray(out).reshape(4, 3, 3), expected)


def test_the_default_orders_are_row_major_and_lexicographic() -> None:
    assert TensorAxis().entries(2) == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert TensorAxis(layout="symmetric").entries(3) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    )


def test_the_frame_size_is_read_from_the_number_of_entries() -> None:
    assert TensorAxis().frame_ndim(9) == 3
    assert TensorAxis(layout="symmetric").frame_ndim(6) == 3
    assert TensorAxis(indices=("vector",) * 3).frame_ndim(8) == 2
    symmetric = TensorAxis(indices=("vector",) * 3, layout="symmetric")
    assert symmetric.frame_ndim(10) == 3


@pytest.mark.parametrize("layout, size", [("full", 5), ("symmetric", 5)])
def test_a_size_that_holds_no_tensor_is_refused(
    layout: str, size: int
) -> None:
    with pytest.raises(ValueError):
        TensorAxis(layout=layout).frame_ndim(size)


def test_an_incomplete_order_is_refused() -> None:
    axis = TensorAxis(
        layout="symmetric", order=((0, 0), (1, 1), (0, 1), (0, 1))
    )

    with pytest.raises(ValueError):
        axis.entries(2)


def test_an_order_with_entries_of_the_wrong_rank_is_refused() -> None:
    with pytest.raises(ValueError):
        TensorAxis(order=((0,), (1,)))


def test_a_symmetric_tensor_with_mixed_indices_is_refused() -> None:
    with pytest.raises(ValueError):
        TensorAxis(indices=("vector", "covector"), layout="symmetric")


def test_a_tensor_without_indices_is_refused() -> None:
    with pytest.raises(ValueError):
        TensorAxis(indices=())


def test_a_symmetric_order_may_list_indices_in_any_order() -> None:
    axis = TensorAxis(layout="symmetric", order=((0, 0), (1, 0), (1, 1)))

    assert axis.entries(2) == ((0, 0), (0, 1), (1, 1))


def test_indexing_part_of_a_component_axis_gives_a_channel_axis() -> None:
    from brainhops.datamodel.geometry import _index2transform
    from brainhops.datamodel.systems import CoordinateSystem

    system = CoordinateSystem([Axis("i", "space"), TensorAxis("c")])

    sub2full, _ = _index2transform((slice(None), slice(0, 4)), (2, 9), system)

    assert isinstance(sub2full.input.axes[1], ChannelAxis)
    assert sub2full.input.axes[1].name == "c"
