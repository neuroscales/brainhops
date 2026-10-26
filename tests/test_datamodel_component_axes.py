"""Tests for the component axes, whose type says what their entries hold."""

import itertools

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel._frames import (
    _resolve_layout,
    _tensor_ind2sub,
    _tensor_sub2ind,
    get_frame_ndim,
)
from brainhops.datamodel._reframe import _reframed_axis, apply_linear_map
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    PointAxis,
    TensorAxis,
    VectorAxis,
    is_component_type,
)
from brainhops.datamodel.enums import AxisType, IndexVariance, TensorLayout
from brainhops.io.transformations.zarr._axes import _to_axis

_J = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 3.0]])
_K = np.linalg.inv(_J)
_MRTRIX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _is_closed(axis: TensorAxis, ndim: int) -> bool:
    # A layout is closed when no component is zero by construction.
    return -1 not in _tensor_sub2ind(axis, ndim, None)


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
        ("tensor", TensorAxis),
    ],
)
def test_a_component_axis_is_built_from_its_type(kind: str, cls: type) -> None:
    axis = Axis("c", kind)

    assert isinstance(axis, cls)
    assert axis.type == AxisType(kind)
    assert is_component_type(kind)
    assert isinstance(_to_axis({"name": "c", "type": kind}), cls)


@pytest.mark.parametrize(
    "kind", ["space", "time", "channel", "covariant", None, "foo"]
)
def test_other_types_are_not_component_types(kind: str) -> None:
    assert not is_component_type(kind)


def test_a_tensor_axis_is_read_from_a_mapping() -> None:
    axis = Axis(
        "c",
        "tensor",
        variance=["covariant", "covariant"],
        layout=[list(entry) for entry in _MRTRIX],
    )

    assert isinstance(axis, TensorAxis)
    assert axis.variance == ("covariant", "covariant")
    assert np.array_equal(axis.layout, _MRTRIX)
    assert axis != TensorAxis("c")


# --- vectors, covectors and points ------------------------------------


@pytest.mark.parametrize("cls", [VectorAxis, DisplacementAxis])
def test_a_vector_is_multiplied_by_the_matrix(cls: type) -> None:
    values = _random(4, 3, 5)

    out = apply_linear_map(values, _J, cls(), 1)

    assert np.allclose(out, np.einsum("ij,ajb->aib", _J, values))


def test_a_gradient_is_multiplied_by_the_inverse_transpose() -> None:
    values = _random(4, 3)
    gradient = Axis("g", "vector", variance="covariant")

    out = apply_linear_map(values, _J, gradient, 1)

    assert isinstance(gradient, VectorAxis)

    assert np.allclose(out, values @ _K)
    # The pairing of a covector with a vector does not change.
    vector = np.array([1.0, 2.0, 3.0])
    assert np.allclose(out @ (_J @ vector), values @ vector)


@pytest.mark.parametrize("cls", [PointAxis, CoordinateAxis])
def test_points_cannot_be_transformed_by_a_jacobian(cls: type) -> None:
    with pytest.raises(NotImplementedError):
        apply_linear_map(np.zeros((2, 3)), np.eye(3), cls(), 1)


# --- tensors ----------------------------------------------------------


def _symmetric(*shape: int) -> np.ndarray:
    full = _random(*shape)
    return full + np.swapaxes(full, -1, -2)


def _pack(
    full: np.ndarray, entries: tx.Sequence[tx.Tuple[int, ...]]
) -> tx.Any:
    return np.stack([full[(Ellipsis, *e)] for e in entries], -1)


@pytest.mark.parametrize(
    "layout, entries",
    [
        (
            "C",
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
            + [(2, 1), (2, 2)],
        ),
        (
            "F",
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)]
            + [(1, 2), (2, 2)],
        ),
        ("symmetric", [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]),
        ("symmetric-lower", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]),
        ("symmetric-diagonal-first", list(_MRTRIX)),
        ("diagonal", [(0, 0), (1, 1), (2, 2)]),
    ],
)
def test_each_named_layout_has_its_order(
    layout: str, entries: tx.List[tx.Tuple[int, int]]
) -> None:
    entries = [list(entry) for entry in entries]
    assert _tensor_ind2sub(TensorAxis(layout=layout), 3).tolist() == entries


@pytest.mark.parametrize(
    "layout",
    ["C", "F", "symmetric", "symmetric-lower", "symmetric-diagonal-first"],
)
def test_a_diffusion_tensor_is_transformed_on_both_sides(layout: str) -> None:
    full = _symmetric(4, 3, 3)
    axis = TensorAxis(layout=layout)
    entries = _tensor_ind2sub(axis, 3)

    out = apply_linear_map(_pack(full, entries), _J, axis, 1)

    assert _reframed_axis(axis, 3) is axis
    assert np.allclose(out, _pack(_J @ full @ _J.T, entries))


def test_the_c_and_f_layouts_are_transposes_of_each_other() -> None:
    full = _random(4, 3, 3)
    c = TensorAxis(variance=("contra", "co"), layout="C")
    f = TensorAxis(variance=("contra", "co"), layout="F")

    out_c = apply_linear_map(full.reshape(4, 9), _J, c, 1).reshape(4, 3, 3)
    out_f = apply_linear_map(np.swapaxes(full, 1, 2).reshape(4, 9), _J, f, 1)

    assert np.allclose(out_c, _J @ full @ _K)
    assert np.allclose(np.swapaxes(out_f.reshape(4, 3, 3), 1, 2), out_c)


def test_a_hessian_is_transformed_by_the_inverse_on_both_sides() -> None:
    full = _symmetric(4, 3, 3)
    axis = TensorAxis(variance=("co", "co"), layout="symmetric")
    entries = _tensor_ind2sub(axis, 3)

    out = apply_linear_map(_pack(full, entries), _J, axis, 1)

    assert np.allclose(out, _pack(_K.T @ full @ _K, entries))


def test_a_linear_map_is_conjugated_by_the_matrix() -> None:
    full = _random(4, 3, 3)
    axis = TensorAxis(variance=("contra", "co"))

    out = apply_linear_map(full.reshape(4, 9), _J, axis, 1)

    assert np.allclose(out.reshape(4, 3, 3), _J @ full @ _K)


def test_a_tensor_of_rank_three_is_transformed_on_each_index() -> None:
    full = _random(2, 3, 3, 3)
    axis = TensorAxis(variance=("contra", "contra", "co"))

    out = apply_linear_map(full.reshape(2, 27), _J, axis, 1)

    expected = np.einsum("ai,bj,ck,nijk->nabc", _J, _J, _K.T, full)
    assert np.allclose(out.reshape(2, 3, 3, 3), expected)


@pytest.mark.parametrize("layout", ["symmetric", "symmetric-lower"])
def test_a_symmetric_tensor_of_rank_three_is_packed_again(layout: str) -> None:
    full = _random(2, 3, 3, 3)
    full = sum(
        np.transpose(full, (0, *perm))
        for perm in itertools.permutations((1, 2, 3))
    )
    axis = TensorAxis(variance=("contra",) * 3, layout=layout)
    entries = _tensor_ind2sub(axis, 3)

    out = apply_linear_map(_pack(full, entries), _J, axis, 1)

    expected = np.einsum("ai,bj,ck,nijk->nabc", _J, _J, _J, full)
    assert len(entries) == 10
    assert np.allclose(out, _pack(expected, entries))


def test_a_diagonal_tensor_is_expanded_to_a_symmetric_one() -> None:
    diagonal = _random(4, 3)
    axis = TensorAxis(layout="diagonal")

    out = apply_linear_map(diagonal, _J, axis, 1)

    reframed = _reframed_axis(axis, 3)
    expected = _J @ (diagonal[:, :, None] * np.eye(3)) @ _J.T
    assert reframed.layout == "symmetric"
    assert np.allclose(out, _pack(expected, _tensor_ind2sub(reframed, 3)))


def test_a_diagonal_tensor_of_mixed_variance_is_expanded_to_a_full_one() -> (
    None
):
    diagonal = _random(4, 3)
    axis = TensorAxis(variance=("contra", "co"), layout="diagonal")

    out = apply_linear_map(diagonal, _J, axis, 1)

    expected = _J @ (diagonal[:, :, None] * np.eye(3)) @ _K
    assert _reframed_axis(axis, 3).layout == "C"
    assert np.allclose(out.reshape(4, 3, 3), expected)


def test_an_explicit_layout_fills_the_permuted_components() -> None:
    full = _symmetric(4, 3, 3)
    axis = TensorAxis(layout=_MRTRIX)

    out = apply_linear_map(_pack(full, _MRTRIX), _J, axis, 1)

    assert _is_closed(axis, 3)
    assert np.allclose(out, _pack(_J @ full @ _J.T, _MRTRIX))


def test_an_explicit_layout_with_zeros_is_expanded() -> None:
    # The upper-left block of a matrix, whose other components are zero.
    layout = ((0, 0), (1, 0), (1, 1))
    full = _symmetric(4, 3, 3)
    full[:, 2, :] = full[:, :, 2] = 0
    axis = TensorAxis(layout=layout)

    out = apply_linear_map(_pack(full, layout), _J, axis, 1)

    reframed = _reframed_axis(axis, 3)
    assert not _is_closed(axis, 3)
    assert reframed.layout == "symmetric"
    assert np.allclose(
        out, _pack(_J @ full @ _J.T, _tensor_ind2sub(reframed, 3))
    )


def test_a_tensor_takes_a_batch_of_matrices() -> None:
    full = _random(5, 3, 3)
    matrices = np.stack([_J * (1 + k) for k in range(5)])
    axis = TensorAxis()

    out = apply_linear_map(full.reshape(5, 9), matrices, axis, 1)

    expected = matrices @ full @ np.swapaxes(matrices, -1, -2)
    assert np.allclose(out.reshape(5, 3, 3), expected)


def test_a_tensor_is_transformed_lazily_with_dask() -> None:
    da = pytest.importorskip("dask.array")
    full = _random(4, 3, 3)
    axis = TensorAxis(variance=("co", "contra"))

    out = apply_linear_map(
        da.from_array(full.reshape(4, 9), chunks=2), _J, axis, 1
    )

    assert isinstance(out, da.Array)
    expected = _K.T @ full @ _J.T
    assert np.allclose(np.asarray(out).reshape(4, 3, 3), expected)


def test_a_diagonal_tensor_is_expanded_lazily_with_dask() -> None:
    da = pytest.importorskip("dask.array")
    diagonal = _random(4, 3)
    axis = TensorAxis(layout="diagonal")

    out = apply_linear_map(da.from_array(diagonal, chunks=2), _J, axis, 1)

    assert isinstance(out, da.Array)
    assert out.shape == (4, 6)
    assert np.allclose(
        np.asarray(out), apply_linear_map(diagonal, _J, axis, 1)
    )


def test_the_layout_is_read_from_the_number_of_entries() -> None:
    assert _resolve_layout(TensorAxis(), 3, 9) == "C"
    assert _resolve_layout(TensorAxis(), 3, 6) == "symmetric"
    assert get_frame_ndim(TensorAxis(), 9) == 3
    assert get_frame_ndim(TensorAxis(), 6) == 3
    assert get_frame_ndim(TensorAxis(variance=("contra",) * 3), 8) == 2
    assert get_frame_ndim(TensorAxis(variance=("contra",) * 3), 10) == 3
    # The size of the frame tells the layout of 36 entries.
    assert get_frame_ndim(TensorAxis(), 36, 6) == 6
    assert _resolve_layout(TensorAxis(), 6, 36) == "C"
    assert _resolve_layout(TensorAxis(), 8, 36) == "symmetric"


def test_a_layout_is_not_inferred_from_an_ambiguous_size() -> None:
    with pytest.raises(ValueError):
        get_frame_ndim(TensorAxis(), 36)


@pytest.mark.parametrize("size", [3, 5, 7])
def test_a_size_that_holds_no_tensor_is_refused(size: int) -> None:
    with pytest.raises(ValueError):
        _resolve_layout(TensorAxis(), 3, size)
    with pytest.raises(ValueError):
        get_frame_ndim(TensorAxis(layout="C"), size)


def test_a_diagonal_layout_is_never_inferred() -> None:
    with pytest.raises(ValueError):
        apply_linear_map(np.zeros((2, 3)), np.eye(3), TensorAxis(), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance": ()},
        {"layout": "symmetric-diagonal-first", "variance": ("contra",) * 3},
        {"layout": ((0, 0), (1, 1), (0, 0))},
        {"layout": ((0, 1), (0, 1))},
        {"layout": ((0,), (1,))},
    ],
)
def test_an_invalid_layout_is_refused(kwargs: tx.Dict[str, tx.Any]) -> None:
    with pytest.raises(ValueError):
        TensorAxis(**kwargs)


def test_a_diagonal_explicit_layout_accepts_a_mixed_variance() -> None:
    axis = TensorAxis(variance=("contra", "co"), layout=((0, 0), (1, 1)))

    assert _reframed_axis(axis, 2).layout == "C"


def test_indexing_part_of_a_component_axis_gives_a_channel_axis() -> None:
    from brainhops.datamodel.geometry import _index2transform
    from brainhops.datamodel.systems import CoordinateSystem

    system = CoordinateSystem([Axis("i", "space"), TensorAxis("c")])

    sub2full, _ = _index2transform((slice(None), slice(0, 4)), (2, 9), system)

    assert isinstance(sub2full.input.axes[1], ChannelAxis)
    assert sub2full.input.axes[1].name == "c"


def test_a_vector_with_an_unknown_variance_is_refused() -> None:
    with pytest.raises(ValueError):
        VectorAxis(variance="tensor")


def test_a_displacement_has_no_variance_to_change() -> None:
    assert DisplacementAxis().variances == (IndexVariance.contravariant,)
    with pytest.raises(ValueError):
        DisplacementAxis(variance="co")


def test_a_single_variance_applies_to_every_index() -> None:
    # Without a layout, a single variance describes a tensor of rank 2,
    # and an explicit layout gives the rank.
    assert TensorAxis(variance=("co",)).variance == (IndexVariance.co,) * 2
    axis = TensorAxis(variance=("co",), layout=((0, 0, 0), (1, 1, 1)))
    assert axis.variance == (IndexVariance.co,) * 3


def test_the_gradient_of_a_moving_image_pairs_with_its_displacement() -> None:
    # The pairing of a gradient with a displacement does not change when
    # both are re-expressed in the same new frame.
    gradient = apply_linear_map(
        _random(4, 3), _J, VectorAxis(variance="co"), 1
    )
    vectors = apply_linear_map(_random(4, 3)[::-1], _J, DisplacementAxis(), 1)

    expected = np.sum(_random(4, 3) * _random(4, 3)[::-1], axis=1)
    assert np.allclose(np.sum(gradient * vectors, axis=1), expected)


def test_a_full_tensor_may_be_listed_in_any_order() -> None:
    full = _random(4, 3, 3)
    order = ((2, 1), (0, 0), (1, 2), (0, 2), (1, 1), (2, 0), (1, 0))
    order += ((0, 1), (2, 2))
    axis = TensorAxis(variance=("contra", "co"), layout=order)
    c = TensorAxis(variance=("contra", "co"), layout="C")

    out = apply_linear_map(_pack(full, order), _J, axis, 1)

    expected = apply_linear_map(full.reshape(4, 9), _J, c, 1).reshape(4, 3, 3)
    assert _is_closed(axis, 3)
    assert np.allclose(out, _pack(expected, order))


def test_listed_permutations_are_independent_components() -> None:
    # The two off-diagonal components differ, and the third axis is zero.
    layout = ((0, 0), (0, 1), (1, 0), (1, 1))
    full = _random(4, 3, 3)
    full[:, 2, :] = full[:, :, 2] = 0
    axis = TensorAxis(layout=layout)

    out = apply_linear_map(_pack(full, layout), _J, axis, 1)

    reframed = _reframed_axis(axis, 3)
    assert reframed.layout == "C"
    assert np.allclose(out.reshape(4, 3, 3), _J @ full @ _J.T)


def test_an_unknown_layout_name_is_refused() -> None:
    with pytest.raises(TypeError):
        TensorAxis(layout="upper")


@pytest.mark.parametrize("variance", ["covariant", IndexVariance.covector])
def test_a_variance_is_a_member_or_its_value(variance: tx.Any) -> None:
    axis = VectorAxis(variance=variance)

    assert axis.variance is IndexVariance.covariant
    assert axis == VectorAxis(variance="covariant")
    assert axis.variances == (IndexVariance.covariant,)


@pytest.mark.parametrize(
    "layout", [member.value for member in TensorLayout] + list(TensorLayout)
)
def test_a_layout_is_a_member_or_its_value(layout: tx.Any) -> None:
    axis = TensorAxis(layout=layout)

    assert axis.layout is TensorLayout(layout)
    assert axis == TensorAxis(layout=str(TensorLayout(layout).value))


def test_tensor_variances_are_members_or_values() -> None:
    axis = TensorAxis(variance=["co", IndexVariance.contra])

    assert axis.variance == (IndexVariance.co, IndexVariance.contra)
    assert all(isinstance(v, IndexVariance) for v in axis.variance)
