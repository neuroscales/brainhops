"""Tests of the batched transformations, one transformation per element."""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import kinds
from brainhops.datamodel._transformations.compute.utils import axis_counts
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    BatchedAffine,
    BatchedLinear,
    BatchedPermutation,
    BatchedScaling,
    BatchedTransformation,
    BatchedTranslation,
    Identity,
    Linear,
    Permutation,
    Rotation,
    Scaling,
    Transformation,
    Translation,
    is_kind,
)


def _affines(n: int = 5) -> np.ndarray:
    # A stack of invertible affines that differ from one element to the
    # next.
    rng = np.random.default_rng(0)
    stack = rng.normal(scale=0.1, size=(n, 3, 4))
    stack[..., :, :3] += np.eye(3)
    return stack


def _rotations(n: int = 4) -> np.ndarray:
    angles = np.linspace(0.1, 1.0, n)
    stack = np.zeros((n, 3, 3))
    stack[:, 0, 0] = stack[:, 1, 1] = np.cos(angles)
    stack[:, 0, 1], stack[:, 1, 0] = -np.sin(angles), np.sin(angles)
    stack[:, 2, 2] = 1
    return stack


def _system(names: str) -> CoordinateSystem:
    return CoordinateSystem(axes=[Axis(name=n) for n in names])


# --- construction -----------------------------------------------------


def test_batch_puts_the_batch_axes_first_by_default() -> None:
    batch = Affine.batch(_affines())
    assert isinstance(batch, BatchedAffine)
    assert batch.element_type is Affine
    assert batch.batch_shape == (5,)
    assert batch.counts == (4, 4)
    assert batch.input_axes == batch.output_axes == [1, 2, 3]
    assert batch.batch_input_axes == batch.batch_output_axes == [0]


def test_input_axes_name_the_axes_of_the_elements() -> None:
    batch = Affine.batch(_affines(), input_axes=[0, 1, 2])
    assert batch.output_axes == [0, 1, 2]
    assert batch.batch_input_axes == batch.batch_output_axes == [3]


def test_a_batch_axis_may_sit_at_different_positions_on_each_side() -> None:
    batch = Affine.batch(
        _affines(), input_axes=[0, 1, 2], output_axes=[1, 2, 3]
    )
    assert batch.batch_input_axes == [3]
    assert batch.batch_output_axes == [0]


def test_several_batch_axes_follow_the_leading_dimensions() -> None:
    batch = Scaling.batch(np.ones((2, 3, 4)), input_axes=[1, 3, 4, 5])
    assert isinstance(batch, BatchedScaling)
    assert batch.batch_shape == (2, 3)
    assert batch.counts == (6, 6)
    assert batch.batch_input_axes == [0, 2]


@pytest.mark.parametrize(
    "element_type, data, cls",
    [
        (Affine, np.zeros((2, 3, 4)), BatchedAffine),
        (Linear, np.zeros((2, 3, 3)), BatchedLinear),
        (Rotation, np.stack([np.eye(3)] * 2), BatchedLinear),
        (Scaling, np.ones((2, 3)), BatchedScaling),
        (Translation, np.zeros((2, 3)), BatchedTranslation),
        (Permutation, np.tile(np.arange(3), (2, 1)), BatchedPermutation),
    ],
)
def test_the_subclass_follows_the_element_type(
    element_type: tx.Type[Transformation],
    data: np.ndarray,
    cls: tx.Type[BatchedTransformation],
) -> None:
    batch = element_type.batch(data)
    assert type(batch) is cls
    assert batch.element_type is element_type


def test_log_selects_the_tangent_class() -> None:
    batch = Affine.batch(np.zeros((4, 3, 4)), log=True)
    assert batch.element_type.__name__ == "AffineExponential"
    np.testing.assert_allclose(batch.matrix, np.stack([np.eye(3, 4)] * 4))


def test_a_view_may_be_given_in_place_of_the_data() -> None:
    stack = _affines()
    homogeneous = Affine.batch(stack).homogeneous_matrix
    assert homogeneous.shape == (5, 4, 4)
    batch = Affine.batch(homogeneous_matrix=homogeneous)
    np.testing.assert_allclose(batch.matrix, stack)
    scales = Scaling.batch(scale=np.full((2, 3), 2.0), log=True)
    np.testing.assert_allclose(scales.data, np.log(2.0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"input_axes": [0, 1]}, "must name 3 axes"),
        ({"input_axes": [0, 0, 1]}, "names an axis twice"),
        ({"input_axes": [1, 2, 4]}, "positions of a space of 4 axes"),
        ({"positions": [np.arange(4)]}, "vector of 5 values"),
        ({"positions": [np.arange(5)] * 2}, "one vector per batch axis"),
        ({"input": CoordinateSystem(axes=[Axis()] * 3)}, "has 3 axes"),
    ],
)
def test_inconsistent_arguments_are_refused(
    kwargs: tx.Dict[str, tx.Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Affine.batch(_affines(), **kwargs)


def test_a_batch_needs_a_batch_dimension() -> None:
    with pytest.raises(ValueError, match="more than 2 dimensions"):
        Affine.batch(np.eye(3, 4))


@pytest.mark.parametrize("element_type", [Identity, BatchedAffine])
def test_only_matrix_and_vector_types_are_batched(
    element_type: tx.Type[Transformation],
) -> None:
    with pytest.raises(TypeError, match="matrix or vector transformations"):
        BatchedTransformation(element_type, np.zeros((2, 3, 4)))


def test_a_lazy_inverse_is_not_an_element_type() -> None:
    lazy = type(Affine(np.eye(3, 4)).inverse())
    with pytest.raises(TypeError, match="matrix or vector transformations"):
        BatchedTransformation(lazy, np.zeros((2, 3, 4)))


def test_a_batch_is_not_registered_as_a_kind() -> None:
    batch = Affine.batch(_affines())
    assert not isinstance(batch, kinds.Affine)
    assert not is_kind(batch, kinds.Affine)
    assert is_kind(batch, kinds.Transformation)


def test_axis_counts_reads_a_batch() -> None:
    batch = Affine.batch(np.zeros((2, 3, 2, 3)))
    assert axis_counts(batch) == (4, 4)


# --- elements ---------------------------------------------------------


def test_an_element_is_a_view_of_the_stack() -> None:
    stack = _affines()
    element = Affine.batch(stack)[3]
    assert type(element) is Affine
    np.testing.assert_array_equal(element.matrix, stack[3])
    assert np.shares_memory(element.matrix, stack)


def test_iteration_gives_the_elements_in_order() -> None:
    stack = _affines()
    elements = list(Affine.batch(stack))
    assert len(elements) == 5
    for k, element in enumerate(elements):
        np.testing.assert_array_equal(element.matrix, stack[k])


def test_a_slice_keeps_the_batch_axis_and_its_positions() -> None:
    batch = Affine.batch(_affines(), positions=[np.arange(5) * 2.5])
    part = batch[1:4]
    assert isinstance(part, BatchedAffine)
    assert part.batch_shape == (3,)
    np.testing.assert_array_equal(part.positions[0], [2.5, 5.0, 7.5])


def test_an_integer_removes_its_batch_axis_from_the_space() -> None:
    system = _system("ixyzt")
    batch = Scaling.batch(
        np.ones((2, 3, 3)),
        input_axes=[1, 2, 3],
        input=system,
        output=system,
    )
    part = batch[1]
    assert part.batch_shape == (3,)
    assert part.counts == (4, 4)
    assert part.input_axes == [0, 1, 2]
    assert [a.name for a in part.input.axes] == ["x", "y", "z", "t"]
    element = batch[1, 2]
    assert type(element) is Scaling
    assert [a.name for a in element.input.axes] == ["x", "y", "z"]


def test_an_element_takes_the_systems_of_its_axes() -> None:
    system = _system("xyzt")
    batch = Affine.batch(
        _affines(), input_axes=[0, 1, 2], input=system, output=system
    )
    assert [a.name for a in batch.element_input.axes] == ["x", "y", "z"]
    assert batch[0].output == batch.element_output


def test_an_open_system_gives_no_system_to_the_elements() -> None:
    system = CoordinateSystem(axes=[Axis(name="t"), ...])
    batch = Affine.batch(_affines(), input=system, output=system)
    assert batch.element_input is None
    assert batch[0].input is None


def test_a_key_with_too_many_indices_is_refused() -> None:
    with pytest.raises(IndexError, match="1 batch axes"):
        Affine.batch(_affines())[0, 0]


# --- inverse ----------------------------------------------------------


@pytest.mark.parametrize(
    "element_type, data",
    [
        (Affine, _affines()),
        (Linear, _affines()[..., :3]),
        (Rotation, _rotations()),
        (Scaling, np.linspace(1, 2, 6).reshape(2, 3)),
        (Translation, np.arange(6.0).reshape(2, 3)),
        (Permutation, np.array([[1, 2, 0], [2, 0, 1]])),
    ],
)
def test_the_inverse_is_the_batch_of_the_inverses(
    element_type: tx.Type[Transformation], data: np.ndarray
) -> None:
    batch = element_type.batch(data)
    inverse = batch.inverse()
    assert inverse.element_type is batch.element_type
    for k in range(batch.batch_shape[0]):
        expected = batch[k].inverse(compute=True).data
        np.testing.assert_allclose(inverse[k].data, expected, atol=1e-12)


def test_the_inverse_of_a_tangent_negates_it() -> None:
    batch = Affine.batch(_affines(), log=True)
    np.testing.assert_allclose(batch.inverse().data, -batch.data)


def test_the_inverse_swaps_the_axes_and_the_systems() -> None:
    source, target = _system("xyzt"), _system("tijk")
    batch = Affine.batch(
        _affines(),
        input_axes=[0, 1, 2],
        output_axes=[1, 2, 3],
        input=source,
        output=target,
    )
    inverse = batch.inverse()
    assert inverse.input_axes == [1, 2, 3]
    assert inverse.output_axes == [0, 1, 2]
    assert inverse.input == target and inverse.output == source


# --- stack ------------------------------------------------------------


def test_stack_builds_a_batch_from_its_elements() -> None:
    stack = _affines()
    system = _system("xyz")
    elements = [Affine(m, input=system, output=system) for m in stack]
    batch = BatchedTransformation.stack(elements, batch_axis=Axis(name="t"))
    assert isinstance(batch, BatchedAffine)
    np.testing.assert_array_equal(batch.data, stack)
    assert [a.name for a in batch.input.axes] == ["t", "x", "y", "z"]
    assert batch[2].input == system


@pytest.mark.parametrize(
    "elements, match",
    [
        ([], "at least one element"),
        ([Affine(np.eye(3, 4)), Linear(np.eye(3))], "same type"),
        ([Affine(np.eye(3, 4)), Affine(np.eye(2, 3))], "same shape"),
        (
            [Affine(np.eye(3, 4)), Affine(np.eye(3, 4), input=_system("xyz"))],
            "same input and output systems",
        ),
    ],
)
def test_stack_refuses_elements_that_differ(
    elements: tx.List[Transformation], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BatchedTransformation.stack(elements)


def test_the_inverse_of_a_directed_type_is_the_type_of_its_pair() -> None:
    from brainhops.io.transformations.base.affines import (
        LPSToVoxel,
        VoxelToLPS,
    )

    batch = VoxelToLPS.batch(_affines())
    assert batch.element_type is VoxelToLPS
    assert batch.inverse().element_type is LPSToVoxel
