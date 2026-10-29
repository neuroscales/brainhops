"""Tests of the compute rules of the batched transformations.

Every rule is checked against the element-by-element result, in both
forms of a batch. The general form is built from the elements of the
stacked form, so that the two forms hold the same maps.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import kinds
from brainhops.datamodel._transformations.batched import (
    match_positions,
    result_type,
)
from brainhops.datamodel._transformations.compute.prune import (
    read_pattern,
    restrict_block,
)
from brainhops.datamodel._transformations.compute.simplify import (
    SimplifyTable,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    BatchedTransformation,
    CartesianField,
    CoordinateField,
    Identity,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.errors import ConversionError

ANALYTIC = SimplifyTable.from_like("analytic")
NUMERIC = SimplifyTable.from_like({None: "numeric"})

# ----------------------------------------------------------------------
#   BATCHES
# ----------------------------------------------------------------------


def _rotations(n: int) -> np.ndarray:
    angles = np.linspace(0.1, 1.0, n)
    stack = np.zeros((n, 3, 3))
    stack[:, 0, 0] = stack[:, 1, 1] = np.cos(angles)
    stack[:, 0, 1], stack[:, 1, 0] = -np.sin(angles), np.sin(angles)
    stack[:, 2, 2] = 1
    return stack


def _stacked(name: str, n: int = 4, seed: int = 0) -> BatchedTransformation:
    # A stacked batch of `n` elements of three axes, whose batch axis is
    # the last axis of a space (x, y, z, t). The matrices have exact
    # zeros, which the numeric policy reads, and the elements differ
    # along the batch axis in their first row only.
    rng = np.random.default_rng(seed)
    axes = {"input_axes": [0, 1, 2]}
    if name in ("affine", "affine-log"):
        stack = np.tile(np.eye(3, 4), (n, 1, 1))
        stack[:, 0, 1] = 0.5
        stack[:, 0, 3] = rng.normal(size=n)
        stack[:, 0, 0] += np.arange(n) * 0.1
        stack[:, 1, 3] = 2.0
        if name == "affine-log":
            return Affine.batch(stack * 0.1, log=True, **axes)
        return Affine.batch(stack, **axes)
    if name == "linear":
        stack = np.tile(np.eye(3), (n, 1, 1))
        stack[:, 0, 2] = rng.normal(size=n)
        stack[:, 1, 0] = 0.3
        return Linear.batch(stack, **axes)
    if name == "rotation":
        return Rotation.batch(_rotations(n), **axes)
    if name in ("scaling", "scaling-log"):
        stack = np.tile([1.0, 2.0, 3.0], (n, 1))
        stack[:, 0] += np.arange(n)
        if name == "scaling-log":
            return Scaling.batch(np.log(stack), log=True, **axes)
        return Scaling.batch(stack, **axes)
    if name == "translation":
        stack = np.tile([1.0, 2.0, 3.0], (n, 1))
        stack[:, 0] += rng.normal(size=n)
        return Translation.batch(stack, **axes)
    if name == "permutation":
        stack = np.array([[1, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
        return Permutation.batch(stack[:n], **axes)
    raise ValueError(name)


_NAMES = [
    "affine",
    "affine-log",
    "linear",
    "rotation",
    "scaling",
    "scaling-log",
    "translation",
    "permutation",
]


def _general(batch: BatchedTransformation) -> BatchedTransformation:
    # The general form of a batch, which holds the elements of `batch`.
    return BatchedTransformation(
        batch.elements,
        input_axes=batch.input_axes,
        output_axes=batch.output_axes,
        positions=batch.positions,
        input=batch.input,
        output=batch.output,
    )


def _points(n_in: int, n: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, n_in))


def _apply(t: Transformation, points: np.ndarray) -> np.ndarray:
    # Apply a transformation to points, through its affine matrix when it
    # has one, and through the composers otherwise.
    if isinstance(t, Identity):
        return points
    try:
        matrix = np.asarray(t.to(Affine).matrix)
    except ConversionError:
        field = CoordinateField(field=points, store="values")
        return np.asarray(Sequence([field, t]).compute().field)
    return points @ matrix[:, :-1].T + matrix[:, -1]


def _element_values(
    batch: BatchedTransformation, points: np.ndarray
) -> np.ndarray:
    # The values of every element at the same points, element by
    # element, with one leading dimension per batch dimension.
    values = [_apply(e, points) for e in batch.elements.flat]
    return np.stack(values).reshape(batch.batch_shape + values[0].shape)


def _same_elements(
    a: BatchedTransformation, b: BatchedTransformation, seed: int = 0
) -> bool:
    points = _points(a.element_counts[0], seed=seed)
    left, right = _element_values(a, points), _element_values(b, points)
    return left.shape == right.shape and np.allclose(
        left, right, equal_nan=True
    )


def _system(names: str) -> CoordinateSystem:
    return CoordinateSystem(axes=[Axis(name=n) for n in names])


# ----------------------------------------------------------------------
#   POSITIONS
# ----------------------------------------------------------------------


def test_a_coordinate_computed_in_floating_point_matches_its_position() -> (
    None
):
    t0, tr = 0.3, 0.7
    positions = t0 + np.arange(10) * tr
    values = [t0 + k * tr for k in range(10)]
    assert match_positions(positions, values).tolist() == list(range(10))


def test_a_coordinate_between_or_outside_the_positions_matches_none() -> None:
    positions = [0.0, 1.0, 2.0]
    values = [0.5, 1.001, -1.0, 3.0, np.nan, np.inf]
    assert match_positions(positions, values).tolist() == [-1] * 6


def test_unsorted_positions_give_the_index_of_each_element() -> None:
    assert match_positions([2.0, 0.0, 1.0], [0.0, 1.0, 2.0]).tolist() == [
        1,
        2,
        0,
    ]


def test_a_single_position_matches_within_a_relative_tolerance() -> None:
    assert match_positions([1000.0], [1000.0001, 1000.01]).tolist() == [0, -1]
    assert match_positions([0.0], [1e-7, 1e-5]).tolist() == [0, -1]


def test_repeated_positions_are_refused() -> None:
    with pytest.raises(ValueError, match="distinct"):
        Scaling.batch(np.ones((3, 2)), positions=[[0.0, 1.0, 1.0]])


# ----------------------------------------------------------------------
#   ELEMENTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("general", [False, True])
def test_elements_are_asks_the_kind_of_every_element(general: bool) -> None:
    batch = _stacked("scaling")
    batch = _general(batch) if general else batch
    assert batch.elements_are(kinds.Affine)
    assert batch.elements_are("scaling")
    assert not batch.elements_are(kinds.Identity, compute=True)
    assert not batch.is_kind(kinds.Affine)
    ones = Scaling.batch(np.ones((3, 2)))
    assert ones.elements_are(kinds.Identity, compute=True)


def test_the_policy_of_a_batch_is_that_of_its_elements() -> None:
    table = SimplifyTable.from_like("numeric")
    assert table.resolve(_stacked("affine")) is table.resolve(Affine())
    field = CoordinateField(field=np.zeros((2, 2, 2, 3)))
    mixed = BatchedTransformation.stack([Affine(np.eye(3, 4)), field])
    assert table.resolve(mixed) is table.resolve(field)
    assert table.resolve(mixed) is not table.resolve(Affine())


def _grid(general: bool) -> BatchedTransformation:
    # A batch with two batch axes, of 2 and 3 elements, on (v, x, y, t),
    # whose elements are scalings of (x, y).
    scale = np.arange(1.0, 13.0).reshape(2, 3, 2)
    batch = Scaling.batch(
        scale,
        input_axes=[1, 2],
        positions=[[10.0, 20.0], [0.0, 0.5, 1.0]],
        input=_system("vxyt"),
        output=_system("vxyt"),
    )
    return _general(batch) if general else batch


@pytest.mark.parametrize("general", [False, True])
def test_select_with_every_batch_axis_gives_the_element(general: bool) -> None:
    batch = _grid(general)
    element = batch.select({0: 20.0, 3: 0.5})
    assert isinstance(element, Scaling)
    assert np.allclose(element.scale, [9.0, 10.0])
    assert element.input == _system("xy")


@pytest.mark.parametrize("general", [False, True])
def test_select_with_some_batch_axes_gives_a_smaller_batch(
    general: bool,
) -> None:
    batch = _grid(general)
    smaller = batch.select({3: 1.0})
    assert isinstance(smaller, BatchedTransformation)
    assert smaller.batch_shape == (2,)
    assert smaller.positions[0].tolist() == [10.0, 20.0]
    assert smaller.input == _system("vxy")
    assert np.allclose(smaller[1].scale, [11.0, 12.0])


@pytest.mark.parametrize("general", [False, True])
def test_select_declines_a_coordinate_that_matches_no_position(
    general: bool,
) -> None:
    batch = _grid(general)
    assert batch.select({0: 15.0}) is None
    assert batch.select({3: np.nan}) is None
    with pytest.raises(ValueError, match="batch axes"):
        batch.select({1: 0.0})


# ----------------------------------------------------------------------
#   KERNELS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", [ANALYTIC, NUMERIC], ids=["a", "n"])
@pytest.mark.parametrize("name", _NAMES)
def test_the_pattern_of_a_stack_bounds_that_of_its_elements(
    name: str, policy: SimplifyTable
) -> None:
    stacked = _stacked(name)
    block, reads_batch = stacked.element_pattern(policy)
    general, _ = _general(stacked).element_pattern(policy)
    # A stack reads its zeros on the whole stack, and the general form
    # reads each element, so the two give the same union.
    assert np.array_equal(block, general)
    if policy is ANALYTIC:
        assert reads_batch.all()
    else:
        # A row reads the batch axis when it differs between elements.
        matrices = np.stack([e.to(Affine).matrix for e in stacked])
        varies = (matrices != matrices[:1]).any(axis=(0, 2))
        assert np.array_equal(reads_batch[:, 0], varies)


def test_a_general_batch_reads_a_batch_axis_unless_its_elements_repeat() -> (
    None
):
    scaling = Scaling([1.0, 2.0])
    same = BatchedTransformation.stack([scaling, scaling])
    other = BatchedTransformation.stack([scaling, Scaling([1.0, 3.0])])
    assert not same.element_pattern(NUMERIC)[1].any()
    assert other.element_pattern(NUMERIC)[1].all()
    assert same.element_pattern(ANALYTIC)[1].all()


_BLOCKS = [([0, 1], [0, 1]), ([2], [2]), ([0, 1, 2], [0]), ([], [1])]


@pytest.mark.parametrize("block", _BLOCKS, ids=str)
@pytest.mark.parametrize("name", _NAMES)
def test_a_stack_is_cut_as_its_elements_are(
    name: str, block: tx.Tuple[tx.List[int], tx.List[int]]
) -> None:
    stacked = _stacked(name)
    inputs, outputs = block
    fields = {
        "input_axes": list(range(len(inputs))),
        "output_axes": list(range(len(outputs))),
    }
    cut = stacked.restrict_elements(inputs, outputs, NUMERIC, **fields)
    reference = _general(stacked).restrict_elements(
        inputs, outputs, NUMERIC, **fields
    )
    assert (cut is None) == (reference is None)
    if cut is not None:
        assert cut.batch_shape == stacked.batch_shape
        if inputs:
            assert _same_elements(cut, reference)


@pytest.mark.parametrize("inner", _NAMES)
@pytest.mark.parametrize("outer", _NAMES)
def test_two_stacks_compose_as_their_elements_do(
    outer: str, inner: str
) -> None:
    first, second = _stacked(inner, seed=1), _stacked(outer, seed=2)
    product = second.compose_with(first, input_axes=[0, 1, 2])
    reference = _general(second).compose_with(
        _general(first), input_axes=[0, 1, 2]
    )
    assert product.element_type is not None
    assert issubclass(
        product.element_type,
        result_type(second.element_type, first.element_type),
    )
    assert _same_elements(product, reference)


def test_a_batch_of_one_element_broadcasts() -> None:
    stacked = _stacked("affine")
    single = Scaling.batch(np.array([[2.0, 3.0, 4.0]]))
    product = single.compose_with(stacked, input_axes=[0, 1, 2])
    assert product.batch_shape == stacked.batch_shape
    reference = _general(single).compose_with(
        _general(stacked), input_axes=[0, 1, 2]
    )
    assert _same_elements(product, reference)


def test_result_type_follows_the_composers_of_the_elements() -> None:
    assert result_type(Rotation, Rotation) is Rotation
    assert result_type(Scaling, Permutation) is Linear
    assert result_type(Rotation, Linear) is Linear
    assert result_type(Scaling, Translation) is Affine
    assert result_type(Translation, Translation) is Translation


@pytest.mark.parametrize("name", _NAMES)
def test_a_stack_applies_to_points_as_its_elements_do(name: str) -> None:
    stacked = _stacked(name)
    points = _points(3, n=8)
    index = np.array([[0], [1], [2], [3], [3], [0], [-1], [2]])
    valid = (index >= 0).all(axis=1)
    values = stacked.apply_to_points(points, index, valid)
    reference = _general(stacked).apply_to_points(points, index, valid)
    assert np.allclose(values, reference, equal_nan=True)
    assert np.isnan(values[6]).all()
    expected = _apply(stacked[3], points[3:4])
    assert np.allclose(values[3], expected[0])


# ----------------------------------------------------------------------
#   PRUNING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("general", [False, True])
def test_the_pattern_of_a_batch_places_its_batch_axis(general: bool) -> None:
    batch = _stacked("scaling")
    batch = _general(batch) if general else batch
    # Only the scale of x changes from one element to the next, which a
    # stack reads row by row, and the general form element by element.
    expected = np.eye(4, dtype=bool)
    expected[0, 3] = True
    if general:
        expected[:3, 3] = True
    assert np.array_equal(read_pattern(batch, NUMERIC), expected)
    expected[:3, 3] = True
    assert np.array_equal(read_pattern(batch, ANALYTIC), expected)


def test_a_batch_axis_along_which_the_elements_are_equal_is_not_read() -> None:
    batch = Affine.batch(
        np.tile(np.eye(3, 4), (5, 1, 1)), input_axes=[1, 2, 3]
    )
    pattern = read_pattern(batch, NUMERIC)
    assert not pattern[1:, 0].any()
    assert pattern[1:, 0].sum() == 0 and pattern[0, 0]


_CUTS = [([0, 1, 3], [0, 1, 3]), ([0, 3], [0, 3]), ([2, 3], [2, 3])]


@pytest.mark.parametrize("block", _CUTS, ids=str)
@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("name", ["affine", "scaling", "permutation"])
def test_the_cut_of_a_batch_is_the_batch_of_the_cuts(
    name: str, general: bool, block: tx.Tuple[tx.List[int], tx.List[int]]
) -> None:
    batch = _stacked(name)
    batch = _general(batch) if general else batch
    inputs, outputs = block
    others = [i for i in range(4) if i not in inputs]
    if read_pattern(batch, NUMERIC)[np.ix_(outputs, others)].any():
        # The block is not closed, so no restrictor is asked for it.
        return
    cut = restrict_block(batch, inputs, outputs, NUMERIC)
    inner = [i for i in inputs if i != 3]
    for k, element in enumerate(batch):
        expected = restrict_block(element, inner, inner, NUMERIC)
        if expected is None:
            assert cut is None
            continue
        points = _points(len(inner))
        if isinstance(cut, Identity):
            # A block in which every element is the identity is cut into
            # the identity.
            assert np.allclose(_apply(expected, points), points)
            continue
        assert cut.batch_input_axes == [len(inner)]
        assert np.allclose(_apply(cut[k], points), _apply(expected, points))


@pytest.mark.parametrize("general", [False, True])
def test_the_cut_of_a_batch_declines_a_free_batch_axis(general: bool) -> None:
    batch = _stacked("scaling")
    batch = _general(batch) if general else batch
    assert restrict_block(batch, [1, 2], [1, 2], NUMERIC) is None
    assert restrict_block(batch, [1, 2, 3], [1, 2], NUMERIC) is None


def test_a_batch_of_identity_pieces_is_cut_into_the_identity() -> None:
    batch = _stacked("scaling")
    cut = restrict_block(batch, [1, 2, 3], [1, 2, 3], NUMERIC)
    assert isinstance(cut, BatchedTransformation)
    batch = Permutation.batch(np.array([[1, 0, 2], [0, 1, 2]]))
    cut = restrict_block(batch, [0, 3], [0, 3], NUMERIC)
    assert isinstance(cut, Identity)


@pytest.mark.parametrize("general", [False, True])
def test_simplify_drops_an_axis_that_nothing_reads_through_a_batch(
    general: bool,
) -> None:
    batch = _stacked("scaling")
    batch = _general(batch) if general else batch
    chain = Sequence([batch, Projection(dropped=[2])])
    pruned = chain.simplify("numeric")
    first, second = pruned.transformations
    assert isinstance(first, Projection) and list(first.dropped) == [2]
    assert isinstance(second, BatchedTransformation)
    assert second.counts == (3, 3)
    for k in range(batch.batch_shape[0]):
        assert np.allclose(second[k].to(Affine).matrix[0, 0], 1.0 + k)


@pytest.mark.parametrize("general", [False, True])
def test_the_factor_pass_keeps_a_batch_whole(general: bool) -> None:
    batch = _stacked("affine")
    batch = _general(batch) if general else batch
    system = CoordinateSystem().expand(5)
    chain = Sequence(
        [
            CartesianField(shape=(2, 3, 4, 4, 5)),
            SubspaceTransformation(
                batch, [0, 1, 2, 3], [0, 1, 2, 3], input=system, output=system
            ),
            SubspaceTransformation(
                Scaling([2.0]), [4], [4], input=system, output=system
            ),
        ]
    )
    nf = chain.factor("numeric")
    grid, first, second = nf.transformations
    assert first.transformation is batch
    assert list(first.input_axes) == [0, 1, 2, 3]
    assert list(second.input_axes) == [4]
