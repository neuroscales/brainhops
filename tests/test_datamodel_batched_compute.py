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
    apply_to_points,
    compose_with,
    element_pattern,
    match_positions,
    restrict_elements,
    result_type,
)
from brainhops.datamodel._transformations.compute.compose import compose
from brainhops.datamodel._transformations.compute.prune import (
    read_pattern,
    restrict_block,
)
from brainhops.datamodel._transformations.compute.simplify import (
    SimplifyTable,
)
from brainhops.datamodel._transformations.compute.utils import axis_counts
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    AffineExponential,
    BatchedAffine,
    BatchedScaling,
    BatchedTransformation,
    CartesianField,
    CoordinateField,
    DisplacementField,
    Identity,
    Linear,
    Permutation,
    Projection,
    Rotation,
    Scaling,
    ScalingExponential,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
)
from brainhops.errors import (
    CompositionError,
    ConversionError,
    RestrictionError,
)

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
    block, reads_batch = element_pattern(stacked, policy)
    general, _ = element_pattern(_general(stacked), policy)
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
    assert not element_pattern(same, NUMERIC)[1].any()
    assert element_pattern(other, NUMERIC)[1].all()
    assert element_pattern(same, ANALYTIC)[1].all()


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
    cut = restrict_elements(stacked, inputs, outputs, NUMERIC, **fields)
    reference = restrict_elements(
        _general(stacked), inputs, outputs, NUMERIC, **fields
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
    product = compose_with(second, first, input_axes=[0, 1, 2])
    reference = compose_with(
        _general(second), _general(first), input_axes=[0, 1, 2]
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
    product = compose_with(single, stacked, input_axes=[0, 1, 2])
    assert product.batch_shape == stacked.batch_shape
    reference = compose_with(
        _general(single), _general(stacked), input_axes=[0, 1, 2]
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
    values = apply_to_points(stacked, points, index, valid)
    reference = apply_to_points(_general(stacked), points, index, valid)
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


# ----------------------------------------------------------------------
#   SIMPLIFICATION
# ----------------------------------------------------------------------


def _series_batch(
    name: str, general: bool, seed: int = 0
) -> BatchedTransformation:
    # A batch of elements of type `name` at the times 0, 2, 4 and 6 of
    # the last axis of (x, y, z, t).
    batch = _stacked(name, seed=seed)
    batch = BatchedTransformation(
        element_type=batch.element_type,
        data=batch.data,
        input_axes=[0, 1, 2],
        positions=[[0.0, 2.0, 4.0, 6.0]],
    )
    return _general(batch) if general else batch


def _section(batch: BatchedTransformation, at: float) -> Sequence:
    # The element of a batch on (x, y, z, t) at time `at`, as a chain.
    return Sequence(
        [Projection(created=[3], at=[at]), batch, Projection(dropped=[3])]
    )


@pytest.mark.parametrize("general", [False, True])
def test_a_section_of_a_batch_simplifies_to_its_element(general: bool) -> None:
    batch = _series_batch("scaling", general)
    element = _section(batch, 4.0).simplify()
    assert isinstance(element, Scaling)
    assert np.allclose(element.scale, batch[2].scale)
    if general:
        assert element is batch[2]


@pytest.mark.parametrize("general", [False, True])
def test_a_free_batch_axis_raises_when_the_section_is_computed(
    general: bool,
) -> None:
    chain = _section(_series_batch("scaling", general), np.nan)
    assert isinstance(chain.simplify(), Sequence)
    with pytest.raises(RestrictionError):
        chain.compute()


@pytest.mark.parametrize("general", [False, True])
def test_a_coordinate_between_two_positions_selects_nothing(
    general: bool,
) -> None:
    batch = _series_batch("scaling", general)
    chain = _section(batch, 3.0).simplify()
    assert any(t is batch or isinstance(t, type(batch)) for t in chain)


@pytest.mark.parametrize("general", [False, True])
def test_a_partial_section_gives_a_smaller_batch(general: bool) -> None:
    batch = _grid(general)
    chain = Sequence(
        [Projection(created=[0], at=[20.0]), batch, Projection(dropped=[0])]
    ).simplify()
    assert isinstance(chain, BatchedTransformation)
    assert chain.batch_shape == (3,)
    assert np.allclose(chain[1].scale, batch[1, 1].scale)


def test_a_general_batch_of_one_type_becomes_a_stack() -> None:
    general = _general(_stacked("affine"))
    stacked = general.simplify()
    assert isinstance(stacked, BatchedAffine)
    assert _same_elements(stacked, general)
    assert stacked.simplify() is stacked


def test_the_element_type_stacks_or_widens_a_batch() -> None:
    general = _general(_stacked("scaling"))
    assert isinstance(general.to(element_type=Scaling), BatchedScaling)
    widened = general.to(element_type=Scaling).to(element_type=Affine)
    assert isinstance(widened, BatchedAffine)
    assert _same_elements(widened, general)
    with pytest.raises(ConversionError):
        widened.to(element_type=Scaling)
    with pytest.raises(ConversionError):
        general.to(element_type=Sequence)


def test_elements_with_different_systems_are_not_stacked() -> None:
    elements = [
        Scaling([1.0, 2.0], input=_system("xy"), output=_system("xy")),
        Scaling([1.0, 2.0], input=_system("uv"), output=_system("uv")),
    ]
    batch = BatchedTransformation(elements)
    with pytest.raises(ConversionError, match="systems"):
        batch.to(element_type=Scaling)
    assert batch.simplify() is batch


def test_a_general_batch_cancels_with_its_inverse() -> None:
    batch = _general(_stacked("affine"))
    assert isinstance(Sequence([batch, batch.inverse()]).simplify(), Identity)
    assert isinstance(Sequence([batch.inverse(), batch]).simplify(), Identity)


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_axis_of_length_one_is_passed_through(general: bool) -> None:
    batch = Affine.batch(_stacked("affine").data[:1], input_axes=[0, 1, 2])
    batch = _general(batch) if general else batch
    simplified = batch.simplify()
    assert isinstance(simplified, SubspaceTransformation)
    assert list(simplified.input_axes) == [0, 1, 2]
    assert np.allclose(simplified.transformation.matrix, batch[0].matrix)
    assert simplified.simplify() is simplified


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_axis_along_which_nothing_changes_is_removed(
    general: bool,
) -> None:
    scale = np.tile([[[1.0, 2.0]], [[3.0, 4.0]]], (1, 3, 1))
    batch = Scaling.batch(scale, input_axes=[1, 2])
    batch = _general(batch) if general else batch
    if general:
        # The general form tells equal elements by their identity.
        elements = batch.elements
        elements[:, 1:] = elements[:, :1]
        batch = BatchedTransformation(elements, input_axes=[1, 2])
    assert batch.simplify("analytic") is batch or general
    simplified = batch.simplify("numeric")
    assert isinstance(simplified, SubspaceTransformation)
    assert list(simplified.input_axes) == [0, 1, 2]
    inner = simplified.transformation
    assert inner.batch_shape == (2,)
    assert np.allclose(inner[1].scale, [3.0, 4.0])
    assert simplified.simplify("numeric") is simplified


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_of_identities_is_the_identity(general: bool) -> None:
    batch = Scaling.batch(np.ones((3, 2)))
    batch = _general(batch) if general else batch
    assert isinstance(batch.simplify("numeric"), Identity)


# ----------------------------------------------------------------------
#   COMPOSITION
# ----------------------------------------------------------------------


def _apply_batch(
    batch: BatchedTransformation, points: np.ndarray
) -> np.ndarray:
    # Apply a batch to points by hand: each point reaches the element at
    # its batch coordinates and keeps these coordinates.
    n_out = batch.counts[1]
    values = np.full((len(points), n_out), np.nan)
    values[:, batch.batch_output_axes] = points[:, batch.batch_input_axes]
    for p, point in enumerate(points):
        key = []
        for axis, positions in zip(batch.batch_input_axes, batch.positions):
            key.append(int(match_positions(positions, [point[axis]])[0]))
        if min(key) < 0:
            continue
        element = batch[tuple(key)]
        inner = _apply(element, point[batch.input_axes][None])
        values[p, batch.output_axes] = inner[0]
    return values


def _reference(
    chain: tx.List[Transformation], points: np.ndarray
) -> np.ndarray:
    # Apply a chain to points, element by element. A projection that
    # does not state its number of axes takes it from the points.
    for t in chain:
        if isinstance(t, BatchedTransformation):
            points = _apply_batch(t, points)
            continue
        if isinstance(t, Projection) and t.input is None:
            t = t.to(input=CoordinateSystem().expand(points.shape[1]))
        points = _apply(t, points)
    return points


def _series_points(n: int = 12, seed: int = 0) -> np.ndarray:
    # Points of (x, y, z, t) whose times are the positions of `_series`.
    points = _points(4, n=n, seed=seed)
    points[:, 3] = np.arange(n) % 4 * 2.0
    return points


def _matching(chain: tx.List[Transformation], result: Transformation) -> bool:
    points = _series_points()
    expected = _reference(chain, points)
    got = _reference([result], points)
    return np.allclose(got, expected, equal_nan=True)


# A spatial affine of (x, y, z, t) that carries t through, and the same
# affine of (t, x, y, z), whose time comes first.
_SPATIAL = np.array(
    [
        [2.0, 0.5, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 2.0],
        [0.0, 0.3, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)
_TIME_FIRST = np.array(
    [
        [0.0, 2.0, 0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 0.3, 3.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("name", ["affine", "scaling", "permutation"])
def test_a_transformation_of_the_elements_composes_before_a_batch(
    name: str, general: bool
) -> None:
    batch = _series_batch(name, general)
    spatial = Affine(_SPATIAL)
    product = compose(batch, spatial)
    assert isinstance(product, BatchedTransformation)
    assert (product.element_type is None) == general
    assert _matching([spatial, batch], product)
    for k in range(4):
        expected = compose(batch[k], Affine(_SPATIAL[:3, [0, 1, 2, 4]]))
        points = _points(3)
        assert np.allclose(
            _apply(product[k], points), _apply(expected, points)
        )


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("name", ["affine", "scaling", "translation"])
def test_a_transformation_of_the_elements_composes_after_a_batch(
    name: str, general: bool
) -> None:
    batch = _series_batch(name, general)
    spatial = Affine(_SPATIAL)
    product = compose(spatial, batch)
    assert isinstance(product, BatchedTransformation)
    assert _matching([batch, spatial], product)


@pytest.mark.parametrize("general", [False, True])
def test_a_neighbour_may_move_the_batch_axis(general: bool) -> None:
    batch = _series_batch("affine", general)
    moved = Affine(_TIME_FIRST)
    product = compose(batch, moved)
    assert product.batch_input_axes == [0]
    assert product.input_axes == [1, 2, 3]
    points = _series_points()[:, [3, 0, 1, 2]]
    expected = _reference([moved, batch], points)
    assert np.allclose(_reference([product], points), expected)


@pytest.mark.parametrize("general", [False, True])
def test_a_neighbour_that_scales_the_batch_axis_stays_beside_it(
    general: bool,
) -> None:
    batch = _series_batch("affine", general)
    scaled = Affine(np.diag([1.0, 1.0, 1.0, 2.0, 1.0])[:4])
    with pytest.raises(CompositionError):
        compose(batch, scaled)
    with pytest.raises(CompositionError):
        compose(scaled, batch)
    first, second = Sequence([scaled, batch]).compute()
    assert isinstance(first, Affine)
    assert isinstance(second, BatchedTransformation)


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("inner", ["affine", "scaling", "permutation"])
@pytest.mark.parametrize("outer", ["affine", "linear", "translation"])
def test_two_batches_compose_element_by_element(
    outer: str, inner: str, general: bool
) -> None:
    first = _series_batch(inner, general)
    second = _series_batch(outer, general, seed=3)
    product = compose(second, first)
    assert isinstance(product, BatchedTransformation)
    assert _matching([first, second], product)


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_axis_of_length_one_is_broadcast(general: bool) -> None:
    batch = _series_batch("affine", general)
    single = Scaling.batch(
        np.array([[2.0, 3.0, 4.0]]), input_axes=[0, 1, 2], positions=[[9.0]]
    )
    product = compose(single, batch)
    assert product.batch_shape == (4,)
    assert product.positions[0].tolist() == [0.0, 2.0, 4.0, 6.0]
    for k in range(4):
        points = _points(3)
        expected = _apply(single[0], _apply(batch[k], points))
        assert np.allclose(_apply(product[k], points), expected)


def test_two_batches_at_different_positions_do_not_compose() -> None:
    batch = _series_batch("affine", False)
    other = BatchedTransformation(
        element_type=Affine,
        data=batch.data,
        input_axes=[0, 1, 2],
        positions=[[0.0, 1.0, 2.0, 3.0]],
    )
    with pytest.raises(CompositionError, match="positions"):
        compose(other, batch)


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_applied_to_points_selects_their_elements(
    general: bool,
) -> None:
    batch = _series_batch("affine", general)
    points = _series_points()
    points[[1, 5], 3] = [1.0, 7.5]
    field = CoordinateField(field=points.reshape(3, 4, 4))
    result = compose(batch, field)
    assert isinstance(result, CoordinateField)
    values = np.asarray(result.field).reshape(-1, 4)
    assert np.isnan(values[[1, 5], :3]).all()
    assert values[[1, 5], 3].tolist() == [1.0, 7.5]
    assert np.allclose(values, _apply_batch(batch, points), equal_nan=True)


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_applied_to_displacements_gives_displacements(
    general: bool,
) -> None:
    batch = _series_batch("translation", general)
    field = DisplacementField(field=np.zeros((2, 2, 2, 4, 4)))
    result = compose(batch, field)
    assert isinstance(result, DisplacementField)
    grid = np.asarray(field.to(CoordinateField).field)
    expected = _apply_batch(batch, grid.reshape(-1, 4)).reshape(grid.shape)
    got = np.asarray(result.field) + grid
    assert np.allclose(got, expected, equal_nan=True)


@pytest.mark.parametrize("general", [False, True])
def test_mode_affine_composes_on_each_side_of_a_batch(general: bool) -> None:
    batch = _series_batch("affine", general)
    spatial = Affine(_SPATIAL)
    chain = Sequence([spatial, spatial, batch, spatial])
    computed = chain.compute(mode="affine")
    first, middle, last = computed
    assert isinstance(middle, BatchedTransformation)
    assert _same_elements(middle, batch)
    assert isinstance(first, Affine) and isinstance(last, Affine)


@pytest.mark.parametrize("general", [False, True])
def test_mode_true_composes_a_batch_with_its_neighbours(general: bool) -> None:
    batch = _series_batch("affine", general)
    spatial = Affine(_SPATIAL)
    chain = [spatial, batch, spatial]
    computed = Sequence(chain).compute()
    assert isinstance(computed, BatchedAffine)
    assert _matching(chain, computed)
    assert isinstance(Sequence(chain).compute(mode="batch"), Sequence)


def test_runs_of_batches_compose_under_the_batch_mode() -> None:
    batch = _series_batch("affine", False)
    computed = Sequence([batch, batch]).compute(mode="batch")
    assert isinstance(computed, BatchedAffine)
    assert _matching([batch, batch], computed)


@pytest.mark.parametrize("general", [False, True])
def test_the_composers_of_a_batch_do_not_tie(general: bool) -> None:
    batch = _series_batch("affine", general)
    chain = Sequence([Affine(_SPATIAL), Scaling([1.0, 2.0, 3.0, 1.0])])
    for product in (
        compose(batch, chain),
        compose(chain, batch),
        compose(batch, batch),
        compose(batch, CoordinateField(field=_series_points())),
    ):
        assert isinstance(product, Transformation)


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_after_the_creation_of_its_batch_axis_is_selected(
    general: bool,
) -> None:
    batch = _series_batch("affine", general)
    creation = Affine(np.vstack([np.eye(3, 4), [0.0, 0.0, 0.0, 4.0]]))
    product = compose(batch, creation)
    assert isinstance(product, Affine)
    points = _points(3)
    expected = np.hstack([_apply(batch[2], points), np.full((6, 1), 4.0)])
    assert np.allclose(_apply(product, points), expected)


# ----------------------------------------------------------------------
#   CHAINS
# ----------------------------------------------------------------------

_TIMES = [0.0, 2.0, 4.0, 6.0]


@pytest.mark.parametrize("general", [False, True])
def test_a_volume_of_a_series_has_the_normal_form_of_one_volume(
    general: bool,
) -> None:
    # The grid of one volume of a series, at index 2, is mapped to world
    # coordinates and time, through the motion of that volume, and back
    # to the voxels of the series. The chain reduces to the factors of
    # the affine of that volume, and a projection that creates the time
    # axis of the series at the volume index.
    grid_to_world = np.eye(4, 5)
    grid_to_world[:3, :3] = np.diag([2.0, 2.0, 3.0])
    grid_to_world[:3, 4] = [1.0, -2.0, 0.5]
    grid_to_world[3, 3:] = [2.0, 0.0]
    world = Affine(grid_to_world)
    batch = _series_batch("affine", general)
    chain = Sequence(
        [
            CartesianField(shape=(4, 5, 6)),
            Projection(created=[3], at=[2.0]),
            world,
            batch,
            world.inverse(),
        ]
    )
    nf = chain.compute(mode="affine", factor=True, simplify="numeric")
    grid, *factors, creation = nf
    assert isinstance(grid, CartesianField)
    # The motion of the volume mixes x and y, and leaves z alone, so that
    # the factor of z is the identity, which the factor pass drops.
    assert [list(f.input_axes) for f in factors] == [[0, 1]]
    assert all(isinstance(f, SubspaceTransformation) for f in factors)
    assert isinstance(creation, Projection)
    assert list(creation.created) == [3] and list(creation.at) == [2.0]
    points = _points(3)
    expected = _reference(chain.transformations[1:], points)
    assert np.allclose(_reference(nf.transformations[1:], points), expected)


def _random_series_element(rng: np.random.Generator) -> Transformation:
    # Draw an element of (x, y, z, t) that carries t through, or a batch
    # along t. Matrices have random exact zeros, which the numeric policy
    # reads.
    kind = int(rng.integers(6))
    if kind == 0:
        matrix = np.eye(4, 5)
        matrix[:3, :3] += rng.normal(size=(3, 3)) * (rng.random((3, 3)) < 0.5)
        matrix[:3, 4] = rng.normal(size=3)
        return Affine(matrix)
    if kind == 1:
        return Scaling(np.append(rng.normal(size=3) + 2.0, 1.0))
    if kind == 2:
        return Permutation(np.append(rng.permutation(3), 3))
    if kind == 3:
        return Affine(np.eye(4, 5) + 0.1 * np.eye(4, 5)).inverse()
    name = ["affine", "scaling", "permutation", "linear"][int(rng.integers(4))]
    return _series_batch(
        name, bool(rng.integers(2)), seed=int(rng.integers(9))
    )


def _random_series_chain(seed: int) -> tx.Tuple[tx.List[Transformation], int]:
    # Draw a chain of (x, y, z, t), which may be one volume of the series,
    # between a creation of t at a time and the projection that drops t.
    rng = np.random.default_rng(seed)
    chain = [_random_series_element(rng) for _ in range(rng.integers(2, 6))]
    if rng.random() < 0.5:
        time = _TIMES[int(rng.integers(4))]
        chain = [
            Projection(created=[3], at=[time]),
            *chain,
            Projection(dropped=[3]),
        ]
        return chain, 3
    return chain, 4


def _chain_points(n_in: int, seed: int = 0) -> np.ndarray:
    points = _points(n_in, n=8, seed=seed)
    if n_in == 4:
        points[:, 3] = np.array(_TIMES * 2)
    return points


def _same_chain_map(a: tx.List, b: tx.List, n_in: int, seed: int) -> bool:
    points = _chain_points(n_in, seed)
    left = _reference(_flat(a), points)
    right = _reference(_flat(b), points)
    return left.shape == right.shape and np.allclose(
        left, right, equal_nan=True
    )


def _flat(t: tx.Any) -> tx.List[Transformation]:
    leaves = t if isinstance(t, list) else [t]
    out: tx.List[Transformation] = []
    for leaf in leaves:
        if isinstance(leaf, Sequence):
            out.extend(_flat(list(leaf.transformations)))
        elif isinstance(leaf, Identity):
            continue
        else:
            out.append(leaf)
    return out


def _structure(t: tx.Any) -> tx.List[str]:
    return [type(leaf).__name__ for leaf in _flat(t)]


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
@pytest.mark.parametrize("seed", range(16))
def test_random_series_chains_keep_their_map_when_simplified(
    seed: int, policy: str
) -> None:
    chain, n_in = _random_series_chain(seed)
    simplified = Sequence(chain).simplify(policy)
    assert len(_flat(simplified)) <= len(chain)
    assert _same_chain_map(chain, _flat(simplified), n_in, seed)
    again = Sequence(_flat(simplified) or [Identity()]).simplify(policy)
    assert _structure(again) == _structure(simplified)


@pytest.mark.parametrize("mode", [True, "batch"])
@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("policy", ["analytic", "numeric"])
@pytest.mark.parametrize("seed", range(16))
def test_compute_is_idempotent_on_random_series_chains(
    seed: int, policy: str, factor: bool, mode: tx.Any
) -> None:
    chain, n_in = _random_series_chain(seed)
    options = {"simplify": policy, "factor": factor}
    once = Sequence(chain).compute(mode, **options)
    assert _same_chain_map(chain, _flat(once), n_in, seed)
    twice = Sequence(_flat(once) or [Identity()]).compute(mode, **options)
    assert _structure(twice) == _structure(once)
    assert _same_chain_map(_flat(once), _flat(twice), n_in, seed)


# ----------------------------------------------------------------------
#   REVIEW
# ----------------------------------------------------------------------


def test_the_elements_of_a_batch_of_chains_keep_their_systems() -> None:
    # A chain that the batch selects keeps the systems that it declares.
    chains = [
        Sequence(
            [Scaling([1.0 + k, 2.0, 3.0]), Translation([k, 0.0, 0.0])],
            input=_system("xyz"),
            output=_system("abc"),
        )
        for k in range(3)
    ]
    batch = BatchedTransformation(
        chains,
        input_axes=[1, 2, 3],
        input=_system("txyz"),
        output=_system("tabc"),
    )
    section = Sequence(
        [Projection(created=[0], at=[1.0]), batch, Projection(dropped=[0])]
    )
    for result in (section.compute(), section.simplify()):
        assert result.input == _system("xyz")
        assert result.output == _system("abc")
        points = _points(3)
        assert np.allclose(_apply(result, points), _apply(chains[1], points))


def test_compose_selects_the_element_after_a_creation() -> None:
    batch = _series_batch("affine", False)
    creation = Projection(created=[3], at=[4.0], input=_system("xyz"))
    product = compose(batch, creation)
    assert isinstance(product, Affine)
    points = _points(3)
    expected = np.hstack([_apply(batch[2], points), np.full((6, 1), 4.0)])
    assert np.allclose(_apply(product, points), expected)


def test_a_neighbour_with_other_numbers_of_axes_is_refused() -> None:
    batch = _series_batch("affine", False)
    with pytest.raises(CompositionError):
        compose(Affine(np.eye(3, 4)), batch)
    with pytest.raises(CompositionError):
        compose(batch, Affine(np.eye(3, 4)))


def test_simplifying_a_stack_creates_no_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = np.tile(np.eye(3, 4), (50, 1, 1))
    stack[:, 0, 3] = np.arange(50)
    batch = Affine.batch(stack, input_axes=[0, 1, 2])
    # The element that stands for the others is created once.
    batch.simplify()
    created = []
    element = type(batch)._element

    def counted(self: tx.Any, key: tx.Any) -> Transformation:
        created.append(key)
        return element(self, key)

    monkeypatch.setattr(type(batch), "_element", counted)
    for policy in ("analytic", "numeric", {None: "numeric"}):
        assert batch.simplify(policy) is batch
    assert not batch.elements_are(kinds.Identity, compute=True)
    assert batch.elements_are(kinds.Affine)
    assert created == []


@pytest.mark.parametrize("general", [False, True])
def test_a_batch_with_open_systems_keeps_its_numbers_of_axes(
    general: bool,
) -> None:
    batch = Affine.batch(
        _stacked("affine").data[:1],
        input_axes=[0, 1, 2],
        input=CoordinateSystem(),
        output=CoordinateSystem(),
    )
    batch = _general(batch) if general else batch
    simplified = batch.simplify()
    assert isinstance(simplified, SubspaceTransformation)
    assert axis_counts(simplified) == (4, 4)


@pytest.mark.parametrize("general", [False, True])
def test_a_spatial_neighbour_composes_with_two_batch_axes(
    general: bool,
) -> None:
    batch = _grid(general)
    spatial = np.eye(4, 5)
    spatial[1:3, 1:3] = [[2.0, 0.5], [0.0, 3.0]]
    spatial[1:3, 4] = [1.0, -1.0]
    product = compose(batch, Affine(spatial))
    assert isinstance(product, BatchedTransformation)
    assert product.batch_shape == (2, 3)
    points = _points(4, n=6)
    points[:, 0] = [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
    points[:, 3] = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    expected = _reference([Affine(spatial), batch], points)
    assert np.allclose(_reference([product], points), expected)


@pytest.mark.parametrize("general", [False, True])
def test_a_permutation_that_moves_the_batch_axis_composes(
    general: bool,
) -> None:
    batch = _series_batch("scaling", general)
    moved = Permutation([1, 2, 3, 0])
    product = compose(batch, moved)
    assert product.batch_input_axes == [0]
    points = _series_points()[:, [3, 0, 1, 2]]
    expected = _reference([moved, batch], points)
    assert np.allclose(_reference([product], points), expected)
    after = compose(Permutation([3, 0, 1, 2]), batch)
    assert after.batch_output_axes == [0]
    points = _series_points()
    expected = _reference([batch, Permutation([3, 0, 1, 2])], points)
    assert np.allclose(_reference([after], points), expected)


def _fields(n: int = 4) -> tx.List[Transformation]:
    rng = np.random.default_rng(1)
    return [
        DisplacementField(field=rng.normal(size=(4, 5, 6, 3)) * 0.1)
        for _ in range(n)
    ]


@pytest.mark.parametrize("kind", ["chains", "fields"])
def test_a_general_batch_applies_each_element_to_its_points(
    kind: str,
) -> None:
    if kind == "chains":
        elements = [
            Sequence([Scaling([1.0 + k, 2.0, 3.0]), Affine(_SPATIAL[:3, 1:])])
            for k in range(4)
        ]
    else:
        elements = _fields()
    batch = BatchedTransformation(
        elements, input_axes=[0, 1, 2], positions=[_TIMES]
    )
    points = _series_points()
    points[:, :3] = np.abs(points[:, :3]) + 1.0
    result = compose(batch, CoordinateField(field=points))
    expected = _apply_batch(batch, points)
    assert np.allclose(np.asarray(result.field), expected, equal_nan=True)


def test_a_subspace_of_a_batch_of_fields_interpolates() -> None:
    batch = BatchedTransformation(_fields(), input_axes=[0, 1, 2])
    system = CoordinateSystem().expand(5)
    subspace = SubspaceTransformation(
        batch, [0, 1, 2, 3], [0, 1, 2, 3], input=system, output=system
    )
    with pytest.raises(CompositionError, match="wraps a field"):
        compose(Affine(np.eye(5, 6)), subspace)


def test_elements_are_on_elements_of_several_types() -> None:
    batch = BatchedTransformation.stack(
        [Scaling([1.0, 2.0]), Affine(np.eye(2, 3)), Translation([1.0, 0.0])]
    )
    assert batch.elements_are(kinds.Affine)
    assert not batch.elements_are("scaling")
    assert not batch.elements_are(kinds.Identity, compute=True)


def test_tangent_elements_stack_and_widen() -> None:
    logs = [np.diag([0.1 * k, 0.2, 0.0]) for k in range(3)]
    affines = [
        AffineExponential(np.hstack([m, np.ones((3, 1))])) for m in logs
    ]
    stacked = BatchedTransformation.stack(affines).to(
        element_type=AffineExponential
    )
    assert stacked.element_type is AffineExponential
    assert np.allclose(stacked.data, np.stack([a.data for a in affines]))
    scales = BatchedTransformation.stack(
        [Scaling([0.1 * k, 0.2], log=True) for k in range(3)]
    ).to(element_type=ScalingExponential)
    widened = scales.to(element_type=Linear)
    assert widened.element_type is Linear
    for k in range(3):
        assert np.allclose(widened[k].matrix, np.diag(np.exp([0.1 * k, 0.2])))
    assert _same_elements(widened, scales)
