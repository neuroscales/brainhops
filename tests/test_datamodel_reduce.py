"""Tests of the reduction of the axes that a chain does not need."""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops._core.affines import matmul
from brainhops.datamodel._transformations.compute import reduce
from brainhops.datamodel._transformations.compute.factor import (
    factor_sequence,
)
from brainhops.datamodel._transformations.compute.reduce import (
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
    CartesianField,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    Permutation,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)
from brainhops.errors import RestrictionError

ANALYTIC = SimplifyTable.from_like("analytic")
NUMERIC = SimplifyTable.from_like("numeric")

# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _dense(t: object, n_in: int) -> np.ndarray:
    # Return the affine matrix of an element that reads `n_in` axes,
    # without its last row. Every type is written out by hand, so that
    # the expected maps do not depend on the code under test.
    if isinstance(t, Projection):
        dropped = [int(i) for i in t.dropped]
        created = [int(o) for o in t.created]
        n_out = n_in - len(dropped) + len(created)
        kept = [i for i in range(n_in) if i not in dropped]
        rows = [o for o in range(n_out) if o not in created]
        matrix = np.zeros((n_out, n_in + 1))
        matrix[rows, kept] = 1.0
        matrix[created, -1] = list(t.at)
        return matrix
    if isinstance(t, Identity):
        return np.eye(n_in, n_in + 1)
    if isinstance(t, Sequence):
        return _product(t, n_in)
    if isinstance(t, SubspaceTransformation):
        in_axes = [int(i) for i in t.input_axes]
        out_axes = [int(i) for i in t.output_axes]
        n_out = n_in - len(in_axes) + len(out_axes)
        inner = t.transformation
        if inner is None:
            block = np.eye(len(in_axes), len(in_axes) + 1)
        else:
            block = _dense(inner, len(in_axes))
        matrix = np.zeros((n_out, n_in + 1))
        pass_in = [i for i in range(n_in) if i not in in_axes]
        pass_out = [o for o in range(n_out) if o not in out_axes]
        matrix[pass_out, pass_in] = 1.0
        matrix[np.ix_(out_axes, in_axes)] = block[:, :-1]
        matrix[out_axes, -1] = block[:, -1]
        return matrix
    if isinstance(t, Inverse):
        forward = _dense(t.forward, n_in)
        square = np.vstack([forward, np.eye(1, n_in + 1, n_in)])
        return np.linalg.inv(square)[:-1]
    if isinstance(t, Translation):
        shift = np.asarray(t.translation, dtype=float)
        return np.hstack([np.eye(n_in), shift[:, None]])
    if isinstance(t, Scaling):
        scale = np.asarray(t.scale, dtype=float)
        return np.hstack([np.diag(scale), np.zeros((n_in, 1))])
    if isinstance(t, Permutation):
        matrix = np.zeros((n_in, n_in + 1))
        matrix[list(range(n_in)), [int(i) for i in t.permutation]] = 1.0
        return matrix
    if isinstance(t, Linear):
        matrix = np.asarray(t.matrix, dtype=float)
        return np.hstack([matrix, np.zeros((matrix.shape[0], 1))])
    if isinstance(t, Affine):
        return np.asarray(t.matrix, dtype=float)
    raise TypeError(type(t))


def _product(chain: object, n_in: int) -> np.ndarray:
    # Compose the elements of a chain by hand, in the order of
    # application.
    leaves = _leaves(chain)
    product = np.eye(n_in, n_in + 1)
    for leaf in leaves:
        dense = _dense(leaf, product.shape[0])
        product = matmul(dense, product, zero_cancels_nan=True)
    return product


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :-1].T + matrix[:, -1]


def _same_map(a: object, b: object, n_in: int, seed: int = 0) -> bool:
    points = np.random.default_rng(seed).normal(size=(16, n_in))
    left = _apply(_product(a, n_in), points)
    right = _apply(_product(b, n_in), points)
    return left.shape == right.shape and np.allclose(
        left, right, equal_nan=True
    )


def _leaves(t: object) -> list:
    if isinstance(t, list):
        return t
    return list(getattr(t, "transformations", None) or [t])


def _restricted(affine: Affine, at: float) -> Sequence:
    # The affine of (x, y, t) read from (x, y) to (x, y), with t at
    # `at`.
    return Sequence(
        [Projection(created=[2], at=[at]), affine, Projection(dropped=[2])]
    )


_A1 = Affine(matrix=np.array([[2, 0, 0, 1], [0, 3, 0, 2], [0, 0, 1, 5.0]]))
_A2 = Affine(matrix=np.array([[2, 0, 0.5, 1], [0, 3, 0, 2], [0, 0, 1, 5.0]]))
_A3 = Affine(matrix=np.array([[2, 0, 0, 1], [0, 3, 0, 2], [0, 1, 1, 5.0]]))

_SHEAR = Linear(matrix=np.array([[1.0, 1, 1], [0, 1, 1], [0, 0, 1]]))


# ----------------------------------------------------------------------
#   THE RESTRICTION TABLE
# ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
@pytest.mark.parametrize(
    "affine, expected",
    [
        (_A1, [[2.0, 0, 1], [0, 3, 2]]),
        (_A2, [[2.0, 0, 3], [0, 3, 2]]),
        (_A3, [[2.0, 0, 1], [0, 3, 2]]),
    ],
    ids=["A1", "A2", "A3"],
)
def test_the_restriction_table_comes_out_of_compute(
    affine: Affine, expected: list, policy: str
) -> None:
    # The projection that drops t' cuts the affine down to x' and y'.
    # Simplification computes nothing, so the embedding of t stays
    # before the affine when the affine may read t, and compute folds
    # its value into the translation.
    chain = _restricted(affine, 4.0)
    reduced = chain.simplify(policy)
    *before, cut = _leaves(reduced)
    assert isinstance(cut, Affine) and cut.matrix.shape[0] == 2
    reads_t = policy == "analytic" or affine is _A2
    assert len(before) == (1 if reads_t else 0)
    assert np.allclose(chain.compute(simplify=policy).matrix, expected)


@pytest.mark.parametrize("affine", [_A1, _A3], ids=["A1", "A3"])
def test_a_free_axis_that_nothing_reads_cancels_under_numeric(
    affine: Affine,
) -> None:
    reduced = _restricted(affine, np.nan).simplify("numeric")
    assert isinstance(reduced, Affine)
    assert np.allclose(reduced.matrix, [[2.0, 0, 1], [0, 3, 2]])


@pytest.mark.parametrize("affine", [_A1, _A3], ids=["A1", "A3"])
def test_a_free_axis_stays_under_analytic_and_compute_removes_it(
    affine: Affine,
) -> None:
    # An affine reads as fully coupled under the analytic policy, so the
    # free axis may be read and its embedding stays. The projection that
    # drops t' is still absorbed into the affine.
    chain = _restricted(affine, np.nan)
    reduced = chain.simplify("analytic")
    first, second = reduced.transformations
    assert isinstance(first, Projection) and np.isnan(first.at[0])
    assert isinstance(second, Affine) and second.matrix.shape == (2, 4)
    expected = np.array([[2.0, 0, 1], [0, 3, 2]])
    assert np.allclose(chain.compute().matrix, expected)


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
def test_a_free_axis_that_a_kept_output_reads_stays_in_place(
    policy: str,
) -> None:
    chain = _restricted(_A2, np.nan)
    reduced = chain.simplify(policy)
    assert isinstance(reduced.transformations[0], Projection)
    with pytest.raises(RestrictionError, match="free"):
        chain.compute(simplify=policy)


# ----------------------------------------------------------------------
#   BACKWARD PASS
# ----------------------------------------------------------------------


def test_a_drop_moves_back_through_several_elements() -> None:
    chain = Sequence(
        [
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Translation(translation=np.array([1.0, 2.0, 3.0])),
            Permutation(permutation=np.array([1, 0, 2])),
            Projection(dropped=[2]),
        ]
    )
    reduced = chain.simplify()
    drop, scaling, translation, permutation = reduced.transformations
    assert isinstance(drop, Projection) and list(drop.dropped) == [2]
    assert isinstance(scaling, Scaling)
    assert list(scaling.scale) == [2.0, 3.0]
    assert isinstance(translation, Translation)
    assert list(translation.translation) == [1.0, 2.0]
    assert isinstance(permutation, Permutation)
    assert list(permutation.permutation) == [1, 0]
    assert _same_map(chain, reduced, 3)


def test_a_drop_is_absorbed_by_the_first_element_that_reads_it() -> None:
    # The affine reads every input, so the drop cuts its outputs and no
    # projection is left.
    affine = Affine(matrix=np.arange(12.0).reshape(3, 4) + 1)
    chain = Sequence(
        [
            affine,
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Projection(dropped=[1]),
        ]
    )
    reduced = chain.simplify()
    first, second = reduced.transformations
    assert isinstance(first, Affine) and first.matrix.shape == (2, 4)
    assert isinstance(second, Scaling) and list(second.scale) == [2.0, 4.0]
    assert _same_map(chain, reduced, 3)


# ----------------------------------------------------------------------
#   FORWARD PASS
# ----------------------------------------------------------------------


def test_a_created_axis_is_carried_through_a_permutation() -> None:
    # The permutation moves the created axis to the first output, and
    # the permutation of the other two axes is the identity. The scaling
    # reads the created axis, so the axis is created just before it.
    scaling = Scaling(scale=np.array([2.0, 3.0, 4.0]))
    chain = Sequence(
        [
            Projection(created=[2], at=[5.0]),
            Permutation(permutation=np.array([2, 0, 1])),
            scaling,
        ]
    )
    reduced = chain.simplify()
    create, kept = reduced.transformations
    assert isinstance(create, Projection)
    assert list(create.created) == [0] and list(create.at) == [5.0]
    assert kept is scaling
    assert _same_map(chain, reduced, 2)
    assert _same_map(chain, chain.compute(), 2)


def test_a_created_axis_passes_around_a_field_in_a_subspace() -> None:
    # The subspace passes t through, so the value of t reaches the end
    # of the chain without touching the field, which is kept as it is.
    field = DisplacementField(field=np.zeros((3, 3, 2)))
    subspace = SubspaceTransformation(
        field,
        input_axes=[0, 1],
        output_axes=[0, 1],
        input=CoordinateSystem().expand(3),
        output=CoordinateSystem().expand(3),
    )
    chain = Sequence([Projection(created=[2], at=[4.0]), subspace])
    reduced = chain.simplify()
    inner, create = reduced.transformations
    assert inner is field
    assert list(create.created) == [2] and list(create.at) == [4.0]


def test_a_free_axis_is_carried_to_the_end_and_refused() -> None:
    chain = Sequence(
        [
            Projection(created=[2], at=[np.nan]),
            Permutation(permutation=np.array([2, 0, 1])),
        ]
    )
    reduced = chain.simplify()
    (create,) = _leaves(reduced)
    assert list(create.created) == [0] and np.isnan(create.at[0])
    with pytest.raises(RestrictionError, match="free"):
        chain.compute()


# ----------------------------------------------------------------------
#   ZERO ROWS AND COLUMNS
# ----------------------------------------------------------------------


def test_an_affine_with_a_zero_column_is_split_under_numeric() -> None:
    # The affine does not read its third input, so the free axis created
    # there cancels under the numeric policy. The analytic policy reads
    # the affine as fully coupled and leaves the chain alone.
    affine = Affine(
        matrix=np.array([[2.0, 0, 0, 1], [0, 3, 0, 2], [1, 1, 0, 5]])
    )
    chain = Sequence([Projection(created=[2], at=[np.nan]), affine])
    assert chain.simplify("analytic") is chain
    reduced = chain.simplify("numeric")
    assert isinstance(reduced, Affine)
    assert np.allclose(reduced.matrix, affine.matrix[:, [0, 1, 3]])


def test_an_affine_with_a_zero_row_is_split_under_numeric() -> None:
    # The third output of the affine is the constant 5, so under the
    # numeric policy it is created at 5 instead. The permutation passes
    # it through to the first output and is left as the identity on the
    # two others.
    affine = Affine(
        matrix=np.array([[1.0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 5]])
    )
    chain = Sequence([affine, Permutation(permutation=np.array([2, 0, 1]))])
    assert chain.simplify("analytic") is chain
    reduced = chain.simplify("numeric")
    cut, create = reduced.transformations
    assert _dense(cut, 3).shape == (2, 4)
    assert list(create.created) == [0] and list(create.at) == [5.0]
    assert _same_map(chain, reduced, 3)


def test_simplify_does_not_lengthen_a_chain_to_split_an_affine() -> None:
    # Splitting this affine would add a projection before it, so the
    # simplified chain keeps it, and only the factor pass splits it.
    affine = Affine(
        matrix=np.array([[2.0, 0, 0, 1], [0, 3, 0, 2], [0, 0, 0, 5]])
    )
    chain = Sequence([affine, Scaling(scale=np.array([1.0, 2.0, 3.0]))])
    assert chain.simplify("numeric") is chain


# ----------------------------------------------------------------------
#   LAZY INVERSES
# ----------------------------------------------------------------------


def test_the_pattern_of_an_inverse_is_bounded_by_the_closure() -> None:
    shear = _SHEAR
    inverse = shear.inverse()
    # A matrix is fully coupled under the analytic policy, and so is the
    # closure of its pattern.
    assert read_pattern(inverse, ANALYTIC).all()
    # The numeric policy reads the coefficients of the inverse, whose
    # first row does not read z, and the inverse stays lazy.
    expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
    assert np.array_equal(read_pattern(inverse, NUMERIC), expected)
    assert isinstance(inverse, Inverse)


def test_the_closure_follows_the_structure_of_the_forward() -> None:
    # The forward couples x and y and leaves z alone, so its inverse
    # does too, although the inverse is never computed.
    subspace = SubspaceTransformation(
        Affine(matrix=np.array([[1.0, 1, 0], [0, 1, 0]])),
        input_axes=[0, 1],
        output_axes=[0, 1],
        input=CoordinateSystem().expand(3),
        output=CoordinateSystem().expand(3),
    )
    pattern = read_pattern(Inverse(forward=subspace), ANALYTIC)
    expected = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    assert np.array_equal(pattern, expected)


def test_the_inverse_of_a_shear_drops_z_only_under_numeric() -> None:
    shear = _SHEAR
    chain = Sequence([shear.inverse(), Projection(dropped=[1, 2])])
    analytic = chain.simplify("analytic")
    numeric = chain.simplify("numeric")
    # Under the analytic policy, x' may read z, and the inverse could
    # only be cut by computing it, so the chain is left alone.
    assert analytic is chain
    # Under the numeric policy, z is dropped first, and the inverse is
    # computed only because it is cut.
    drop, row = numeric.transformations
    assert list(drop.dropped) == [2]
    assert np.allclose(_dense(row, 2), [[1.0, -1.0, 0.0]])
    assert _same_map(chain, numeric, 3)
    assert _same_map(chain, analytic, 3)


def test_a_closed_block_of_an_inverse_stays_lazy() -> None:
    # The forward outputs y and z read only y and z, so the block of the
    # inverse over y and z is the lazy inverse of the block of the
    # forward.
    shear = _SHEAR
    piece = restrict_block(shear.inverse(), [1, 2], [1, 2], NUMERIC)
    assert isinstance(piece, Inverse)
    assert np.allclose(piece.forward.matrix, [[1.0, 1], [0, 1]])
    chain = Sequence([shear.inverse(), Projection(dropped=[0])])
    reduced = chain.simplify("numeric")
    drop, inverse = reduced.transformations
    assert list(drop.dropped) == [0]
    assert isinstance(inverse, Inverse)
    assert _same_map(chain, reduced, 3)


def test_an_inverse_still_cancels_with_its_forward() -> None:
    shear = _SHEAR
    chain = Sequence([shear, shear.inverse(), Projection(dropped=[1, 2])])
    for policy in ("analytic", "numeric"):
        reduced = chain.simplify(policy)
        assert isinstance(reduced, Projection)
        assert list(reduced.dropped) == [1, 2]


def test_a_lazy_inverse_that_is_not_cut_is_kept() -> None:
    shear = _SHEAR
    inverse = shear.inverse()
    chain = Sequence([Projection(created=[2], at=[np.nan]), inverse])
    reduced = chain.simplify("numeric")
    assert reduced is chain
    assert reduced.transformations[1] is inverse


# ----------------------------------------------------------------------
#   FIELDS
# ----------------------------------------------------------------------


def test_a_field_next_to_a_projection_is_not_cut() -> None:
    field = DisplacementField(field=np.zeros((2, 2, 2, 3)))
    after = Sequence([field, Projection(dropped=[2])])
    assert after.simplify() is after
    before = Sequence([Projection(created=[2], at=[1.0]), field])
    assert before.simplify() is before
    # The field reads every axis, so a free axis before it is refused.
    free = Sequence([Projection(created=[2], at=[np.nan]), field])
    assert free.simplify() is free
    with pytest.raises(RestrictionError):
        free.compute()


# ----------------------------------------------------------------------
#   IDENTITY
# ----------------------------------------------------------------------


def test_simplify_returns_the_same_object_when_nothing_is_dropped() -> None:
    chain = Sequence(
        [
            Affine(matrix=np.arange(12.0).reshape(3, 4)),
            Projection(created=[3], at=[2.0]),
        ]
    )
    assert chain.simplify() is chain
    assert chain.simplify("numeric") is chain


def test_a_reduced_chain_is_not_reduced_again() -> None:
    reduced = _restricted(_A2, np.nan).simplify("analytic")
    assert reduced.simplify("analytic") is reduced


# ----------------------------------------------------------------------
#   NORMAL FORM
# ----------------------------------------------------------------------


def test_the_normal_form_drops_the_unread_inputs_first() -> None:
    grid = CartesianField(shape=(4, 5, 6, 7))
    chain = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0, 4.0, 5.0])),
            Projection(dropped=[3]),
        ]
    )
    nf = chain.compute(factor=True)
    first, drop, *factors = nf.transformations
    assert first is grid
    assert list(drop.dropped) == [3] and not list(drop.created)
    assert [list(f.input_axes) for f in factors] == [[0], [1], [2]]
    assert _same_map(chain[1:], nf[1:], 4)


def test_the_normal_form_creates_the_constant_outputs_last() -> None:
    # The ends of this chain differ in their numbers of axes.
    grid = CartesianField(shape=(4, 5))
    chain = Sequence(
        [
            grid,
            Scaling(scale=np.array([2.0, 3.0])),
            Permutation(permutation=np.array([1, 0])),
            Projection(created=[2], at=[7.0]),
        ]
    )
    nf = chain.compute(factor=True)
    first, *factors, permutation, create = nf.transformations
    assert first is grid
    assert [list(f.input_axes) for f in factors] == [[0], [1]]
    assert isinstance(permutation, Permutation)
    assert list(create.created) == [2] and list(create.at) == [7.0]
    assert _same_map(chain[1:], nf[1:], 2)


def test_the_normal_form_splits_an_affine_with_zero_lines() -> None:
    # The third input is unread and the third output is constant, which
    # only the numeric policy sees.
    grid = CartesianField(shape=(4, 5, 6))
    affine = Affine(
        matrix=np.array([[2.0, 0, 0, 1], [0, 3, 0, 2], [0, 0, 0, 5]])
    )
    chain = Sequence([grid, affine])
    analytic = chain.compute(factor=True)
    assert not any(isinstance(t, Projection) for t in analytic)
    nf = chain.compute(factor=True, simplify="numeric")
    first, drop, fx, fy, create = nf.transformations
    assert first is grid
    assert list(drop.dropped) == [2]
    assert list(fx.input_axes) == [0] and list(fy.input_axes) == [1]
    assert list(create.created) == [2] and list(create.at) == [5.0]
    assert _same_map(chain[1:], nf[1:], 3)


def test_the_normal_form_with_ends_is_a_fixpoint() -> None:
    chain = Sequence(
        [
            CartesianField(shape=(4, 5, 6)),
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Projection(dropped=[2]),
            Projection(created=[0], at=[1.0]),
        ]
    )
    nf = chain.compute(factor=True)
    assert factor_sequence(nf) is nf
    again = nf.compute(factor=True)
    assert all(a is b for a, b in zip(again, nf))


# ----------------------------------------------------------------------
#   SYSTEMS AND ROBUSTNESS
# ----------------------------------------------------------------------


def test_an_identity_that_renames_the_axes_keeps_its_systems() -> None:
    xyz = CoordinateSystem(axes=[Axis(name=n) for n in "xyz"])
    abc = CoordinateSystem(axes=[Axis(name=n) for n in "abc"])
    chain = Sequence(
        [
            Scaling(scale=np.array([2.0, 3.0, 4.0])),
            Identity(input=xyz, output=abc),
            Projection(dropped=[2]),
        ]
    )
    reduced = chain.simplify()
    names = [axis.name for axis in reduced.transformations[-1].output.axes]
    assert names == ["a", "b"]
    assert _same_map(chain, reduced, 3)


@pytest.mark.parametrize(
    "chain",
    [
        Sequence([Scaling(scale=np.ones(3)), Projection(dropped=[5])]),
        Sequence([Projection(created=[7]), Scaling(scale=np.ones(2))]),
        Sequence([Projection(dropped=[0]), Projection(created=[0, 0])]),
        Sequence([Affine(matrix=np.ones((2, 4))), Scaling(scale=np.ones(3))]),
        Sequence(
            [
                SubspaceTransformation(Scaling(scale=np.ones(1)), [3], [3]),
                Projection(dropped=[1]),
            ]
        ),
    ],
    ids=["drop", "create", "merge", "counts", "subspace"],
)
def test_simplify_never_raises_on_odd_chains(chain: Sequence) -> None:
    chain.simplify("analytic")
    chain.simplify("numeric")


def test_simplify_never_raises_when_a_rule_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("broken rule")

    monkeypatch.setattr(reduce, "_restrict_block", broken)
    chain = _restricted(_A1, 4.0)
    assert chain.simplify() is chain
    monkeypatch.setattr(reduce, "_read_pattern", broken)
    assert chain.simplify() is chain
    monkeypatch.setattr(reduce, "_Reduction", broken)
    assert chain.simplify() is chain


def test_a_direct_factor_call_cancels_before_it_cuts() -> None:
    # Under the numeric policy, cutting the lazy inverse would compute
    # it, but it cancels with its forward transformation first.
    shear = _SHEAR
    grid = CartesianField(shape=(3, 3, 3))
    chain = Sequence([grid, shear, shear.inverse(), Projection(dropped=[2])])
    nf = factor_sequence(chain, policy="numeric")
    first, drop = nf.transformations
    assert first is grid
    assert list(drop.dropped) == [2]


# ----------------------------------------------------------------------
#   EQUIVALENCE
# ----------------------------------------------------------------------


def _square(rng: np.random.Generator, n: int) -> np.ndarray:
    # Draw an invertible affine matrix with random exact zeros.
    matrix = rng.normal(size=(n, n + 1))
    matrix[:, :-1] *= rng.random((n, n)) < 0.6
    matrix[:, :-1] += 2 * np.eye(n)
    return matrix


def _random_element(
    rng: np.random.Generator, n: int, depth: int = 0, free: bool = True
) -> object:
    # Draw an element that reads `n` axes. Matrices have random exact
    # zeros, so that the numeric policy finds structure in them.
    kind = int(rng.integers(12))
    if kind == 0:
        n_out = int(rng.integers(max(n - 1, 1), n + 2))
        matrix = rng.normal(size=(n_out, n + 1))
        matrix[:, :-1] *= rng.random((n_out, n)) < 0.6
        return Affine(matrix=matrix)
    if kind == 1:
        return Scaling(scale=rng.normal(size=n) * (rng.random(n) < 0.8))
    if kind == 2:
        return Permutation(permutation=rng.permutation(n))
    if kind == 3 and n > 1:
        size = int(rng.integers(1, n))
        dropped = sorted(rng.choice(n, size=size, replace=False))
        return Projection(dropped=[int(i) for i in dropped])
    if kind == 4:
        size = int(rng.integers(1, 3))
        created = sorted(rng.choice(n + size, size=size, replace=False))
        at = rng.normal(size=size).round(1)
        if free and rng.random() < 0.3:
            at[int(rng.integers(size))] = np.nan
        return Projection(created=[int(o) for o in created], at=list(at))
    if kind == 5:
        return Affine(matrix=_square(rng, n)).inverse()
    if kind == 6:
        return Translation(translation=rng.normal(size=n))
    if kind == 7:
        return Identity()
    if kind == 8 and n > 1:
        # A subspace over some axes, which it may reorder.
        size = int(rng.integers(1, n))
        in_axes = [int(i) for i in rng.choice(n, size=size, replace=False)]
        out_axes = [int(i) for i in rng.choice(n, size=size, replace=False)]
        inner = [
            Affine(matrix=_square(rng, size)),
            Scaling(scale=rng.normal(size=size) * (rng.random(size) < 0.8)),
            None,
        ][int(rng.integers(3))]
        system = CoordinateSystem().expand(n)
        return SubspaceTransformation(
            inner, in_axes, out_axes, input=system, output=system
        )
    if kind == 9 and depth < 1:
        members, m = [], n
        for _ in range(int(rng.integers(2, 4))):
            member = _random_element(rng, m, depth + 1, free=False)
            members.append(member)
            m = _dense(member, m).shape[0]
        return Sequence(members)
    if kind == 10:
        return Permutation(permutation=rng.permutation(n)).inverse()
    if kind == 11:
        return Scaling(scale=rng.normal(size=n) + 2).inverse()
    return Translation(translation=rng.normal(size=n))


def _random_chain(seed: int) -> tx.Tuple[Sequence, int]:
    rng = np.random.default_rng(seed)
    n_in = int(rng.integers(1, 5))
    elements, n = [], n_in
    for _ in range(int(rng.integers(2, 6))):
        element = _random_element(rng, n)
        elements.append(element)
        n = _dense(element, n).shape[0]
    return Sequence(elements), n_in


def _flat(t: object) -> list:
    out = []
    for leaf in _leaves(t):
        out.extend(_flat(leaf) if isinstance(leaf, Sequence) else [leaf])
    return out


_POLICIES = [
    "analytic",
    "numeric",
    {None: "numeric"},
    {None: "analytic", Affine: "none"},
    {None: "numeric", Projection: "none"},
]


@pytest.mark.parametrize(
    "policy", _POLICIES, ids=["a", "n", "all-n", "frozen-affine", "frozen-p"]
)
@pytest.mark.parametrize("seed", range(20))
def test_random_chains_keep_their_map(seed: int, policy: object) -> None:
    chain, n_in = _random_chain(seed)
    reduced = chain.simplify(policy)
    assert len(_flat(reduced)) <= len(_flat(chain))
    assert _same_map(chain, reduced, n_in, seed)
    assert reduced.simplify(policy) is reduced


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
@pytest.mark.parametrize("seed", [2, 3, 5, 7, 8, 11])
def test_random_chains_compute_and_factor(seed: int, policy: str) -> None:
    # Composition is slow on some random chains, so a few seeds are
    # enough to check that compute and the factor pass agree with the
    # map of the chain.
    chain, n_in = _random_chain(seed)
    if np.isnan(_product(chain, n_in)).any():
        with pytest.raises(RestrictionError):
            chain.compute(simplify=policy)
        return
    computed = chain.compute(simplify=policy)
    assert _same_map(chain, computed, n_in, seed)
    grid = CartesianField(shape=(3,) * n_in)
    nf = Sequence([grid, *chain.transformations]).compute(
        factor=True, simplify=policy
    )
    first, *rest = _leaves(nf)
    assert isinstance(first, CartesianField)
    assert _same_map(chain, rest or [Identity()], n_in, seed)


def test_a_cut_keeps_the_type() -> None:
    scaling = Scaling(scale=np.array([2.0, 3.0, 4.0]))
    assert isinstance(
        restrict_block(scaling, [0, 2], [0, 2], ANALYTIC), Scaling
    )
    linear = Linear(matrix=np.arange(9.0).reshape(3, 3))
    assert type(restrict_block(linear, [0, 1, 2], [1], ANALYTIC)) is Linear
