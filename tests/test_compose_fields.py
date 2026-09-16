"""Regression tests for folding an affine into a stored field.

Composing an affine-like transformation with a ``CoordinatesField`` or a
``DisplacementField`` folds the affine into the stored field. The folded
field must keep the interpolation settings of the input field (``order``,
``bound`` and ``coeff``); otherwise it is later re-interpolated at the wrong
settings. A field of spline coefficients must be converted to sampled values
before the affine arithmetic and converted back afterwards; otherwise the
arithmetic runs on coefficients and yields garbage.

Both faults were present in the affine-into-field composers. This file locks
the fix in two ways: it checks that the folded field reports the same
``order``, ``bound`` and ``coeff`` as its input, and it checks that
evaluating the folded field at interior points reproduces the in-order
reference ``A(interp(f))``.
"""

import numpy as np
import pytest

from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import (
    A,
    R,
    S,
    TimeAxis,
)
from brainhops.datamodel.enums import BoundaryCondition
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CompositionError,
    ConversionError,
    CoordinatesField,
    DisplacementField,
    Identity,
    Sequence,
    SubspaceTransformation,
    _compose,
    _ensure_proper_modes,
    _merge_adjacent_subspaces,
)

# An anisotropic affine with shear and a shift, so a dropped setting or
# coefficient arithmetic shows up as a large discrepancy rather than a small
# one. The linear part is invertible and far from the identity.
AFFINE_MATRIX = np.array([[1.7, 0.4, 2.0], [-0.3, 0.9, -1.5]])

# A grid large enough to hold query points several nodes from every edge.
GRID_SHAPE = (14, 15)

# Query points in the interior of the grid, offset from the nodes so that a
# wrong interpolation order changes the result.
QUERY_POINTS = np.array(
    [[5.5, 6.5], [7.2, 8.1], [6.3, 5.7], [8.0, 9.0], [5.8, 7.4]]
)

# Coefficients require a spline order of at least two, so `coeff=True` is
# paired only with the cubic order.
ORDER_COEFF = [(1, False), (3, False), (3, True)]


def _evaluate(field, points):  # noqa: ANN001, ANN202
    """The coordinate map of ``field`` sampled at ``points``.

    Leading with the query points as a sampling domain evaluates the field on
    those points and returns the resulting coordinates.
    """
    computed = Sequence(
        transformations=[CoordinatesField(field=points), field]
    ).compute()
    return np.asarray(computed.to(CoordinatesField).field)


@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("order, coeff", ORDER_COEFF)
def test_fold_affine_into_field_keeps_interpolation_settings(
    field_type: type,
    order: int,
    coeff: bool,
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale
    field = field_type(
        field=values, order=order, bound=BoundaryCondition.mirror
    ).to(coeff=coeff)

    folded = (Affine(matrix=AFFINE_MATRIX) @ field).compute()

    assert folded.order == field.order
    assert folded.bound == field.bound
    assert folded.coeff == field.coeff


@pytest.mark.parametrize("field_type", [CoordinatesField, DisplacementField])
@pytest.mark.parametrize("order, coeff", ORDER_COEFF)
def test_fold_affine_into_field_matches_inorder_reference(
    field_type: type,
    order: int,
    coeff: bool,
) -> None:
    rng = np.random.default_rng(0)
    scale = 1.0 if field_type is CoordinatesField else 0.1
    values = rng.standard_normal((*GRID_SHAPE, 2)) * scale
    field = field_type(
        field=values, order=order, bound=BoundaryCondition.mirror
    ).to(coeff=coeff)

    matrix = AFFINE_MATRIX
    sampled = _evaluate(field, QUERY_POINTS)
    reference = sampled @ matrix[:, :-1].T + matrix[:, -1]

    folded = (Affine(matrix=matrix) @ field).compute()
    result = _evaluate(folded, QUERY_POINTS)

    # Folding a coordinate field is exact within the field of view, because
    # interpolation is linear and the affine is affine. Folding a
    # displacement field carries an additional interior term under a spline
    # order above one: the representation subtracts the node grid, whose
    # cubic interpolation departs from the identity by an amount that decays
    # away from the edges but does not vanish on a finite grid.
    interior_term = field_type is DisplacementField and order > 1
    atol = 1e-3 if interior_term else 1e-10
    np.testing.assert_allclose(result, reference, atol=atol, rtol=0)


# ----------------------------------------------------------------------
#   SUBSPACE COMPOSERS: A 3D TRANSFORM ACROSS A 4D FIELD
# ----------------------------------------------------------------------


def _full4(name: str) -> CoordinateSystem:
    return CoordinateSystem(name=name, axes=[R, A, S, TimeAxis(name="t")])


def _sub3(name: str) -> CoordinateSystem:
    return CoordinateSystem(name=name, axes=[R, A, S])


# A non-trivial 3D affine with shear and a shift, so a dropped or misplaced
# component shows up plainly.
SUB_AFFINE = np.array(
    [[1.3, 0.2, -0.1, 4.0], [0.0, 0.9, 0.3, -2.0], [0.1, 0.0, 1.1, 1.0]]
)


def _subspace_affine(positions) -> SubspaceTransformation:  # noqa: ANN001
    positions = np.asarray(positions, dtype=int)
    inner = Affine(matrix=SUB_AFFINE, input=_sub3("lps"), output=_sub3("lps"))
    return SubspaceTransformation(
        transformation=inner,
        input_axes=positions,
        output_axes=positions,
        input=_full4("in"),
        output=_full4("out"),
    )


def _coords_4d(seed: int = 0) -> CoordinatesField:
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((5, 6, 4))
    return CoordinatesField(
        field=field,
        output=_full4("in"),
        order=3,
        bound=BoundaryCondition.mirror,
    )


def test_subspace_coords_matches_affine_reduction() -> None:
    # C1. Applying a subspace transform whose inner is an affine equals
    # reducing the whole thing to an affine and folding that into the field.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = _compose(To, Ti)
    ref = _compose(To.to(Affine), Ti)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_subspace_coords_passthrough_is_bit_exact() -> None:
    # C1/I1. The time component is copied straight through, not recomputed,
    # so it is bit-for-bit identical.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = np.asarray(_compose(To, Ti).field)
    np.testing.assert_array_equal(got[..., 3], np.asarray(Ti.field)[..., 3])


def test_subspace_coords_permuted_positions_align() -> None:
    # C1/I2. Non-monotonic positions place each component where the axis
    # vector says, matching the affine reduction, and the pass-through
    # component stays bit-exact.
    To = _subspace_affine([2, 0, 1])
    Ti = _coords_4d()
    got = _compose(To, Ti)
    ref = _compose(To.to(Affine), Ti)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(got.field)[..., 3], np.asarray(Ti.field)[..., 3]
    )


def test_subspace_coords_preserves_interpolation_settings() -> None:
    # C1. The order, bound and coeff of the input field are preserved.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = _compose(To, Ti)
    assert got.order == Ti.order
    assert got.bound == Ti.bound
    assert got.coeff == Ti.coeff


def test_subspace_coords_promotes_an_integer_domain() -> None:
    # C1. An integer coordinate field promotes to floating point when an
    # affine inner is folded in.
    ab = get_array_backend()
    field = ab.asarray(np.arange(5 * 6 * 4).reshape(5, 6, 4), dtype="int64")
    Ti = CoordinatesField(field=field, output=_full4("in"))
    To = _subspace_affine([0, 1, 2])
    got = _compose(To, Ti)
    assert np.asarray(got.field).dtype.kind == "f"


def test_subspace_disp_matches_affine_reduction() -> None:
    # C2. A subspace transform applied to a displacement field equals the
    # affine reduction folded into the same displacement field.
    rng = np.random.default_rng(2)
    disp = rng.standard_normal((3, 4, 5, 2, 4)) * 0.1
    Ti = DisplacementField(field=disp, output=_full4("in"))
    To = _subspace_affine([0, 1, 2])
    got = _compose(To, Ti)
    ref = _compose(To.to(Affine), Ti)
    assert isinstance(got, DisplacementField)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_subspace_compose_subspace_matches_into_one_wrapper() -> None:
    # C3. Two subspace transforms over the same axes compose into a single
    # subspace transform.
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    composed = _compose(second, first)
    assert isinstance(composed, SubspaceTransformation)
    np.testing.assert_array_equal(composed.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(composed.output_axes, [0, 1, 2])


def test_subspace_compose_its_inverse_is_identity_without_inverting(
    monkeypatch,  # noqa: ANN001
) -> None:
    # C3. A subspace-wrapped field composed with its own inverse cancels to
    # the identity, and the numeric field inversion is never reached.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.random.default_rng(3).standard_normal((4, 4, 4, 3)) * 0.1,
        input=voxel,
        output=voxel,
    )
    wrapper = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )
    composed = _compose(wrapper, wrapper.inverse())
    assert isinstance(composed, Identity)


def test_subspace_compose_subspace_mismatch_raises() -> None:
    # C3. Two subspace transforms whose axes do not line up cannot compose.
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([1, 2, 3])
    with pytest.raises(CompositionError):
        _compose(second, first)


_DEFAULT_MODE = _ensure_proper_modes(None)


def test_merge_adjacent_subspaces_folds_a_matching_pair() -> None:
    # C4. The pre-pass folds two adjacent subspace transforms over the same
    # axes into one.
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    seq = Sequence([first, second])
    merged = _merge_adjacent_subspaces(seq, _DEFAULT_MODE)
    assert len(merged.transformations) == 1
    assert isinstance(merged.transformations[0], SubspaceTransformation)


def _field_wrapper(seed: int) -> SubspaceTransformation:
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.random.default_rng(seed).standard_normal((4, 4, 4, 3)) * 0.1,
        input=voxel,
        output=voxel,
    )
    return SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )


def test_merge_adjacent_subspaces_drops_an_inverse_pair() -> None:
    # C4. A subspace transform next to its own inverse cancels to the
    # identity, so the pair leaves the identity behind rather than an empty
    # sequence.
    wrapper = _field_wrapper(4)
    seq = Sequence([wrapper.inverse(), wrapper])
    merged = _merge_adjacent_subspaces(seq, _DEFAULT_MODE)
    assert isinstance(merged, Identity)
    assert merged.input == _full4("s")
    assert merged.output == _full4("s")


def test_merge_adjacent_subspaces_gated_by_mode() -> None:
    # C4. A restrictive mode composes only the inner types it admits. Two
    # subspace-wrapped fields are left separate under an affine-only mode,
    # because composing them would resample one field through the other, and
    # the affine mode does not compose fields. The default mode composes
    # everything, so the same pair merges into one wrapper.
    first = _field_wrapper(4)
    second = _field_wrapper(5)
    affine_mode = _ensure_proper_modes("Affine")
    unmerged = _merge_adjacent_subspaces(
        Sequence([first, second]), affine_mode
    )
    assert len(unmerged.transformations) == 2
    merged = _merge_adjacent_subspaces(
        Sequence([first, second]), _DEFAULT_MODE
    )
    assert len(merged.transformations) == 1
    assert isinstance(merged.transformations[0], SubspaceTransformation)


def test_field_subspaces_are_not_composed_under_affine_mode(
    monkeypatch,  # noqa: ANN001
) -> None:
    # C4. Two subspace-wrapped fields kept separate by an affine-only mode
    # are never resampled, so the numeric field composition is never reached.
    # The same pair merges under the default mode.
    from brainhops.datamodel import _xform_composers

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("a field was composed numerically")

    monkeypatch.setattr(_xform_composers, "pull_field", _boom)
    first = _field_wrapper(4)
    second = _field_wrapper(5)
    result = Sequence([first, second]).compute(mode="Affine")
    assert isinstance(result, Sequence)
    assert len(result.transformations) == 2


def test_full_subspace_cancellation_computes_to_the_identity() -> None:
    # F3. A subspace-wrapped field next to its own inverse cancels to the
    # identity under compute, rather than leaving an empty sequence behind.
    wrapper = _field_wrapper(6)
    result = Sequence([wrapper.inverse(), wrapper]).compute()
    assert isinstance(result, Identity)
    assert result.input == _full4("s")
    assert result.output == _full4("s")


def test_subspace_to_affine_on_a_field_inner_raises() -> None:
    # C5. A subspace transform whose inner is a field cannot be reduced to
    # an affine, because a field is applied by composing it with a domain.
    voxel = _sub3("voxel")
    warp = DisplacementField(
        field=np.zeros((4, 4, 4, 3)), input=voxel, output=voxel
    )
    wrapper = SubspaceTransformation(
        transformation=warp,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=_full4("s"),
        output=_full4("s"),
    )
    with pytest.raises(ConversionError):
        wrapper.to(Affine)
