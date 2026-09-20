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

from brainhops.backends import backend, get_array_backend
from brainhops.datamodel._transformations.compose import compose
from brainhops.datamodel._transformations.sequence import (
    _ensure_proper_modes,
)
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
    InverseDisplacementField,
    Sequence,
    SubspaceTransformation,
    Transformation,
    is_identity,
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

    # Pinned to the exact backend. Folding builds its node grid with
    # `CartesianField`, so under the dask backend the folded field is a
    # dask array and its spline prefilter is `dask_image`'s, which is
    # applied patch-wise rather than along the whole axis. That is a
    # deliberate trade -- a real IIR filter over a large volume is what
    # dask exists to avoid -- and it costs about 3e-2 here, far above the
    # tolerance this test is about. The arithmetic under test is the
    # composer's, not the interpolator's, so it is checked where the
    # interpolation is exact.
    with backend("numpy"):
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
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_subspace_coords_passthrough_is_bit_exact() -> None:
    # C1/I1. The time component is copied straight through, not recomputed,
    # so it is bit-for-bit identical.
    To = _subspace_affine([0, 1, 2])
    Ti = _coords_4d()
    got = np.asarray(compose(To, Ti).field)
    np.testing.assert_array_equal(got[..., 3], np.asarray(Ti.field)[..., 3])


def test_subspace_coords_permuted_positions_align() -> None:
    # C1/I2. Non-monotonic positions place each component where the axis
    # vector says, matching the affine reduction, and the pass-through
    # component stays bit-exact.
    To = _subspace_affine([2, 0, 1])
    Ti = _coords_4d()
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
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
    got = compose(To, Ti)
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
    got = compose(To, Ti)
    assert np.asarray(got.field).dtype.kind == "f"


def test_subspace_disp_matches_affine_reduction() -> None:
    # C2. A subspace transform applied to a displacement field equals the
    # affine reduction folded into the same displacement field.
    rng = np.random.default_rng(2)
    disp = rng.standard_normal((3, 4, 5, 2, 4)) * 0.1
    Ti = DisplacementField(field=disp, output=_full4("in"))
    To = _subspace_affine([0, 1, 2])
    got = compose(To, Ti)
    ref = compose(To.to(Affine), Ti)
    assert isinstance(got, DisplacementField)
    np.testing.assert_allclose(
        np.asarray(got.field), np.asarray(ref.field), atol=1e-12
    )


def test_subspace_compose_subspace_matches_into_one_wrapper() -> None:
    # C3. Two subspace transforms over the same axes compose into a single
    # subspace transform.
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    composed = compose(second, first)
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
    composed = compose(wrapper, wrapper.inverse())
    assert isinstance(composed, Identity)


def test_subspace_compose_subspace_mismatch_raises() -> None:
    # C3. Two subspace transforms whose axes do not line up cannot compose.
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([1, 2, 3])
    with pytest.raises(CompositionError):
        compose(second, first)


_DEFAULT_MODE = _ensure_proper_modes(None)


def test_merge_adjacent_subspaces_folds_a_matching_pair() -> None:
    # C4. Composing two adjacent subspace transforms over the same axes
    # folds them into one. (This used to be a dedicated pre-pass,
    # `_merge_adjacent_subspaces`; it is now the subspace/subspace composer,
    # reached by `compose`/`Sequence.compute`.)
    first = _subspace_affine([0, 1, 2])
    second = _subspace_affine([0, 1, 2])
    folded = compose(second, first)
    assert isinstance(folded, SubspaceTransformation)
    np.testing.assert_array_equal(folded.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(folded.output_axes, [0, 1, 2])


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
    # identity, so computing the sequence leaves the identity behind rather
    # than an empty sequence. The single stack sweep in `_compute_sequence`
    # annihilates the pair symbolically, before any field is materialized.
    wrapper = _field_wrapper(4)
    computed = Sequence([wrapper.inverse(), wrapper]).compute(
        mode=_DEFAULT_MODE
    )
    assert isinstance(computed, Identity)
    assert computed.input == _full4("s")
    assert computed.output == _full4("s")


def test_field_subspaces_are_not_composed_under_affine_mode(
    monkeypatch,  # noqa: ANN001
) -> None:
    # C4. Two subspace-wrapped fields kept separate by an affine-only mode
    # are never resampled, so the numeric field composition is never reached.
    # The same pair merges under the default mode.
    from brainhops.datamodel._transformations import composers

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("a field was composed numerically")

    monkeypatch.setattr(composers, "pull_field", _boom)
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


def _subspace_reindex(input_axes, output_axes) -> SubspaceTransformation:  # noqa: ANN001
    # A subspace transform whose inner is the identity but whose axis vectors
    # move components from `input_axes` positions to `output_axes` positions.
    return SubspaceTransformation(
        transformation=None,
        input_axes=np.asarray(input_axes, dtype=int),
        output_axes=np.asarray(output_axes, dtype=int),
        input=_full4("in"),
        output=_full4("out"),
    )


def test_subspace_compose_identity_inner_keeps_axis_reindex() -> None:
    # C3. Two subspace transforms whose inners cancel to the identity but
    # whose axis vectors describe a genuine permutation must NOT collapse to
    # a bare Identity: the composition is a pure axis reindex and has to be
    # preserved. Ti sends input axes [0, 1, 2] to [1, 2, 0]; To reads those
    # same [1, 2, 0] (so the pair composes) and writes them back to [1, 2, 0].
    # The net map is input [0, 1, 2] -> output [1, 2, 0], a real reindex.
    Ti = _subspace_reindex([0, 1, 2], [1, 2, 0])
    To = _subspace_reindex([1, 2, 0], [1, 2, 0])

    composed = compose(To, Ti)  # same path _merge_adjacent_subspaces uses

    # Not a bare Identity -- the reindex survives.
    assert not isinstance(composed, Identity)
    assert isinstance(composed, SubspaceTransformation)
    assert not is_identity(composed, compute=True)
    np.testing.assert_array_equal(composed.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(composed.output_axes, [1, 2, 0])

    # The embedded affine maps input axis i -> output axis o for each
    # (o, i) in zip(output_axes, input_axes) == (1,0), (2,1), (0,2), with the
    # unnamed time axis (3) passing through. So y = [x2, x0, x1, x3].
    matrix = np.asarray(composed.to(Affine).matrix)
    expected = np.zeros((4, 5))
    expected[1, 0] = 1.0
    expected[2, 1] = 1.0
    expected[0, 2] = 1.0
    expected[3, 3] = 1.0
    np.testing.assert_array_equal(matrix, expected)


def test_subspace_compose_identity_inner_matching_axes_is_identity() -> None:
    # C3. When the axis vectors also match (input [0, 1, 2] -> output
    # [0, 1, 2]), the same identity-inner composition really is the identity
    # and collapses to a bare Identity, carrying the composed endpoints.
    Ti = _subspace_reindex([0, 1, 2], [0, 1, 2])
    To = _subspace_reindex([0, 1, 2], [0, 1, 2])

    composed = compose(To, Ti)

    assert isinstance(composed, Identity)
    assert composed.input == _full4("in")
    assert composed.output == _full4("out")


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


# ----------------------------------------------------------------------
#   EMBED COMPOSERS AND THE mode-THREADED IDENTITY CANCEL
# ----------------------------------------------------------------------


def test_subspace_affine_embed_folds_a_non_interpolating_subspace() -> None:
    # A subspace transform that merely lifts an affine into a larger space is
    # non-interpolating, so composing it with an affine folds the two into a
    # single affine rather than keeping the wrapper. The folded matrix equals
    # reducing the subspace to an affine and composing.
    # ``sub`` has input=_full4("in") and output=_full4("out").
    sub = _subspace_affine([0, 1, 2])
    aff = Affine(
        matrix=np.eye(4, 5), input=_full4("out"), output=_full4("world")
    )
    folded = compose(aff, sub)
    assert isinstance(folded, Affine)
    assert not isinstance(folded, SubspaceTransformation)
    ref = compose(aff, sub.to(Affine))
    np.testing.assert_allclose(
        np.asarray(folded.matrix), np.asarray(ref.matrix)
    )

    # The mirror direction (subspace applied after the affine) folds too.
    aff2 = Affine(
        matrix=np.eye(4, 5), input=_full4("world"), output=_full4("in")
    )
    folded2 = compose(sub, aff2)
    assert isinstance(folded2, Affine)
    assert not isinstance(folded2, SubspaceTransformation)


def test_interpolating_subspace_does_not_embed_into_an_affine() -> None:
    # A subspace transform that wraps a field is interpolating, so it is NOT
    # reduced to an affine: composing it with an affine declines, and in a
    # sequence the wrapper survives rather than folding away.
    wrapper = _field_wrapper(7)  # input=output=_full4("s")
    aff = Affine(matrix=np.eye(4, 5), input=_full4("s"), output=_full4("s"))
    with pytest.raises(CompositionError):
        compose(aff, wrapper)
    result = Sequence([wrapper, aff]).compute()
    leaves = list(result) if isinstance(result, Sequence) else [result]
    assert any(isinstance(t, SubspaceTransformation) for t in leaves)


def test_compose_cancels_inverse_by_identity_without_materializing(
    monkeypatch,  # noqa: ANN001
) -> None:
    # Cancellation is an ANALYTIC-priority composer, so `compose` tries it
    # ahead of any numeric composer (higher priority beats hierarchy
    # distance). A transform composed with its own inverse collapses to the
    # identity without the numeric inversion the (Affine, Affine) composer
    # would otherwise run had it been reached by distance dispatch.
    real_inv = np.linalg.inv
    calls = {"n": 0}

    def counting(matrix):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return real_inv(matrix)

    monkeypatch.setattr(np.linalg, "inv", counting)
    affine = Affine(matrix=np.array([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]]))
    result = compose(affine, affine.inverse())
    assert isinstance(result, Identity)
    assert calls["n"] == 0

    # The same holds for a field, whose inverse would be materialized by a
    # numeric mesh inversion if the pair did not cancel symbolically first.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    field = DisplacementField(field=np.zeros((5, 6, 2)))
    cancelled = compose(field, field.inverse())
    assert isinstance(cancelled, Identity)


def test_restrictive_mode_prevents_field_through_field_composition() -> None:
    # A restrictive mode composes only the inner types it admits. Two
    # subspace-wrapped fields are left separate under an affine-only mode,
    # so neither field is resampled through the other; under the default mode
    # the same pair folds into a single wrapper.
    first = _field_wrapper(4)
    second = _field_wrapper(5)
    unmerged = Sequence([first, second]).compute(mode="Affine")
    assert isinstance(unmerged, Sequence)
    assert len(unmerged.transformations) == 2
    merged = Sequence([first, second]).compute()
    assert isinstance(merged, SubspaceTransformation)


# ----------------------------------------------------------------------
#   DISPATCH ORDER AND THE ANALYTIC CANCEL TIER
# ----------------------------------------------------------------------


def test_compose_distinct_inverse_falls_through_to_affine() -> None:
    # For distinct A, B the pair `A @ B^-1` does not cancel: the analytic
    # cancel composer declines (its `_cancels` sees no identity link), so
    # dispatch falls through past the cancel tier to the numeric affine
    # composer, which returns a plain Affine.
    A = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]))
    B = Affine(matrix=np.array([[1.5, 0.0, -1.0], [0.0, 0.5, 4.0]]))
    result = compose(A, B.inverse())
    assert type(result) is Affine
    assert not isinstance(result, Identity)


def test_compose_identity_with_lazy_inverse_stays_unmaterialized(
    monkeypatch,  # noqa: ANN001
) -> None:
    # `compose(Identity(), df.inverse())` returns the lazy inverse itself,
    # unmaterialized. The cancel composer declines (the inverse's forward is
    # `df`, not the identity), and the identity composer returns the operand
    # untouched -- without the `.compute()` that would materialize the field.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    df = DisplacementField(field=np.zeros((5, 6, 2)))
    lazy = df.inverse()
    result = compose(Identity(), lazy)
    assert result is lazy
    assert isinstance(result, InverseDisplacementField)


def test_dispatch_priority_tier_and_terminal_composition_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    # The dispatch order (see the `compose` module docstring): a composer in
    # the ANALYTIC priority tier is tried ahead of a priority-0 family
    # composer, and a `CompositionError` raised by a composer is terminal --
    # dispatch stops, so later candidates are never reached.
    from brainhops.datamodel._transformations import compose as compose_mod
    from brainhops.datamodel._transformations.registries import ANALYTIC

    a = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]))
    b = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    calls: list = []

    # An analytic-priority composer that declines, plus a broader family
    # composer at the default priority. The analytic one must be tried first
    # and, on declining with NotImplemented, hand off to the family.
    def declining_analytic(x1, x2):  # noqa: ANN001, ANN202
        calls.append("analytic")
        return NotImplemented

    def family(x1, x2):  # noqa: ANN001, ANN202
        calls.append("family")
        return Identity()

    monkeypatch.setattr(
        compose_mod,
        "COMPOSERS",
        {
            (Affine, Affine): (declining_analytic, ANALYTIC),
            (Transformation, Transformation): (family, 0),
        },
    )
    monkeypatch.setattr(compose_mod, "COMPOSERS_FASTMAP", {})

    result = compose(a, b)
    assert isinstance(result, Identity)
    assert calls == ["analytic", "family"]

    # A composer that raises `CompositionError` stops dispatch: the family
    # composer, a later (lower-priority) candidate, is never reached.
    calls.clear()

    def raising_analytic(x1, x2):  # noqa: ANN001, ANN202
        calls.append("analytic")
        raise CompositionError("right types, cannot combine")

    monkeypatch.setattr(
        compose_mod,
        "COMPOSERS",
        {
            (Affine, Affine): (raising_analytic, ANALYTIC),
            (Transformation, Transformation): (family, 0),
        },
    )
    monkeypatch.setattr(compose_mod, "COMPOSERS_FASTMAP", {})

    with pytest.raises(CompositionError):
        compose(a, b)
    assert calls == ["analytic"]
