"""Tests for the simplify grammar, resolution, and `compute`/`simplify`.

The policy is a `SimplifyPolicy` (`none < analytic < numeric`); a
`simplify=` value lowers to a *simplify table* (`dict` mapping lowered kind
pairs, plus a `None` fallback, to policies). Resolution is always analytic.
"""

from unittest import mock

import numpy as np
import pytest

from brainhops.datamodel import hierarchy
from brainhops.datamodel._transformations import inverse as _inv
from brainhops.datamodel._transformations import sequence as _seq
from brainhops.datamodel._transformations.modes import (
    _lower_key,
    _lower_simplify,
    _resolve_simplify,
)
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    Linear,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

none, analytic, numeric = (
    SimplifyPolicy.none,
    SimplifyPolicy.analytic,
    SimplifyPolicy.numeric,
)
AFF = (hierarchy.AffineTransformation, None)
LIN = (hierarchy.LinearTransformation, None)


def _small(shape: tuple = (6, 7, 2), seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape) * 0.05


# ----------------------------------------------------------------------
#   LOWERING TO A TABLE
# ----------------------------------------------------------------------


def test_scalar_forms() -> None:
    for v in (None, False, "none", "NONE", none):
        assert _lower_simplify(v) == {None: none}
    for v in ("analytic", analytic):
        assert _lower_simplify(v) == {None: analytic}
    for v in (True, "numeric", "Numeric", numeric):
        assert _lower_simplify(v) == {None: numeric}


def test_key_and_list_restrict_to_those() -> None:
    assert _lower_simplify("affine") == {None: none, AFF: analytic}
    assert _lower_simplify(Affine) == {None: none, AFF: analytic}
    assert _lower_simplify(hierarchy.LinearTransformation) == {
        None: none,
        LIN: analytic,
    }
    table = _lower_simplify(["scaling", "translation"])
    assert table[None] is none
    assert table[(hierarchy.DiagonalTransformation, None)] is analytic
    assert table[(hierarchy.Translation, None)] is analytic


def test_mapping_fallback_and_collisions() -> None:
    # `None`-absent mapping defaults its fallback to analytic.
    t = _lower_simplify({"affine": "numeric"})
    assert t == {AFF: numeric, None: analytic}
    assert t == _lower_simplify({None: "analytic", "affine": "numeric"})
    assert _lower_simplify({None: "none", "affine": "numeric"}) == {
        AFF: numeric,
        None: none,
    }
    # Colliding lowered keys keep the safest policy.
    assert _lower_simplify({"affine": "numeric", Affine: "none"})[AFF] is none
    assert _lower_simplify({Affine: "none", "affine": "numeric"})[AFF] is none
    assert (
        _lower_simplify({"affine": "numeric", "AFFINE": "none"})[AFF] is none
    )


def test_lowering_is_idempotent() -> None:
    t = _lower_simplify({"affine": "numeric"})
    assert _lower_simplify(t) == t


def test_special_and_symbol_keys() -> None:
    assert _lower_key("SO(3)") == (
        hierarchy.SpecialOrthogonalTransformation,
        3,
    )
    assert _lower_key(3) == (hierarchy.Transformation, 3)
    assert _lower_key("scaling") == (hierarchy.DiagonalTransformation, None)
    from brainhops.datamodel.transformations import (
        CartesianField,
        Inverse,
        InverseAffine,
        Projection,
    )

    for key in (
        "subspace",
        "inverse",
        "projection",
        "meta",
        "field",
        "displacementfield",
        "coordinatesfield",
    ):
        kind, ndim = _lower_key(key)
        assert isinstance(kind, (type, tuple))
    assert _lower_key("bijection")[0] is hierarchy.BijectiveTransformation
    assert _lower_key("rotation")[0] is (
        hierarchy.SpecialOrthogonalTransformation
    )
    assert _lower_key(Rotation)[0] is hierarchy.SpecialOrthogonalTransformation
    assert _lower_key(Identity)[0] is hierarchy.IdentityTransformation
    assert _lower_key(InverseAffine) == (InverseAffine, None)  # class kind
    assert _lower_key(DisplacementField) == (DisplacementField, None)
    assert _lower_key(Projection) == (Projection, None)
    assert _lower_key(CartesianField) == (CartesianField, None)
    _ = Inverse


def test_invalid_keys_raise() -> None:
    from brainhops.datamodel.transformations import MultiscaleField

    with pytest.raises(ValueError):
        _lower_key(Sequence)
    with pytest.raises(ValueError):
        _lower_key(MultiscaleField)
    with pytest.raises(ValueError):
        _lower_simplify("not-a-name")
    with pytest.raises(ValueError):
        _lower_simplify(42.0)
    with pytest.raises(ValueError):
        _lower_simplify({"affine": "sideways"})


# ----------------------------------------------------------------------
#   RESOLUTION (always analytic)
# ----------------------------------------------------------------------


def test_resolution() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert _resolve_simplify(lin, _lower_simplify({"affine": "numeric"})) is (
        numeric
    )
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    # A general affine is not linear, so the None-absent analytic fallback.
    assert _resolve_simplify(aff, _lower_simplify({"linear": "numeric"})) is (
        analytic
    )


def test_resolution_safest_among_matched() -> None:
    sca = Scaling(scale=[2.0, 3.0])
    assert (
        _resolve_simplify(
            sca, _lower_simplify({"affine": "numeric", "scaling": "none"})
        )
        is none
    )
    assert (
        _resolve_simplify(
            sca, _lower_simplify({"affine": "numeric", "linear": "analytic"})
        )
        is analytic
    )


def test_resolution_fallback_and_bare_list() -> None:
    df = DisplacementField(field=_small())
    assert _resolve_simplify(df, _lower_simplify({None: "numeric"})) is numeric
    assert _resolve_simplify(df, _lower_simplify(["affine"])) is none
    # Same lowered key -> collision -> safest.
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert (
        _resolve_simplify(
            lin, _lower_simplify({Affine: "none", "affine": "numeric"})
        )
        is none
    )


# ----------------------------------------------------------------------
#   DEFAULTS AND SUGAR
# ----------------------------------------------------------------------


def test_default_compute_is_analytic() -> None:
    # `Linear(None)` -> `Identity` (structure-only downcast restores main).
    assert isinstance(Linear().compute(), Identity)
    # A diagonal linear stays linear (analytic reads no values), and the
    # object identity is preserved by the no-op-downcast guard.
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert lin.compute() is lin


def test_explicit_none_is_untouched() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    for v in (False, None, "none"):
        assert lin.compute(simplify=v) is lin


def test_numeric_downcast() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert isinstance(lin.simplify("numeric"), Scaling)
    assert isinstance(lin.compute(simplify=True), Scaling)


def test_simplify_sugar_compute_keyword() -> None:
    aff = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 3.0]]))
    # `compute=` gates on a mode: an affine is not admitted by "translation".
    assert aff.simplify("numeric", compute="Translation") is aff
    assert isinstance(
        Linear(matrix=np.diag([2.0, 3.0])).simplify("numeric"), Scaling
    )


def test_subspace_compute() -> None:
    # A subspace of a `None` linear on the same axes computes to the identity.
    sub = SubspaceTransformation(
        transformation=Linear(), input_axes=[0, 1], output_axes=[0, 1]
    )
    assert isinstance(sub.compute(), Identity)
    # A subspace of a numerically-diagonal linear, under a numeric affine
    # policy, downcasts its inner to a scaling.
    sub2 = SubspaceTransformation(
        transformation=Linear(matrix=np.diag([2.0, 3.0])),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    out = sub2.compute(simplify={"affine": "numeric"})
    assert isinstance(out, SubspaceTransformation)
    assert isinstance(out.transformation, Scaling)
    # A non-simplifiable inner leaves the wrapper identical.
    sub3 = SubspaceTransformation(
        transformation=Translation(translation=[1.0, 2.0]),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    assert sub3.compute() is sub3
    # A non-admitting mode returns the wrapper unchanged.
    assert sub3.compute(mode="rotation") is sub3


# ----------------------------------------------------------------------
#   EARLY NUMERIC DOWNCAST + CANCELLATION PRESERVED
# ----------------------------------------------------------------------


def test_numeric_downcast_feeds_composition() -> None:
    zero = DisplacementField(field=np.zeros((5, 5, 2)))
    t1 = Translation(translation=[1.0, 2.0])
    t2 = Translation(translation=[3.0, 4.0])
    seq = Sequence(transformations=[t1, zero, t2])
    numeric_result = seq.compute(simplify="numeric")
    assert isinstance(numeric_result, Translation)
    np.testing.assert_allclose(
        np.asarray(numeric_result.translation), [4.0, 6.0]
    )
    # Under the analytic default the zero field survives (not read).
    assert not isinstance(seq.compute(), Translation)


def test_restricted_numeric_leaves_other_types() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    result = Sequence(transformations=[lin]).compute(
        simplify={"translation": "numeric"}
    )
    assert isinstance(result, Linear)
    assert not isinstance(result, Scaling)


@pytest.mark.parametrize("policy", [True, "numeric"])
def test_affine_inverse_pair_cancels_under_numeric(policy: object) -> None:
    aff = Affine(matrix=[[2.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    result = Sequence(transformations=[aff, aff.inverse()]).compute(
        simplify=policy
    )
    assert isinstance(result, Identity)


@pytest.mark.parametrize("policy", [True, "numeric"])
def test_displacement_pair_cancels_zero_inversions(policy: object) -> None:
    df = DisplacementField(field=_small())
    calls = {"n": 0}
    real = _inv.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_inv, "inverse_disp", counting):
        result = Sequence(transformations=[df, df.inverse()]).compute(
            simplify=policy
        )
    assert isinstance(result, Identity)
    assert calls["n"] == 0


def test_analytic_never_materializes_a_coordinate_inverse() -> None:
    cf = CoordinatesField(field=_small())
    result = Sequence(transformations=[cf, cf.inverse()]).compute(
        simplify="analytic"
    )
    assert isinstance(result, Identity)  # cancelled, not materialized


def test_analytic_does_not_invert_a_displacement() -> None:
    df = DisplacementField(field=_small())
    calls = {"n": 0}
    real = _inv.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_inv, "inverse_disp", counting):
        Sequence(transformations=[df.inverse()]).compute(simplify="analytic")
    assert calls["n"] == 0


def test_opposite_translations_to_identity_under_numeric() -> None:
    seq = Sequence(
        transformations=[
            Translation(translation=[1.0, 2.0]),
            Translation(translation=[-1.0, -2.0]),
        ]
    )
    assert isinstance(seq.compute(simplify=True), Identity)


def test_default_compute_still_cancels() -> None:
    df = DisplacementField(field=_small())
    assert isinstance(
        Sequence(transformations=[df, df.inverse()]).compute(), Identity
    )


# ----------------------------------------------------------------------
#   FINAL PASS IS NUMERIC-GATED
# ----------------------------------------------------------------------


def test_final_pass_only_runs_for_numeric_tables() -> None:
    seq = Sequence(
        transformations=[Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))]
    )
    real = _seq._simplify_result
    calls = {"n": 0}

    def counting(result: object, table: object) -> object:
        calls["n"] += 1
        return real(result, table)

    with mock.patch.object(_seq, "_simplify_result", counting):
        seq.compute(simplify="analytic")
    assert calls["n"] == 0

    calls["n"] = 0
    with mock.patch.object(_seq, "_simplify_result", counting):
        seq.compute(simplify="numeric")
    assert calls["n"] == 1
