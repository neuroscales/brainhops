"""Tests for the simplify grammar, its resolution, and compute."""

from unittest import mock

import numpy as np
import pytest

from brainhops.datamodel import kinds
from brainhops.datamodel._transformations import concrete as _concrete
from brainhops.datamodel._transformations.compute.simplify import SimplifyTable
from brainhops.datamodel._transformations.modes import normalize_family
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.kinds import TransformationFamily
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
AFF = TransformationFamily(kinds.Affine, None)
LIN = TransformationFamily(kinds.Linear, None)


def _small(shape: tuple = (6, 7, 2), seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape) * 0.05


# Former helper names, expressed with SimplifyTable.


# --- old helpers wrapping new helpers ---------------------------------


def normalize_simplify(value: object) -> SimplifyTable:
    return SimplifyTable.from_like(value)


def resolve_simplify(t: object, table: SimplifyTable) -> SimplifyPolicy:
    return table.resolve(t)


# ----------------------------------------------------------------------
#   LOWERING TO A TABLE
# ----------------------------------------------------------------------


def test_scalar_forms() -> None:
    for v in (None, False, "none", "NONE", none):
        assert normalize_simplify(v) == {None: none}
    for v in ("analytic", analytic):
        assert normalize_simplify(v) == {None: analytic}
    for v in (True, "numeric", "Numeric", numeric):
        assert normalize_simplify(v) == {None: numeric}


def test_key_and_list_restrict_to_those() -> None:
    assert normalize_simplify("affine") == {None: none, AFF: analytic}
    # A class key means isinstance; a name means the set.
    assert normalize_simplify(Affine) == {
        None: none,
        TransformationFamily(Affine, None): analytic,
    }
    assert normalize_simplify(kinds.Linear) == {
        None: none,
        LIN: analytic,
    }
    table = normalize_simplify(["scaling", "translation"])
    assert table[None] is none
    assert table[TransformationFamily(kinds.Diagonal, None)] is analytic
    assert table[TransformationFamily(kinds.Translation, None)] is analytic


def test_mapping_fallback_and_collisions() -> None:
    # Without a None key, the fallback is analytic.
    t = normalize_simplify({"affine": "numeric"})
    assert t == {AFF: numeric, None: analytic}
    assert t == normalize_simplify({None: "analytic", "affine": "numeric"})
    assert normalize_simplify({None: "none", "affine": "numeric"}) == {
        AFF: numeric,
        None: none,
    }
    # Colliding keys keep the safest policy; a class key does not collide with
    # a name.
    assert (
        normalize_simplify({"affine": "numeric", "AFFINE": "none"})[AFF]
        is none
    )
    t = normalize_simplify({"affine": "numeric", Affine: "none"})
    assert t[AFF] is numeric
    assert t[TransformationFamily(Affine, None)] is none


def test_lowering_is_idempotent() -> None:
    t = normalize_simplify({"affine": "numeric"})
    assert normalize_simplify(t) == t


def test_special_and_symbol_keys() -> None:
    assert normalize_family("SO(3)") == TransformationFamily(
        kinds.SpecialOrthogonal, 3
    )
    assert normalize_family(3) == TransformationFamily(kinds.Transformation, 3)
    assert normalize_family("scaling") == TransformationFamily(
        kinds.Diagonal, None
    )
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
        "displacements",
        "coordinates",
    ):
        assert isinstance(normalize_family(key).kind, type)
    assert normalize_family("bijection").kind is kinds.Bijection
    assert normalize_family("rotation").kind is kinds.SpecialOrthogonal
    # A class is an isinstance kind, never its hierarchy set.
    assert normalize_family(Rotation) == TransformationFamily(Rotation, None)
    assert normalize_family(Identity) == TransformationFamily(Identity, None)
    assert normalize_family(InverseAffine) == TransformationFamily(
        InverseAffine, None
    )
    assert normalize_family(DisplacementField) == TransformationFamily(
        DisplacementField, None
    )
    assert normalize_family(Projection) == TransformationFamily(
        Projection, None
    )
    assert normalize_family(CartesianField) == TransformationFamily(
        CartesianField, None
    )
    _ = Inverse


def test_invalid_keys_raise() -> None:
    from brainhops.datamodel.transformations import MultiscaleField

    # Classes without a hierarchy node are normalized to a family without a
    # node rather than raising.
    assert normalize_family(Sequence) == TransformationFamily(Sequence, None)
    assert normalize_family(MultiscaleField) == TransformationFamily(
        MultiscaleField, None
    )
    with pytest.raises(ValueError):
        normalize_simplify("not-a-name")
    with pytest.raises(ValueError):
        normalize_simplify(42.0)
    with pytest.raises(ValueError):
        normalize_simplify({"affine": "sideways"})


# ----------------------------------------------------------------------
#   RESOLUTION (always analytic)
# ----------------------------------------------------------------------


def test_resolution() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert (
        resolve_simplify(lin, normalize_simplify({"affine": "numeric"}))
        is numeric
    )
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    # A general affine is not linear and falls back to analytic.
    assert (
        resolve_simplify(aff, normalize_simplify({"linear": "numeric"}))
        is analytic
    )


def test_resolution_safest_among_matched() -> None:
    sca = Scaling(scale=[2.0, 3.0])
    assert (
        resolve_simplify(
            sca,
            normalize_simplify({"affine": "numeric", "scaling": "none"}),
        )
        is none
    )
    assert (
        resolve_simplify(
            sca,
            normalize_simplify({"affine": "numeric", "linear": "analytic"}),
        )
        is analytic
    )


def test_resolution_fallback_and_bare_list() -> None:
    df = DisplacementField(field=_small())
    assert (
        resolve_simplify(df, normalize_simplify({None: "numeric"})) is numeric
    )
    assert resolve_simplify(df, normalize_simplify(["affine"])) is none
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert (
        resolve_simplify(
            lin, normalize_simplify({"AFFINE": "none", "affine": "numeric"})
        )
        is none
    )


# ----------------------------------------------------------------------
#   DEFAULTS AND SUGAR
# ----------------------------------------------------------------------


def test_default_compute_is_analytic() -> None:
    # A default Linear is downcast to Identity from its structure alone.
    assert isinstance(Linear().compute(), Identity)
    # A diagonal Linear stays a Linear under analytic.
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
    # "Translation" does not admit this affine.
    assert aff.simplify("numeric", compute="Translation") is aff
    assert isinstance(
        Linear(matrix=np.diag([2.0, 3.0])).simplify("numeric"), Scaling
    )


def test_subspace_compute() -> None:
    sub = SubspaceTransformation(
        transformation=Linear(), input_axes=[0, 1], output_axes=[0, 1]
    )
    assert isinstance(sub.compute(), Identity)
    sub2 = SubspaceTransformation(
        transformation=Linear(matrix=np.diag([2.0, 3.0])),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    out = sub2.compute(simplify={"affine": "numeric"})
    assert isinstance(out, SubspaceTransformation)
    assert isinstance(out.transformation, Scaling)
    sub3 = SubspaceTransformation(
        transformation=Translation(translation=[1.0, 2.0]),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    assert sub3.compute() is sub3
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
    # Analytic does not read the zero field.
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
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
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
    assert isinstance(result, Identity)


def test_analytic_does_not_invert_a_displacement() -> None:
    df = DisplacementField(field=_small())
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
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
#   THE PRODUCTS OF A COMPOSITION ARE SIMPLIFIED TOO
# ----------------------------------------------------------------------


def test_composition_products_are_downcast() -> None:
    # A final pass also downcasts the products of composition.
    seq = Sequence(
        transformations=[
            Translation(translation=[1.0, 2.0]),
            Translation(translation=[-1.0, -2.0]),
        ]
    )
    assert isinstance(seq.compute(simplify="numeric"), Identity)
    assert not isinstance(seq.compute(simplify="analytic"), Identity)


class _GuardedField:
    """Field values that raise when read.

    Structural checks read only `field is None` and the shape. A plain object
    is used because the array comparison protocol varies across NumPy versions.
    """

    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def _refuse(self, *args: object) -> None:
        raise AssertionError("the field's values were read")

    __eq__ = _refuse
    __ne__ = _refuse
    __lt__ = _refuse
    __gt__ = _refuse
    __getitem__ = _refuse
    __array__ = _refuse
    __hash__ = None


def _guarded_sequence() -> Sequence:
    return Sequence(
        transformations=[
            Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]])),
            DisplacementField(field=_GuardedField((6, 7, 2))),
        ]
    )


def test_analytic_never_reads_a_value() -> None:
    _guarded_sequence().compute(mode=False, simplify="analytic")


def test_the_value_guard_is_not_vacuous() -> None:
    # Numeric does read the field, so the guard can fire.
    with pytest.raises(AssertionError, match="values were read"):
        _guarded_sequence().compute(mode=False, simplify="numeric")


# ----------------------------------------------------------------------
#   THE DOWNCAST LADDER
# ----------------------------------------------------------------------


def test_ladder_stops_at_the_cheapest_type_and_never_widens() -> None:
    # A transform is rewritten as the first type on the ladder that it is
    # shown to belong to. Otherwise the same object is returned, so that a
    # lazy inverse still cancels by identity.
    rot = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]])
    # The ladder never widens a Rotation to a Linear.
    assert rot.simplify("numeric") is rot
    # The inverse of a narrowed Rotation is then a transpose rather than a
    # linear solve.
    narrowed = Linear(matrix=[[0.0, -1.0], [1.0, 0.0]]).simplify("numeric")
    assert type(narrowed) is Rotation
    np.testing.assert_allclose(
        np.asarray(narrowed.inverse().matrix), [[0.0, 1.0], [-1.0, 0.0]]
    )
    # A permutation is cheaper than a rotation.
    swap3 = Linear(matrix=np.eye(3)[[2, 0, 1]])
    assert type(swap3.simplify("numeric")).__name__ == "Permutation"


def test_ladder_leaves_a_grid_alone() -> None:
    # Collapsing a grid to Identity would lose its sampling domain.
    from brainhops.datamodel.transformations import CartesianField

    grid = CartesianField(shape=(4, 5))
    assert grid.simplify("numeric") is grid
    assert grid.simplify("analytic") is grid


def test_a_lazy_inverse_is_never_materialized_by_a_downcast() -> None:
    # Resolving an inverse is computation, not simplification.
    df = DisplacementField(field=_small())
    calls = {"n": 0}
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        lazy = df.inverse()
        assert lazy.simplify("numeric") is lazy
    assert calls["n"] == 0


def test_a_downcast_forward_rewraps_as_its_own_family() -> None:
    # Simplifying the operand can change the family of the wrapper.
    from brainhops.datamodel.transformations import InverseScaling

    lazy = Linear(matrix=np.diag([2.0, 4.0])).inverse()
    assert type(lazy).__name__ == "InverseLinear"
    assert isinstance(lazy.simplify("numeric"), InverseScaling)


# ----------------------------------------------------------------------
#   THE TWO-ARGUMENT DISPATCHER
# ----------------------------------------------------------------------


def test_simplify_takes_one_or_two_transforms() -> None:
    from brainhops.datamodel._transformations.compute.simplify import simplify

    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    assert simplify(aff, policy="analytic") is aff
    assert isinstance(
        simplify(aff, aff.inverse(), policy="analytic"), Identity
    )
    assert isinstance(
        simplify(aff.inverse(), aff, policy="analytic"), Identity
    )
    # A pair that does not collapse declines with None.
    other = Affine(matrix=np.array([[5.0, 0, 0], [0, 7, 0]]))
    assert simplify(aff, other, policy="analytic") is None


def test_simplify_rejects_a_policy_passed_positionally() -> None:
    from brainhops.datamodel._transformations.compute.simplify import simplify

    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    with pytest.raises(TypeError, match="positionally"):
        simplify(aff, "analytic")


def test_a_none_policy_declines_every_pair() -> None:
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    seq = Sequence(transformations=[aff, aff.inverse()])
    assert isinstance(seq.compute(mode=False, simplify=False), Sequence)
    assert isinstance(seq.compute(mode=False, simplify="analytic"), Identity)


# ----------------------------------------------------------------------
#   A PAIR RULE ASSUMES THE BOUNDARY LINES UP
# ----------------------------------------------------------------------


def _reordered_pair() -> tuple:
    # A real axis permutation sits between the two, and nothing may drop it.
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem

    ras = CoordinateSystem(name="ras", axes=[R(), A(), S()])
    reordered = CoordinateSystem(name="reordered", axes=[S(), R(), A()])
    return (
        Identity(input=ras, output=ras),
        Affine(matrix=np.eye(4)[:3], input=reordered, output=reordered),
    )


def test_a_pair_over_a_disagreeing_boundary_is_declined() -> None:
    # The identity rule would swallow the reordering, so the pair is refused.
    ident, aff = _reordered_pair()
    simplified = Sequence(transformations=[ident, aff]).simplify()
    assert isinstance(simplified, Sequence)
    assert len(simplified) == 2


def test_compose_says_so_rather_than_swallowing_the_boundary() -> None:
    from brainhops.datamodel._transformations.compute.compose import compose
    from brainhops.errors import CompositionError

    ident, aff = _reordered_pair()
    with pytest.raises(CompositionError, match="disagree on the system"):
        compose(aff, ident)


def test_compute_bridges_the_boundary_it_declined_to_swallow() -> None:
    # The reconciled boundary keeps the permutation.
    ident, aff = _reordered_pair()
    result = Sequence(transformations=[ident, aff]).compute()
    np.testing.assert_allclose(
        np.asarray(result.matrix), np.eye(4)[[2, 0, 1]][:, :4]
    )


def test_an_agreeing_boundary_still_collapses() -> None:
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem

    ras = CoordinateSystem(name="ras", axes=[R(), A(), S()])
    pair = Sequence(
        transformations=[
            Identity(input=ras, output=ras),
            Affine(matrix=np.eye(4)[:3], input=ras, output=ras),
        ]
    )
    assert isinstance(pair.simplify(), Affine)
