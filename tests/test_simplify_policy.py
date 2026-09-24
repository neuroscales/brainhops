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
from brainhops.datamodel._transformations.modes import normalize_family
from brainhops.datamodel._transformations.simplify import SimplifyTable
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.hierarchy import TransformationFamily
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
AFF = TransformationFamily(hierarchy.AffineTransformation, None)
LIN = TransformationFamily(hierarchy.LinearTransformation, None)


def _small(shape: tuple = (6, 7, 2), seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(*shape) * 0.05

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
    # A concrete class key is an `isinstance` kind (the class itself), NOT
    # the affine *set*: `Affine` means `isinstance(t, Affine)`, `"affine"`
    # means the set.
    assert normalize_simplify(Affine) == {
        None: none,
        TransformationFamily(Affine, None): analytic
    }
    assert normalize_simplify(hierarchy.LinearTransformation) == {
        None: none,
        LIN: analytic,
    }
    table = normalize_simplify(["scaling", "translation"])
    assert table[None] is none
    assert table[
        TransformationFamily(hierarchy.DiagonalTransformation, None)
    ] is analytic
    assert table[
        TransformationFamily(hierarchy.Translation, None)
    ] is analytic


def test_mapping_fallback_and_collisions() -> None:
    # `None`-absent mapping defaults its fallback to analytic.
    t = normalize_simplify({"affine": "numeric"})
    assert t == {AFF: numeric, None: analytic}
    assert t == normalize_simplify({None: "analytic", "affine": "numeric"})
    assert normalize_simplify({None: "none", "affine": "numeric"}) == {
        AFF: numeric,
        None: none,
    }
    # Colliding lowered keys keep the safest policy. Two spellings of the
    # same set NAME collide; a concrete-class key (`Affine`) is a distinct
    # `isinstance` key and does not collide with the set NAME "affine".
    assert (
        normalize_simplify({"affine": "numeric", "AFFINE": "none"})[AFF]
        is none
    )
    t = normalize_simplify({"affine": "numeric", Affine: "none"})
    assert t[AFF] is numeric  # the set key
    assert t[(Affine, None)] is none  # the distinct class key


def test_lowering_is_idempotent() -> None:
    t = normalize_simplify({"affine": "numeric"})
    assert normalize_simplify(t) == t


def test_special_and_symbol_keys() -> None:
    assert normalize_family("SO(3)") == (
        hierarchy.SpecialOrthogonalTransformation,
        3,
    )
    assert normalize_family(3) == (hierarchy.Transformation, 3)
    assert normalize_family("scaling") == (
        hierarchy.DiagonalTransformation,
        None,
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
        "displacements",  # plural (class name stays DisplacementField)
        "coordinates",  # plural (class name stays CoordinatesField)
    ):
        kind, ndim = normalize_family(key)
        assert isinstance(kind, (type, tuple))
    assert normalize_family("bijection")[0] is (
        hierarchy.BijectiveTransformation
    )
    assert normalize_family("rotation")[0] is (
        hierarchy.SpecialOrthogonalTransformation
    )
    # A concrete class is an `isinstance` kind (the class itself), never the
    # hierarchy set it registered to.
    assert normalize_family(Rotation) == (Rotation, None)
    assert normalize_family(Identity) == (Identity, None)
    # a class kind
    assert normalize_family(InverseAffine) == (InverseAffine, None)
    assert normalize_family(DisplacementField) == (DisplacementField, None)
    assert normalize_family(Projection) == (Projection, None)
    assert normalize_family(CartesianField) == (CartesianField, None)
    _ = Inverse


def test_invalid_keys_raise() -> None:
    from brainhops.datamodel.transformations import MultiscaleField

    # A class with no hierarchy node is now a valid `isinstance` kind (it no
    # longer has to resolve to a set), so it lowers rather than raising.
    assert normalize_family(Sequence) == (Sequence, None)
    assert normalize_family(MultiscaleField) == (MultiscaleField, None)
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
    assert resolve_simplify(
        lin, normalize_simplify({"affine": "numeric"})
    ) is numeric
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    # A general affine is not linear, so the None-absent analytic fallback.
    assert resolve_simplify(
        aff, normalize_simplify({"linear": "numeric"})
    ) is analytic


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
    assert resolve_simplify(
        df, normalize_simplify({None: "numeric"})
    ) is numeric
    assert resolve_simplify(df, normalize_simplify(["affine"])) is none
    # Two spellings of the same set NAME collide on one lowered key -> safest.
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
#   THE PRODUCTS OF A COMPOSITION ARE SIMPLIFIED TOO
# ----------------------------------------------------------------------


def test_composition_products_are_downcast() -> None:
    # The simplify pass runs before each composition, so the leaves a
    # composition *produces* would otherwise escape it. `_compute_sequence`
    # runs one last pass over them on its way out: two translations compose
    # to a translation whose vector is zero, which a numeric policy then
    # downcasts to the identity.
    seq = Sequence(
        transformations=[
            Translation(translation=[1.0, 2.0]),
            Translation(translation=[-1.0, -2.0]),
        ]
    )
    assert isinstance(seq.compute(simplify="numeric"), Identity)
    # Under the analytic default the composed vector is not read, so the
    # result stays a translation.
    assert not isinstance(seq.compute(simplify="analytic"), Identity)


class _GuardedField:
    """Stands in for a field and refuses to be read.

    A structural check reads only `field is None` and, at most, the shape;
    every numeric check compares or indexes the values. So each of those
    raises, wherever it is reached from. It is a plain object rather than
    an `ndarray` subclass because the comparison protocol an array
    subclass takes part in differs across NumPy versions.
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
    # The cost invariant: an analytic run downcasts from structure alone, so
    # it never touches the values of a field -- not in the per-round pass,
    # and not in the pass over the composition products.
    _guarded_sequence().compute(mode=False, simplify="analytic")


def test_the_value_guard_is_not_vacuous() -> None:
    # The companion to the test above: under a numeric policy the very same
    # sequence *does* read the field, so the guard is known to fire.
    with pytest.raises(AssertionError, match="values were read"):
        _guarded_sequence().compute(mode=False, simplify="numeric")


# ----------------------------------------------------------------------
#   THE DOWNCAST LADDER
# ----------------------------------------------------------------------


def test_ladder_stops_at_the_cheapest_type_and_never_widens() -> None:
    # The ladder rewrites a transform as the first *set* it is established
    # in, paired with the class that represents it. A transform already at
    # or below that rung is left alone -- and left as the same object, so a
    # lazy inverse next to it still cancels by identity.
    rot = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]])
    # A rotation is in the linear set, but `Linear` is a *wider* type: the
    # ladder must not widen it.
    assert rot.simplify("numeric") is rot
    # A `Linear` that happens to be a rotation does narrow to one, which is
    # what makes its inverse a transpose rather than a solve.
    narrowed = Linear(matrix=[[0.0, -1.0], [1.0, 0.0]]).simplify("numeric")
    assert type(narrowed) is Rotation
    np.testing.assert_allclose(
        np.asarray(narrowed.inverse().matrix), [[0.0, 1.0], [-1.0, 0.0]]
    )
    # A permutation is cheaper than the rotation it also is, so an even
    # permutation matrix stops at the permutation rung.
    swap3 = Linear(matrix=np.eye(3)[[2, 0, 1]])
    assert type(swap3.simplify("numeric")).__name__ == "Permutation"


def test_ladder_leaves_a_grid_alone() -> None:
    # A `CartesianField` is the identity map over its own grid, so the
    # ladder would collapse it to `Identity` and lose the sampling domain.
    # It is never downcast, at any policy.
    from brainhops.datamodel.transformations import CartesianField

    grid = CartesianField(shape=(4, 5))
    assert grid.simplify("numeric") is grid
    assert grid.simplify("analytic") is grid


def test_a_lazy_inverse_is_never_materialized_by_a_downcast() -> None:
    # Resolving an inverse is computation, not simplification, so no policy
    # makes the downcast do it -- not even `numeric`.
    df = DisplacementField(field=_small())
    calls = {"n": 0}
    real = _inv.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_inv, "inverse_disp", counting):
        lazy = df.inverse()
        assert lazy.simplify("numeric") is lazy
    assert calls["n"] == 0


def test_a_downcast_forward_rewraps_as_its_own_family() -> None:
    # Simplifying what an inverse wraps can change the wrapper's family:
    # the inverse of a linear that turns out to be a scaling is an inverse
    # *scaling*, which reciprocates rather than solving.
    from brainhops.datamodel.transformations import InverseScaling

    lazy = Linear(matrix=np.diag([2.0, 4.0])).inverse()
    assert type(lazy).__name__ == "InverseLinear"
    assert isinstance(lazy.simplify("numeric"), InverseScaling)


# ----------------------------------------------------------------------
#   THE TWO-ARGUMENT DISPATCHER
# ----------------------------------------------------------------------


def test_simplify_takes_one_or_two_transforms() -> None:
    from brainhops.datamodel._transformations.simplify import simplify

    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    # One in, one out: total, always a transform back.
    assert simplify(aff, policy="analytic") is aff
    # Two in, one out: partial. `first` is applied before `second`.
    assert isinstance(
        simplify(aff, aff.inverse(), policy="analytic"), Identity
    )
    assert isinstance(
        simplify(aff.inverse(), aff, policy="analytic"), Identity
    )
    # A pair that does not collapse declines with `None` rather than
    # raising -- that is the ordinary answer over a sequence.
    other = Affine(matrix=np.array([[5.0, 0, 0], [0, 7, 0]]))
    assert simplify(aff, other, policy="analytic") is None


def test_simplify_rejects_a_policy_passed_positionally() -> None:
    from brainhops.datamodel._transformations.simplify import simplify

    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    with pytest.raises(TypeError, match="positionally"):
        simplify(aff, "analytic")


def test_a_none_policy_declines_every_pair() -> None:
    # The policy gate has `analytic` as its floor: a pair whose resolved
    # policy is `none` is left alone, so the two survive as a sequence.
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    seq = Sequence(transformations=[aff, aff.inverse()])
    assert isinstance(seq.compute(mode=False, simplify=False), Sequence)
    assert isinstance(seq.compute(mode=False, simplify="analytic"), Identity)


# ----------------------------------------------------------------------
#   A PAIR RULE ASSUMES THE BOUNDARY LINES UP
# ----------------------------------------------------------------------


def _reordered_pair() -> tuple:
    # An identity that leaves coordinates in RAS, followed by a transform
    # that reads them in a reordered system. A real axis permutation sits
    # between the two; nothing may drop it.
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem

    ras = CoordinateSystem(name="ras", axes=[R, A, S])
    reordered = CoordinateSystem(name="reordered", axes=[S, R, A])
    return (
        Identity(input=ras, output=ras),
        Affine(matrix=np.eye(4)[:3], input=reordered, output=reordered),
    )


def test_a_pair_over_a_disagreeing_boundary_is_declined() -> None:
    # Every pair rule assumes the two ends of the boundary line up: the
    # identity rule would drop the identity and hand back the affine,
    # swallowing the reordering. The dispatcher refuses the pair instead,
    # so both survive.
    ident, aff = _reordered_pair()
    simplified = Sequence(transformations=[ident, aff]).simplify()
    assert isinstance(simplified, Sequence)
    assert len(simplified) == 2


def test_compose_says_so_rather_than_swallowing_the_boundary() -> None:
    from brainhops.datamodel._transformations.compose import compose
    from brainhops.datamodel.transformations import CompositionError

    ident, aff = _reordered_pair()
    with pytest.raises(CompositionError, match="disagree on the system"):
        compose(aff, ident)


def test_compute_bridges_the_boundary_it_declined_to_swallow() -> None:
    # `compute` reconciles the boundary before it simplifies, so the
    # reordering ends up in the result rather than being lost: the composed
    # affine is the permutation matrix, not the identity.
    ident, aff = _reordered_pair()
    result = Sequence(transformations=[ident, aff]).compute()
    np.testing.assert_allclose(
        np.asarray(result.matrix), np.eye(4)[[2, 0, 1]][:, :4]
    )


def test_an_agreeing_boundary_still_collapses() -> None:
    from brainhops.datamodel.axes import A, R, S
    from brainhops.datamodel.systems import CoordinateSystem

    ras = CoordinateSystem(name="ras", axes=[R, A, S])
    pair = Sequence(
        transformations=[
            Identity(input=ras, output=ras),
            Affine(matrix=np.eye(4)[:3], input=ras, output=ras),
        ]
    )
    assert isinstance(pair.simplify(), Affine)
