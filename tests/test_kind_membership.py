"""Tests for level-aware kind membership (`is_member`) and its hooks.

Membership is a predicate `is_member(t, kind, policy)` on the ladder
`none < analytic < numeric`. Resolution (mode / simplify) only ever asks at
`analytic`; `numeric` is reachable via explicit calls and may retract an
analytic shape assumption. See the spec and the `SimplifyPolicy` docstring.
"""

from unittest import mock

import numpy as np
import pytest

from brainhops.datamodel import hierarchy as H
from brainhops.datamodel._transformations import inverse as _inv
from brainhops.datamodel._transformations.modes import (
    _bijective_targets,
    _lift_targets,
    _permute_targets,
    _registered_node,
)
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    InverseAffine,
    Linear,
    Projection,
    Rotation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
    is_member,
)


def _M(t: object, node: str, policy: object = "analytic") -> bool:
    return is_member(t, getattr(H, node), policy)


# ----------------------------------------------------------------------
#   LATTICE TABLES
# ----------------------------------------------------------------------


def test_lift_targets() -> None:
    assert _lift_targets(H.Translation) == (H.Translation,)
    # ConformalEuclidean is not liftable; its maximal liftable subnodes are
    # computed generically from the lattice (Euclidean + the branches the
    # lattice does not place under Euclidean, e.g. Translation / SE).
    ce = _lift_targets(H.ConformalEuclideanTransformation)
    assert H.EuclideanTransformation in ce
    assert H.ConformalEuclideanTransformation not in ce


def test_permute_targets() -> None:
    assert _permute_targets(H.Translation, True) == ()
    assert _permute_targets(H.EuclideanTransformation, False) == (
        H.EuclideanTransformation,
    )
    assert _permute_targets(H.SpecialOrthogonalTransformation, True) == (
        H.SpecialOrthogonalTransformation,
    )
    assert H.SpecialOrthogonalTransformation not in _permute_targets(
        H.SpecialOrthogonalTransformation, False
    )


def test_bijective_targets() -> None:
    assert _bijective_targets(H.AffineTransformation) == (
        H.InvertibleAffineTransformation,
    )
    assert _bijective_targets(H.Transformation) == (H.BijectiveTransformation,)
    assert _bijective_targets(H.InjectiveTransformation) == (
        H.BijectiveTransformation,
    )
    assert _bijective_targets(H.DiagonalTransformation) == (
        H.InvertibleDiagonalTransformation,
    )


def test_registered_node() -> None:
    assert _registered_node(Affine) is H.AffineTransformation
    assert _registered_node(Linear) is H.LinearTransformation
    assert _registered_node(Rotation) is H.SpecialOrthogonalTransformation
    assert _registered_node(Scaling) is H.DiagonalTransformation
    assert _registered_node(Translation) is H.Translation
    assert _registered_node(Identity) is H.IdentityTransformation

    class MyAffine(Affine):
        pass

    assert _registered_node(MyAffine) is H.AffineTransformation
    with pytest.raises(ValueError):
        _registered_node(Sequence)
    with pytest.raises(ValueError):
        _registered_node(DisplacementField)  # a field has no hierarchy set


# ----------------------------------------------------------------------
#   CONCRETE, ANALYTIC (optimistic by shape)
# ----------------------------------------------------------------------


def test_none_parameter_is_identity() -> None:
    lin = Linear()
    for node in (
        "IdentityTransformation",
        "Translation",
        "SpecialOrthogonalTransformation",
        "AffineTransformation",
    ):
        assert _M(lin, node), node


def test_square_affine_is_optimistically_invertible() -> None:
    a = Affine(matrix=np.eye(4)[:3])  # 3x4 -> 3-D -> 3-D, square
    assert _M(a, "AffineTransformation")
    assert _M(a, "InvertibleAffineTransformation")
    assert _M(a, "BijectiveTransformation")
    assert _M(a, "InjectiveTransformation")
    assert _M(a, "SurjectiveTransformation")


def test_wide_affine_is_surjective_only() -> None:
    w = Affine(matrix=np.zeros((2, 4)))  # 3-D -> 2-D
    assert _M(w, "AffineTransformation")
    assert _M(w, "SurjectiveTransformation")
    assert not _M(w, "InvertibleAffineTransformation")
    assert not _M(w, "BijectiveTransformation")
    assert not _M(w, "InjectiveTransformation")


def test_tall_affine_is_injective_only() -> None:
    t = Affine(matrix=np.zeros((4, 3)))  # 2-D -> 4-D
    assert _M(t, "InjectiveTransformation")
    assert not _M(t, "SurjectiveTransformation")
    assert not _M(t, "InvertibleAffineTransformation")


def test_linear_shapes() -> None:
    assert _M(Linear(matrix=np.eye(3)), "InvertibleLinearTransformation")
    assert _M(Linear(matrix=np.zeros((2, 3))), "SurjectiveTransformation")
    assert not _M(
        Linear(matrix=np.zeros((2, 3))), "InvertibleLinearTransformation"
    )


def test_scaling_is_optimistically_invertible() -> None:
    assert _M(Scaling(scale=[2.0, 0.0]), "InvertibleDiagonalTransformation")


def test_trivial_families_are_affine() -> None:
    assert _M(
        Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]]),
        "InvertibleAffineTransformation",
    )
    assert _M(
        Translation(translation=[1.0, 2.0]), "InvertibleAffineTransformation"
    )


def test_fields_establish_nothing_affine() -> None:
    df = DisplacementField(field=np.zeros((4, 4, 2)))
    assert not _M(df, "AffineTransformation")
    assert not _M(df, "InjectiveTransformation")
    assert not _M(df, "SurjectiveTransformation")


def test_cartesian_field_membership_never_builds_field() -> None:
    from brainhops.datamodel.transformations import CartesianField

    empty = CartesianField()
    assert _M(empty, "IdentityTransformation")
    grid = CartesianField(shape=(4, 4))
    with mock.patch.object(
        type(grid),
        "field",
        new=property(
            lambda self: (_ for _ in ()).throw(AssertionError("built field"))
        ),
    ):
        # Structural membership must not touch `.field`.
        assert not _M(grid, "AffineTransformation")
        assert not _M(grid, "IdentityTransformation")


# ----------------------------------------------------------------------
#   CONCRETE, NUMERIC (rank retraction)
# ----------------------------------------------------------------------


def test_numeric_retracts_singular_square() -> None:
    sing = Affine(
        matrix=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    )  # 3x4 square, rank 2
    assert _M(sing, "InvertibleAffineTransformation", "analytic")
    assert not _M(sing, "InvertibleAffineTransformation", "numeric")
    assert not _M(sing, "BijectiveTransformation", "numeric")
    assert _M(sing, "AffineTransformation", "numeric")


def test_numeric_retracts_rank_deficient_wide_and_tall() -> None:
    # 3-D -> 2-D, linear block rank 1.
    wide = Affine(matrix=np.array([[1.0, 1, 1, 0], [2, 2, 2, 0]]))
    assert _M(wide, "SurjectiveTransformation", "analytic")
    assert not _M(wide, "SurjectiveTransformation", "numeric")
    # 2-D -> 4-D, linear block rank 1.
    tall = Affine(
        matrix=np.array([[1.0, 2, 0], [2, 4, 0], [0, 0, 0], [0, 0, 0]])
    )
    assert _M(tall, "InjectiveTransformation", "analytic")
    assert not _M(tall, "InjectiveTransformation", "numeric")


def test_numeric_scaling_zero_factor() -> None:
    s = Scaling(scale=[2.0, 0.0])
    assert not _M(s, "InvertibleDiagonalTransformation", "numeric")
    assert _M(s, "DiagonalTransformation", "numeric")


def test_numeric_only_facts() -> None:
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    assert not _M(lin, "DiagonalTransformation", "analytic")
    assert _M(lin, "DiagonalTransformation", "numeric")
    assert _M(lin, "InvertibleDiagonalTransformation", "numeric")
    assert _M(Linear(matrix=np.eye(2)), "IdentityTransformation", "numeric")


# ----------------------------------------------------------------------
#   WRAPPERS
# ----------------------------------------------------------------------


def test_subspace_lift() -> None:
    same = SubspaceTransformation(
        transformation=Affine(matrix=np.eye(4)[:3]),
        input_axes=[0, 1, 2],
        output_axes=[0, 1, 2],
    )
    assert _M(same, "AffineTransformation")
    assert _M(same, "InvertibleAffineTransformation")
    assert _M(same, "BijectiveTransformation")
    subT = SubspaceTransformation(
        transformation=Translation(translation=[1.0, 2.0]),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    assert _M(subT, "Translation")


def test_subspace_reindex_permutes() -> None:
    # A reindexing subspace composes the lift with a coordinate permutation.
    subT = SubspaceTransformation(
        transformation=Translation(translation=[1.0, 2.0]),
        input_axes=[0, 1],
        output_axes=[1, 0],
    )
    # It is bijective (and affine) but no longer a pure translation.
    assert _M(subT, "BijectiveTransformation")
    assert _M(subT, "AffineTransformation")
    assert not _M(subT, "Translation")
    # NOTE: `Euclidean` is NOT established here because the hierarchy lattice
    # does not place `Translation` under `EuclideanTransformation` (it sits
    # on the dilation/conformal branch). See the report's flagged lattice-gap
    # item. A rotation reindex, which the lattice does place under Euclidean
    # via Orthogonal, establishes it (below).
    subScale = SubspaceTransformation(
        transformation=Scaling(scale=[2.0, 3.0]),
        input_axes=[0, 1],
        output_axes=[1, 0],
    )
    assert _M(subScale, "GeneralizedPermutation")
    subP = SubspaceTransformation(input_axes=[0, 1], output_axes=[1, 0])
    assert _M(subP, "Permutation")


def test_subspace_reindex_rotation_parity() -> None:
    r = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]])
    odd = SubspaceTransformation(
        transformation=r, input_axes=[0, 1, 2], output_axes=[1, 0, 2]
    )  # single swap = odd
    even = SubspaceTransformation(
        transformation=r, input_axes=[0, 1, 2], output_axes=[2, 0, 1]
    )  # 3-cycle = even
    assert not _M(odd, "SpecialOrthogonalTransformation")
    assert _M(odd, "OrthogonalTransformation")
    assert _M(even, "SpecialOrthogonalTransformation")


def test_subspace_of_field_establishes_nothing() -> None:
    sub = SubspaceTransformation(
        transformation=DisplacementField(field=np.zeros((4, 4, 2))),
        input_axes=[0, 1],
        output_axes=[0, 1],
    )
    assert not _M(sub, "AffineTransformation")


def test_inverse_delegates_without_reading_its_parameter() -> None:
    inv = Affine(matrix=np.eye(4)[:3]).inverse()  # InverseAffine
    calls = {"n": 0}
    real = np.linalg.inv

    def counting(m: object) -> object:
        calls["n"] += 1
        return real(m)

    with mock.patch.object(np.linalg, "inv", counting):
        assert _M(inv, "AffineTransformation", "analytic")
        assert _M(inv, "InvertibleAffineTransformation", "analytic")
        assert _M(inv, "AffineTransformation", "numeric")
    assert calls["n"] == 0
    assert _M(Inverse(), "IdentityTransformation")
    assert _M(
        Scaling(scale=[2.0, 3.0]).inverse(), "InvertibleDiagonalTransformation"
    )


def test_field_inverse_never_inverts_and_never_raises() -> None:
    df = DisplacementField(field=np.random.RandomState(0).randn(6, 7, 2))
    calls = {"n": 0}
    real = _inv.inverse_disp

    def counting(f: object) -> object:
        calls["n"] += 1
        return real(f)

    with mock.patch.object(_inv, "inverse_disp", counting):
        for policy in ("analytic", "numeric"):
            assert not _M(df.inverse(), "AffineTransformation", policy)
            # A coordinate-field inverse must never raise from membership.
            _M(
                CoordinatesField(field=df.field).inverse(),
                "AffineTransformation",
                policy,
            )
    assert calls["n"] == 0


def test_inverse_of_projection_is_not_a_function() -> None:
    inv = Inverse(forward=Projection(dropped=[2]))
    assert not _M(inv, "InjectiveTransformation")
    assert not _M(inv, "SurjectiveTransformation")


def test_bijection_membership() -> None:
    r = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]])
    assert _M(Bijection(forward=r), "SpecialOrthogonalTransformation")
    assert _M(Bijection(forward=r), "InvertibleAffineTransformation")
    df = DisplacementField(field=np.zeros((4, 4, 2)))
    assert _M(Bijection(forward=df), "BijectiveTransformation")
    assert _M(Bijection(forward=df), "InjectiveTransformation")
    assert not _M(Bijection(forward=df), "AffineTransformation")
    assert _M(Bijection(backward=r), "SpecialOrthogonalTransformation")


def test_projection_membership() -> None:
    assert _M(Projection(), "IdentityTransformation")
    assert _M(Projection(dropped=[2]), "SurjectiveTransformation")
    assert not _M(Projection(dropped=[2]), "InjectiveTransformation")
    assert not _M(Projection(dropped=[2]), "AffineTransformation")
    assert _M(Projection(created=[2]), "InjectiveTransformation")
    assert not _M(
        Projection(dropped=[2], created=[0]), "InjectiveTransformation"
    )
    assert not _M(
        Projection(dropped=[2], created=[0]), "SurjectiveTransformation"
    )


# ----------------------------------------------------------------------
#   CLASS / CALLABLE KINDS
# ----------------------------------------------------------------------


def test_class_kinds() -> None:
    inv = Affine(matrix=np.eye(4)[:3]).inverse()
    assert is_member(inv, Inverse)
    assert not is_member(Affine(matrix=np.eye(4)[:3]), Inverse)
    df = DisplacementField(field=np.zeros((4, 4, 2)))
    assert is_member(df, (DisplacementField, CoordinatesField))
    assert issubclass(InverseAffine, Inverse)


def test_callable_kind_receives_t_and_policy() -> None:
    seen = {}

    def kind(t: object, policy: object) -> bool:
        seen["t"] = t
        seen["policy"] = policy
        return True

    t = Translation(translation=[1.0])
    assert is_member(t, kind)
    assert seen["t"] is t
    assert str(seen["policy"]) == "analytic"


# ----------------------------------------------------------------------
#   COST INVARIANT: analytic membership never reads values
# ----------------------------------------------------------------------


def test_analytic_membership_calls_no_numeric_routines() -> None:
    # Analytic membership reads only structure, so the numeric routines
    # (rank via `_rank`, matrix inverse, field inversion) are never called.
    import contextlib

    from brainhops.datamodel._transformations import concrete as _concrete

    counts = {"n": 0}

    def bump(fn: object) -> object:
        def wrapped(*a: object, **k: object) -> object:
            counts["n"] += 1
            return fn(*a, **k)

        return wrapped

    cases = [
        Affine(matrix=np.zeros((3, 4))),
        Linear(matrix=np.eye(2)),
        Scaling(scale=[2.0, 3.0]),
        Translation(translation=[1.0, 2.0]),
        SubspaceTransformation(
            transformation=Affine(matrix=np.zeros((3, 4))),
            input_axes=[0, 1, 2],
            output_axes=[0, 1, 2],
        ),
        Bijection(forward=Affine(matrix=np.zeros((3, 4)))),
        Affine(matrix=np.eye(4)[:3]).inverse(),
        DisplacementField(field=np.zeros((4, 4, 2))).inverse(),
    ]
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(_concrete, "_rank", bump(_concrete._rank))
        )
        stack.enter_context(
            mock.patch.object(np.linalg, "inv", bump(np.linalg.inv))
        )
        stack.enter_context(
            mock.patch.object(_inv, "inverse_disp", bump(_inv.inverse_disp))
        )
        for t in cases:
            for node in (
                "AffineTransformation",
                "InvertibleAffineTransformation",
                "BijectiveTransformation",
                "SurjectiveTransformation",
                "Transformation",
            ):
                is_member(t, getattr(H, node), "analytic")
    assert counts["n"] == 0


# ----------------------------------------------------------------------
#   MODE INTEGRATION (composition through membership)
# ----------------------------------------------------------------------


def test_mode_composes_affine_run_of_mixed_kinds() -> None:
    # A run of established affines (of different concrete types) composes to
    # one affine under `mode="affine"`.
    trans = Translation(translation=[1.0, 2.0])
    aff = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]))
    rot = Rotation(matrix=[[0.0, -1.0], [1.0, 0.0]])
    result = Sequence(transformations=[trans, aff, rot]).compute(mode="affine")
    assert is_member(result, H.AffineTransformation)
    assert not isinstance(result, Sequence)


def test_mode_class_means_the_set() -> None:
    # `mode=Affine` (a concrete class) now means the affine SET: a run of
    # affine-set members composes to one leaf.
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    sca = Scaling(scale=[2.0, 3.0])
    aff = Affine(matrix=np.array([[2.0, 0, 1], [0, 3, 2]]))
    result = Sequence(transformations=[aff, lin, sca]).compute(mode=Affine)
    assert is_member(result, H.AffineTransformation)
    assert not isinstance(result, Sequence)


def test_mode_Aff_admits_only_invertible_affines() -> None:
    # Membership drives mode admission: a wide (surjective, non-invertible)
    # affine is admitted by `mode="affine"` but not by `mode="Aff"`.
    wide = Affine(matrix=np.zeros((2, 4)))  # 3-D -> 2-D
    assert is_member(wide, H.AffineTransformation)
    assert not is_member(wide, H.InvertibleAffineTransformation)
