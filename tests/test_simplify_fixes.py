"""Regression tests for the #89-review fixes F1-F6.

F1 - a simplify pass downcasts leaves only; it never composes an inner
     `Sequence` nor materializes a lazy inverse (analytic).
F2 - `Sequence.compute` defaults to `"analytic"` like every other compute.
F3 - public `is_kind` accepts name / int / concrete-class keys.
F4 - `is_translation(compute=True)` needs an identity linear part.
F5 - `Bijection.compute` returns self when unchanged.
F6 - `_lower_simplify` accepts any Mapping.
"""

import inspect
import types
from unittest import mock

import numpy as np

from brainhops.datamodel import hierarchy as H
from brainhops.datamodel._transformations import inverse as _inv
from brainhops.datamodel._transformations.concrete import is_translation
from brainhops.datamodel._transformations.simplify import SimplifyTable
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    DisplacementField,
    Rotation,
    Sequence,
    SubspaceTransformation,
    is_kind,
)


def _small(seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(6, 7, 2) * 0.05


def _aff() -> Affine:
    return Affine(matrix=[[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])


def _zero_inversions(thunk: object) -> object:
    calls = {"n": 0}
    real = _inv.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_inv, "inverse_disp", counting):
        result = thunk()
    assert calls["n"] == 0
    return result

# --- old helpers wrapping new helpers ---------------------------------

def normalize_simplify(value: object) -> SimplifyTable:
    return SimplifyTable.from_like(value)


def resolve_simplify(t: object, table: SimplifyTable) -> SimplifyPolicy:
    return table.resolve(t)

# ----------------------------------------------------------------------
#   F1 - simplify pass never composes or materializes
# ----------------------------------------------------------------------


def test_f1_subspace_of_sequence_never_composes_under_analytic() -> None:
    df = DisplacementField(field=_small(seed=1))
    sub = SubspaceTransformation(
        transformation=Sequence(transformations=[df.inverse(), _aff()]),
        input_axes=[0, 1, 2],
        output_axes=[0, 1, 2],
    )
    _zero_inversions(sub.compute)


def test_f1_sequence_of_subspace_of_sequence_never_composes() -> None:
    for kwargs in ({"simplify": "analytic"}, {"mode": "affine"}):
        df = DisplacementField(field=_small(seed=1))
        sub = SubspaceTransformation(
            transformation=Sequence(transformations=[df.inverse(), _aff()]),
            input_axes=[0, 1, 2],
            output_axes=[0, 1, 2],
        )
        _zero_inversions(
            lambda s=sub, k=kwargs: Sequence(transformations=[s]).compute(**k)
        )


def test_f1_bijection_with_lazy_inverse_never_materializes() -> None:
    df = DisplacementField(field=_small(seed=3))
    bij = Bijection(forward=df.inverse(), backward=df)
    result = _zero_inversions(bij.compute)
    assert result is bij  # F5: unchanged -> identity preserved
    df2 = DisplacementField(field=_small(seed=4))
    bij2 = Bijection(forward=df2.inverse(), backward=df2)
    _zero_inversions(
        lambda: Sequence(transformations=[bij2]).compute(simplify="analytic")
    )


def test_f1_existing_subspace_of_inverse_stays_zero() -> None:
    df = DisplacementField(field=_small(seed=5))
    sub = SubspaceTransformation(
        transformation=df.inverse(), input_axes=[0, 1], output_axes=[0, 1]
    )
    _zero_inversions(lambda: Sequence(transformations=[sub]).compute())


# ----------------------------------------------------------------------
#   F2 - Sequence.compute default
# ----------------------------------------------------------------------


def test_f2_sequence_compute_default_is_analytic() -> None:
    sig = inspect.signature(Sequence.compute)
    assert sig.parameters["simplify"].default == "analytic"


# ----------------------------------------------------------------------
#   F3 - is_kind accepts name / concrete-class keys
# ----------------------------------------------------------------------


def test_f3_is_kind_name_keys() -> None:
    aff = _aff()
    assert is_kind(aff, "affine")
    assert is_kind(aff, "Aff")
    rot = Rotation(matrix=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert is_kind(rot, "SO(3)")  # ndim ignored, set membership
    assert not is_kind(aff, "inverse")
    assert is_kind(aff.inverse(), "inverse")
    # A NAME string means the SET; a concrete class key means `isinstance`.
    assert is_kind(aff, Affine)  # aff is an Affine instance
    assert is_kind(rot, "affine")  # a rotation is in the affine *set*
    assert not is_kind(rot, Affine)  # but is not an `Affine` instance


# ----------------------------------------------------------------------
#   F4 - is_translation identity linear part
# ----------------------------------------------------------------------


def test_f4_is_translation_numeric() -> None:
    const = Affine(matrix=[[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]])  # constant map
    transl = Affine(matrix=[[1.0, 0.0, 5.0], [0.0, 1.0, 6.0]])
    assert not is_translation(const, compute=True)
    assert is_translation(transl, compute=True)
    # A constant map is not invertible; a genuine translation is a Translation.
    assert not is_kind(const, H.InvertibleAffineTransformation, compute=True)
    assert is_kind(transl, H.Translation, compute=True)


# ----------------------------------------------------------------------
#   F6 - _lower_simplify accepts any Mapping
# ----------------------------------------------------------------------


def test_f6_lower_simplify_accepts_mappingproxy() -> None:
    proxy = types.MappingProxyType({"affine": "numeric"})
    table = normalize_simplify(proxy)
    assert table[(H.AffineTransformation, None)] is SimplifyPolicy.numeric
    assert table[None] is SimplifyPolicy.analytic
    # And it resolves like the dict form.
    assert resolve_simplify(_aff(), table) is SimplifyPolicy.numeric
