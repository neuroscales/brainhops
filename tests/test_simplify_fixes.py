"""Regression tests for the review fixes F1 to F6 of #89."""

import inspect
import types
from unittest import mock

import numpy as np

from brainhops.datamodel import kinds as H
from brainhops.datamodel._transformations import concrete as _concrete
from brainhops.datamodel._transformations.compute.simplify import SimplifyTable
from brainhops.datamodel._transformations.concrete import is_translation
from brainhops.datamodel.enums import SimplifyPolicy
from brainhops.datamodel.kinds import TransformationFamily
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
    real = _concrete.inverse_disp

    def counting(field: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return real(field)

    with mock.patch.object(_concrete, "inverse_disp", counting):
        result = thunk()
    assert calls["n"] == 0
    return result


# Former helper names, expressed with SimplifyTable.


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
    assert result is bij  # F5
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
    assert is_kind(rot, "SO(3)")  # set membership
    assert not is_kind(aff, "inverse")
    assert is_kind(aff.inverse(), "inverse")
    # A name means the set; a class means isinstance.
    assert is_kind(aff, Affine)
    assert is_kind(rot, "affine")
    assert not is_kind(rot, Affine)


# ----------------------------------------------------------------------
#   F4 - is_translation identity linear part
# ----------------------------------------------------------------------


def test_f4_is_translation_numeric() -> None:
    const = Affine(matrix=[[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]])
    transl = Affine(matrix=[[1.0, 0.0, 5.0], [0.0, 1.0, 6.0]])
    assert not is_translation(const, compute=True)
    assert is_translation(transl, compute=True)
    # A constant map is not invertible.
    assert not is_kind(const, H.InvertibleAffine, compute=True)
    assert is_kind(transl, H.Translation, compute=True)


# ----------------------------------------------------------------------
#   F6 - _lower_simplify accepts any Mapping
# ----------------------------------------------------------------------


def test_f6_lower_simplify_accepts_mappingproxy() -> None:
    proxy = types.MappingProxyType({"affine": "numeric"})
    table = normalize_simplify(proxy)
    assert table[TransformationFamily(H.Affine, None)] is (
        SimplifyPolicy.numeric
    )
    assert table[None] is SimplifyPolicy.analytic
    assert resolve_simplify(_aff(), table) is SimplifyPolicy.numeric
