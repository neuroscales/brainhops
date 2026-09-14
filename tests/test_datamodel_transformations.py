"""Tests for transformation rebuilds routed through ``replace``.

These cover the same-class construction paths in the transformation data
model: flattening a sequence, same-type conversions with field
overrides, and the non-idempotent coefficient conversion that must not
run twice.
"""

import numpy as np

from brainhops.datamodel import _xform_converters as xc
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    DisplacementField,
    Sequence,
    Translation,
    _flatten,
)


def test_flatten_removes_nesting_and_keeps_endpoints() -> None:
    inp = CoordinateSystem(name="in")
    out = CoordinateSystem(name="out")
    inner = Sequence(transformations=[Translation(translation=[1.0, 2.0])])
    outer = Sequence(
        transformations=[inner, Translation(translation=[3.0, 4.0])],
        input=inp,
        output=out,
    )
    flat = _flatten(outer)
    assert isinstance(flat, Sequence)
    assert flat.input is inp
    assert flat.output is out
    assert len(flat.transformations) == 2
    assert all(not isinstance(t, Sequence) for t in flat.transformations)


def test_flatten_propagates_endpoints_to_first_and_last() -> None:
    inp = CoordinateSystem(name="in")
    out = CoordinateSystem(name="out")
    first = Translation(translation=[1.0, 2.0])
    last = Translation(translation=[3.0, 4.0])
    seq = Sequence(transformations=[first, last], input=inp, output=out)
    flat = _flatten(seq)
    assert flat.transformations[0].input is inp
    assert flat.transformations[-1].output is out


def test_same_type_conversion_applies_field_override() -> None:
    # Regression: the ``(Transformation, Transformation)`` converter was
    # registered twice and the terminal pass-through overwrote the
    # override-aware one, so ``to(SameType, input=...)`` silently dropped
    # the override.
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    affine = Affine(matrix=matrix)
    target = CoordinateSystem(name="target")
    converted = affine.to(Affine, input=target)
    assert isinstance(converted, Affine)
    assert converted.input is target
    assert np.allclose(converted.matrix, matrix)


def test_same_type_conversion_without_overrides_is_passthrough() -> None:
    affine = Affine(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert affine.to(Affine) is affine


def test_cartesian_field_same_type_rebuild_keeps_shape() -> None:
    # A CartesianField serves `field` through a property backed by
    # `shape`, and its setter rejects a non-None `field`. The rebuild
    # must route to the CartesianField converter (which drops `field`),
    # not the generic CoordinatesField one that feeds the generated
    # field back into the constructor and raises.
    system = CoordinateSystem(name="grid")
    rebuilt = CartesianField(shape=(4, 5)).to(input=system)
    assert isinstance(rebuilt, CartesianField)
    assert rebuilt.shape == (4, 5)
    assert rebuilt.input is system
    assert rebuilt.field.shape == (4, 5, 2)


def test_cartesian_field_flattens_and_computes_in_a_sequence() -> None:
    output = CoordinateSystem(name="B")
    affine = Affine(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    seq = Sequence(
        transformations=[affine, CartesianField(shape=(4, 5))],
        output=output,
    )
    # `compute` flattens the sequence first, which rebuilds the
    # CartesianField through the same-type converter.
    result = seq.compute()
    assert result is not None


def test_coeff_conversion_runs_once(monkeypatch) -> None:  # noqa: ANN001
    # The value-to-coefficient conversion is deliberately non-idempotent.
    # Rebuilding through ``replace`` must reuse the already-converted
    # field rather than converting it a second time.
    calls = {"count": 0}

    def spy(field, order, bound, inplace=False):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return field + 1.0

    monkeypatch.setattr(xc, "value2coeff_field", spy)
    values = np.zeros((5, 6, 2))
    field = DisplacementField(field=values.copy(), order=3, coeff=False)
    coeffs = field.to(coeff=True)
    assert coeffs.coeff is True
    assert calls["count"] == 1
    np.testing.assert_allclose(coeffs.field, values + 1.0)

    # Passing `field=` explicitly supplies the already-converted field,
    # so the conversion is suppressed rather than run a second time.
    calls["count"] = 0
    supplied = np.full((5, 6, 2), 7.0)
    result = field.to(coeff=True, field=supplied)
    assert calls["count"] == 0
    np.testing.assert_allclose(result.field, supplied)
