"""Tests for transformation rebuilds routed through ``replace``.

These cover the same-class construction paths in the transformation data
model: flattening a sequence, same-type conversions with field
overrides, and the non-idempotent coefficient conversion that must not
run twice.
"""

import numpy as np
from bagof.magic import fields_dict, replace

from brainhops.datamodel import _xform_converters as xc
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Sequence,
    SubspaceTransformation,
    Translation,
    is_identity,
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
    flat = outer._flattened()
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
    flat = seq._flattened()
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


def test_gridded_cartesian_field_is_the_identity_only_under_compute() -> None:
    # A `CartesianField` is the identity map over its grid by
    # construction. Under `compute=True` it is recognized as the identity,
    # which is the predicate the sequence simplifier uses to factor away an
    # interior grid. Under `compute=False` the check stays conservative,
    # because the grid's `field` parameter is set.
    grid = CartesianField(shape=(4, 5))
    assert is_identity(grid) is False
    assert is_identity(grid, compute=True) is True


def test_empty_cartesian_field_is_the_identity() -> None:
    # With no grid, the `field` parameter is unset, so an empty
    # `CartesianField` is recognized as the identity by the parameter
    # check.
    assert is_identity(CartesianField()) is True


def test_compute_preserves_a_leading_cartesian_field() -> None:
    # A leading grid is a domain restriction, not an identity. Computing a
    # sequence that starts from a grid must keep the grid as a coordinate
    # field rather than fold it away, so the resulting field carries the
    # grid's shape.
    grid = CartesianField(shape=(4, 5))
    affine = Affine(matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = Sequence(transformations=[grid, affine]).compute()
    assert result.field is not None
    assert result.field.shape == (4, 5, 2)
    # The affine scales the grid coordinates by two.
    np.testing.assert_allclose(result.field, np.asarray(grid.field) * 2.0)


def test_computing_an_identity_only_sequence_still_simplifies() -> None:
    # Removing the `CartesianField` branch from `is_identity` must not
    # affect a genuine identity, which still simplifies to `Identity`.
    seq = Sequence(transformations=[Affine(matrix=np.eye(3)[:-1])])
    assert isinstance(seq.compute().to(Identity), Identity)


def test_cartesian_field_is_not_an_init_field_but_base_is() -> None:
    # `field` is computed from `shape` on a CartesianField, so it is not a
    # constructor-taken field there. The base CoordinatesField keeps
    # `field` as a normal init field.
    assert "field" not in fields_dict(CartesianField)
    assert "field" in fields_dict(CoordinatesField)


def test_replace_cartesian_field_changes_endpoints_and_keeps_shape() -> None:
    inp = CoordinateSystem(name="in")
    out = CoordinateSystem(name="out")
    cf = CartesianField(shape=(4, 5, 6))
    replaced = replace(cf, input=inp, output=out)
    assert isinstance(replaced, CartesianField)
    assert replaced.shape == (4, 5, 6)
    assert replaced.input is inp
    assert replaced.output is out
    # The field is regenerated lazily from the shape.
    assert replaced.field.shape == (4, 5, 6, 3)
    expected = np.stack(
        np.meshgrid(*(np.arange(s) for s in (4, 5, 6)), indexing="ij"), -1
    )
    np.testing.assert_array_equal(np.asarray(replaced.field), expected)


def test_replace_cartesian_field_changes_order_and_bound() -> None:
    cf = CartesianField(shape=(3, 4))
    replaced = replace(
        cf,
        order=InterpolationOrder.cubic,
        bound=BoundaryCondition.reflect,
    )
    assert isinstance(replaced, CartesianField)
    assert replaced.order == InterpolationOrder.cubic
    assert replaced.bound == BoundaryCondition.reflect
    # Everything not named is carried over unchanged.
    assert replaced.shape == (3, 4)
    assert replaced.coeff is False
    assert replaced.field.shape == (3, 4, 2)


def test_to_same_type_cartesian_field_changes_output() -> None:
    # The converter path (`.to`) rebuilds a CartesianField with the
    # endpoint changed, without its explicit `field=None` workaround, and
    # the field is still absent from the constructor and lazily computable.
    output = CoordinateSystem(name="out")
    rebuilt = CartesianField(shape=(4, 5)).to(output=output)
    assert isinstance(rebuilt, CartesianField)
    assert rebuilt.output is output
    assert rebuilt.shape == (4, 5)
    assert "field" not in fields_dict(type(rebuilt))
    assert rebuilt.field.shape == (4, 5, 2)


def test_replace_coordinates_field_round_trips_explicit_field() -> None:
    # Guard against regressing the base: CoordinatesField takes `field` as
    # a normal init field, so replace carries an explicit array over.
    values = np.zeros((5, 6, 2))
    cf = CoordinatesField(field=values.copy(), order=3, coeff=True)
    replaced = replace(cf, order=1)
    assert isinstance(replaced, CoordinatesField)
    assert not isinstance(replaced, CartesianField)
    assert replaced.order == 1
    assert replaced.coeff is True
    np.testing.assert_array_equal(np.asarray(replaced.field), values)


def _contains_cartesian_field(result) -> bool:  # noqa: ANN001
    # Walk a computed result and report whether any `CartesianField`
    # survives, descending into a `Sequence` result.
    if isinstance(result, CartesianField):
        return True
    if isinstance(result, Sequence):
        return any(_contains_cartesian_field(t) for t in result)
    return False


def test_interior_grid_is_factored_away() -> None:
    # A grid that sits strictly between two transformations is the
    # identity map over its grid, and its neighbours overwrite those
    # coordinates. Computing the sequence must drop the grid and yield the
    # same transformation as the sequence without it.
    a = Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]])
    b = Affine(matrix=[[1.0, 0.0, 4.0], [0.0, 1.0, 5.0]])
    grid = CartesianField(shape=(4, 5))
    result = Sequence(transformations=[a, grid, b]).compute()
    assert not _contains_cartesian_field(result)
    reference = Sequence(transformations=[a, b]).compute()
    np.testing.assert_allclose(result.matrix, reference.matrix)


def test_multiple_interior_grids_are_all_factored_away() -> None:
    # Every strictly interior grid is factored away, not just the first.
    a = Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]])
    b = Affine(matrix=[[1.0, 0.0, 4.0], [0.0, 1.0, 5.0]])
    grid1 = CartesianField(shape=(4, 5))
    grid2 = CartesianField(shape=(6, 7))
    result = Sequence(transformations=[a, grid1, grid2, b]).compute()
    assert not _contains_cartesian_field(result)
    reference = Sequence(transformations=[a, b]).compute()
    np.testing.assert_allclose(result.matrix, reference.matrix)


def test_interior_grid_preserves_endpoint_systems() -> None:
    # Factoring an interior grid away must keep the endpoints of the
    # chain: the result spans the input of the first transformation and
    # the output of the last one.
    inp = CoordinateSystem(name="in")
    mid = CoordinateSystem(name="mid")
    out = CoordinateSystem(name="out")
    a = Affine(
        matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]], input=inp, output=mid
    )
    grid = CartesianField(shape=(4, 5), input=mid, output=mid)
    b = Affine(
        matrix=[[1.0, 0.0, 4.0], [0.0, 1.0, 5.0]], input=mid, output=out
    )
    result = Sequence(transformations=[a, grid, b]).compute()
    assert result.input is inp
    assert result.output is out


def test_trailing_grid_is_preserved() -> None:
    # A grid in the last position defines the sampling domain and is not
    # interior, so computing the sequence must keep it.
    affine = Affine(matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    grid = CartesianField(shape=(4, 5))
    result = Sequence(transformations=[affine, grid]).compute()
    assert _contains_cartesian_field(result)


def test_standalone_grid_is_preserved() -> None:
    # A grid computed on its own is neither interior nor part of a
    # sequence, so it keeps its grid rather than collapsing to an
    # `Identity`.
    grid = CartesianField(shape=(4, 5))
    result = grid.compute()
    assert isinstance(result, CartesianField)
    assert not isinstance(result, Identity)
    assert result.field.shape == (4, 5, 2)


def test_interior_non_identity_field_is_preserved() -> None:
    # Only a grid is the identity map by construction. A displacement
    # field carries its own values that the neighbours do not reproduce,
    # so an interior displacement field must survive computation.
    a = Affine(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = Affine(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    disp = DisplacementField(field=np.ones((4, 5, 2)))
    assert not is_identity(disp, compute=True)
    result = Sequence(transformations=[a, disp, b]).compute()

    def _has_displacement(res) -> bool:  # noqa: ANN001
        if isinstance(res, DisplacementField):
            return True
        if isinstance(res, Sequence):
            return any(_has_displacement(t) for t in res)
        return False

    assert _has_displacement(result)


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


def test_subspace_inverse_without_transform_swaps_axes() -> None:
    # With no inner transformation, `inverse` only swaps the input/output
    # spaces and their axes; it must not touch a `transformations`
    # attribute that does not exist.
    inp = CoordinateSystem(name="in")
    out = CoordinateSystem(name="out")
    subspace = SubspaceTransformation(
        input=inp,
        output=out,
        input_axes=[0, 1],
        output_axes=[2, 3],
    )
    inverse = subspace.inverse()
    assert inverse.transformation is None
    assert inverse.input is out
    assert inverse.output is inp
    np.testing.assert_array_equal(inverse.input_axes, [2, 3])
    np.testing.assert_array_equal(inverse.output_axes, [0, 1])


def test_subspace_inverse_wraps_inner_transform() -> None:
    # With an inner transformation, `inverse` inverts it and swaps the
    # input/output spaces and their axes.
    inner = Translation(translation=np.array([1.0, 2.0]))
    subspace = SubspaceTransformation(
        transformation=inner,
        input_axes=[0, 1],
        output_axes=[2, 3],
    )
    inverse = subspace.inverse()
    assert isinstance(inverse.transformation, Translation)
    np.testing.assert_allclose(
        inverse.transformation.translation, [-1.0, -2.0]
    )
    np.testing.assert_array_equal(inverse.input_axes, [2, 3])
    np.testing.assert_array_equal(inverse.output_axes, [0, 1])
