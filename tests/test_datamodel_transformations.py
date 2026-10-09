"""Tests for rebuilding, converting, composing and computing transforms."""

import inspect

import numpy as np
import typing_extensions as tx
from bagof.magic import fields_dict, replace

from brainhops._core.properties import smartproperty
from brainhops.datamodel._transformations import concrete as xconcrete
from brainhops.datamodel._transformations.compute import converters as xc
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.enums import BoundaryCondition, InterpolationOrder
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    Linear,
    Permutation,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
    is_identity,
)
from brainhops.errors import ConversionError


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
    # Regression: a second same-type converter dropped the field overrides.
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


def test_same_type_rebuild_of_an_unlisted_type_applies_overrides() -> None:
    # An unlisted subclass reaches the catch-all converter, which must apply
    # the overrides rather than return the original.
    class MyAffine(Affine):
        """Affine subclass that no converter names."""

    target = CoordinateSystem(name="target")
    original = MyAffine(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rebuilt = original.to(input=target)
    assert type(rebuilt) is MyAffine
    assert rebuilt is not original
    assert rebuilt.input is target
    assert original.input is None
    np.testing.assert_allclose(rebuilt.matrix, original.matrix)


def test_same_type_rebuild_of_a_sequence_replaces_its_chain() -> None:
    # Flattening honours the new chain and carries the given endpoints over.
    inp = CoordinateSystem(name="in")
    seq = Sequence(
        transformations=[Translation(translation=[1.0, 2.0])], input=inp
    )
    chain = [Scaling(scale=[2.0, 3.0])]
    rebuilt = seq.to(transformations=chain)
    assert type(rebuilt) is type(seq)
    assert list(rebuilt.transformations) == chain
    assert rebuilt.input is inp


def test_cartesian_field_same_type_rebuild_keeps_shape() -> None:
    # The rebuild must not feed the derived `field` back to the constructor.
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
    # compute flattens the sequence, rebuilding the CartesianField first.
    result = seq.compute()
    assert result is not None


def test_gridded_cartesian_field_is_the_identity_only_under_compute() -> None:
    # A grid is the identity over itself, but only compute=True checks that.
    grid = CartesianField(shape=(4, 5))
    assert is_identity(grid) is False
    assert is_identity(grid, compute=True) is True


def test_empty_cartesian_field_is_the_identity() -> None:
    # Without a grid, the field parameter is unset.
    assert is_identity(CartesianField()) is True


def test_compute_preserves_a_leading_cartesian_field() -> None:
    # A leading grid restricts the domain, so the result keeps its shape.
    grid = CartesianField(shape=(4, 5))
    affine = Affine(matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    result = Sequence(transformations=[grid, affine]).compute()
    assert result.field is not None
    assert result.field.shape == (4, 5, 2)
    # The affine scales the grid coordinates by two.
    np.testing.assert_allclose(result.field, np.asarray(grid.field) * 2.0)


def test_computing_an_identity_only_sequence_still_simplifies() -> None:
    # A genuine identity still simplifies to Identity.
    seq = Sequence(transformations=[Affine(matrix=np.eye(3)[:-1])])
    assert isinstance(seq.compute().to(Identity), Identity)


def test_cartesian_field_is_not_an_init_field_but_base_is() -> None:
    # CartesianField derives data and field from its shape.
    assert "data" not in fields_dict(CartesianField)
    assert "data" in fields_dict(CoordinatesField)
    assert "field" not in inspect.signature(CartesianField).parameters
    assert "field" in inspect.signature(CoordinatesField).parameters


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


def test_replace_cartesian_field_changes_degree_and_bound() -> None:
    cf = CartesianField(shape=(3, 4))
    replaced = replace(
        cf,
        degree=InterpolationOrder.cubic,
        bound=BoundaryCondition.reflect,
    )
    assert isinstance(replaced, CartesianField)
    assert replaced.degree == InterpolationOrder.cubic
    assert replaced.bound == BoundaryCondition.reflect
    # Attributes that are not named are carried over.
    assert replaced.shape == (3, 4)
    assert replaced.store == "values"
    assert replaced.field.shape == (3, 4, 2)


def test_to_same_type_cartesian_field_changes_output() -> None:
    # `to` rebuilds a CartesianField without an explicit field=None.
    output = CoordinateSystem(name="out")
    rebuilt = CartesianField(shape=(4, 5)).to(output=output)
    assert isinstance(rebuilt, CartesianField)
    assert rebuilt.output is output
    assert rebuilt.shape == (4, 5)
    assert "field" not in fields_dict(type(rebuilt))
    assert rebuilt.field.shape == (4, 5, 2)


def test_replace_coordinates_field_round_trips_explicit_field() -> None:
    # The base CoordinatesField carries its array over as is.
    values = np.zeros((5, 6, 2))
    cf = CoordinatesField(data=values.copy(), degree=3, store="coefficients")
    replaced = replace(cf, degree=1)
    assert isinstance(replaced, CoordinatesField)
    assert not isinstance(replaced, CartesianField)
    assert replaced.degree == 1
    assert replaced.store == "coefficients"
    np.testing.assert_array_equal(np.asarray(replaced.data), values)


def _contains_cartesian_field(result) -> bool:  # noqa: ANN001
    # Whether a CartesianField survives anywhere in a computed result.
    if isinstance(result, CartesianField):
        return True
    if isinstance(result, Sequence):
        return any(_contains_cartesian_field(t) for t in result)
    return False


def test_interior_grid_is_factored_away() -> None:
    # An interior grid is overwritten by its neighbours, so compute drops it.
    a = Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]])
    b = Affine(matrix=[[1.0, 0.0, 4.0], [0.0, 1.0, 5.0]])
    grid = CartesianField(shape=(4, 5))
    result = Sequence(transformations=[a, grid, b]).compute()
    assert not _contains_cartesian_field(result)
    reference = Sequence(transformations=[a, b]).compute()
    np.testing.assert_allclose(result.matrix, reference.matrix)


def test_multiple_interior_grids_are_all_factored_away() -> None:
    # Every interior grid is removed, not only the first.
    a = Affine(matrix=[[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]])
    b = Affine(matrix=[[1.0, 0.0, 4.0], [0.0, 1.0, 5.0]])
    grid1 = CartesianField(shape=(4, 5))
    grid2 = CartesianField(shape=(6, 7))
    result = Sequence(transformations=[a, grid1, grid2, b]).compute()
    assert not _contains_cartesian_field(result)
    reference = Sequence(transformations=[a, b]).compute()
    np.testing.assert_allclose(result.matrix, reference.matrix)


def test_interior_grid_preserves_endpoint_systems() -> None:
    # Removing an interior grid keeps the endpoints of the chain.
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
    # A trailing grid defines the sampling domain and is kept.
    affine = Affine(matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    grid = CartesianField(shape=(4, 5))
    result = Sequence(transformations=[affine, grid]).compute()
    assert _contains_cartesian_field(result)


def test_standalone_grid_is_preserved() -> None:
    # A standalone grid does not collapse to Identity.
    grid = CartesianField(shape=(4, 5))
    result = grid.compute()
    assert isinstance(result, CartesianField)
    assert not isinstance(result, Identity)
    assert result.field.shape == (4, 5, 2)


def test_interior_non_identity_field_is_preserved() -> None:
    # Only grids are identities by construction; other fields survive.
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


def test_store_conversion_runs_once(monkeypatch) -> None:  # noqa: ANN001
    # Converting values to coefficients is not idempotent, so `replace`
    # must reuse the converted field rather than convert again.
    calls = {"count": 0}

    def spy(field, degree, bound, inplace=False):  # noqa: ANN001, ANN202
        calls["count"] += 1
        return field + 1.0

    monkeypatch.setattr(xconcrete, "value2coeff_field", spy)
    values = np.zeros((5, 6, 2))
    field = DisplacementField(field=values.copy(), degree=3, store="values")
    coeffs = field.to(store="coefficients")
    assert coeffs.store == "coefficients"
    assert calls["count"] == 1
    np.testing.assert_allclose(coeffs.data, values + 1.0)

    # An explicit data= supplies the converted array and skips conversion.
    calls["count"] = 0
    supplied = np.full((5, 6, 2), 7.0)
    result = field.to(store="coefficients", data=supplied)
    assert calls["count"] == 0
    np.testing.assert_allclose(result.data, supplied)


def test_identity_composes_with_affine_in_both_orders() -> None:
    # Regression #60: composing with Identity passed the other transform
    # positionally; composition now rebuilds it with `replace`.
    a = CoordinateSystem(name="A")
    b = CoordinateSystem(name="B")
    affine = Affine(
        matrix=[[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], input=a, output=a
    )

    # An identity applied first leaves the affine and its endpoints.
    first = Sequence([Identity(input=a, output=a), affine]).compute()
    assert isinstance(first, Affine)
    assert first.input is a
    assert first.output is a
    np.testing.assert_allclose(first.matrix, affine.matrix)

    # An identity applied last gives the result its output.
    last = Sequence([affine, Identity(input=a, output=b)]).compute()
    assert isinstance(last, Affine)
    assert last.input is a
    assert last.output is b
    np.testing.assert_allclose(last.matrix, affine.matrix)


def test_permutation_composes_in_application_order() -> None:
    # Regression: composing permutations indexed the outer one by the inner
    # one. These two permutations do not commute.
    a = CoordinateSystem(name="A")
    b = CoordinateSystem(name="B")
    c = CoordinateSystem(name="C")
    inner = Permutation(permutation=np.array([1, 0, 2]), input=a, output=b)
    outer = Permutation(permutation=np.array([1, 2, 0]), input=b, output=c)

    result = Sequence([inner, outer]).compute()
    assert isinstance(result, Permutation)
    assert result.input is a
    assert result.output is c
    np.testing.assert_array_equal(result.permutation, [0, 2, 1])

    # Check by applying both to a point in turn.
    x = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(
        x[result.permutation], x[inner.permutation][outer.permutation]
    )

    # Check by a dense matrix product.
    dense = xc.convert(result, Linear).matrix
    expected = (
        xc.convert(outer, Linear).matrix @ xc.convert(inner, Linear).matrix
    )
    np.testing.assert_array_equal(dense, expected)


def test_subspace_inverse_without_transform_swaps_axes() -> None:
    # Without an inner transform, the inverse only swaps spaces and axes.
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
    # With an inner transform, the inverse also inverts it.
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


def test_interpolates_truth_table() -> None:
    # Whether applying a transformation interpolates data with splines.
    from brainhops.datamodel._transformations.sequence import _interpolates

    affine = Affine(matrix=np.eye(3, 4))
    translation = Translation(translation=[1.0, 2.0, 3.0])
    grid = CartesianField(shape=(4, 5, 6))
    disp = DisplacementField(field=np.zeros((4, 5, 6, 3)))
    coords = CoordinatesField(field=np.zeros((4, 5, 6, 3)))

    # Neither non-fields nor grids interpolate.
    assert _interpolates(None) is False
    assert _interpolates(Identity()) is False
    assert _interpolates(affine) is False
    assert _interpolates(translation) is False
    assert _interpolates(grid) is False

    # Displacement fields and non-grid coordinate fields do.
    assert _interpolates(disp) is True
    assert _interpolates(coords) is True

    # A sequence does when any element does.
    assert _interpolates(Sequence([affine, translation])) is False
    assert _interpolates(Sequence([affine, disp])) is True

    # A subspace is transparent.
    axes = np.asarray([0, 1, 2])
    assert (
        _interpolates(
            SubspaceTransformation(
                transformation=affine, input_axes=axes, output_axes=axes
            )
        )
        is False
    )
    assert (
        _interpolates(
            SubspaceTransformation(
                transformation=disp, input_axes=axes, output_axes=axes
            )
        )
        is True
    )

    # An inverse does when the inverted transformation does.
    assert _interpolates(affine.inverse()) is False
    assert _interpolates(disp.inverse()) is True


# ----------------------------------------------------------------------
#   UNIFIED compute() SIGNATURE: mode-gating and simplify
# ----------------------------------------------------------------------
# Every transformation has compute(mode=None, *, simplify=False). The mode
# selects which kinds are materialized, and simplify downcasts the result to
# the cheapest type.


def test_inverse_of_field_is_not_materialized_under_restrictive_mode() -> None:
    # The costly inverse of a field is not computed when mode excludes it.
    field = DisplacementField(field=np.random.default_rng(0).random((4, 4, 2)))
    inv = Inverse(forward=field)
    result = inv.compute(mode="Affine")
    assert result is inv


def test_inverse_of_field_is_materialized_when_mode_admits_it() -> None:
    # The default mode admits every kind.
    field = DisplacementField(field=np.random.default_rng(1).random((4, 4, 2)))
    inv = Inverse(forward=field)
    result = inv.compute(mode=None)
    assert isinstance(result, DisplacementField)
    assert result is not inv


def test_inverse_materializes_when_mode_admits_the_wrapped_kind() -> None:
    # A mode that admits the wrapped kind lets the inverse materialize.
    lin = Linear(matrix=np.diag([2.0, 3.0]))
    inv = Inverse(forward=lin)
    result = inv.compute(mode="Linear")
    assert isinstance(result, Linear)
    np.testing.assert_allclose(result.matrix, np.diag([0.5, 1.0 / 3.0]))


def test_leaf_not_admitted_by_mode_is_returned_unchanged() -> None:
    # A leaf that the mode excludes is returned as the same object.
    affine = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 3.0]]))
    result = affine.compute(mode="Translation")
    assert result is affine


def test_leaf_admitted_by_mode_is_computed() -> None:
    # A leaf that the mode admits is computed normally.
    affine = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 3.0]]))
    result = affine.compute(mode="Affine")
    assert isinstance(result, Affine)


def test_simplify_downcasts_a_leaf_to_the_cheapest_type() -> None:
    # simplify=True downcasts to the cheapest compatible type.
    identity_like = Linear(matrix=np.eye(2))
    assert isinstance(identity_like.compute(simplify=True), Identity)

    scaling_like = Linear(matrix=np.diag([2.0, 3.0]))
    assert isinstance(scaling_like.compute(simplify=True), Scaling)
    # Without simplify, nothing is downcast.
    assert isinstance(scaling_like.compute(), Linear)


def test_sequence_compute_applies_simplify_to_the_result() -> None:
    # Sequence.compute(simplify=True) downcasts the composed result.
    scaling_like = Linear(matrix=np.diag([2.0, 3.0]))
    seq = Sequence(transformations=[scaling_like])
    assert isinstance(seq.compute(simplify=True), Scaling)
    # Without simplify, the result stays linear.
    assert isinstance(seq.compute(), Linear)


def test_simplify_is_keyword_only() -> None:
    # mode may be positional, simplify may not.
    import pytest

    affine = Affine(matrix=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 3.0]]))
    with pytest.raises(TypeError):
        affine.compute("Affine", True)


# ----------------------------------------------------------------------
#   simplify=True HARDENING: kind-checks must never raise
# ----------------------------------------------------------------------


def test_simplify_downcasts_a_swap_to_a_permutation() -> None:
    # Regression: the permutation check compared whole arrays with `and`.
    swap = Linear(matrix=[[0.0, 1.0], [1.0, 0.0]])
    result = swap.compute(simplify=True)
    assert isinstance(result, Permutation)


def test_simplify_does_not_raise_on_a_shear() -> None:
    # A shear is neither permutation, scaling nor rotation, and stays linear.
    shear = Affine(matrix=[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
    result = shear.compute(simplify=True)
    assert isinstance(result, Linear)
    assert not isinstance(result, Permutation)


def test_simplify_does_not_raise_on_a_non_square_matrix() -> None:
    # The square-only checks return False for a non-square matrix.
    rectangular = Linear(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = rectangular.compute(simplify=True)
    assert isinstance(result, Linear)


def test_simplify_does_not_raise_on_a_rotation() -> None:
    # A quarter turn passes the numeric checks without raising.
    rotation = Linear(matrix=[[0.0, -1.0], [1.0, 0.0]])
    result = rotation.compute(simplify=True)
    assert result is not None


# ----------------------------------------------------------------------
#   simplify=True must not drop a sampling grid
# ----------------------------------------------------------------------


def test_simplify_keeps_a_leading_grid() -> None:
    # A leading grid is the sampling domain and survives simplification.
    grid = CartesianField(shape=(4, 5))
    shear = Affine(matrix=[[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
    result = Sequence([grid, shear]).compute(simplify=True)
    assert not isinstance(result, Identity)
    assert result.field is not None
    assert result.field.shape == (4, 5, 2)


def test_simplify_keeps_a_trailing_grid_as_a_cartesian_field() -> None:
    # A trailing grid is not downcast to Identity by simplification.
    affine = Affine(matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    grid = CartesianField(shape=(4, 5))
    result = Sequence([affine, grid]).compute(simplify=True)
    assert _contains_cartesian_field(result)


# ----------------------------------------------------------------------
#   SubspaceTransformation endpoint reconstruction
# ----------------------------------------------------------------------


def test_subspace_input_reconstructs_full_space_for_high_axes() -> None:
    # Regression: high subspace axes indexed the inner system (IndexError).
    # The input is now an open system with the inner axes at their positions.
    inner = Identity(
        input=CoordinateSystem(
            axes=[Axis(name="x"), Axis(name="y"), Axis(name="z")]
        )
    )
    subspace = SubspaceTransformation(
        transformation=inner, input_axes=[1, 2, 3], output_axes=[1, 2, 3]
    )
    system = subspace.input
    assert system is not None
    assert system.ndim is None
    assert system.axes == [
        Axis(),
        Axis(name="x"),
        Axis(name="y"),
        Axis(name="z"),
        ...,
    ]


def test_subspace_endpoint_reconstruction_is_backward_compatible() -> None:
    # Axes from 0 give the inner axes in order, then `...`.
    inner = Identity(
        input=CoordinateSystem(axes=[Axis(name="x"), Axis(name="y")])
    )
    subspace = SubspaceTransformation(
        transformation=inner, input_axes=[0, 1], output_axes=[0, 1]
    )
    system = subspace.input
    assert system.axes == [Axis(name="x"), Axis(name="y"), ...]
    assert system.axes.names == ("x", "y", ...)


def test_subspace_declared_endpoint_is_returned_as_is() -> None:
    # A declared endpoint is returned as is.
    declared = CoordinateSystem(name="full", axes=[Axis(), Axis(), Axis()])
    inner = Identity(
        input=CoordinateSystem(axes=[Axis(name="x"), Axis(name="y")])
    )
    subspace = SubspaceTransformation(
        transformation=inner, input=declared, input_axes=[0, 1]
    )
    assert subspace.input is declared


# ----------------------------------------------------------------------
#   compute() has no silent default: the base raises
# ----------------------------------------------------------------------


def test_base_transformation_compute_raises() -> None:
    # A subclass that forgets to implement compute fails loudly.
    import pytest

    with pytest.raises(NotImplementedError):
        Transformation().compute()


def test_subspace_compute_returns_self_unchanged() -> None:
    # A subspace returns itself, in the default and in an excluding mode.
    inner = Translation(translation=np.array([1.0, 2.0]))
    subspace = SubspaceTransformation(
        transformation=inner, input_axes=[0, 1], output_axes=[0, 1]
    )
    assert subspace.compute() is subspace
    assert subspace.compute(mode="Affine") is subspace


# ----------------------------------------------------------------------
#   DERIVED CHAINS SURVIVE REPLACE
# ----------------------------------------------------------------------


class _DerivedSequence(Sequence):
    """Sequence whose chain derives from a declared parameter."""

    shift: float = 0.0

    @smartproperty(cache=True)
    def transformations(self) -> list:
        return [Translation(translation=[self.shift, self.shift])]


def test_sequence_stores_its_chain_under_a_private_name() -> None:
    # The given chain is stored privately so that a subclass can derive it.
    field = fields_dict(Sequence)["transformations"]
    assert field.name == "_transformations"
    assert field.public_name == "transformations"
    seq = Sequence(transformations=[Identity()])
    assert seq._transformations == seq.transformations


def test_replace_does_not_freeze_a_derived_chain() -> None:
    # The copy rebuilds its chain from the new parameter.
    seq = _DerivedSequence(shift=1.0)
    np.testing.assert_allclose(
        np.asarray(seq.transformations[0].translation), [1.0, 1.0]
    )
    copy = replace(seq, shift=5.0)
    np.testing.assert_allclose(
        np.asarray(copy.transformations[0].translation), [5.0, 5.0]
    )
    # Copies do not share the cached list.
    assert copy.transformations is not seq.transformations


def test_replace_with_no_changes_leaves_a_derived_chain_derived() -> None:
    seq = _DerivedSequence(shift=2.0)
    _ = seq.transformations
    copy = replace(seq)
    assert copy._transformations is None
    assert copy.transformations is not seq.transformations


def test_replace_carries_over_an_assigned_chain() -> None:
    # An assigned chain is a declared value and is carried over.
    seq = _DerivedSequence(shift=1.0)
    seq.transformations = [Identity()]
    copy = replace(seq, shift=5.0)
    assert len(copy) == 1
    assert isinstance(copy.transformations[0], Identity)


# ----------------------------------------------------------------------
#   to(): WHAT COMES BACK, AND WHAT HAPPENS WHEN IT CANNOT
# ----------------------------------------------------------------------


def test_to_never_returns_a_type_that_was_not_asked_for() -> None:
    # The catch-all converter matches every target, so a type that no
    # converter produces must raise rather than return the original type.
    import pytest

    from brainhops.datamodel.transformations import Transformation
    from brainhops.errors import ConversionError

    class Unreachable(Transformation):
        """Transformation of no family, which no converter produces."""

    with pytest.raises(ConversionError):
        Affine(matrix=np.eye(3)[:2]).to(Unreachable)

    # A refinement of a family is rebuilt as the requested class.
    class Refinement(Affine):
        """Affine of its own class, as an io format is."""

    built = Affine(matrix=np.eye(3)[:2]).to(Refinement)
    assert type(built) is Refinement
    np.testing.assert_array_equal(built.matrix, np.eye(3)[:2])


def test_to_reports_a_lossy_conversion_rather_than_performing_it() -> None:
    import pytest

    from brainhops.datamodel.transformations import Rotation
    from brainhops.errors import LossyConversionError

    lin = Linear(matrix=np.diag([2.0, 3.0]))
    # A lossy conversion is refused by default.
    with pytest.raises(LossyConversionError):
        lin.to(Rotation)
    # lossy=True returns the lossy result rather than the exception.
    lossy = lin.to(Rotation, lossy=True)
    assert type(lossy) is Rotation
    # error=<value> stands in for the result.
    assert lin.to(Rotation, error=False) is False
    # error=<exception> raises that exception.
    with pytest.raises(TypeError):
        lin.to(Rotation, error=TypeError)


def test_transformations_compare_by_identity() -> None:
    # == is `is`, and it never raises, whatever the other operand.
    system = CoordinateSystem(name="world", axes=[Axis(name="x")] * 2)
    matrix = np.array([[2.0, 0.5, 1.0], [0.0, 3.0, -1.0]])
    affine = Affine(matrix=matrix, input=system, output=system)
    twin = Affine(matrix=matrix.copy(), input=system, output=system)
    others = (
        twin,
        Identity(),
        Sequence(transformations=[affine]),
        None,
        1,
    )
    for this in (affine, Identity(), Scaling(scale=[1.0, 2.0])):
        assert this == this
        assert not (this != this)
        for other in others:
            if other is this:
                continue
            assert not (this == other)
            assert this != other
            assert not (other == this)
        assert hash(this) == object.__hash__(this)
    assert Identity() != Identity()
    # Hashing is by identity too.
    assert len({affine, twin, affine}) == 2
    names = {affine: "affine", twin: "twin"}
    assert names[affine] == "affine" and names[twin] == "twin"


def test_every_transformation_type_compares_by_identity() -> None:
    # Identity comparison holds for every type, whatever its other bases.
    import brainhops.io  # noqa: F401  (registers every format)

    def subclasses(cls: type) -> tx.Iterator[type]:
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    for cls in (Transformation, *subclasses(Transformation)):
        assert cls.__eq__ is object.__eq__, cls
        assert cls.__ne__ is object.__ne__, cls
        assert cls.__hash__ is object.__hash__, cls


def test_a_sequence_finds_its_members_by_identity() -> None:
    # Membership and lookup in a sequence are by identity.
    import pytest

    first = Affine(matrix=np.eye(2, 3))
    twin = Affine(matrix=np.eye(2, 3))
    seq = Sequence(transformations=[first, Scaling(scale=[2.0, 2.0]), first])
    assert first in seq
    assert twin not in seq
    assert seq.index(first) == 0
    assert seq.index(first, 1) == 2
    assert seq.count(first) == 2
    assert seq.count(twin) == 0
    with pytest.raises(ValueError):
        seq.index(twin)
    with pytest.raises(ValueError):
        seq.remove(twin)
    seq.remove(first)
    assert len(seq) == 2 and seq[1] is first


def _subspace(inner: Transformation, axes: list, inp, out) -> object:  # noqa: ANN001
    return SubspaceTransformation(
        transformation=inner,
        input_axes=axes,
        output_axes=axes,
        input=inp,
        output=out,
    )


def test_a_sequence_of_disjoint_subspaces_converts_to_a_block_affine() -> None:
    # Steps on disjoint axes give the block-diagonal product of their affines.
    import pytest

    from brainhops.datamodel.axes import SpaceAxis, TimeAxis

    def system(*axes: Axis) -> CoordinateSystem:
        return CoordinateSystem(axes=list(axes))

    xyz = [SpaceAxis(name=n, unit="index") for n in "xyz"]
    ras = [SpaceAxis(name=n, unit="mm") for n in "xyz"]
    t_index = TimeAxis(name="t", unit="index")
    t_sec = TimeAxis(name="t", unit="s")
    voxel, middle = system(*xyz, t_index), system(*ras, t_index)
    world = system(*ras, t_sec)
    spatial = np.array(
        [[0.0, -2.0, 0.0, 10.0], [1.5, 0.0, 0.0, -3.0], [0.0, 0.0, 2.5, 4.0]]
    )
    temporal = Sequence(
        transformations=[
            Scaling(scale=[2.0], input=system(t_index), output=system(t_sec)),
            Translation(
                translation=[0.5], input=system(t_sec), output=system(t_sec)
            ),
        ]
    )
    product = Sequence(
        transformations=[
            _subspace(
                Affine(
                    matrix=spatial, input=system(*xyz), output=system(*ras)
                ),
                [0, 1, 2],
                voxel,
                middle,
            ),
            _subspace(temporal, [3], middle, world),
        ]
    )

    affine = product.to(Affine)

    expected = np.zeros((4, 5))
    expected[:3, [0, 1, 2, 4]] = spatial
    expected[3, 3:] = [2.0, 0.5]
    assert np.allclose(affine.matrix, expected)
    assert affine.input is voxel and affine.output is world

    # Every piece must be affine.
    field = _subspace(
        DisplacementField(field=np.zeros((2, 2, 2, 3))), [0, 1, 2], None, None
    )
    with pytest.raises(ConversionError):
        Sequence(transformations=[field, product[1]]).to(Affine)
    # An empty sequence is the identity.
    assert np.array_equal(
        Sequence(transformations=[], input=voxel, output=voxel)
        .to(Affine)
        .matrix,
        np.eye(4, 5),
    )
    # Pieces without systems take the full space from the endpoints.
    bare = Sequence(
        transformations=[
            SubspaceTransformation(
                transformation=product[0].transformation,
                input_axes=[0, 1, 2],
                output_axes=[0, 1, 2],
            ),
            SubspaceTransformation(
                transformation=temporal, input_axes=[3], output_axes=[3]
            ),
        ],
        input=voxel,
        output=world,
    )
    assert np.allclose(bare.to(Affine).matrix, expected)
    # A sequence of non-subspace elements has no affine form.
    with pytest.raises(ConversionError):
        Sequence(
            transformations=[
                Affine(matrix=np.eye(4, 5)),
                Projection(dropped=[3]),
            ]
        ).to(Affine)
