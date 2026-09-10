"""
Tests for `Transformation._expand_dims` and the axis adaptor
(`_adapt` / the `@_adaptor`-registered permutation+scaling function).

These tests exercise the real `brainhops` code, so they need to run in
an environment where `brainhops` is importable. A few notes on
assumptions baked into these tests -- adjust if your API differs:

* `LeftToRightAxis()` / `AnteriorToPosteriorAxis()` /
    `InferiorToSuperiorAxis()`
  take no constructor args (matches the example script this suite is
  based on).
* Importing `brainhops.datamodel._xform_adaptors` is what registers the
  permutation/scaling adaptor with `_adapt` -- update that import path
  if the module was renamed/moved (it was referred to as both
  `adaptors.py` and `_xform_adaptors.py` in earlier discussion).
* The adaptor tests that would exercise *unit* mismatches (as opposed
  to axis *order* mismatches) are left as skipped placeholders, since
  they depend on the exact `Unit` API (`unit.scale`), which wasn't
  available to verify here -- fill in `make_unit(...)` and un-skip
  once you confirm that API.
* `_make_same_axes(x1, x2)` and `_adapt(x1, x2)` treat `x2` as the
  transform applied FIRST (matched via its `.output`) and `x1` as the
  transform applied SECOND (matched via its `.input`). Tests that call
  these directly use fixtures named `earlier`/`later` (== `x2`/`x1`)
  to keep this explicit. `Sequence`'s own public ordering (transforms
  apply in list order) is unaffected by this and is not renamed.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel.axes import (
    AnteriorToPosteriorAxis,
    InferiorToSuperiorAxis,
    LeftToRightAxis,
)
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    Identity,
    Linear,
    Permutation,
    Scaling,
    Sequence,
    Translation,
    _adapt,
    _make_same_axes,
)

# Importing this module registers the permutation/scaling adaptor
# function with `_adapt` via the `@_adaptor` decorator.


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------


@pytest.fixture
def axes() -> dict:
    """Fresh L/R, A/P, I/S axis instances."""
    return {
        "X": LeftToRightAxis(),
        "Y": AnteriorToPosteriorAxis(),
        "Z": InferiorToSuperiorAxis(),
    }


@pytest.fixture
def xyz(axes: dict) -> CoordinateSystem:
    return CoordinateSystem(axes=[axes["X"], axes["Y"], axes["Z"]])


@pytest.fixture
def xy(axes: dict) -> CoordinateSystem:
    return CoordinateSystem(axes=[axes["X"], axes["Y"]])


@pytest.fixture
def zyx(axes: dict) -> CoordinateSystem:
    return CoordinateSystem(axes=[axes["Z"], axes["Y"], axes["X"]])


def make_affine(
    cs_in: CoordinateSystem,
    cs_out: CoordinateSystem,
    matrix: np.ndarray = None,
) -> Affine:
    ni = len(cs_in.axes)
    no = len(cs_out.axes)
    if matrix is None:
        matrix = np.eye(no, ni + 1)
    return Affine(matrix=matrix, input=cs_in, output=cs_out)


def make_linear(
    cs_in: CoordinateSystem,
    cs_out: CoordinateSystem,
    matrix: np.ndarray = None,
) -> Linear:
    ni = len(cs_in.axes)
    no = len(cs_out.axes)
    if matrix is None:
        matrix = np.eye(no, ni)
    return Linear(matrix=matrix, input=cs_in, output=cs_out)


def make_field(
    cs_in: CoordinateSystem,
    cs_out: CoordinateSystem,
    shape: tx.Tuple[int, ...],
) -> CoordinateSystem:
    ndim_out = len(cs_out.axes)
    field = np.zeros((*shape, ndim_out))
    return CoordinatesField(field=field, input=cs_in, output=cs_out)


# =======================================================================
#   expand_dims
# =======================================================================


class TestExpandDimsAffine:
    def test_expand_output_adds_zero_rows(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = make_affine(
            xy,
            xy,
            matrix=np.array(
                [
                    [2.0, 0.0, 10.0],
                    [0.0, 3.0, 20.0],
                ]
            ),
        )
        expanded = t._expand_dims([axes["Z"]], side="output")

        assert expanded.matrix.shape == (3, 3)
        # existing rows/cols untouched
        np.testing.assert_array_equal(expanded.matrix[:2, :], t.matrix)
        # new row is constant zero (independent of any input)
        np.testing.assert_array_equal(expanded.matrix[2, :], [0.0, 0.0, 0.0])

        assert len(expanded.input.axes) == 2
        assert len(expanded.output.axes) == 3
        assert expanded.output.axes[-1] is axes["Z"]

    def test_expand_input_adds_zero_columns(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = make_affine(
            xy,
            xy,
            matrix=np.array(
                [
                    [2.0, 0.0, 10.0],
                    [0.0, 3.0, 20.0],
                ]
            ),
        )
        expanded = t._expand_dims([axes["Z"]], side="input")

        assert expanded.matrix.shape == (2, 4)
        # new input column (before translation) has no effect on output
        np.testing.assert_array_equal(expanded.matrix[:, 2], [0.0, 0.0])
        # translation column preserved in the same place
        np.testing.assert_array_equal(expanded.matrix[:, -1], [10.0, 20.0])

        assert len(expanded.input.axes) == 3
        assert len(expanded.output.axes) == 2

    def test_expand_both_is_identity_pass_through(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = make_affine(xy, xy)
        expanded = t._expand_dims([axes["Z"]], side="both")

        assert expanded.matrix.shape == (3, 4)
        assert expanded.ndim == 3  # input == output dimensionality

        # a point on the new axis should pass through unchanged
        point = np.array([1.0, 2.0, 5.0, 1.0])  # homogeneous coords
        out = expanded.matrix @ point
        assert out[-1] == pytest.approx(5.0)

    def test_expand_empty_missing_is_noop(self, xy: CoordinateSystem) -> None:
        t = make_affine(xy, xy)
        result = t._expand_dims([], side="both")
        assert result is t


class TestExpandDimsLinear:
    def test_expand_output(self, xy: CoordinateSystem, axes: dict) -> None:
        t = make_linear(xy, xy, matrix=np.eye(2))
        expanded = t._expand_dims([axes["Z"]], side="output")
        assert expanded.matrix.shape == (3, 2)
        np.testing.assert_array_equal(expanded.matrix[-1], [0.0, 0.0])

    def test_expand_input(self, xy: CoordinateSystem, axes: dict) -> None:
        t = make_linear(xy, xy, matrix=np.eye(2))
        expanded = t._expand_dims([axes["Z"]], side="input")
        assert expanded.matrix.shape == (2, 3)
        np.testing.assert_array_equal(expanded.matrix[:, -1], [0.0, 0.0])

    def test_expand_both(self, xy: CoordinateSystem, axes: dict) -> None:
        t = make_linear(xy, xy, matrix=np.eye(2))
        expanded = t._expand_dims([axes["Z"]], side="both")
        assert expanded.matrix.shape == (3, 3)
        np.testing.assert_array_equal(expanded.matrix, np.eye(3))


class TestExpandDimsFields:
    def test_expand_output_pads_channel_axis(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = make_field(xy, xy, shape=(4, 5))
        expanded = t._expand_dims([axes["Z"]], side="output")

        assert expanded.field.shape == (4, 5, 3)
        np.testing.assert_array_equal(expanded.field[..., -1], 0.0)
        assert len(expanded.output.axes) == 3
        assert len(expanded.input.axes) == 2

    def test_expand_input_adds_singleton_grid_axis(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = make_field(xy, xy, shape=(4, 5))
        expanded = t._expand_dims([axes["Z"]], side="input")

        assert expanded.field.shape == (4, 5, 1, 2)
        assert len(expanded.input.axes) == 3
        assert len(expanded.output.axes) == 2

    def test_expand_both(self, xy: CoordinateSystem, axes: dict) -> None:
        t = make_field(xy, xy, shape=(4, 5))
        expanded = t._expand_dims([axes["Z"]], side="both")
        assert expanded.field.shape == (4, 5, 1, 3)


class TestExpandDimsSquareParametric:
    """Scaling / Translation / Permutation, which delegate to Linear/Affine."""

    def test_scaling_expand_both(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = Scaling(scale=np.array([2.0, 3.0]), input=xy, output=xy)
        expanded = t._expand_dims([axes["Z"]], side="both")
        # delegates to Linear -- result is a Linear, not a Scaling
        assert isinstance(expanded, Scaling)
        assert len(expanded.scale) == 3

    def test_translation_expand_both(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = Translation(translation=np.array([1.0, 2.0]), input=xy, output=xy)
        expanded = t._expand_dims([axes["Z"]], side="both")
        # delegates to Affine -- result is an Affine, not a Translation
        assert isinstance(expanded, Translation)
        assert len(expanded.translation) == 3

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (Scaling, {"scale": np.array([2.0, 3.0])}),
            (Translation, {"translation": np.array([1.0, 2.0])}),
            (Permutation, {"permutation": [1, 0]}),
        ],
    )
    def test_empty_missing_returns_original_object_unchanged(
        self, xy: CoordinateSystem, cls: type, kwargs: dict
    ) -> None:
        # Regression test: previously, delegating to Linear/Affine with
        # an empty `missing` list silently returned a converted
        # Linear/Affine instance instead of the original object.
        t = cls(input=xy, output=xy, **kwargs)
        result = t._expand_dims([], side="both")
        assert result is t
        assert type(result) is cls


class TestExpandDimsIdentity:
    def test_expand_updates_axes_only(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        t = Identity(input=xy, output=xy)
        expanded = t._expand_dims([axes["Z"]], side="both")
        assert len(expanded.input.axes) == 3
        assert len(expanded.output.axes) == 3
        # original untouched
        assert len(t.input.axes) == 2


class TestExpandDimsSequence:
    def test_expand_input_only_touches_first_transform(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        first = make_affine(xy, xy)
        second = make_affine(xy, xy)
        seq = Sequence(transformations=[first, second], input=xy, output=xy)

        expanded = seq._expand_dims([axes["Z"]], side="input")

        assert expanded.transformations[0].ndims.input == 3
        # second transform untouched
        assert expanded.transformations[1].ndims.input == 2
        assert len(expanded.input.axes) == 3

    def test_expand_output_only_touches_last_transform(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        first = make_affine(xy, xy)
        second = make_affine(xy, xy)
        seq = Sequence(transformations=[first, second], input=xy, output=xy)

        expanded = seq._expand_dims([axes["Z"]], side="output")

        assert expanded.transformations[-1].ndims.output == 3
        assert expanded.transformations[0].ndims.output == 2
        assert len(expanded.output.axes) == 3

    def test_original_sequence_and_children_untouched(
        self, xy: CoordinateSystem, axes: dict
    ) -> None:
        first = make_affine(xy, xy)
        seq = Sequence(transformations=[first], input=xy, output=xy)
        seq._expand_dims([axes["Z"]], side="both")
        assert seq.transformations[0].ndims.input == 2


# =======================================================================
#   _make_same_axes
#
#   Current convention: `_make_same_axes(x1, x2)` -- and `_adapt`,
#   `_compose` below -- treat `x2` as the transform applied FIRST
#   (its `.output` is what's being matched) and `x1` as the transform
#   applied SECOND (its `.input` is what's being matched). i.e.
#   `x2`'s output should end up compatible with `x1`'s input.
#   Fixtures below are named `earlier` (== x2 role) / `later` (== x1
#   role) to keep this unambiguous, since it's easy to get backwards.
# =======================================================================


class TestMakeSameAxes:
    def test_already_matching_is_noop(self, xyz: CoordinateSystem) -> None:
        later = make_affine(xyz, xyz)  # x1
        earlier = make_affine(xyz, xyz)  # x2
        r1, r2 = _make_same_axes(later, earlier)
        assert r1 is later
        assert r2 is earlier

    def test_expands_missing_axis(
        self, xy: CoordinateSystem, xyz: CoordinateSystem, axes: dict
    ) -> None:
        # later's input (xyz) has Z; earlier's output (xy) is missing it.
        later = make_affine(xyz, xyz)
        earlier = make_affine(xy, xy)
        r1, r2 = _make_same_axes(later, earlier)

        assert r1.ndims.input == 3  # later: unchanged, already 3
        assert r2.ndims.output == 3  # earlier: expanded to add Z

    def test_identity_ndims_zero_short_circuits(
        self, xy: CoordinateSystem, xyz: CoordinateSystem
    ) -> None:
        later = Identity(input=xy, output=xy)  # ndims == (0, 0) -> wildcard
        earlier = make_affine(xyz, xyz)
        r1, r2 = _make_same_axes(later, earlier)
        assert r1 is later
        assert r2 is earlier

    def test_returns_results_in_same_order_as_arguments(
        self, xyz: CoordinateSystem
    ) -> None:
        # Regression test: `_make_same_axes(x1, x2)` must return
        # `(result_for_x1, result_for_x2)` -- the same order as the
        # arguments -- and not silently swap them. Uses two distinct
        # (non-equal) matrices so an accidental swap can't hide behind
        # object/value equality.
        later = make_affine(xyz, xyz, matrix=np.eye(3, 4) * 2)
        earlier = make_affine(xyz, xyz, matrix=np.eye(3, 4) * 3)
        r1, r2 = _make_same_axes(later, earlier)
        assert r1 is later
        assert r2 is earlier


# =======================================================================
#   _adapt / the axis-permutation+scaling adaptor
#
#   Same convention as above: `_adapt(x1, x2)` -- `x2` (`earlier`) is
#   matched via its `.output`, `x1` (`later`) is matched via its
#   `.input`.
# =======================================================================


class TestAdapt:
    def test_matching_axes_returns_identity(
        self, xyz: CoordinateSystem
    ) -> None:
        later = make_affine(xyz, xyz)
        earlier = make_affine(xyz, xyz)
        adapted = _adapt(later, earlier)
        assert isinstance(adapted, Identity)

    def test_reordered_axes_returns_permutation(
        self, xyz: CoordinateSystem, zyx: CoordinateSystem
    ) -> None:
        # earlier's output is X,Y,Z; later's input is Z,Y,X (reordered)
        later = make_affine(zyx, zyx)
        earlier = make_affine(xyz, xyz)

        adapted = _adapt(later, earlier)

        assert isinstance(adapted, Permutation)
        assert adapted.input == earlier.output
        assert adapted.output == later.input

    def test_permutation_actually_reorders_correctly(
        self, xyz: CoordinateSystem, zyx: CoordinateSystem
    ) -> None:
        # 3-axis, non-involutive-friendly check: verify applying the
        # returned Permutation's `permutation` array to earlier's
        # output axes, in the class's own
        # `output[i] = input[permutation[i]]` convention, reproduces
        # later's input axes order.
        later = make_affine(zyx, zyx)
        earlier = make_affine(xyz, xyz)
        adapted = _adapt(later, earlier)

        reordered = [earlier.output.axes[i] for i in adapted.permutation]
        assert reordered == list(later.input.axes)

    def test_dimensionality_mismatch_raises(
        self, xy: CoordinateSystem, xyz: CoordinateSystem
    ) -> None:
        later = make_affine(xyz, xyz)  # later.input has 3 dims
        earlier = make_affine(xy, xy)  # earlier.output has 2 dims
        with pytest.raises(NotImplementedError):
            _adapt(later, earlier)

    def test_no_matching_axis_raises_value_error(self, axes: dict) -> None:
        # two coordinate systems that are the same size but share no
        # common axis types at all
        cs_a = CoordinateSystem(axes=[axes["X"], axes["Y"]])
        cs_b = CoordinateSystem(axes=[axes["Z"], axes["Z"]])
        later = make_affine(cs_a, cs_a)
        earlier = make_affine(cs_b, cs_b)
        with pytest.raises(ValueError):
            _adapt(later, earlier)

    def test_zero_ndims_side_returns_identity(
        self, xyz: CoordinateSystem
    ) -> None:
        # an Identity transform has ndims (0, 0), which `_adapt`
        # (mirroring `_make_same_axes`) treats as "accepts anything".
        later = make_affine(xyz, xyz)
        earlier = Identity(input=xyz, output=xyz)
        adapted = _adapt(later, earlier)
        assert isinstance(adapted, Identity)


# =======================================================================
#   end-to-end: adapt + expand_dims working together inside a Sequence
#
#   NOTE: `Sequence(transformations=[t1, t2, ...])` semantics are
#   unaffected by the x1/x2 convention above -- `_compute_sequence`
#   already calls `_compose`/`_adapt`/`_make_same_axes` with the right
#   internal argument order, so from the outside, transformations in a
#   Sequence still simply apply in list order (t1 first, then t2, ...).
# =======================================================================


class TestSequenceComputeWithAdaptAndExpand:
    def test_reordered_axes_compose_through_sequence(
        self, xyz: CoordinateSystem, zyx: CoordinateSystem
    ) -> None:
        t1 = make_affine(xyz, xyz)
        t2 = make_affine(zyx, zyx)

        seq = Sequence(transformations=[t1, t2])
        result = seq.compute()

        # composition should succeed without raising, and the
        # resulting transform's axes should end up matching t2's output
        assert result is not None
        assert result.ndims.output == 3

    def test_missing_axis_count_mismatch_is_auto_expanded(
        self, xy: CoordinateSystem, xyz: CoordinateSystem
    ) -> None:
        # t1's output (xy) is missing the Z axis that t2's input (xyz)
        # has; `_make_same_axes` should transparently expand t1's output
        # (and/or t2's input) before composition, rather than erroring.
        t1 = make_affine(xy, xy)
        t2 = make_affine(xyz, xyz)
        seq = Sequence(transformations=[t1, t2])

        result = seq.compute()
        assert result.ndims.output == 3

    @pytest.mark.xfail(
        reason=(
            "known issue: `ndims == 0` is used as a 'this side accepts "
            "anything' wildcard in _make_same_axes/_adapt, but transforms "
            "like Scaling/Linear/Translation also report ndims == (0, 0) "
            "whenever their parameter is left as `None`, not only true "
            "Identity transforms. If such an under-specified transform "
            "sits between two real, mismatched-dimensionality transforms "
            "in a Sequence, the mismatch can slip past _make_same_axes "
            "instead of being caught."
        ),
        strict=False,
    )
    def test_none_parameter_transform_does_not_mask_real_dim_mismatch(
        self, xy: CoordinateSystem, xyz: CoordinateSystem
    ) -> None:
        # A `Scaling` with no `scale` set reports ndims == (0, 0), same
        # as a true Identity, even though it's sandwiched between a 2D
        # and a 3D transform that don't actually agree on dimensionality.
        t1 = make_affine(xy, xy)
        underspecified = Scaling(scale=None)
        t2 = make_affine(xyz, xyz)
        seq = Sequence(transformations=[t1, underspecified, t2])

        # Ideally this should either raise (dims genuinely mismatched)
        # or correctly expand `t1`'s output to 3D before composing --
        # not silently produce a transform with inconsistent axes.
        result = seq.compute()
        assert result.ndims.input == 2
        assert result.ndims.output == 3
