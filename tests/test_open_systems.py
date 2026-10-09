"""Tests for axis readers on open coordinate systems.

An open system holds `...` among its axes. A missing endpoint and
CoordinateSystem() both state no axis; axes are counted only in a closed system
and are never guessed behind `...`.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel._transformations.compute import separable as sep
from brainhops.datamodel._transformations.compute.adaptors import (
    _grid_extents,
    adapt,
    bridge,
    embed,
)
from brainhops.datamodel._transformations.compute.check import is_family
from brainhops.datamodel._transformations.compute.compose import compose
from brainhops.datamodel._transformations.compute.utils import (
    axis_counts,
    get_ndim,
    systems_disagree,
)
from brainhops.datamodel.axes import A, Axis, R, S, SpaceAxis
from brainhops.datamodel.geometry import _index2transform
from brainhops.datamodel.systems import (
    CoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    CartesianField,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)
from brainhops.errors import (
    AdaptationError,
    CompositionError,
    ConversionError,
)

CS = CoordinateSystem
X, Y, Z = Axis(name="x"), Axis(name="y"), Axis(name="z")

# The ways to state that no axis is known.
UNKNOWN = {
    "missing": None,
    "default": CS(),
    "axes=[...]": CS(axes=[...]),
}


@pytest.fixture(params=list(UNKNOWN), ids=list(UNKNOWN))
def unknown(request: pytest.FixtureRequest) -> tx.Optional[CS]:
    """A system with no known axes, or no system at all."""
    return UNKNOWN[request.param]


@pytest.fixture(params=["[...]", "(...,)"])
def unknown_axes(request: pytest.FixtureRequest) -> tx.Sequence:
    """The axes of a system with no known axes."""
    return [...] if request.param == "[...]" else (...,)


def _xyz() -> CS:
    return CS(axes=[X, Y, Z])


def _scale_x(inner: tx.Optional[CS] = None) -> SubspaceTransformation:
    # Build a subspace scaling of axis 0 that names no full-space system.
    return SubspaceTransformation(
        transformation=Scaling(
            scale=np.array([2.0]), input=inner, output=inner
        ),
        input_axes=np.array([0]),
        output_axes=np.array([0]),
    )


# ----------------------------------------------------------------------
#   THE _subsystem REGRESSION
# ----------------------------------------------------------------------


def test_a_subspace_reports_an_open_full_space_system() -> None:
    # Regression: the derived system used to stop at the last axis acted on.
    sub = _scale_x(CS(axes=[X]))
    assert sub.input == CS(axes=[X, ...])
    assert sub.input.ndim is None
    assert sub.output == CS(axes=[X, ...])


def test_a_subspace_with_an_open_system_is_not_embedded_by_guess() -> None:
    # Regression: to(Affine) used to build a 1x2 matrix.
    with pytest.raises(ConversionError, match="axis count .* is unknown"):
        _scale_x(CS(axes=[X])).to(Affine)


@pytest.mark.parametrize("where", ["before", "after", "between"])
def test_a_subspace_composes_in_a_3d_chain(where: str) -> None:
    # Regression: this raised a matmul ValueError. The 3D affines close the
    # system.
    sub = _scale_x(CS(axes=[X]))
    shift = Translation(translation=np.array([1.0, 2.0, 3.0]))
    chain = {
        "before": [shift, sub],
        "after": [sub, shift],
        "between": [shift, sub, shift],
    }[where]
    result = Sequence(transformations=chain).compute()
    assert isinstance(result, Affine)
    expected = np.eye(4)
    for t in chain:
        step = np.eye(4)
        if t is sub:
            step[0, 0] = 2.0
        else:
            step[:3, 3] = [1.0, 2.0, 3.0]
        expected = step @ expected
    assert np.allclose(result.homogeneous_matrix, expected)


def test_a_subspace_closed_by_its_neighbour_keeps_its_axes() -> None:
    sub = _scale_x(CS(axes=[X]))
    result = compose(sub, Affine(matrix=np.eye(3, 4), output=_xyz()))
    assert result.output == CS(name=None, axes=[X, Axis(), Axis()])


@pytest.mark.parametrize("inner", [None, CS(axes=[X])])
def test_a_neighbour_too_small_for_the_subspace_is_refused(
    inner: tx.Optional[CS],
) -> None:
    # The neighbour has no axis 3, and the axis count is not stretched to
    # include it.
    sub = SubspaceTransformation(
        transformation=Scaling(
            scale=np.array([2.0]), input=inner, output=inner
        ),
        input_axes=np.array([3]),
        output_axes=np.array([3]),
    )
    shift = Translation(translation=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(CompositionError, match="space of 3 axes"):
        compose(sub, shift)
    with pytest.raises(CompositionError, match="space of 3 axes"):
        compose(shift, sub)


def test_a_declared_full_space_system_is_still_used() -> None:
    sub = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0])),
        input_axes=np.array([1]),
        output_axes=np.array([1]),
        input=_xyz(),
        output=_xyz(),
    )
    assert sub.input is not None and sub.input.ndim == 3
    assert np.allclose(sub.to(Affine).matrix[:, :3], np.diag([1, 2, 1]))


def test_a_subspace_with_an_unknown_inner_system(
    unknown: tx.Optional[CS],
) -> None:
    # The full-space size is not guessed until a neighbour closes it.
    sub = _scale_x(unknown)
    assert sub.input == unknown
    assert sub.input is None or sub.input.ndim is None
    with pytest.raises(ConversionError, match="axis count"):
        sub.to(Affine)
    shift = Translation(translation=np.array([1.0, 2.0, 3.0]))
    result = Sequence(transformations=[shift, sub]).compute()
    assert np.allclose(result.matrix[:, :3], np.diag([2.0, 1.0, 1.0]))


def test_a_missing_full_space_system_is_derived() -> None:
    sub = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0]), input=CS(axes=[X])),
        input_axes=np.array([1]),
        output_axes=np.array([1]),
    )
    assert sub.input == CS(axes=[Axis(), X, ...])


def test_a_declared_unknown_system_is_kept(unknown_axes: tx.Sequence) -> None:
    # Only a missing system is derived.
    sub = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0]), input=CS(axes=[X])),
        input_axes=np.array([1]),
        output_axes=np.array([1]),
        input=CS(axes=unknown_axes),
    )
    assert sub.input == CS() and sub._input.axes == [...]


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("axes", [[...], [X, ...], [..., X], [X, ..., Z]])
def test_subspace_to_affine_refuses_an_open_declared_system(
    axes: list,
) -> None:
    sub = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0])),
        input_axes=np.array([0]),
        output_axes=np.array([0]),
        input=CS(axes=axes),
        output=_xyz(),
    )
    with pytest.raises(ConversionError, match="axis count"):
        sub.to(Affine)


# ----------------------------------------------------------------------
#   UTILS
# ----------------------------------------------------------------------


def test_axis_counts_read_closed_systems_only(
    unknown: tx.Optional[CS],
) -> None:
    assert axis_counts(Identity(input=_xyz(), output=_xyz())) == (3, 3)
    assert axis_counts(Identity(input=unknown, output=_xyz())) == (None, 3)
    for axes in ([X, ...], [..., X], [X, ..., Y]):
        t = Identity(input=CS(axes=axes), output=CS(axes=axes))
        assert axis_counts(t) == (None, None)


def test_get_ndim_reads_closed_systems_only(unknown: tx.Optional[CS]) -> None:
    assert get_ndim(Identity(input=unknown, output=_xyz())) == 3
    assert get_ndim(Identity(input=CS(axes=[X, ...])), default=7) == 7
    assert get_ndim(Identity(input=unknown, output=unknown)) is None


def test_systems_disagree_on_open_systems(unknown: tx.Optional[CS]) -> None:
    assert not systems_disagree(unknown, _xyz())
    assert not systems_disagree(_xyz(), unknown)
    assert not systems_disagree(CS(axes=[X, ...]), _xyz())
    assert not systems_disagree(CS(axes=[..., Z]), CS(axes=[X, ...]))
    assert systems_disagree(CS(axes=[Y, ...]), _xyz())
    assert systems_disagree(_xyz(), CS(axes=[..., Y]))


def test_systems_disagree_on_closed_systems_is_unchanged() -> None:
    assert not systems_disagree(_xyz(), _xyz())
    assert systems_disagree(_xyz(), CS(axes=[X, Y, Axis()]))
    assert systems_disagree(CS(name="a", axes=[X]), CS(name="b", axes=[X]))


def test_a_family_dimension_is_contradicted_only_by_known_axes(
    unknown: tx.Optional[CS],
) -> None:
    def t(system: tx.Optional[CS]) -> Identity:
        return Identity(input=system, output=system)

    assert is_family(t(unknown), 3)
    assert is_family(t(_xyz()), 3)
    assert not is_family(t(_xyz()), 2)
    assert is_family(t(CS(axes=[X, ...])), 3)
    assert is_family(t(CS(axes=[X, ..., Z])), 2)
    assert not is_family(t(CS(axes=[X, Y, ..., Z])), 2)


# ----------------------------------------------------------------------
#   ENDPOINTS: None DEFERS, AN EXPLICIT SYSTEM IS KEPT
# ----------------------------------------------------------------------
# None defers to the context, while a system, even CoordinateSystem(), is kept
# as given.

EMPTY = {"default": CS(), "axes=[...]": CS(axes=[...])}


@pytest.fixture(params=list(EMPTY), ids=list(EMPTY))
def empty(request: pytest.FixtureRequest) -> CS:
    """An explicit system that states no axis."""
    return EMPTY[request.param]


def test_a_sequence_derives_a_missing_endpoint() -> None:
    child = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[child], input=None, output=None)
    assert seq.input == _xyz() and seq.output == _xyz()


def test_a_sequence_keeps_an_explicit_endpoint(empty: CS) -> None:
    child = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[child], input=empty, output=empty)
    assert seq.input is empty and seq.output is empty


def test_an_inverse_derives_a_missing_endpoint() -> None:
    forward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    inverse = Inverse(forward=forward)
    assert inverse.input == RASCoordinateSystem()
    assert inverse.output == _xyz()
    assert inverse.inverse().input == _xyz()


def test_an_inverse_keeps_an_explicit_endpoint(empty: CS) -> None:
    forward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    inverse = Inverse(forward=forward, input=empty, output=empty)
    assert inverse.input is empty and inverse.output is empty
    assert inverse.inverse().input is empty


def test_a_bijection_reads_past_a_missing_side() -> None:
    forward = Affine(matrix=np.eye(3, 4))
    backward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    bijection = Bijection(forward=forward, backward=backward)
    assert bijection.input == RASCoordinateSystem()
    assert bijection.output == _xyz()


def test_a_bijection_keeps_an_explicit_side(empty: CS) -> None:
    forward = Affine(matrix=np.eye(3, 4), input=empty, output=empty)
    backward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    bijection = Bijection(forward=forward, backward=backward)
    assert bijection.input is empty and bijection.output is empty


def test_a_missing_endpoint_is_not_propagated_over_a_known_one() -> None:
    # An identity without an input does not overwrite a known one.
    affine = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[Identity(), affine])
    assert seq.simplify().input == _xyz()


def test_an_explicit_endpoint_is_propagated(empty: CS) -> None:
    affine = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[Identity(input=empty), affine])
    assert seq.simplify().input is empty


def test_a_known_endpoint_is_propagated_onto_a_missing_one() -> None:
    first = Scaling(scale=np.array([2.0, 2.0, 2.0]))
    seq = Sequence(transformations=[first], input=_xyz())
    assert seq._flattened().transformations[0].input == _xyz()


def test_a_known_endpoint_does_not_replace_an_explicit_one(empty: CS) -> None:
    first = Scaling(scale=np.array([2.0, 2.0, 2.0]), input=empty)
    seq = Sequence(transformations=[first], input=_xyz())
    assert seq._flattened().transformations[0].input is empty


# ----------------------------------------------------------------------
#   ADAPTORS
# ----------------------------------------------------------------------


def test_bridge_from_or_to_an_unknown_system_is_the_identity(
    unknown: tx.Optional[CS],
) -> None:
    for source, target in ((unknown, _xyz()), (_xyz(), unknown)):
        assert isinstance(bridge(source, target), Identity)


@pytest.mark.parametrize(
    "source, target",
    [
        (CS(axes=[X, ...]), CS(axes=[X, Y, Z])),
        (CS(axes=[..., Z]), CS(axes=[X, Y, Z])),
        (CS(axes=[X, Y, Z]), CS(axes=[X, ..., Z])),
        (CS(axes=[X, ...]), CS(axes=[..., Z])),
    ],
)
def test_bridge_between_compatible_systems_is_the_identity(
    source: CS, target: CS
) -> None:
    assert isinstance(bridge(source, target), Identity)


@pytest.mark.parametrize(
    "source, target",
    [
        # Reordering would move axes hidden by `...`.
        (CS(axes=[X, ...]), CS(axes=[Y, X, Z])),
        (CS(axes=[Y, X, Z]), CS(axes=[..., Y])),
        # The flip would also carry the axes behind `...`.
        (CS(axes=[R(), A(), S(), ...]), LPSCoordinateSystem()),
    ],
)
def test_bridge_refuses_to_move_axes_it_cannot_see(
    source: CS, target: CS
) -> None:
    with pytest.raises(AdaptationError, match="open"):
        bridge(source, target)


def test_bridge_between_closed_systems_is_unchanged() -> None:
    flip = bridge(RASCoordinateSystem(), LPSCoordinateSystem())
    assert isinstance(flip, Scaling)
    assert list(flip.scale) == [-1.0, -1.0, 1.0]


def test_adapt_across_an_open_compatible_boundary_adds_nothing() -> None:
    first = Affine(matrix=np.eye(3, 4), output=CS(axes=[X, ...]))
    second = Affine(matrix=np.eye(3, 4), input=_xyz())
    seq = adapt(first, second)
    assert list(seq.transformations) == [first, second]


def test_adapt_across_an_open_incompatible_boundary_raises() -> None:
    first = Affine(matrix=np.eye(3, 4), output=CS(axes=[R(), A(), S(), ...]))
    second = Affine(matrix=np.eye(3, 4), input=LPSCoordinateSystem())
    with pytest.raises(AdaptationError, match="open"):
        adapt(first, second)


def test_embed_needs_closed_systems(unknown: tx.Optional[CS]) -> None:
    inner = Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    open_full = CS(axes=[R(), A(), S(), ...])
    assert embed(inner, full=open_full, side="input") is None
    assert embed(inner, full=unknown, side="input") is None
    full = CS(axes=[R(), A(), S(), Axis(name="t", type="time")])
    assert embed(inner, full=full, side="input") is not None


def test_grid_extents_close_an_open_grid_system_from_its_shape() -> None:
    i, k = Axis(name="i"), Axis(name="k")
    for axes, expected in (
        ([i, ...], {"i": 3}),
        ([..., k], {"k": 5}),
        ([...], {}),
    ):
        system = CS(axes=axes)
        grid = CartesianField(shape=(3, 4, 5), input=system, output=system)
        assert _grid_extents(grid, at_output=True) == expected


# ----------------------------------------------------------------------
#   DISCRETE AXES
# ----------------------------------------------------------------------


def test_the_discrete_check_reads_an_open_derived_system() -> None:
    # The discrete axis is found at its inner position in the derived open
    # system.
    discrete = Axis(name="c", discrete=True)
    inner_system = CS(axes=[discrete, SpaceAxis(name="y")])
    field = DisplacementField(
        field=np.zeros((4, 5, 2)), input=inner_system, output=inner_system
    )
    sub = SubspaceTransformation(
        transformation=field,
        input_axes=np.array([1, 2]),
        output_axes=np.array([1, 2]),
    )
    assert sub.input.ndim is None
    assert sub.input.axes[1] is discrete
    coords = CoordinatesField(field=np.zeros((3, 4, 5, 3)))
    with pytest.raises(CompositionError, match="discrete axis 'c'"):
        compose(sub, coords)


def test_the_separable_discrete_check_reads_an_open_system() -> None:
    discrete = Axis(name="c", discrete=True)
    group = {"G": [1], "D": [0]}
    assert sep._discrete_axis(group, CS(axes=[X, discrete, ...]), None) == (
        True,
        "c",
    )
    assert sep._discrete_axis(group, CS(axes=[..., discrete]), None) == (
        False,
        None,
    )
    assert sep._discrete_axis(group, CS(axes=[...]), None) == (False, None)


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def test_indexing_closes_an_open_system_from_the_shape() -> None:
    affine, shape = _index2transform(
        (0, slice(None), slice(None)), (3, 4, 5), CS(axes=[..., Y, Z])
    )
    assert shape == (4, 5)
    assert affine.input == CS(axes=[Y, Z])


def test_indexing_an_unknown_system_names_no_axis(
    unknown: tx.Optional[CS],
) -> None:
    affine, _ = _index2transform((0, slice(None)), (3, 4), unknown)
    assert affine.input is None
