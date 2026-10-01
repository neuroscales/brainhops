"""Tests for the readers of coordinate systems once systems may be open.

An open system (its axes hold `...`, or are `None`) states only some of
its axes. Every reader treats `axes=None`, `axes=[...]` and a missing
endpoint alike, counts axes only when a system is closed, and never
guesses what `...` stands for.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops._core.properties import smartproperty
from brainhops.datamodel._transformations import separable as sep
from brainhops.datamodel._transformations.adaptors import (
    _grid_extents,
    adapt,
    bridge,
    embed,
)
from brainhops.datamodel._transformations.check import is_family
from brainhops.datamodel._transformations.compose import compose
from brainhops.datamodel._transformations.utils import (
    axis_counts,
    get_ndim,
    systems_disagree,
)
from brainhops.datamodel.axes import A, Axis, R, S, SpatialAxis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.geometry import _index2transform
from brainhops.datamodel.systems import (
    CoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    AdaptationError,
    Affine,
    Bijection,
    CartesianField,
    CompositionError,
    ConversionError,
    CoordinatesField,
    DisplacementField,
    Identity,
    Inverse,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

CS = CoordinateSystem
X, Y, Z = Axis(name="x"), Axis(name="y"), Axis(name="z")

# The ways of saying "nothing is known about this system".
UNKNOWN = {
    "missing": None,
    "axes=None": CS(axes=None),
    "axes=[...]": CS(axes=[...]),
}


@pytest.fixture(params=list(UNKNOWN), ids=list(UNKNOWN))
def unknown(request: pytest.FixtureRequest) -> tx.Optional[CS]:
    """A system about which nothing is known, in each spelling."""
    return UNKNOWN[request.param]


@pytest.fixture(params=["axes=None", "axes=[...]"])
def unknown_axes(request: pytest.FixtureRequest) -> tx.Optional[list]:
    """The axes of a system about which nothing is known, both spellings."""
    return None if request.param == "axes=None" else [...]


def _xyz() -> CS:
    return CS(axes=[X, Y, Z])


def _scale_x(inner: tx.Optional[CS] = None) -> SubspaceTransformation:
    # `Sub(Scaling([2], input=inner, output=inner), axes=[0])`: a subspace
    # that names no full-space system.
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
    # Regression. The full-space system a subspace derives from its inner
    # system used to stop at the last acted-on axis, so this subspace
    # reported a one-axis system inside a three-axis chain.
    sub = _scale_x(CS(axes=[X]))
    assert sub.input == CS(axes=[X, ...])
    assert sub.input.ndim is None
    assert sub.output == CS(axes=[X, ...])


def test_a_subspace_with_an_open_system_is_not_embedded_by_guess() -> None:
    # Regression. `to(Affine)` used to build a 1x2 matrix from the
    # one-axis system it reported.
    with pytest.raises(ConversionError, match="axis count .* is unknown"):
        _scale_x(CS(axes=[X])).to(Affine)


@pytest.mark.parametrize("where", ["before", "after", "between"])
def test_a_subspace_composes_in_a_3d_chain(where: str) -> None:
    # Regression. Composing the subspace with 3D affines used to raise a
    # matmul `ValueError`. The affines state the number of axes of the
    # space they share with the subspace, which closes it.
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
    # The subspace acts on axis 3, which a 3D neighbour does not have: the
    # count is not stretched to fit, the composition is refused.
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
    # An inner system that states no axis gives nothing to place: whatever
    # its spelling, the subspace knows nothing of its full space, refuses
    # to guess its size, and is closed by a neighbour that knows it.
    sub = _scale_x(unknown)
    assert sub.input == None  # noqa: E711
    with pytest.raises(ConversionError, match="axis count"):
        sub.to(Affine)
    shift = Translation(translation=np.array([1.0, 2.0, 3.0]))
    result = Sequence(transformations=[shift, sub]).compute()
    assert np.allclose(result.matrix[:, :3], np.diag([2.0, 1.0, 1.0]))


def test_a_declared_unknown_system_derives_like_a_missing_one(
    unknown_axes: tx.Optional[list],
) -> None:
    # A declared system that says nothing is read as no declaration, so the
    # subspace derives its full-space system from its inner system.
    sub = SubspaceTransformation(
        transformation=Scaling(scale=np.array([2.0]), input=CS(axes=[X])),
        input_axes=np.array([1]),
        output_axes=np.array([1]),
        input=CS(axes=unknown_axes),
    )
    assert sub.input == CS(axes=[Axis(), X, ...])
    # It is still stored as given.
    assert sub._input.axes == unknown_axes


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "axes", [None, [...], [X, ...], [..., X], [X, ..., Z]]
)
def test_subspace_to_affine_refuses_an_open_declared_system(
    axes: tx.Optional[list],
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
    # Unknown or compatible: no disagreement. A conflict between the axes
    # that are known: a disagreement.
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
#   ENDPOINTS: None, axes=None AND axes=[...] READ ALIKE
# ----------------------------------------------------------------------


def test_a_sequence_derives_its_endpoints_past_an_unknown_one(
    unknown: tx.Optional[CS],
) -> None:
    child = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[child], input=unknown, output=unknown)
    assert seq.input == _xyz() and seq.output == _xyz()


def test_an_inverse_derives_its_endpoints_past_an_unknown_one(
    unknown: tx.Optional[CS],
) -> None:
    forward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    inverse = Inverse(forward=forward, input=unknown, output=unknown)
    assert inverse.input == RASCoordinateSystem()
    assert inverse.output == _xyz()
    assert inverse.inverse().input == _xyz()


def test_a_bijection_reads_past_an_unknown_side(
    unknown: tx.Optional[CS],
) -> None:
    forward = Affine(matrix=np.eye(3, 4), input=unknown, output=unknown)
    backward = Affine(
        matrix=np.eye(3, 4), input=_xyz(), output=RASCoordinateSystem()
    )
    bijection = Bijection(forward=forward, backward=backward)
    assert bijection.input == RASCoordinateSystem()
    assert bijection.output == _xyz()


def test_an_unknown_endpoint_is_not_propagated_over_a_known_one(
    unknown: tx.Optional[CS],
) -> None:
    # An identity that says nothing about its input does not overwrite the
    # input of the transform it simplifies into.
    affine = Affine(matrix=np.eye(3, 4), input=_xyz(), output=_xyz())
    seq = Sequence(transformations=[Identity(input=unknown), affine])
    result = seq.simplify()
    assert result.input == _xyz()


def test_a_known_endpoint_is_propagated_onto_an_unknown_one(
    unknown: tx.Optional[CS],
) -> None:
    # A sequence's own endpoint reaches a first element that says nothing
    # about its input, however that element spells it.
    first = Scaling(scale=np.array([2.0, 2.0, 2.0]), input=unknown)
    seq = Sequence(transformations=[first], input=_xyz())
    assert seq._flattened().transformations[0].input == _xyz()


def test_smartproperty_missing_keeps_the_stored_value() -> None:
    class Box(DataModelBase):
        _value: tx.Optional[CS] = None

        @smartproperty(missing=lambda v: v == None)  # noqa: E711
        def value(self) -> CS:
            return _xyz()

    unknown = CS(axes=[...])
    box = Box(value=unknown)
    assert box.value == _xyz()
    assert box._value is unknown
    assert Box(value=CS(axes=[X])).value == CS(axes=[X])


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
        # x is known on one side at the start, and elsewhere on the other:
        # reordering would move axes that `...` hides.
        (CS(axes=[X, ...]), CS(axes=[Y, X, Z])),
        (CS(axes=[Y, X, Z]), CS(axes=[..., Y])),
        # RAS to LPS is a flip of the known axes, but the bridge would also
        # have to carry the axes that `...` stands for.
        (CS(axes=[R, A, S, ...]), LPSCoordinateSystem()),
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
    first = Affine(matrix=np.eye(3, 4), output=CS(axes=[R, A, S, ...]))
    second = Affine(matrix=np.eye(3, 4), input=LPSCoordinateSystem())
    with pytest.raises(AdaptationError, match="open"):
        adapt(first, second)


def test_embed_needs_closed_systems(unknown: tx.Optional[CS]) -> None:
    inner = Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    open_full = CS(axes=[R, A, S, ...])
    assert embed(inner, full=open_full, side="input") is None
    assert embed(inner, full=unknown, side="input") is None
    full = CS(axes=[R, A, S, Axis(name="t", type="time")])
    assert embed(inner, full=full, side="input") is not None


def test_grid_extents_close_an_open_grid_system_from_its_shape() -> None:
    i, k = Axis(name="i"), Axis(name="k")
    for axes, expected in (
        ([i, ...], {"i": 3}),
        ([..., k], {"k": 5}),
        (None, {}),
        ([...], {}),
    ):
        system = CS(axes=axes)
        grid = CartesianField(shape=(3, 4, 5), input=system, output=system)
        assert _grid_extents(grid, at_output=True) == expected


# ----------------------------------------------------------------------
#   DISCRETE AXES
# ----------------------------------------------------------------------


def test_the_discrete_check_reads_an_open_derived_system() -> None:
    # The subspace declares no full-space system, so it derives an open
    # one from its inner field's system. The check still finds the
    # discrete axis at the position the inner axis is placed at.
    discrete = Axis(name="c", discrete=True)
    inner_system = CS(axes=[discrete, SpatialAxis(name="y")])
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
