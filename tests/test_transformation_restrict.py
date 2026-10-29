"""Tests of `Transformation.restrict`.

A restriction reads a transformation from a block of its input axes to
a block of its output axes. It is the lazy composition of a projection
that creates the removed inputs, the transformation itself, and a
projection that drops the removed outputs, and `compute` reduces it.
A removed input that `at` names takes that value, and any other removed
input is free, so that `compute` raises when a kept output reads it.
"""

import numpy as np
import pytest

from brainhops.datamodel.axes import Axis, SpaceAxis, TimeAxis
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    Identity,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)
from brainhops.errors import RestrictionError

# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _xyt() -> CoordinateSystem:
    return CoordinateSystem(
        axes=[SpaceAxis(name="x"), SpaceAxis(name="y"), TimeAxis(name="t")]
    )


def _xyzt() -> CoordinateSystem:
    return CoordinateSystem(
        axes=[
            SpaceAxis(name="x"),
            SpaceAxis(name="y"),
            SpaceAxis(name="z"),
            TimeAxis(name="t"),
        ]
    )


def _affine(rows: list, system: object = None) -> Affine:
    system = _xyt() if system is None else system
    return Affine(
        matrix=np.array(rows, dtype=float), input=system, output=system
    )


# The three rows of the table of the memo, for an affine of (x, y, t).
# In the second, x' reads t. In the third, only t', which is dropped,
# reads a kept input.
_A1 = [[2, 0, 0, 1], [0, 3, 0, 2], [0, 0, 1, 5]]
_A2 = [[2, 0, 0.5, 1], [0, 3, 0, 2], [0, 0, 1, 5]]
_A3 = [[2, 0, 0, 1], [0, 3, 0, 2], [0, 1, 1, 5]]

_SPATIAL = [[2.0, 0, 1], [0, 3, 2]]
_SECTION = [[2.0, 0, 3], [0, 3, 2]]


def _names(system: object) -> list:
    return list(system.axes.names)


# ----------------------------------------------------------------------
#   THE TABLE OF FIXED AND FREE AXES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
@pytest.mark.parametrize(
    "rows, free, fixed",
    [
        (_A1, _SPATIAL, _SPATIAL),
        (_A2, None, _SECTION),
        (_A3, _SPATIAL, _SPATIAL),
    ],
    ids=["A1", "A2", "A3"],
)
def test_the_table_comes_out_of_compute(
    rows: list, free: list, fixed: list, policy: str
) -> None:
    t = _affine(rows)
    restricted = t.restrict(["x", "y"], ["x", "y"])
    if free is None:
        with pytest.raises(RestrictionError, match="free"):
            restricted.compute(simplify=policy)
    else:
        assert np.allclose(restricted.compute(simplify=policy).matrix, free)
    section = t.restrict(["x", "y"], ["x", "y"], at={"t": 4})
    assert np.allclose(section.compute(simplify=policy).matrix, fixed)


def test_a_free_axis_that_is_read_raises_only_at_compute() -> None:
    t = _affine(_A2)
    restricted = t.restrict(SpaceAxis, SpaceAxis)
    for policy in ("analytic", "numeric"):
        simplified = restricted.simplify(policy)
        assert any(
            isinstance(leaf, Projection) and np.isnan(leaf.at).any()
            for leaf in simplified.transformations
        )
    with pytest.raises(RestrictionError, match="free"):
        restricted.compute()


def test_nan_in_at_makes_an_axis_free() -> None:
    restricted = _affine(_A2).restrict(SpaceAxis, SpaceAxis, at={"t": np.nan})
    with pytest.raises(RestrictionError, match="free"):
        restricted.compute()


# ----------------------------------------------------------------------
#   LAZINESS AND SYSTEMS
# ----------------------------------------------------------------------


def test_the_restriction_is_a_lazy_composition_of_projections() -> None:
    t = _affine(_A2)
    restricted = t.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
    assert isinstance(restricted, Sequence)
    create, middle, drop = restricted.transformations
    assert middle is t
    assert isinstance(create, Projection) and list(create.created) == [2]
    assert list(create.at) == [4.0]
    assert isinstance(drop, Projection) and list(drop.dropped) == [2]
    assert create.output is t.input and drop.input is t.output


def test_the_systems_are_those_of_t_restricted_to_the_block() -> None:
    restricted = _affine(_A1).restrict(SpaceAxis, ["x"], at={"t": 0})
    assert _names(restricted.input) == ["x", "y"]
    assert _names(restricted.output) == ["x"]
    computed = restricted.compute()
    assert _names(computed.input) == ["x", "y"]
    assert _names(computed.output) == ["x"]
    assert np.allclose(computed.matrix, [[2.0, 0, 1]])


def test_the_full_block_returns_t_itself() -> None:
    t = _affine(_A2)
    assert t.restrict([0, 1, 2], [0, 1, 2]) is t
    assert t.restrict(Axis, Axis) is t
    assert t.restrict(["x", "y", "t"], [SpaceAxis, "t"]) is t


def test_keeping_every_input_adds_only_the_drop() -> None:
    t = _affine(_A3)
    restricted = t.restrict([0, 1, 2], ["x", "y"])
    first, drop = restricted.transformations
    assert first is t and isinstance(drop, Projection)
    expected = [[2.0, 0, 0, 1], [0, 3, 0, 2]]
    assert np.allclose(restricted.compute().matrix, expected)


def test_an_empty_block_of_inputs_gives_the_constant_outputs() -> None:
    t = _affine(_A1)
    restricted = t.restrict([], ["x", "y"], at={"x": 1, "y": 2, "t": 3})
    computed = restricted.compute()
    assert np.allclose(computed.matrix, [[3.0], [8.0]])


# ----------------------------------------------------------------------
#   AXIS QUERIES
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "refs",
    [
        ["x", "y"],
        [0, 1],
        [0, -2],
        SpaceAxis,
        SpaceAxis(),
        {"type": "space"},
        lambda axis: axis.name in ("x", "y"),
        ["x", 1],
        [SpaceAxis],
    ],
    ids=[
        "names",
        "positions",
        "negative",
        "class",
        "template",
        "fields",
        "function",
        "mixed",
        "list-of-class",
    ],
)
def test_every_form_of_query_designates_the_spatial_axes(refs: object) -> None:
    restricted = _affine(_A2).restrict(refs, refs, at={TimeAxis: 4})
    assert np.allclose(restricted.compute().matrix, _SECTION)


@pytest.mark.parametrize("key", ["t", 2, -1, TimeAxis])
def test_at_is_keyed_like_the_input_axes(key: object) -> None:
    restricted = _affine(_A2).restrict(SpaceAxis, SpaceAxis, at={key: 4})
    assert np.allclose(restricted.compute().matrix, _SECTION)


def test_at_accepts_numpy_scalars() -> None:
    restricted = _affine(_A2).restrict(
        SpaceAxis, SpaceAxis, at={np.int64(2): np.float32(4)}
    )
    assert np.allclose(restricted.compute().matrix, _SECTION)


def test_an_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="named 'z'"):
        _affine(_A1).restrict(["x", "z"], ["x"])
    with pytest.raises(KeyError, match="named 'z'"):
        _affine(_A1).restrict(["x"], ["x"], at={"z": 1})


def test_a_shared_name_raises() -> None:
    system = CoordinateSystem(
        axes=[SpaceAxis(name="x"), SpaceAxis(name="x"), TimeAxis(name="t")]
    )
    with pytest.raises(ValueError, match="2 axes"):
        _affine(_A1, system).restrict(["x"], [0])


def test_a_query_that_matches_nothing_raises() -> None:
    system = CoordinateSystem(axes=[SpaceAxis(name=n) for n in "xyz"])
    with pytest.raises(ValueError, match="TimeAxis matches no"):
        _affine(_A1, system).restrict(TimeAxis, [0])


def test_an_axis_designated_twice_raises() -> None:
    with pytest.raises(ValueError, match="more than once"):
        _affine(_A1).restrict(["x", 0], ["x"])


def test_the_order_of_the_references_is_ignored() -> None:
    # A block is a set of axes, which keep the order of the
    # transformation.
    t = _affine(_A2)
    forward = t.restrict(["x", "y"], ["x", "y"], at={"t": 4})
    backward = t.restrict(["y", "x"], ["y", "x"], at={"t": 4})
    assert np.allclose(backward.compute().matrix, _SECTION)
    assert _names(backward.input) == _names(forward.input) == ["x", "y"]
    mixed = t.restrict([TimeAxis, "x"], ["t", 0])
    assert np.allclose(mixed.compute().matrix, [[2.0, 0.5, 1], [0, 1, 5]])


def test_a_position_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        _affine(_A1).restrict([0, 3], [0])


def test_a_reference_of_another_type_raises() -> None:
    with pytest.raises(TypeError):
        _affine(_A1).restrict(0, [0])
    with pytest.raises(TypeError):
        _affine(_A1).restrict([0.5], [0])


def test_at_for_a_kept_input_raises() -> None:
    with pytest.raises(ValueError, match="keeps"):
        _affine(_A1).restrict(SpaceAxis, SpaceAxis, at={"x": 1})


def test_at_with_two_values_for_one_axis_raises() -> None:
    with pytest.raises(ValueError, match="more than one value"):
        _affine(_A1).restrict(SpaceAxis, SpaceAxis, at={"t": 1, 2: 1})


@pytest.mark.parametrize("at", [[("t", 4)], {"t": "4"}, {"t": True}])
def test_at_that_is_not_a_mapping_of_numbers_raises(at: object) -> None:
    with pytest.raises(TypeError):
        _affine(_A1).restrict(SpaceAxis, SpaceAxis, at=at)


# ----------------------------------------------------------------------
#   OPEN AND MISSING SYSTEMS
# ----------------------------------------------------------------------


def test_positions_work_without_systems() -> None:
    t = Affine(matrix=np.array(_A2, dtype=float))
    restricted = t.restrict([0, 1], [0, 1], at={-1: 4})
    assert restricted.input is None and restricted.output is None
    assert np.allclose(restricted.compute().matrix, _SECTION)


def test_a_name_or_a_query_without_systems_raises() -> None:
    t = Affine(matrix=np.array(_A2, dtype=float))
    with pytest.raises(KeyError, match="no input system"):
        t.restrict(["x"], [0])
    with pytest.raises(ValueError, match="no output system"):
        t.restrict([0], SpaceAxis)


def test_names_resolve_on_both_sides_of_an_open_system() -> None:
    # The name after `...` counts from the end, and the number of axes
    # of the matrix turns it into a position from the start.
    system = CoordinateSystem(
        axes=[SpaceAxis(name="x"), ..., TimeAxis(name="t")]
    )
    t = _affine(_A2, system)
    restricted = t.restrict(["x", 1], ["x", 1], at={"t": 4})
    assert np.allclose(restricted.compute().matrix, _SECTION)
    assert restricted.input.axes.ndim == 2
    assert restricted.input.axes[0].name == "x"


def test_a_query_never_matches_the_axes_of_an_ellipsis() -> None:
    system = CoordinateSystem(axes=[SpaceAxis(name="x"), ...])
    t = _affine(_A1, system)
    restricted = t.restrict(SpaceAxis, SpaceAxis, at={1: 0, 2: 0})
    assert np.allclose(restricted.compute().matrix, [[2.0, 1]])


def test_an_unknown_number_of_axes_raises() -> None:
    with pytest.raises(RestrictionError, match="unknown"):
        Identity().restrict([0], [0])
    open_system = CoordinateSystem(axes=[SpaceAxis(name="x"), ...])
    with pytest.raises(RestrictionError, match="unknown"):
        Identity(input=open_system, output=open_system).restrict(["x"], ["x"])


# ----------------------------------------------------------------------
#   SUBSPACES, SEQUENCES AND INVERSES
# ----------------------------------------------------------------------


def test_the_block_of_a_subspace_is_its_inner_transformation() -> None:
    inner = Affine(matrix=np.array([[2.0, 1, 1], [0, 3, 2]]))
    subspace = SubspaceTransformation(inner, [0, 2], [0, 2])
    assert subspace.restrict([0, 2], [0, 2]) is inner
    assert subspace.restrict([0, 2], [0, 2]).compute() is inner


def test_the_block_of_a_subspace_with_systems_computes_to_its_inner() -> None:
    inner = Scaling(scale=[2.0, 3.0])
    system = CoordinateSystem(axes=[SpaceAxis(name=n) for n in "xyz"])
    subspace = SubspaceTransformation(
        inner, [0, 2], [0, 2], input=system, output=system
    )
    computed = subspace.restrict(["x", "z"], ["x", "z"]).compute()
    assert isinstance(computed, Scaling)
    assert np.allclose(computed.scale, [2.0, 3.0])
    assert _names(computed.input) == ["x", "z"]


def test_a_named_block_of_an_open_subspace_is_its_inner() -> None:
    system = CoordinateSystem(axes=[SpaceAxis(name="x"), SpaceAxis(name="z")])
    inner = Scaling(scale=[2.0, 3.0], input=system, output=system)
    subspace = SubspaceTransformation(inner, [0, 2], [0, 2])
    assert subspace.restrict(["x", "z"], ["x", "z"]) is inner


def test_a_subspace_that_reorders_its_axes_keeps_the_full_order() -> None:
    # The subspace reads the full axes 2 and 0 in that order, and the
    # restriction keeps them in the order 0 and 2, so the result is the
    # wrapped affine with its two axes swapped on each side. The same
    # map comes out of compute when the subspace states its axes.
    inner = Affine(matrix=np.array([[2.0, 1, 1], [0, 3, 2]]))
    expected = [[3.0, 0, 2], [1, 2, 1]]
    subspace = SubspaceTransformation(inner, [2, 0], [2, 0])
    restricted = subspace.restrict([2, 0], [0, 2])
    assert np.allclose(restricted.compute().matrix, expected)
    system = CoordinateSystem().expand(3)
    closed = SubspaceTransformation(
        inner, [2, 0], [2, 0], input=system, output=system
    )
    computed = closed.restrict([0, 2], [0, 2]).compute()
    assert np.allclose(computed.to(Affine).matrix, expected)


def test_another_block_of_an_open_subspace_raises() -> None:
    subspace = SubspaceTransformation(
        Scaling(scale=[2.0, 3.0]), [0, 2], [0, 2]
    )
    with pytest.raises(RestrictionError, match="unknown"):
        subspace.restrict([0], [0])


def test_a_sequence_is_restricted_as_a_whole() -> None:
    shift = Translation(translation=[1.0, 1, 1], input=_xyt(), output=_xyt())
    chain = Sequence([_affine(_A2), shift])
    restricted = chain.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
    assert np.allclose(restricted.compute().matrix, [[2.0, 0, 4], [0, 3, 3]])


def test_a_lazy_inverse_is_restricted_without_being_replaced() -> None:
    t = _affine(_A2)
    inverse = t.inverse()
    restricted = inverse.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
    assert restricted.transformations[1] is inverse
    full = np.linalg.inv(np.vstack([np.array(_A2, float), [0, 0, 0, 1]]))
    # The inverse maps (x', y', t') back, and t' = 4 gives t = -1.
    expected = full[:2, [0, 1, 3]] + np.outer(full[:2, 2], [0, 0, 4])
    assert np.allclose(restricted.compute().matrix, expected)


# ----------------------------------------------------------------------
#   GEOMETRIES
# ----------------------------------------------------------------------


def _series() -> Geometry:
    # The geometry of a series of 3 volumes of shape (4, 5, 6). The
    # position of each volume moves along x with time, as after motion
    # correction, so x reads t.
    voxels = CoordinateSystem(
        axes=[SpaceAxis(name=n, unit="voxel") for n in "ijk"]
        + [TimeAxis(name="v", unit="index")]
    )
    world = CoordinateSystem(
        axes=[SpaceAxis(name=n, unit="mm") for n in "xyz"]
        + [TimeAxis(name="t", unit="s")]
    )
    matrix = np.zeros((4, 5))
    matrix[:3, :3] = [[2, 0.1, 0], [0, 3, 0], [0, 0, 4]]
    matrix[:, 3] = [0.5, 0, 0, 2]
    matrix[:, 4] = [1, 2, 3, 0]
    grid = CartesianField(shape=(4, 5, 6, 3), input=voxels, output=voxels)
    return Geometry((grid, Affine(matrix=matrix, input=voxels, output=world)))


def _world(geometry: Geometry, index: np.ndarray) -> np.ndarray:
    # Map indices of the grid through the computed transformation.
    matrix = np.asarray(geometry.compute().transformation.matrix)
    return index @ matrix[:, :-1].T + matrix[:, -1]


def _samples(shape: tuple, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, size=8) for n in shape], -1)


def test_one_volume_of_a_series_is_a_geometry() -> None:
    series = _series()
    volume = series.restrict(["i", "j", "k"], ["x", "y", "z"], at={"v": 2})
    assert isinstance(volume, Geometry)
    assert volume.shape == (4, 5, 6)
    assert _names(volume.grid.input) == ["i", "j", "k"]
    assert _names(volume.output) == ["x", "y", "z"]
    computed = volume.compute()
    assert isinstance(computed, Geometry) and computed.shape == (4, 5, 6)
    index = _samples((4, 5, 6))
    full = np.hstack([index, np.full((len(index), 1), 2)])
    expected = _world(series, full)[:, :3]
    assert np.allclose(_world(volume, index), expected)


def test_a_slice_of_a_volume_is_a_geometry() -> None:
    series = _series()
    plane = series.restrict(["i", "j"], Axis, at={"k": 1, "v": 0})
    assert isinstance(plane, Geometry) and plane.shape == (4, 5)
    index = _samples((4, 5))
    full = np.hstack([index, np.ones((len(index), 1)), np.zeros((8, 1))])
    assert np.allclose(_world(plane, index), _world(series, full))


def test_a_slice_matches_the_indexing_of_the_geometry() -> None:
    series = _series()
    plane = series.restrict(["i", "k", "v"], Axis, at={"j": 3})
    indexed = series[:, 3]
    assert plane.shape == indexed.shape
    index = _samples(plane.shape)
    assert np.allclose(_world(plane, index), _world(indexed, index))


def test_removing_outputs_only_keeps_the_grid() -> None:
    series = _series()
    spatial = series.restrict(Axis, SpaceAxis)
    assert isinstance(spatial, Geometry)
    assert spatial.grid is series.grid
    assert spatial.compute().transformation.matrix.shape == (3, 5)


def test_the_full_block_of_a_geometry_is_the_geometry() -> None:
    series = _series()
    assert series.restrict(Axis, Axis) is series


@pytest.mark.parametrize(
    "at",
    [None, {"v": 1.5}, {"v": 3}, {"v": -1}],
    ids=["free", "between", "beyond", "negative"],
)
def test_a_removed_axis_without_a_sample_is_not_a_geometry(at: object) -> None:
    # The kept outputs read v, so the plain restriction raises at
    # compute when v is free, and is the section at v otherwise.
    series = _series()
    restricted = series.restrict(["i", "j", "k"], ["x", "y", "z"], at=at)
    assert not isinstance(restricted, Geometry)
    if at is None:
        with pytest.raises(RestrictionError, match="free"):
            restricted.compute()
        return
    v = at["v"]
    index = _samples((4, 5, 6)).astype(float)
    full = np.hstack([index, np.full((len(index), 1), v)])
    expected = _world(series, full)[:, :3]
    matrix = np.asarray(restricted.compute().matrix)
    assert np.allclose(index @ matrix[:, :-1].T + matrix[:, -1], expected)


def test_a_free_axis_that_no_output_reads_gives_a_map() -> None:
    # The spatial outputs of the series do not read v once x does not,
    # so the restriction without a volume is a valid map, but it has no
    # grid to sample.
    series = _series()
    matrix = np.array(series.transformation.matrix)
    matrix[0, 3] = 0.0
    still = Geometry((series.grid, series.transformation.to(matrix=matrix)))
    restricted = still.restrict(SpaceAxis, ["x", "y", "z"])
    assert not isinstance(restricted, Geometry)
    expected = [[2.0, 0.1, 0, 1], [0, 3, 0, 2], [0, 0, 4, 3]]
    assert np.allclose(restricted.compute().matrix, expected)


# ----------------------------------------------------------------------
#   BATCHED TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_one_element_of_a_batch_is_the_restriction_at_its_index() -> None:
    # One affine per volume: the spatial outputs read t, and time is
    # carried through unchanged, so time is a batch axis.
    batch = _affine([[2, 0, 0.5, 1], [0, 3, -1, 2], [0, 0, 1, 0]])
    element = batch.restrict(SpaceAxis, SpaceAxis, at={"t": 4})
    assert np.allclose(element.compute().matrix, [[2.0, 0, 3], [0, 3, -2]])
    with pytest.raises(RestrictionError, match="free"):
        batch.restrict(SpaceAxis, SpaceAxis).compute()


@pytest.mark.parametrize("policy", ["analytic", "numeric"])
def test_slice_timing_restricted_to_space_is_defined(policy: str) -> None:
    # The time output is t + 0.1 z, so the dropped output reads z, but
    # no kept output reads t.
    matrix = np.eye(4, 5)
    matrix[3, 2] = 0.1
    timing = Affine(matrix=matrix, input=_xyzt(), output=_xyzt())
    computed = timing.restrict(SpaceAxis, SpaceAxis).compute(simplify=policy)
    assert computed.is_identity(compute=True)
    assert _names(computed.input) == ["x", "y", "z"]


# ----------------------------------------------------------------------
#   OME-NGFF
# ----------------------------------------------------------------------


def test_a_restriction_with_values_writes_once_it_is_computed() -> None:
    pytest.importorskip("abczarr")
    from brainhops.io.transformations.zarr._map import (
        OmeMappingError,
        to_ome,
    )

    restricted = _affine(_A2).restrict(SpaceAxis, SpaceAxis, at={"t": 4})
    assert to_ome(restricted.compute(), [0, 1], 2) == {
        "type": "sequence",
        "transformations": [
            {"type": "scale", "scale": [2.0, 3.0]},
            {"type": "translation", "translation": [3.0, 2.0]},
        ],
    }
    # Before it is computed, the restriction holds elements that add or
    # remove axes, which a map of the image axes cannot hold.
    with pytest.raises(OmeMappingError, match="axes onto"):
        to_ome(restricted, [0, 1], 2)
    free = _affine(_A2).restrict(SpaceAxis, SpaceAxis)
    with pytest.raises(OmeMappingError, match="free"):
        to_ome(free, [0, 1], 2)
