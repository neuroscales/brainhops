"""Tests for the mapping of OME-NGFF coordinate transformations.

These tests read OME-NGFF 0.6 metadata, as RFC-5 defines it, without a
Zarr store, so that they run wherever abczarr is installed. Most of them
use the examples of the specification (ngff-spec, branch `0.6` at
89da1da, `examples/transformations`). Each example declares an input and
an output coordinate system, and the expected values follow the
normative rule of the specification: entry `i` of a parameter belongs to
axis `i` of the system, and a matrix has one row per output axis and one
column per input axis, followed by the translation. The prose under the
`translation`, `rotation`, `sequence` and second `affine` examples of
the specification disagrees with this rule, while the prose under the
`scale`, `identity`, `mapAxis` and first `affine` examples agrees with
it, so the rule is what the tests check.

brainhops sorts the axes of each system into its own order, so a test
evaluates the mapped transformation on a point given by axis name and
compares the result by axis name.
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel._transformations.multiscale import _as_affine
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Bijection,
    DisplacementField,
    Identity,
    Permutation,
    Projection,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Translation,
)

# The module imports abczarr itself, so the skip must come first.
abczarr = pytest.importorskip("abczarr")

from abczarr.ome import v0_4 as v4  # noqa: E402
from abczarr.ome import v0_6 as v6  # noqa: E402
from abczarr.ome import v0_6dev2 as v6dev2  # noqa: E402

from brainhops.io.images.zarr import (  # noqa: E402
    OmeImageError,
    _axisorder,
    _ome,
)
from brainhops.io.transformations.zarr import _map  # noqa: E402
from brainhops.io.transformations.zarr._axes import _to_axis  # noqa: E402

_JI = [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]
_YX = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
_ZYX = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]


# ---- helpers ---------------------------------------------------------


class _System(tx.NamedTuple):
    # A coordinate system of the metadata, with its brainhops order.
    stored: tx.List[str]
    order: tx.List[int]

    @property
    def names(self) -> tx.List[str]:
        return [self.stored[p] for p in self.order]


def _system(axes: tx.List[dict]) -> _System:
    parsed = [_to_axis(axis) for axis in axes]
    return _System(
        stored=[axis["name"] for axis in axes],
        order=_axisorder.to_canonical(parsed),
    )


def _map_json(
    transform: dict,
    source: _System,
    target: _System,
    reading: tx.Optional[_map.Reading] = None,
) -> tx.Any:
    parsed = v6.CoordinateTransformation.from_json(transform)
    return _map.from_ome(parsed, source.order, target.order, reading)


def _evaluate(
    xform: tx.Any,
    source: _System,
    target: _System,
    point: tx.Dict[str, float],
) -> tx.Dict[str, float]:
    # Evaluate an affine transformation on a point given by axis name.
    if isinstance(xform, Identity):
        matrix = np.eye(len(target.order), len(source.order) + 1)
    else:
        affine = _as_affine(xform)
        assert affine is not None
        matrix = np.asarray(affine.matrix, dtype=float)
    vector = np.asarray([point[name] for name in source.names] + [1.0])
    values = matrix @ vector
    return dict(zip(target.names, values.tolist()))


def _assert_point(
    actual: tx.Dict[str, float], expected: tx.Dict[str, float]
) -> None:
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value), name


_POINT_JI = {"j": 5.0, "i": 7.0}


# ---- the parameters follow the axes of each system -------------------


def test_scale_scales_each_axis_by_its_own_entry() -> None:
    # examples/transformations/scale.json, whose prose gives
    # x = 3.12 * i and y = 2 * j.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json({"type": "scale", "scale": [2, 3.12]}, source, target)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI),
        {"y": 2 * 5.0, "x": 3.12 * 7.0},
    )


def test_a_scale_between_systems_of_one_order_stays_a_scaling() -> None:
    zyx = _system(_ZYX)
    xform = _map_json({"type": "scale", "scale": [3, 2, 1]}, zyx, zyx)
    assert isinstance(xform, Scaling)
    # The stored (z, y, x) entries are read in the brainhops (x, y, z) order.
    np.testing.assert_allclose(np.asarray(xform.scale), [1.0, 2.0, 3.0])


def test_translation_offsets_each_axis_by_its_own_entry() -> None:
    # examples/transformations/translation.json.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json(
        {"type": "translation", "translation": [9, -1.42]}, source, target
    )
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI),
        {"y": 5.0 + 9, "x": 7.0 - 1.42},
    )


def test_identity_carries_each_axis_to_the_axis_of_the_same_index() -> None:
    # examples/transformations/identity.json, whose prose gives x = i and
    # y = j.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json({"type": "identity"}, source, target)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI), {"y": 5.0, "x": 7.0}
    )


def test_affine_rows_are_output_axes_and_columns_are_input_axes() -> None:
    # examples/transformations/affine2d2d.json, whose prose gives
    # y = 1*j + 2*i + 3 and x = 4*j + 5*i + 6.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json(
        {"type": "affine", "affine": [[1, 2, 3], [4, 5, 6]]}, source, target
    )
    assert isinstance(xform, Affine)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI),
        {"y": 1 * 5.0 + 2 * 7.0 + 3, "x": 4 * 5.0 + 5 * 7.0 + 6},
    )


def test_affine_maps_two_axes_onto_three() -> None:
    # examples/transformations/affine2d3d.json, read with the translation
    # in the last column, as the definition of an affine requires.
    source = _system(
        [{"name": "i", "type": "space"}, {"name": "j", "type": "space"}]
    )
    target = _system(_ZYX)
    xform = _map_json(
        {"type": "affine", "affine": [[1, 0, 0], [2, 3, 4], [5, 6, 7]]},
        source,
        target,
    )
    i, j = 5.0, 7.0
    _assert_point(
        _evaluate(xform, source, target, {"i": i, "j": j}),
        {"z": 1 * i, "y": 2 * i + 3 * j + 4, "x": 5 * i + 6 * j + 7},
    )


def test_affine_accepts_a_full_homogeneous_matrix() -> None:
    zyx = _system(_ZYX)
    compact = [[1, 0, 0, 4], [0, 2, 0, 5], [0, 0, 3, 6]]
    full = compact + [[0, 0, 0, 1]]
    one = _map_json({"type": "affine", "affine": compact}, zyx, zyx)
    two = _map_json({"type": "affine", "affine": full}, zyx, zyx)
    np.testing.assert_allclose(np.asarray(one.matrix), np.asarray(two.matrix))


def test_affine_refuses_a_matrix_that_does_not_fit_its_systems() -> None:
    zyx = _system(_ZYX)
    with pytest.raises(_map.OmeMappingError):
        _map_json({"type": "affine", "affine": [[1, 0, 0]]}, zyx, zyx)


def test_rotation_rows_are_output_axes_and_columns_are_input_axes() -> None:
    # examples/transformations/rotation.json.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json(
        {"type": "rotation", "rotation": [[0, -1], [1, 0]]}, source, target
    )
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI), {"y": -7.0, "x": 5.0}
    )


def test_map_axis_reads_the_input_axis_of_each_output_axis() -> None:
    # examples/transformations/mapAxis1.json, whose prose gives x = j and
    # y = i for the permutation.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json({"type": "mapAxis", "mapAxis": [1, 0]}, source, target)
    assert isinstance(xform, Permutation)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI), {"y": 7.0, "x": 5.0}
    )


def test_map_axis_between_two_orders_of_the_same_axes_is_the_identity() -> (
    None
):
    # Both systems sort their axes into (x, y, z), so the reordering that
    # the metadata spells out leaves brainhops coordinates unchanged.
    zyx = _system(_ZYX)
    xyz = _system(list(reversed(_ZYX)))
    xform = _map_json({"type": "mapAxis", "mapAxis": [2, 1, 0]}, zyx, xyz)
    np.testing.assert_array_equal(np.asarray(xform.permutation), [0, 1, 2])


def test_map_axis_refuses_a_map_that_is_not_a_permutation() -> None:
    # abczarr 0.2.3 refuses such a mapAxis when it parses it, and brainhops
    # refuses it when an older abczarr lets it through. Both errors are
    # value errors.
    yx = _system(_YX)
    with pytest.raises(ValueError, match="not a permutation"):
        _map_json({"type": "mapAxis", "mapAxis": [0, 0]}, yx, yx)


def test_sequence_applies_its_first_transformation_first() -> None:
    # examples/transformations/sequence.json.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json(
        {
            "type": "sequence",
            "transformations": [
                {"type": "translation", "translation": [0.1, 0.9]},
                {"type": "scale", "scale": [2, 3]},
            ],
        },
        source,
        target,
    )
    assert isinstance(xform, Sequence)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI),
        {"y": (5.0 + 0.1) * 2, "x": (7.0 + 0.9) * 3},
    )


def test_sequence_follows_a_change_in_the_number_of_axes() -> None:
    # The space between the two steps has two axes, stored as (y, x).
    zyx, yx = _system(_ZYX), _system(_YX)
    xform = _map_json(
        {
            "type": "sequence",
            "transformations": [
                {"type": "projectAxis", "droppedInputs": [0]},
                {"type": "scale", "scale": [2, 3]},
            ],
        },
        zyx,
        yx,
    )
    projection, scaling = xform.transformations[:2]
    assert isinstance(projection, Sequence)
    assert isinstance(scaling, Sequence)
    np.testing.assert_allclose(
        np.asarray(scaling.transformations[0].scale), [2.0, 3.0]
    )


# ---- projections, bijections and byDimension -------------------------


def test_project_axis_is_a_projection_in_the_stored_order() -> None:
    # examples/transformations/projectAxis.json: two zeros are created in
    # front of (i, j). The output axes have no type, and their names sort
    # them as (x, y, z, c), so a permutation follows the projection.
    source = _system([{"name": "i"}, {"name": "j"}])
    target = _system([{"name": n} for n in "czyx"])
    xform = _map_json(
        {"type": "projectAxis", "createdOutputs": [0, 1]}, source, target
    )
    projection, permutation = xform.transformations
    assert isinstance(projection, Projection)
    np.testing.assert_array_equal(np.asarray(projection.created), [0, 1])
    assert len(projection.dropped) == 0
    np.testing.assert_array_equal(
        np.asarray(permutation.permutation), [3, 2, 1, 0]
    )


def test_project_axis_is_surrounded_by_the_permutations_of_its_systems() -> (
    None
):
    zyx, yx = _system(_ZYX), _system(_YX)
    xform = _map_json({"type": "projectAxis", "droppedInputs": [0]}, zyx, yx)
    first, projection, last = xform.transformations
    # (x, y, z) is stored as (z, y, x), z is dropped, and the stored (y, x)
    # is read as (x, y).
    np.testing.assert_array_equal(np.asarray(first.permutation), [2, 1, 0])
    np.testing.assert_array_equal(np.asarray(projection.dropped), [0])
    np.testing.assert_array_equal(np.asarray(last.permutation), [1, 0])


def test_project_axis_refuses_a_count_that_does_not_fit_its_systems() -> None:
    zyx = _system(_ZYX)
    with pytest.raises(_map.OmeMappingError):
        _map_json({"type": "projectAxis", "droppedInputs": [0]}, zyx, zyx)


def test_bijection_keeps_both_directions() -> None:
    yx = _system(_YX)
    xform = _map_json(
        {
            "type": "bijection",
            "forward": {"type": "scale", "scale": [2, 4]},
            "inverse": {"type": "scale", "scale": [0.5, 0.25]},
        },
        yx,
        yx,
    )
    assert isinstance(xform, Bijection)
    np.testing.assert_allclose(np.asarray(xform.forward.scale), [4.0, 2.0])
    np.testing.assert_allclose(np.asarray(xform.backward.scale), [0.25, 0.5])
    assert isinstance(xform.inverse().forward, Scaling)


def test_by_dimension_of_affine_parts_is_one_affine() -> None:
    # examples/transformations/byDimension1.json: x = i - 1 and y = 2 * j.
    source, target = _system(_JI), _system(_YX)
    xform = _map_json(
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {
                        "type": "translation",
                        "translation": [-1.0],
                    },
                    "inputAxes": [1],
                    "outputAxes": [1],
                },
                {
                    "transformation": {"type": "scale", "scale": [2.0]},
                    "inputAxes": [0],
                    "outputAxes": [0],
                },
            ],
        },
        source,
        target,
    )
    assert isinstance(xform, Affine)
    _assert_point(
        _evaluate(xform, source, target, _POINT_JI),
        {"y": 2 * 5.0, "x": 7.0 - 1},
    )


def test_by_dimension_may_change_the_number_of_axes() -> None:
    # examples/transformations/byDimension2.json: from (l, j, k, i) to
    # (z, y, x), with (y, x) = (i, k) + (0.5, 1.5) and z = 2 * j.
    source = _system([{"name": n, "type": "space"} for n in "ljki"])
    target = _system(_ZYX)
    xform = _map_json(
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {
                        "type": "translation",
                        "translation": [0.5, 1.5],
                    },
                    "inputAxes": [3, 2],
                    "outputAxes": [1, 2],
                },
                {
                    "transformation": {"type": "scale", "scale": [2]},
                    "inputAxes": [1],
                    "outputAxes": [0],
                },
            ],
        },
        source,
        target,
    )
    point = {"l": 1.0, "j": 2.0, "k": 3.0, "i": 4.0}
    _assert_point(
        _evaluate(xform, source, target, point),
        {"z": 2 * 2.0, "y": 4.0 + 0.5, "x": 3.0 + 1.5},
    )


def _constant_field(
    transform: tx.Any, kind: str, source: tx.Any, target: tx.Any
) -> DisplacementField:
    # A field reader that ignores the store, for the mapping tests.
    return DisplacementField(field=np.zeros((2, 1)))


def _field_by_dimension(inputs: tx.List[int], outputs: tx.List[int]) -> dict:
    return {
        "type": "byDimension",
        "transformations": [
            {
                "transformation": {"type": "displacements", "path": "f"},
                "inputAxes": inputs,
                "outputAxes": outputs,
            },
            {
                "transformation": {"type": "scale", "scale": [2.0, 3.0]},
                "inputAxes": [1, 2],
                "outputAxes": [1, 2],
            },
        ],
    }


def test_by_dimension_with_a_field_is_a_sequence_of_subspaces() -> None:
    zyx = _system(_ZYX)
    reading = _map.Reading(read_field=_constant_field)
    xform = _map_json(_field_by_dimension([0], [0]), zyx, zyx, reading)
    field, scaling = xform.transformations
    assert isinstance(field, SubspaceTransformation)
    assert isinstance(field.transformation, DisplacementField)
    # The stored z axis is brainhops axis 2.
    np.testing.assert_array_equal(np.asarray(field.input_axes), [2])
    np.testing.assert_array_equal(np.asarray(scaling.output_axes), [1, 0])


def test_by_dimension_refuses_a_field_that_moves_values_between_axes() -> None:
    zyx = _system(_ZYX)
    reading = _map.Reading(read_field=_constant_field)
    transform = _field_by_dimension([1], [0])
    transform["transformations"][1]["inputAxes"] = [0, 2]
    with pytest.raises(_map.OmeMappingError):
        _map_json(transform, zyx, zyx, reading)


def test_by_dimension_refuses_an_output_axis_written_twice() -> None:
    yx = _system(_YX)
    part = {
        "transformation": {"type": "scale", "scale": [2.0]},
        "inputAxes": [0],
        "outputAxes": [0],
    }
    with pytest.raises(_map.OmeMappingError):
        _map_json(
            {"type": "byDimension", "transformations": [part, part]}, yx, yx
        )


# ---- parameters stored in arrays -------------------------------------


def test_parameters_may_be_read_from_an_array_of_the_group() -> None:
    zyx = _system(_ZYX)
    # A mapping stands in for the group: the path is followed through it.
    group = {"params": {"scale": np.asarray([3.0, 2.0, 1.0])}}
    xform = _map_json(
        {"type": "scale", "path": "params/scale"},
        zyx,
        zyx,
        _map.Reading(node=group),
    )
    np.testing.assert_allclose(np.asarray(xform.scale), [1.0, 2.0, 3.0])


def test_parameters_in_an_array_need_a_group() -> None:
    zyx = _system(_ZYX)
    with pytest.raises(_map.OmeMappingError):
        _map_json({"type": "scale", "path": "params/scale"}, zyx, zyx)


# ---- versions and drafts ---------------------------------------------


class _Group:
    # The part of a Zarr group that reading the metadata needs.
    def __init__(self, ome: tx.Any) -> None:
        self.ome = ome


def test_the_removed_inverse_of_draft_type_is_refused() -> None:
    # RFC-5 removed `inverseOf` before version 0.6 (ome/ngff,
    # rfc/5/responses/2/index.md), and abczarr does not normalize it.
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    ome = v6dev2.OME.from_json(
        {
            "version": "0.6.dev2",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {"name": "phys", "axes": axes},
                        {"name": "world", "axes": axes},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": [1, 2],
                                    "input": "0",
                                    "output": "phys",
                                }
                            ],
                        }
                    ],
                    "coordinateTransformations": [
                        {
                            "type": "inverseOf",
                            "transformation": {
                                "type": "scale",
                                "scale": [2, 2],
                            },
                            "input": "phys",
                            "output": "world",
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(OmeImageError, match="InverseOf"):
        _ome.read_multiscale(_Group(ome))


def test_a_level_index_is_the_centre_of_its_voxel() -> None:
    # A level downsampled by two is shifted by half a base voxel in
    # OME-NGFF 0.4, so that its voxel 0 lies at the centre of base voxels 0
    # and 1. The translation is read as stored, because OME-NGFF 0.6 and
    # brainhops both put an integer index at the centre of a voxel.
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    ome = v4.OME.from_json(
        {
            "multiscales": [
                {
                    "version": "0.4",
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [2.0, 2.0]},
                                {
                                    "type": "translation",
                                    "translation": [0.5, 0.5],
                                },
                            ],
                        },
                    ],
                }
            ]
        }
    )
    multiscale, version = _ome.read_multiscale(_Group(ome))
    assert version == "0.4"
    yx = _system(axes)
    geometry = _ome.level_transformation(multiscale.datasets[1], yx.order)
    origin = _evaluate(geometry, yx, yx, {"y": 0.0, "x": 0.0})
    _assert_point(origin, {"y": 0.5, "x": 0.5})


# ---- common transformations of a multiscale --------------------------


def _multiscale(commons: tx.List[dict]) -> tx.Any:
    block = {
        "coordinateSystems": [
            {"name": "intrinsic", "axes": _ZYX},
            {"name": "world", "axes": _ZYX},
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0],
                        "input": {"path": "0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        "coordinateTransformations": commons,
    }
    ome = v6.OME.from_json({"version": "0.6", "multiscales": [block]})
    return ome.multiscales[0]


def _commons(commons: tx.List[dict]) -> tx.List[tx.Any]:
    zyx = _system(_ZYX)
    systems = {
        name: CoordinateSystem(name=name) for name in ("intrinsic", "world")
    }
    return _ome.common_transformations(
        _multiscale(commons),
        {"intrinsic": zyx.order, "world": zyx.order},
        zyx.order,
        input=systems["intrinsic"],
        systems=systems,
    )


_SCALE_TO_WORLD = {
    "type": "scale",
    "scale": [2.0, 4.0, 8.0],
}


def test_a_transformation_into_the_intrinsic_space_is_read_inverted() -> None:
    (placement,) = _commons(
        [
            dict(
                _SCALE_TO_WORLD,
                input={"name": "world"},
                output={"name": "intrinsic"},
            )
        ]
    )
    assert placement.input.name == "intrinsic"
    assert placement.output.name == "world"
    matrix = np.asarray(_as_affine(placement).matrix)
    np.testing.assert_allclose(np.diag(matrix[:, :3]), [1 / 8, 1 / 4, 1 / 2])


def test_the_inverted_placement_of_the_conformance_case_is_read() -> None:
    # ngff-spec tag 0.6, tests/attributes/strict/valid/image/
    # multiscales_example.json, a valid case whose only transformation of
    # the multiscale goes from `world` to `intrinsic`. Its datasets are
    # reduced to the first one here.
    def axes(units: tx.Dict[str, str]) -> tx.List[dict]:
        types = {"t": "time", "c": "channel"}
        return [
            dict(
                {"name": name, "type": types.get(name, "space")},
                **({"unit": units[name]} if name in units else {}),
            )
            for name in "tczyx"
        ]

    units = {"t": "millisecond", "z": "micrometer"}
    units.update(y="micrometer", x="micrometer")
    block = {
        "name": "example",
        "coordinateSystems": [
            {"name": "world", "axes": axes(units)},
            {"name": "intrinsic", "axes": axes(units)},
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 0.5, 0.5, 0.5],
                        "input": {"path": "s0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [0.1, 1.0, 1.0, 1.0, 1.0],
                "input": {"name": "world"},
                "output": {"name": "intrinsic"},
            }
        ],
    }
    ome = v6.OME.from_json({"version": "0.6", "multiscales": [block]})
    tczyx = _system(block["coordinateSystems"][0]["axes"])
    systems = {
        name: CoordinateSystem(name=name) for name in ("intrinsic", "world")
    }
    (placement,) = _ome.common_transformations(
        ome.multiscales[0],
        {"intrinsic": tczyx.order, "world": tczyx.order},
        tczyx.order,
        input=systems["intrinsic"],
        systems=systems,
    )
    assert placement.input.name == "intrinsic"
    assert placement.output.name == "world"
    world = _evaluate(
        placement,
        tczyx,
        tczyx,
        {"t": 1.0, "c": 0.0, "z": 1.0, "y": 1.0, "x": 1.0},
    )
    _assert_point(world, {"t": 10.0, "c": 0.0, "z": 1.0, "y": 1.0, "x": 1.0})


def test_a_transformation_to_another_group_is_skipped() -> None:
    placements = _commons(
        [
            {
                "type": "identity",
                "input": {"name": "intrinsic"},
                "output": {"name": "intrinsic", "path": "labels/cells"},
            },
            dict(
                _SCALE_TO_WORLD,
                input={"name": "intrinsic"},
                output={"name": "world"},
            ),
        ]
    )
    assert [p.output.name for p in placements] == ["world"]
    assert isinstance(placements[0], Scaling)


def test_translation_of_a_sequence_is_exact_in_brainhops_order() -> None:
    # A sequence between two systems of one order keeps its steps.
    zyx = _system(_ZYX)
    xform = _map_json(
        {
            "type": "sequence",
            "transformations": [
                {"type": "scale", "scale": [3.0, 2.0, 1.0]},
                {"type": "translation", "translation": [30.0, 20.0, 10.0]},
            ],
        },
        zyx,
        zyx,
    )
    scaling, translation = xform.transformations
    assert isinstance(scaling, Scaling)
    assert isinstance(translation, Translation)
    np.testing.assert_allclose(
        np.asarray(translation.translation), [10.0, 20.0, 30.0]
    )


@pytest.mark.parametrize(
    "transform",
    [
        {"type": "mapAxis", "mapAxis": [0, 5]},
        {"type": "mapAxis", "mapAxis": [0, 1, 2]},
        {"type": "projectAxis", "droppedInputs": [5]},
        {
            "type": "projectAxis",
            "droppedInputs": [0, 0],
            "createdOutputs": [0],
        },
        {"type": "projectAxis", "droppedInputs": [-1]},
    ],
)
def test_axis_indices_must_be_valid_and_unique(transform: dict) -> None:
    # OME-NGFF 0.6 requires valid, unique and zero-based indices. abczarr
    # 0.2.3 already refuses a mapAxis that is not a permutation when it
    # parses it, with a value error, as brainhops does after an older
    # abczarr.
    with pytest.raises(ValueError):
        _map_json(transform, _system(_ZYX), _system(_YX))


def test_project_axis_refuses_a_created_index_out_of_range() -> None:
    with pytest.raises(_map.OmeMappingError):
        _map_json(
            {
                "type": "projectAxis",
                "droppedInputs": [0],
                "createdOutputs": [3],
            },
            _system(_ZYX),
            _system(_ZYX),
        )


def test_a_dropping_projection_into_the_intrinsic_space_is_refused() -> None:
    # A projection that drops an axis has no inverse
    # (docs/design/compute-api-and-open-systems.md, section A.8).
    with pytest.raises(OmeImageError, match="dropping axes"):
        _commons(
            [
                {
                    "type": "projectAxis",
                    "droppedInputs": [0],
                    "createdOutputs": [0],
                    "input": {"name": "world"},
                    "output": {"name": "intrinsic"},
                }
            ]
        )


# ---- the dataset of a level on write ---------------------------------
# brainhops orders the axes (x, y, z), and OME stores them as (z, y, x).
_TO_ZYX = [2, 1, 0]


def test_a_diagonal_level_is_written_as_a_scale_then_a_translation() -> None:
    # A translation followed by a scale is the scale (2, 3, 4) followed by
    # the translation (2, 6, 12), in the stored order (z, y, x).
    shifted = Sequence(
        [
            Translation(translation=[1.0, 2.0, 3.0]),
            Scaling(scale=[2.0, 3.0, 4.0]),
        ]
    )
    entry = _map.to_dataset_ome(shifted, _TO_ZYX, 3)
    assert entry is not None
    scale, translation = entry["transformations"]
    assert scale == {"type": "scale", "scale": [4.0, 3.0, 2.0]}
    assert translation == {
        "type": "translation",
        "translation": [12.0, 6.0, 2.0],
    }


def test_a_scale_level_is_written_as_it_is() -> None:
    entry = _map.to_dataset_ome(Scaling(scale=[2.0, 3.0, 4.0]), _TO_ZYX, 3)
    assert entry == {"type": "scale", "scale": [4.0, 3.0, 2.0]}


def test_a_level_that_is_not_diagonal_has_no_dataset_transformation() -> None:
    matrix = np.eye(3, 4)
    matrix[0, 1] = 0.5
    assert _map.to_dataset_ome(Affine(matrix=matrix), _TO_ZYX, 3) is None


@pytest.mark.parametrize(
    "entry, allowed",
    [
        ({"type": "identity"}, True),
        ({"type": "scale", "scale": [1.0]}, True),
        (
            {
                "type": "sequence",
                "transformations": [
                    {"type": "scale"},
                    {"type": "translation"},
                ],
            },
            True,
        ),
        (
            {
                "type": "sequence",
                "transformations": [
                    {"type": "translation"},
                    {"type": "scale"},
                ],
            },
            False,
        ),
        ({"type": "translation", "translation": [1.0]}, False),
        ({"type": "affine"}, False),
    ],
)
def test_a_dataset_holds_a_scale_followed_by_a_translation(
    entry: dict, allowed: bool
) -> None:
    # ngff-spec 0.6, "Constraints": "a single scale, identity or sequence of
    # a scale followed by a translation transformation".
    assert _map.is_dataset_transform(entry) is allowed


# ---- names of coordinate systems of other groups ---------------------


@pytest.mark.parametrize(
    "name, group, expected",
    [
        ("physical", "", "physical"),
        ("physical", "labels/cells", "labels/cells//physical"),
        ("a//b", "", "//a//b"),
        ("/x", "labels", "labels///x"),
        ("a//b", "labels", "labels//a//b"),
    ],
)
def test_a_system_is_named_after_its_group(
    name: str, group: str, expected: str
) -> None:
    # The systems of the root group keep their names, and the name splits
    # back into its group path and its OME name.
    assert _ome.system_name(name, group) == expected
    assert _ome.split_system_name(expected) == (group, name)


@pytest.mark.parametrize(
    "group, path, expected",
    [
        ("", None, ""),
        ("", ".", ""),
        ("", "labels/cells/", "labels/cells"),
        ("labels", "./cells", "labels/cells"),
        ("images/a", "../b", "images/b"),
        ("images/a", "labels/cells", "images/a/labels/cells"),
    ],
)
def test_a_group_path_is_resolved_from_the_root(
    group: str, path: tx.Optional[str], expected: str
) -> None:
    assert _ome.group_path(group, path) == expected


@pytest.mark.parametrize("path", ["/labels", "labels//cells", "../up"])
def test_a_group_path_must_stay_relative_and_inside_the_root(
    path: str,
) -> None:
    with pytest.raises(ValueError):
        _ome.group_path("", path)


def test_a_transformation_to_the_same_group_by_path_is_read() -> None:
    # A path that leads back to the group of the metadata refers to its own
    # systems.
    (placement,) = _commons(
        [
            dict(
                _SCALE_TO_WORLD,
                input={"name": "intrinsic", "path": "."},
                output={"name": "world", "path": "labels/.."},
            )
        ]
    )
    assert placement.output.name == "world"


# ---- flips on write --------------------------------------------------


def test_a_negative_scale_is_written_as_an_affine() -> None:
    # A flip is never written as a negative OME scale (memo item 135).
    entry = _map.to_ome(Scaling(scale=[-1.0, 2.0, 3.0]), _TO_ZYX, 3)
    assert entry == {
        "type": "affine",
        "affine": [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ],
    }


def test_a_flipped_diagonal_affine_is_written_as_an_affine() -> None:
    matrix = np.zeros((3, 4))
    matrix[[0, 1, 2], [0, 1, 2]] = [2.0, -3.0, 4.0]
    matrix[:, 3] = [1.0, 2.0, 3.0]
    entry = _map.to_ome(Affine(matrix=matrix), _TO_ZYX, 3)
    assert entry["type"] == "affine"
    assert _map.to_dataset_ome(Affine(matrix=matrix), _TO_ZYX, 3) is None


def test_a_flip_factors_out_of_a_diagonal_geometry() -> None:
    matrix = np.zeros((3, 4))
    matrix[[0, 1, 2], [0, 1, 2]] = [-2.0, 3.0, 4.0]
    matrix[:, 3] = [1.0, 2.0, 3.0]
    geometry = Affine(matrix=matrix)

    signs = _map.diagonal_signs(geometry, 3)
    rest = _map.unflipped(geometry, signs, 3)

    np.testing.assert_array_equal(signs, [-1.0, 1.0, 1.0])
    np.testing.assert_allclose(np.diag(np.asarray(rest.matrix)), [2, 3, 4])
    linear = np.asarray(_map.flip(signs).matrix)
    composed = linear[:, :3] @ np.asarray(rest.matrix)
    np.testing.assert_allclose(composed, matrix)
