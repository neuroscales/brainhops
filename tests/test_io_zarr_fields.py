"""Tests for the OME-Zarr field format.

The placement is anisotropic and rotated, so that the voxel-to-voxel
normalisation is visible.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel import transformations as X
from brainhops.datamodel._transformations.compute.multiscale import (
    nearest_level,
)
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.systems import RASmm
from brainhops.datamodel.transformations import (
    Affine,
    CoordinateField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Scaling,
)
from brainhops.io.base.parsers import FileWriter

# The module imports abczarr itself, so the skip must come first.
abczarr = pytest.importorskip("abczarr")

from brainhops.errors import AxisError  # noqa: E402
from brainhops.io.transformations.zarr import (  # noqa: E402
    OmeFieldError,
    OmeZarrField,
)
from brainhops.io.transformations.zarr._axes import to_axis  # noqa: E402


def write_field_store(
    tmp_path: Path,
    field: np.ndarray,
    axes: list,
    transform: dict,
    name: str = "field.zarr",
) -> str:
    """Write a standalone OME-Zarr field node whose typed metadata names its
    axes and places its one level.
    """
    from abczarr.ome import v0_6 as v6

    path = str(tmp_path / name)
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=field)
    group.ome = v6.OME.from_json(
        {
            "version": "0.6",
            "multiscales": [
                {
                    "coordinateSystems": [
                        {
                            "name": "field",
                            "axes": [
                                {"name": name, "type": type_}
                                for name, type_ in axes
                            ],
                        }
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                dict(
                                    transform,
                                    input={"path": "0"},
                                    output={"name": "field"},
                                )
                            ],
                        }
                    ],
                }
            ],
        }
    )
    return path


def test_from_node_reads_a_coordinate_field(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 1.0
    path = write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
    )
    node = abczarr.open_group(path, mode="r")

    reader = OmeZarrField.from_node(node)
    assert isinstance(reader, MultiscaleField)
    assert reader.nscales == 1
    assert [axis.type for axis in reader.axes] == [
        "space",
        "space",
        "space",
        "coordinate",
    ]
    # One level builds as a two-element coordinate sandwich.
    parts = reader.transformations
    assert isinstance(parts[-1], CoordinateField)
    # The metadata is kept exactly as read.
    assert reader.to_ome() is reader.ome


def test_from_node_reads_a_scaled_displacement_field(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    path = write_field_store(
        tmp_path,
        field,
        [
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
            ("d", "displacement"),
        ],
        {"type": "scale", "scale": [2.0, 3.0, 4.0]},
    )
    node = abczarr.open_group(path, mode="r")

    reader = OmeZarrField.from_node(node)
    assert isinstance(reader, MultiscaleField)
    # A scale placement gives a three-element displacement sandwich.
    parts = reader.transformations
    assert len(parts) == 3
    assert isinstance(parts[1], DisplacementField)
    # The outer parts reduce to affines, for inversion and vector rotation.
    assert parts[0].to_affine(on_error=None) is not None
    assert parts[2].to_affine(on_error=None) is not None


def test_from_node_refuses_a_node_without_ome(tmp_path: Path) -> None:
    path = str(tmp_path / "plain.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.zeros((4, 4, 2), "float32"))
    node = abczarr.open_group(path, mode="r")
    with pytest.raises(OmeFieldError):
        OmeZarrField.from_node(node)


def test_field_is_a_registered_file_format() -> None:
    from brainhops.io.transformations.base import (
        TransformationFormat,
    )

    assert issubclass(OmeZarrField, FileWriter)
    assert OmeZarrField in TransformationFormat._REGISTRY


def test_load_discovers_the_field_format(tmp_path: Path) -> None:
    import brainhops.io.transformations as transformations

    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 1.0
    path = write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
        name="discover.zarr",
    )

    assert transformations.sniff(path) is OmeZarrField
    assert isinstance(transformations.load(path), OmeZarrField)


def test_from_filename_reads_a_field_store(tmp_path: Path) -> None:
    import brainhops.io.transformations as transformations

    # A store is a directory, so `from_filename` must not open it as a
    # file. The dispatcher calls `from_filename` by name.
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 5.0
    path = write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
        name="by_name.zarr",
    )
    expected = OmeZarrField.load(path)

    assert OmeZarrField.sniff_filename(path) > 0
    for back in (
        OmeZarrField.from_filename(path),
        OmeZarrField.from_filename(Path(path)),
        transformations.TransformationFormat.from_filename(path),
    ):
        assert type(back) is OmeZarrField
        assert isinstance(back.transformations[-1], CoordinateField)
        np.testing.assert_allclose(
            np.asarray(back.raw_levels[0]),
            np.asarray(expected.raw_levels[0]),
        )


def test_coordinate_field_round_trips_through_a_store(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 7.0
    path = write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
        name="coord_src.zarr",
    )

    # Read, write to a new path and read again.
    read = OmeZarrField.from_store(path)
    out = str(tmp_path / "coord_dst.zarr")
    read.to_store(out)
    back = OmeZarrField.from_store(out)

    assert isinstance(back, MultiscaleField)
    assert isinstance(back.transformations[-1], CoordinateField)
    np.testing.assert_allclose(
        np.asarray(back.raw_levels[0]), np.asarray(read.raw_levels[0])
    )


def test_displacement_field_round_trips_through_a_store(
    tmp_path: Path,
) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 1] = 3.0
    path = write_field_store(
        tmp_path,
        field,
        [
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
            ("d", "displacement"),
        ],
        {"type": "scale", "scale": [2.0, 3.0, 4.0]},
        name="disp_src.zarr",
    )

    read = OmeZarrField.from_store(path)
    out = str(tmp_path / "disp_dst.zarr")
    read.to_store(out)
    back = OmeZarrField.from_store(out)

    assert isinstance(back, MultiscaleField)
    parts = back.transformations
    assert len(parts) == 3
    assert isinstance(parts[1], DisplacementField)
    np.testing.assert_allclose(
        np.asarray(back.raw_levels[0]), np.asarray(read.raw_levels[0])
    )


def _rotated_anisotropic_affine() -> tuple:
    theta = 0.35
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    linear = rot @ np.diag([2.0, 0.5])
    matrix = np.concatenate([linear, np.array([[5.0], [7.0]])], axis=1)
    return Affine(matrix=matrix), linear


def _displacement_reader() -> tuple:
    placement, linear = _rotated_anisotropic_affine()
    rng = np.random.default_rng(7)
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    metadata = {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}
    reader = OmeZarrField(
        raw_levels=[fine, coarse],
        voxel2world=placement,
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
        axes=[
            Axis(type="space"),
            Axis(type="space"),
            Axis(type="displacement"),
        ],
        ome=metadata,
    )
    return reader, placement, linear, fine, coarse, metadata


def test_reader_is_a_multiscale_field() -> None:
    reader, _, _, _, _, _ = _displacement_reader()
    assert isinstance(reader, MultiscaleField)
    assert isinstance(reader, X.Sequence)
    assert reader.nscales == 2


def test_finest_displacement_level_is_the_three_element_sandwich() -> None:
    reader, placement, _, _, _, _ = _displacement_reader()
    parts = reader.transformations
    assert len(parts) == 3
    assert isinstance(parts[0], Affine)
    assert isinstance(parts[1], DisplacementField)
    assert isinstance(parts[2], Affine)
    # The trailing vox-to-world affine is the placement itself.
    assert parts[2] is placement


def test_displacement_field_is_normalized_voxel_to_voxel() -> None:
    reader, _, linear, fine, _, _ = _displacement_reader()
    field = reader.transformations[1]
    expected = fine @ np.linalg.inv(linear).T
    np.testing.assert_allclose(np.asarray(field.field), expected)


def test_coordinate_level_stores_the_raw_array_without_a_copy() -> None:
    placement, _ = _rotated_anisotropic_affine()
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(6, 5, 2))
    reader = OmeZarrField(
        raw_levels=[raw],
        voxel2world=placement,
        axes=[
            Axis(type="space"),
            Axis(type="space"),
            Axis(type="coordinate"),
        ],
    )
    parts = reader.transformations
    assert len(parts) == 2
    assert isinstance(parts[0], Affine)
    assert isinstance(parts[1], CoordinateField)
    # A coordinate field stores the raw world coordinates unchanged.
    assert parts[1].field is raw


def test_untouched_read_re_emits_metadata_unchanged() -> None:
    reader, _, _, _, _, metadata = _displacement_reader()
    # An untouched read returns the very metadata object it read.
    assert reader.to_ome() is metadata


def test_reader_refuses_mixed_axes() -> None:
    placement, _ = _rotated_anisotropic_affine()
    reader = OmeZarrField(
        raw_levels=[np.zeros((4, 4, 2))],
        voxel2world=placement,
        axes=[
            Axis(type="space"),
            Axis(type="displacement"),
            Axis(type="coordinate"),
        ],
    )
    with pytest.raises(AxisError):
        _ = reader.transformations


def test_reader_refuses_nonlinear_placed_displacement() -> None:
    reader = OmeZarrField(
        raw_levels=[np.zeros((4, 4, 2))],
        voxel2world=DisplacementField(field=np.zeros((4, 4, 2))),
        axes=[
            Axis(type="space"),
            Axis(type="space"),
            Axis(type="displacement"),
        ],
    )
    with pytest.raises(OmeFieldError):
        _ = reader.transformations


def test_reader_refuses_a_level_placed_by_a_nonlinear_map() -> None:
    # The placement of the finest level is affine, but the level
    # transform is not, so the displacements of the level cannot be
    # rescaled to voxel units.
    reader = OmeZarrField(
        raw_levels=[np.zeros((4, 4, 2))],
        level_transforms=[DisplacementField(field=np.zeros((4, 4, 2)))],
        axes=[
            Axis(type="space"),
            Axis(type="space"),
            Axis(type="displacement"),
        ],
    )
    with pytest.raises(OmeFieldError, match="A level"):
        _ = reader.transformations


def test_reader_selects_a_level_by_target_resolution() -> None:
    reader, placement, _, _, _, _ = _displacement_reader()
    # The finest voxel size is (2, 0.5); a doubled target selects level 1.
    fine_target = placement
    coarse_target = (placement @ Scaling(scale=[2.0, 2.0])).compute()
    assert nearest_level(reader, fine_target) == 0
    assert nearest_level(reader, coarse_target) == 1


def test_to_axis_maps_each_ome_type() -> None:
    assert isinstance(to_axis({"type": "space"}), SpaceAxis)
    assert isinstance(to_axis({"type": "time"}), TimeAxis)
    assert isinstance(to_axis({"type": "channel"}), ChannelAxis)
    assert isinstance(to_axis({"type": "displacement"}), DisplacementAxis)
    assert isinstance(to_axis({"type": "coordinate"}), CoordinateAxis)
    # An unrecognised type falls back to a plain axis carrying the type.
    other = to_axis({"type": "array", "name": "c"})
    assert type(other) is Axis
    assert other.type == "array"
    assert other.name == "c"


def test_the_generic_load_reads_a_field_rather_than_an_image(
    tmp_path: Path,
) -> None:
    import brainhops.io as io

    path = write_field_store(
        tmp_path,
        np.zeros((4, 5, 6, 3), dtype="float32"),
        [
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
            ("d", "displacement"),
        ],
        {"type": "identity"},
        name="generic.zarr",
    )
    assert isinstance(io.load(path), OmeZarrField)


def _placement_without_matrix() -> X.Transformation:
    # The lazy inverse of an affine whose matrix is not set converts to an
    # affine that has no matrix either.
    return X.Inverse(forward=Affine())


def test_a_placement_without_a_matrix_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from brainhops.io.transformations.zarr import _xforms

    # Such a placement is taken for the identity, so the test makes it a
    # placement that the check reaches.
    monkeypatch.setattr(_xforms, "is_identity", lambda t: False)
    field = SimpleNamespace(voxel2world=_placement_without_matrix())
    with pytest.raises(OmeFieldError):
        OmeZarrField._check_affine_voxel2world(field)


def test_a_placement_without_a_matrix_becomes_the_identity() -> None:
    from brainhops.io.transformations.zarr._xforms import (
        affine_or_identity,
    )

    affine = affine_or_identity(_placement_without_matrix(), 3)
    np.testing.assert_array_equal(np.asarray(affine.matrix), np.eye(4)[:3])


# ---- checking the metadata before it is written ----------------------


def test_writer_writes_the_metadata_of_a_converted_field(
    tmp_path: Path,
) -> None:
    field = DisplacementField(field=np.zeros((2, 3, 4, 3)))
    path = str(tmp_path / "warp.ome.zarr")

    field.to(OmeZarrField).save(path)

    assert isinstance(OmeZarrField.from_store(path), OmeZarrField)


def test_writer_refuses_metadata_that_breaks_a_rule(tmp_path: Path) -> None:
    from abczarr.ome import v0_6
    from bagof.magic import replace

    from brainhops.errors import WriterError

    field = DisplacementField(field=np.zeros((2, 3, 4, 3))).to(OmeZarrField)
    document = field.to_ome().to_json()
    dataset = document["multiscales"][0]["datasets"][0]
    dataset["coordinateTransformations"][0]["output"] = {"name": "nowhere"}
    broken = replace(field, ome=v0_6.OME.from_json(document))

    with pytest.raises(WriterError, match="names neither"):
        broken.to_store(str(tmp_path / "warp.ome.zarr"))


# ---- oblique fields on write (#447) ------------------------------------


_FIELD_AXES = [
    {"name": "d", "type": "displacement", "discrete": True},
    {"name": "z", "type": "space", "unit": "millimeter"},
    {"name": "y", "type": "space", "unit": "millimeter"},
    {"name": "x", "type": "space", "unit": "millimeter"},
]
"""The stored axes of the fields written here, in the order of OME-NGFF."""


def _oblique_placement() -> np.ndarray:
    # The placement of the finest grid, over the stored axes (z, y, x):
    # a rotation and anisotropic voxels.
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    linear = rotation @ np.diag([3.0, 1.0, 1.0])
    return np.concatenate([linear, [[4.0], [-5.0], [6.0]]], axis=1)


def _downsampling(factors: tx.Sequence[int], centred: bool = True) -> Affine:
    # The map from the grid of a level downsampled by `factors` to the
    # finest grid, by the rule of OME-NGFF 0.6, or to the first node of
    # each window when the level is not centred.
    factors = np.asarray(factors, dtype=float)
    shift = (factors - 1) / 2 if centred else np.zeros_like(factors)
    return Affine(
        matrix=np.concatenate([np.diag(factors), shift[:, None]], axis=1)
    )


def _oblique_field() -> OmeZarrField:
    rng = np.random.default_rng(11)
    fine = rng.normal(size=(3, 6, 10, 8))
    coarse = rng.normal(size=(3, 6, 5, 4))
    return OmeZarrField(
        raw_levels=[fine, coarse],
        voxel2world=Affine(matrix=_oblique_placement()),
        level_transforms=[Identity(), _downsampling([1, 2, 2])],
        axes=[to_axis(axis) for axis in _FIELD_AXES],
    )


def _field_multiscale(path: str) -> dict:
    import json

    with open(Path(path) / "zarr.json") as stream:
        return json.load(stream)["attributes"]["ome"]["multiscales"][0]


def _check_lean_field_datasets(path: str, common: bool = True) -> None:
    # Each dataset holds a positive scale, or a positive scale followed
    # by a translation, and anything else is in the common
    # transformations.
    multiscale = _field_multiscale(path)
    for dataset in multiscale["datasets"]:
        (entry,) = dataset["coordinateTransformations"]
        if entry["type"] == "sequence":
            steps = entry["transformations"]
            assert [step["type"] for step in steps] == ["scale", "translation"]
            scale = steps[0]["scale"]
        else:
            assert entry["type"] == "scale"
            scale = entry["scale"]
        assert all(value > 0 for value in scale)
    assert bool(multiscale.get("coordinateTransformations")) == common


def _level_matrix(field: OmeZarrField, index: int) -> np.ndarray:
    # The placement of a level, from its grid to the world.
    placement = field.voxel2world @ field.level_transforms[index]
    return np.asarray(placement.compute().to(Affine).matrix)


def test_an_oblique_field_is_split_on_write(tmp_path: Path) -> None:
    field = _oblique_field()
    path = str(tmp_path / "oblique.zarr")

    field.to_store(path)
    back = OmeZarrField.from_store(path)

    _check_lean_field_datasets(path)
    assert back.nscales == 2
    for index in range(2):
        np.testing.assert_array_equal(
            np.asarray(back.raw_levels[index]), field.raw_levels[index]
        )
        np.testing.assert_allclose(
            _level_matrix(back, index), _level_matrix(field, index), atol=1e-12
        )


def test_the_vectors_of_an_oblique_field_keep_their_world_units(
    tmp_path: Path,
) -> None:
    # The voxel displacements of a level are the world displacements
    # mapped by the inverse of the linear part of its placement, which
    # the split leaves unchanged.
    field = _oblique_field()
    path = str(tmp_path / "oblique.zarr")

    field.to_store(path)
    back = OmeZarrField.from_store(path)

    for index in range(2):
        expected = field.scales[index].transformations[1].field
        actual = back.scales[index].transformations[1].field
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), atol=1e-12
        )


def test_a_field_whose_levels_turn_differently_is_refused(
    tmp_path: Path,
) -> None:
    from brainhops.errors import UnrepresentableTransformationError

    field = _oblique_field()
    sheared = np.eye(3, 4)
    sheared[0, 1] = 0.1
    field = OmeZarrField(
        raw_levels=field.raw_levels,
        voxel2world=field.voxel2world,
        level_transforms=[Identity(), Affine(matrix=sheared)],
        axes=field.axes,
    )

    with pytest.raises(UnrepresentableTransformationError):
        field.to_store(str(tmp_path / "refused.zarr"))


# ---- fields built from one level (#625) ----------------------------------


def _grid_coordinates(matrix: np.ndarray, shape: tx.Sequence[int]) -> tx.Any:
    # The world coordinates of the voxels of a grid placed by `matrix`,
    # with the axis of the coordinates first, as OME-NGFF 0.6 stores
    # them.
    grid = np.stack(
        np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1
    )
    world = grid @ matrix[:, :3].T + matrix[:, 3]
    return np.moveaxis(world, -1, 0)


def _single_field(kind: str, values: tx.Any) -> OmeZarrField:
    axes = [dict(_FIELD_AXES[0], type=kind)] + _FIELD_AXES[1:]
    return OmeZarrField(
        raw_levels=[values],
        voxel2world=Affine(matrix=_oblique_placement()),
        level_transforms=[Identity()],
        axes=[to_axis(axis) for axis in axes],
    )


@pytest.mark.parametrize("mode", ["mean", "binomial", "gaussian", "nearest"])
def test_with_scales_builds_a_field_pyramid(tmp_path: Path, mode: str) -> None:
    # A field of world coordinates is only averaged, and the average of
    # a linear field stays exact, so its values agree with the placement
    # of each level.
    shape = (12, 40, 36)
    values = _grid_coordinates(_oblique_placement(), shape)
    field = _single_field("coordinate", values).with_scales(
        min_size=5, mode=mode
    )
    path = str(tmp_path / "built.zarr")

    field.to_store(path)
    back = OmeZarrField.from_store(path)

    # The voxels of 3 x 1 x 1 make z wait for two steps. An axis is
    # halved while it has more than 5 voxels, and the alignment "exact"
    # drops the last voxel of an odd axis.
    shapes = [tuple(level.shape) for level in back.raw_levels]
    assert shapes == [
        (3, 12, 40, 36),
        (3, 12, 20, 18),
        (3, 12, 10, 9),
        (3, 6, 5, 4),
        (3, 3, 5, 4),
    ]
    _check_lean_field_datasets(path)
    cumulative = ([1, 1, 1], [1, 2, 2], [1, 4, 4], [2, 8, 8], [4, 8, 8])
    for index, factors in enumerate(cumulative):
        level = np.asarray(back.raw_levels[index])
        np.testing.assert_allclose(
            level, np.asarray(field.raw_levels[index]), atol=1e-12
        )
        expected = Affine(matrix=_oblique_placement()) @ _downsampling(factors)
        expected = np.asarray(expected.compute().to(Affine).matrix)
        matrix = _level_matrix(back, index)
        np.testing.assert_allclose(matrix, expected, atol=1e-12)
        if mode == "nearest":
            # The mode "nearest" keeps the node `(f - 1) // 2` of each
            # window, which is half a voxel before its centre when `f`
            # is even.
            first = [(f - 1) // 2 for f in factors]
            nodes = tuple(
                slice(start, start + f * n, f)
                for start, f, n in zip(first, factors, level.shape[1:])
            )
            np.testing.assert_array_equal(
                level, values[(slice(None),) + nodes]
            )
        else:
            np.testing.assert_allclose(
                level, _grid_coordinates(matrix, level.shape[1:]), atol=1e-10
            )


@pytest.mark.parametrize(
    "align, count, scale, shift",
    [
        ("exact", 13, 3.0, 1.0),
        ("edges", 14, 40 / 14, (40 / 14 - 1) / 2),
        ("centers", 14, 3.0, 0.0),
    ],
)
def test_each_alignment_places_the_levels_of_a_field(
    tmp_path: Path, align: str, count: int, scale: float, shift: float
) -> None:
    values = _grid_coordinates(_oblique_placement(), (12, 40, 36))
    field = _single_field("coordinate", values).with_scales(
        [{"y": 3}], align=align
    )
    path = str(tmp_path / "aligned.zarr")

    field.to_store(path)
    back = OmeZarrField.from_store(path)

    level = np.asarray(back.raw_levels[1])
    assert level.shape == (3, 12, count, 36)
    _check_lean_field_datasets(path)
    # The voxel `i` of the level lies at the voxel `scale * i + shift`
    # of the finest grid along y, the second stored axis.
    downsampling = Affine(
        matrix=np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, scale, 0.0, shift], [0, 0, 1.0, 0]]
        )
    )
    expected = Affine(matrix=_oblique_placement()) @ downsampling
    expected = np.asarray(expected.compute().to(Affine).matrix)
    matrix = _level_matrix(back, 1)
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    # The edges are extended linearly, so a linear field stays exact.
    np.testing.assert_allclose(
        level, _grid_coordinates(matrix, level.shape[1:]), atol=1e-10
    )


def test_the_displacements_of_a_built_field_keep_their_world_units(
    tmp_path: Path,
) -> None:
    # A uniform displacement keeps its value in world units on every
    # level, although the voxels of the coarser levels are larger.
    values = np.empty((3, 12, 40, 36))
    values[:] = np.array([1.5, -2.0, 0.25])[:, None, None, None]
    field = _single_field("displacement", values).with_scales(min_size=5)
    path = str(tmp_path / "uniform.zarr")

    field.to_store(path)
    back = OmeZarrField.from_store(path)

    assert back.nscales == 5
    for level in back.raw_levels:
        np.testing.assert_allclose(
            np.asarray(level),
            np.broadcast_to(values[:, :1, :1, :1], level.shape),
        )


def test_with_scales_keeps_a_dask_field_lazy(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    rng = np.random.default_rng(13)
    values = rng.normal(size=(3, 12, 40, 36))
    lazy = da.from_array(values, chunks=(3, 6, 16, 16))

    built = _single_field("displacement", lazy).with_scales(min_size=5)
    expected = _single_field("displacement", values).with_scales(min_size=5)

    assert all(isinstance(level, da.Array) for level in built.raw_levels)
    path = str(tmp_path / "lazy.zarr")
    built.to_store(path)
    back = OmeZarrField.from_store(path)
    for index in range(back.nscales):
        np.testing.assert_allclose(
            np.asarray(back.raw_levels[index]),
            np.asarray(expected.raw_levels[index]),
            atol=1e-12,
        )


def test_with_scales_builds_from_the_finest_level_of_a_store(
    tmp_path: Path,
) -> None:
    # A field read from a store, which shares a rotation between its
    # levels, is rebuilt from its finest level, and keeps its placement.
    path = str(tmp_path / "oblique.zarr")
    _oblique_field().to_store(path)
    read = OmeZarrField.from_store(path)

    built = read.with_scales(3, min_size=2)
    out = str(tmp_path / "rebuilt.zarr")
    built.to_store(out)
    back = OmeZarrField.from_store(out)

    assert back.nscales == 3
    _check_lean_field_datasets(out)
    np.testing.assert_allclose(
        _level_matrix(back, 0), _oblique_placement(), atol=1e-12
    )


def test_a_field_converts_to_an_ome_pyramid(tmp_path: Path) -> None:
    import brainhops.io as io

    grid = Affine(matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3], output=RASmm())
    values = np.empty((64, 64, 64, 3))
    values[:] = [1.0, 0.0, -1.0]
    t = grid @ DisplacementField(field=values) @ grid.inverse()

    counted = t.to(OmeZarrField, nscales=2)
    path = str(tmp_path / "saved.ome.zarr")
    io.save(t, path, nscales=[2, 4])
    saved = OmeZarrField.from_store(path)

    assert counted.nscales == 2
    assert saved.nscales == 3
    # The grid runs along the world axes, so no placement is shared.
    _check_lean_field_datasets(path, common=False)
    # The vectors stay in millimetres of the world on every level.
    for level in saved.raw_levels:
        np.testing.assert_allclose(np.asarray(level)[:, 2, 2, 2], [2, 0, -2])


def test_save_builds_the_scales_of_a_field_by_default(tmp_path: Path) -> None:
    import brainhops.io as io

    grid = Affine(matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3], output=RASmm())
    values = np.zeros((70, 70, 70, 3))
    t = grid @ DisplacementField(field=values) @ grid.inverse()

    io.save(t, str(tmp_path / "full.ome.zarr"))
    io.save(t, str(tmp_path / "one.ome.zarr"), nscales=1)

    full = OmeZarrField.from_store(str(tmp_path / "full.ome.zarr"))
    one = OmeZarrField.from_store(str(tmp_path / "one.ome.zarr"))
    assert [tuple(level.shape) for level in full.raw_levels] == [
        (3, 70, 70, 70),
        (3, 35, 35, 35),
        (3, 17, 17, 17),
    ]
    assert one.nscales == 1


def test_every_level_of_a_field_has_the_chunks_of_the_finest(
    tmp_path: Path,
) -> None:
    da = pytest.importorskip("dask.array")
    values = da.zeros((3, 12, 40, 36), chunks=(3, 6, 16, 16))
    field = _single_field("displacement", values).with_scales(min_size=5)
    path = str(tmp_path / "chunks.zarr")

    field.to_store(path)

    group = abczarr.open(path, mode="r")
    assert tuple(group["0"].chunks) == (3, 6, 16, 16)
    for index in range(1, 5):
        array = group[str(index)]
        expected = [min(c, n) for c, n in zip((3, 6, 16, 16), array.shape)]
        assert list(array.chunks) == expected


def test_a_field_takes_a_list_of_scales(tmp_path: Path) -> None:
    path = str(tmp_path / "oblique.zarr")
    _oblique_field().to_store(path)
    read = OmeZarrField.from_store(path)

    adopted = read.with_scales(list(read.scales))
    out = str(tmp_path / "adopted.zarr")
    adopted.to_store(out)
    back = OmeZarrField.from_store(out)

    assert back.nscales == 2
    for index in range(2):
        np.testing.assert_allclose(
            np.asarray(back.raw_levels[index]),
            np.asarray(read.raw_levels[index]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            _level_matrix(back, index), _level_matrix(read, index), atol=1e-12
        )


def test_the_vectors_of_an_ome_field_are_never_downsampled() -> None:
    field = _single_field("displacement", np.zeros((3, 12, 40, 36)))

    with pytest.raises(ValueError, match="vectors"):
        field.with_scales([{"d": 2}])
    with pytest.raises(ValueError, match="field"):
        field.with_scales(2, mode="max")
