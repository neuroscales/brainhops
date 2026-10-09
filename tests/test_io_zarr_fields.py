"""Tests for the OME-Zarr field format.

The placement is anisotropic and rotated, so that the voxel-to-voxel
normalisation is visible.
"""

from pathlib import Path

import numpy as np
import pytest

from brainhops.datamodel import transformations as X
from brainhops.datamodel._transformations.multiscale import _as_affine
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Scaling,
)

# The module imports abczarr itself, so the skip must come first.
abczarr = pytest.importorskip("abczarr")

from brainhops.errors import AxisError  # noqa: E402
from brainhops.io.transformations.zarr import (  # noqa: E402
    OmeFieldError,
    OmeZarrField,
)
from brainhops.io.transformations.zarr._axes import _to_axis  # noqa: E402


def _write_field_store(
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
    path = _write_field_store(
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
    assert isinstance(parts[-1], CoordinatesField)
    # The metadata is kept exactly as read.
    assert reader.to_ome() is reader.ome


def test_from_node_reads_a_scaled_displacement_field(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    path = _write_field_store(
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
    assert _as_affine(parts[0]) is not None
    assert _as_affine(parts[2]) is not None


def test_from_node_refuses_a_node_without_ome(tmp_path: Path) -> None:
    path = str(tmp_path / "plain.zarr")
    group = abczarr.open_group(path, mode="w")
    group.create_array("0", data=np.zeros((4, 4, 2), "float32"))
    node = abczarr.open_group(path, mode="r")
    with pytest.raises(OmeFieldError):
        OmeZarrField.from_node(node)


def test_field_is_a_registered_file_format() -> None:
    from brainhops.io.transformations.base import (
        WritableFileBasedTransformation,
    )

    assert issubclass(OmeZarrField, WritableFileBasedTransformation)


def test_load_discovers_the_field_format(tmp_path: Path) -> None:
    import brainhops.io.transformations as transformations

    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 1.0
    path = _write_field_store(
        tmp_path,
        field,
        [("z", "space"), ("y", "space"), ("x", "space"), ("c", "coordinate")],
        {"type": "identity"},
        name="discover.zarr",
    )

    assert transformations.sniff(path) is OmeZarrField
    assert isinstance(transformations.load(path), OmeZarrField)


def test_coordinate_field_round_trips_through_a_store(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 7.0
    path = _write_field_store(
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
    assert isinstance(back.transformations[-1], CoordinatesField)
    np.testing.assert_allclose(
        np.asarray(back.raw_levels[0]), np.asarray(read.raw_levels[0])
    )


def test_displacement_field_round_trips_through_a_store(
    tmp_path: Path,
) -> None:
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 1] = 3.0
    path = _write_field_store(
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
    assert isinstance(parts[1], CoordinatesField)
    # A coordinates field stores the raw world coordinates unchanged.
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


def test_reader_selects_a_level_by_target_resolution() -> None:
    reader, placement, _, _, _, _ = _displacement_reader()
    # The finest voxel size is (2, 0.5); a doubled target selects level 1.
    fine_target = placement
    coarse_target = (placement @ Scaling(scale=[2.0, 2.0])).compute()
    assert reader._nearest_level(fine_target) == 0
    assert reader._nearest_level(coarse_target) == 1


def test_to_axis_maps_each_ome_type() -> None:
    assert isinstance(_to_axis({"type": "space"}), SpaceAxis)
    assert isinstance(_to_axis({"type": "time"}), TimeAxis)
    assert isinstance(_to_axis({"type": "channel"}), ChannelAxis)
    assert isinstance(_to_axis({"type": "displacement"}), DisplacementAxis)
    assert isinstance(_to_axis({"type": "coordinate"}), CoordinateAxis)
    # An unrecognised type falls back to a plain axis carrying the type.
    other = _to_axis({"type": "array", "name": "c"})
    assert type(other) is Axis
    assert other.type == "array"
    assert other.name == "c"
