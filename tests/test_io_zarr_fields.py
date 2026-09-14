"""Tests for the OME-Zarr field reader.

The reader is exercised from in-memory arrays and metadata rather than a
real OME-Zarr store. The placement is anisotropic and rotated so that the
voxel-to-voxel normalization is visible.
"""

import numpy as np
import pytest

from brainhops.datamodel import transformations as X
from brainhops.datamodel.axes import (
    Axis,
    AxisError,
    ChannelAxis,
    CoordinateAxis,
    DisplacementAxis,
    SpatialAxis,
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
from brainhops.io.transformations.zarr import OmeFieldError, OmeZarrField
from brainhops.io.transformations.zarr._axes import _to_axis


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
    assert reader.nlevels == 2


def test_finest_displacement_level_is_the_three_element_sandwich() -> None:
    reader, placement, _, _, _, _ = _displacement_reader()
    parts = reader.transformations
    assert len(parts) == 3
    assert isinstance(parts[0], Affine)
    assert isinstance(parts[1], DisplacementField)
    assert isinstance(parts[2], Affine)
    # The trailing voxel-to-world affine is the placement itself.
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
    # The coordinate field stores the raw world coordinates unchanged.
    assert parts[1].field is raw


def test_untouched_read_re_emits_metadata_unchanged() -> None:
    reader, _, _, _, _, metadata = _displacement_reader()
    # The reader returns the identical metadata object, so a write after
    # an untouched read re-emits the OME metadata byte for byte.
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
    # The finest level's voxel size is (2, 0.5); the coarse level's grid
    # is twice as large, so a doubled target selects the coarse level.
    fine_target = placement
    coarse_target = (placement @ Scaling(scale=[2.0, 2.0])).compute()
    assert reader._nearest_level(fine_target) == 0
    assert reader._nearest_level(coarse_target) == 1


def test_to_axis_maps_each_ome_type() -> None:
    assert isinstance(_to_axis({"type": "space"}), SpatialAxis)
    assert isinstance(_to_axis({"type": "time"}), TimeAxis)
    assert isinstance(_to_axis({"type": "channel"}), ChannelAxis)
    assert isinstance(_to_axis({"type": "displacement"}), DisplacementAxis)
    assert isinstance(_to_axis({"type": "coordinate"}), CoordinateAxis)
    # An unrecognized type falls back to a plain axis carrying the type.
    other = _to_axis({"type": "array", "name": "c"})
    assert type(other) is Axis
    assert other.type == "array"
    assert other.name == "c"
