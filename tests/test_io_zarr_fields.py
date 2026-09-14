"""Tests for the OME-Zarr field reader.

The reader is exercised from in-memory arrays and metadata rather than a
real OME-Zarr store. The placement is anisotropic and rotated so that the
voxel-to-voxel normalization is visible.
"""

import numpy as np
import pytest

from brainhops.datamodel import transformations as X
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    MultiscaleCoordinatesField,
    MultiscaleDisplacementField,
    OmePlacementError,
    Scaling,
)
from brainhops.io.transformations.zarr import OmeZarrField


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
        placement=placement,
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
        axes=[Axis(type="displacement"), Axis(type="displacement")],
        ome_metadata=metadata,
    )
    return reader, placement, linear, fine, coarse, metadata


def test_reader_builds_the_sandwich() -> None:
    reader, placement, _, _, _, _ = _displacement_reader()
    assert isinstance(reader, X.Sequence)
    parts = reader.transformations
    assert len(parts) == 3
    assert isinstance(parts[1], MultiscaleDisplacementField)
    assert parts[2] is placement


def test_displacement_field_is_normalized_voxel_to_voxel() -> None:
    reader, _, linear, fine, _, _ = _displacement_reader()
    field = reader.transformations[1]
    expected = fine @ np.linalg.inv(linear).T
    np.testing.assert_allclose(np.asarray(field.field), expected)


def test_reader_kind_is_read_from_axes() -> None:
    reader, _, _, _, _, _ = _displacement_reader()
    assert reader.kind == "displacement"


def test_coordinate_reader_normalizes_by_the_inverse_placement() -> None:
    placement, _ = _rotated_anisotropic_affine()
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(6, 5, 2))
    reader = OmeZarrField(
        raw_levels=[raw],
        placement=placement,
        axes=[Axis(type="space"), Axis(type="space")],
    )
    assert reader.kind == "coordinate"
    field = reader.transformations[1]
    assert isinstance(field, MultiscaleCoordinatesField)
    inverse = placement.inverse().matrix
    expected = raw @ inverse[:, :-1].T + inverse[:, -1]
    np.testing.assert_allclose(np.asarray(field.field), expected)


def test_untouched_read_re_emits_metadata_unchanged() -> None:
    reader, _, _, _, _, metadata = _displacement_reader()
    # The reader returns the identical metadata object, so a write after
    # an untouched read re-emits the OME metadata byte for byte.
    assert reader.to_ome_metadata() is metadata


def test_reader_refuses_mixed_axes() -> None:
    placement, _ = _rotated_anisotropic_affine()
    reader = OmeZarrField(
        raw_levels=[np.zeros((4, 4, 2))],
        placement=placement,
        axes=[Axis(type="displacement"), Axis(type="space")],
    )
    with pytest.raises(OmePlacementError):
        _ = reader.kind


def test_reader_refuses_nonlinear_placed_displacement() -> None:
    reader = OmeZarrField(
        raw_levels=[np.zeros((4, 4, 2))],
        placement=X.DisplacementField(field=np.zeros((4, 4, 2))),
        axes=[Axis(type="displacement"), Axis(type="displacement")],
    )
    with pytest.raises(OmePlacementError):
        _ = reader.transformations
