"""Tests for the multiscale field.

A `MultiscaleField` carries a pyramid of resolution levels. Each level is
a sequence that maps the multiscale's input space to its output space,
sampled on that level's grid. A coordinate level is a two-element
sequence, and a displacement level a three-element sequence.

The placement used throughout is anisotropic and rotated on purpose. The
voxel-to-voxel normalization that a displacement level applies rescales
the stored vectors by the inverse of the linear part of the placement. On
a one-millimetre isotropic grid that rescaling is the identity, so an
isotropic test would pass even if the normalization were wrong. An
anisotropic and rotated placement makes the rescaling visible.
"""

import numpy as np
import pytest

from brainhops.datamodel.transformations import (
    Affine,
    CoordinatesField,
    DisplacementField,
    Identity,
    MultiscaleField,
    Scaling,
    Sequence,
    _at_resolution,
)


def _rotated_anisotropic_affine(sx: float, sy: float) -> tuple:
    # A 2D placement that scales the axes by (sx, sy), rotates them, and
    # shifts the origin. Its linear part is not a scalar multiple of the
    # identity, so the inverse-linear normalization is not invisible.
    theta = 0.37
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    linear = rot @ np.diag([sx, sy])
    matrix = np.concatenate([linear, np.array([[5.0], [-3.0]])], axis=1)
    return Affine(matrix=matrix), linear


def _world_grid(
    shape: tuple, linear: np.ndarray, affine: Affine
) -> np.ndarray:
    # The world coordinates of every voxel of a grid of the given shape.
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1
    ).astype(float)
    return grid @ linear.T + np.asarray(affine.matrix)[:, -1]


def _displacement_level(
    raw_world: np.ndarray, affine: Affine, linear: np.ndarray
) -> Sequence:
    # A displacement level: the world-to-voxel affine, the displacement in
    # voxel units, and the voxel-to-world affine.
    voxel = raw_world @ np.linalg.inv(linear).T
    return Sequence(
        transformations=[
            affine.inverse(),
            DisplacementField(field=voxel),
            affine,
        ]
    )


def _coordinate_level(raw_world: np.ndarray, affine: Affine) -> Sequence:
    # A coordinate level: the world-to-voxel affine and the coordinate
    # field, which stores the world coordinates without a copy.
    return Sequence(
        transformations=[affine.inverse(), CoordinatesField(field=raw_world)]
    )


def _two_level_displacement(rng: np.random.Generator) -> tuple:
    xf0, lin0 = _rotated_anisotropic_affine(2.0, 0.5)
    xf1, lin1 = _rotated_anisotropic_affine(4.0, 1.0)
    raw0 = rng.normal(size=(6, 5, 2))
    raw1 = rng.normal(size=(3, 3, 2))
    l0 = _displacement_level(raw0, xf0, lin0)
    l1 = _displacement_level(raw1, xf1, lin1)
    field = MultiscaleField(levels=[l0, l1])
    return field, (l0, l1), (xf0, lin0, raw0), (xf1, lin1, raw1)


# --- structure ---------------------------------------------------------


def test_a_level_is_a_sequence() -> None:
    field, (l0, l1), *_ = _two_level_displacement(np.random.default_rng(0))
    assert isinstance(field, Sequence)
    assert isinstance(l0, Sequence)
    assert isinstance(l1, Sequence)


def test_transformations_are_the_finest_levels() -> None:
    field, (l0, _), *_ = _two_level_displacement(np.random.default_rng(1))
    assert field.transformations is l0.transformations
    assert len(field) == len(l0)


def test_nlevels_and_at_level() -> None:
    field, (l0, l1), *_ = _two_level_displacement(np.random.default_rng(2))
    assert field.nlevels == 2
    assert field.finest is l0
    assert field.at_level(0) is l0
    assert field.at_level(1) is l1


def test_at_level_returns_a_plain_sequence() -> None:
    field, (l0, _), *_ = _two_level_displacement(np.random.default_rng(3))
    level = field.at_level(1)
    assert isinstance(level, Sequence)
    assert not isinstance(level, MultiscaleField)


def test_inverse_keeps_the_pyramid() -> None:
    field, *_ = _two_level_displacement(np.random.default_rng(4))
    inverse = field.inverse()
    assert isinstance(inverse, MultiscaleField)
    assert inverse.nlevels == 2


def test_the_container_is_not_mutable() -> None:
    field, (l0, l1), *_ = _two_level_displacement(np.random.default_rng(5))
    with pytest.raises(TypeError):
        field[0] = l1
    with pytest.raises(TypeError):
        del field[0]
    with pytest.raises(TypeError):
        field.insert(0, l1)


# --- transparency in composition ---------------------------------------


def test_composes_like_its_finest_level() -> None:
    field, _, (xf0, lin0, _), _ = _two_level_displacement(
        np.random.default_rng(6)
    )
    outer = Scaling(scale=[3.0, 4.0])
    world = CoordinatesField(field=_world_grid((6, 5), lin0, xf0))
    from_field = ((outer @ field) @ world).compute()
    from_finest = ((outer @ field.finest) @ world).compute()
    np.testing.assert_allclose(
        np.asarray(from_field.field), np.asarray(from_finest.field), atol=1e-6
    )


def test_flattened_splices_the_finest_level() -> None:
    field, (l0, _), *_ = _two_level_displacement(np.random.default_rng(7))
    outer = Scaling(scale=[2.0, 2.0])
    flattened = Sequence([field, outer])._flattened()
    # The finest level's three elements, followed by the outer scaling.
    assert len(flattened.transformations) == len(l0) + 1


def test_compute_returns_a_plain_transformation() -> None:
    field, *_ = _two_level_displacement(np.random.default_rng(8))
    computed = field.compute()
    assert not isinstance(computed, MultiscaleField)


# --- level selection ---------------------------------------------------


def test_nearest_level_matches_the_target_resolution() -> None:
    field, *_ = _two_level_displacement(np.random.default_rng(9))
    # Level 0 has voxel size (2, 0.5); level 1 has voxel size (4, 1).
    assert field._nearest_level(_rotated_anisotropic_affine(2.0, 0.5)[0]) == 0
    assert field._nearest_level(_rotated_anisotropic_affine(4.0, 1.0)[0]) == 1
    # In log scale, (3.2, 0.8) is nearer to (4, 1) than to (2, 0.5).
    assert field._nearest_level(_rotated_anisotropic_affine(3.2, 0.8)[0]) == 1


def test_level_resolution_reads_the_leading_affine() -> None:
    field, *_ = _two_level_displacement(np.random.default_rng(10))
    np.testing.assert_allclose(
        np.asarray(field._level_resolution(0)), [2.0, 0.5]
    )
    np.testing.assert_allclose(
        np.asarray(field._level_resolution(1)), [4.0, 1.0]
    )


def test_an_identity_led_level_has_no_resolution() -> None:
    xf1, _ = _rotated_anisotropic_affine(4.0, 1.0)
    field = MultiscaleField(
        levels=[
            Sequence(
                [Identity(), DisplacementField(field=np.zeros((4, 4, 2)))]
            ),
            Sequence(
                [
                    xf1.inverse(),
                    DisplacementField(field=np.zeros((3, 3, 2))),
                    xf1,
                ]
            ),
        ]
    )
    assert field._level_resolution(0) is None
    assert field._nearest_level(_rotated_anisotropic_affine(4.0, 1.0)[0]) == 0


# --- ground truth ------------------------------------------------------


def test_displacement_level_moves_world_points_by_the_world_field() -> None:
    # Applying a displacement level to the world coordinates of its own
    # voxels shifts each point by exactly the world-unit displacement
    # stored for that voxel.
    xf0, lin0 = _rotated_anisotropic_affine(2.0, 0.5)
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(6, 5, 2))
    level = _displacement_level(raw, xf0, lin0)
    world = _world_grid((6, 5), lin0, xf0)
    moved = (level @ CoordinatesField(field=world)).compute()
    np.testing.assert_allclose(np.asarray(moved.field), world + raw, atol=1e-6)


def test_coordinate_level_recovers_the_stored_coordinates() -> None:
    # Applying a coordinate level to the world coordinates of its own
    # voxels recovers the world coordinate stored for each voxel.
    xf0, lin0 = _rotated_anisotropic_affine(2.0, 0.5)
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(6, 5, 2))
    level = _coordinate_level(raw, xf0)
    world = _world_grid((6, 5), lin0, xf0)
    resolved = (level @ CoordinatesField(field=world)).compute()
    np.testing.assert_allclose(np.asarray(resolved.field), raw, atol=1e-6)


# --- reslice wiring ----------------------------------------------------


def test_at_resolution_selects_the_matching_level() -> None:
    field, (l0, l1), *_ = _two_level_displacement(np.random.default_rng(13))
    coarse = _rotated_anisotropic_affine(4.0, 1.0)[0]
    assert _at_resolution(field, coarse) is l1
    fine = _rotated_anisotropic_affine(2.0, 0.5)[0]
    assert _at_resolution(field, fine) is l0


def test_at_resolution_walks_a_nested_sequence() -> None:
    field, (_, l1), *_ = _two_level_displacement(np.random.default_rng(14))
    outer = Scaling(scale=[2.0, 2.0])
    nested = Sequence([field, outer])
    coarse = _rotated_anisotropic_affine(4.0, 1.0)[0]
    resolved = _at_resolution(nested, coarse)
    assert isinstance(resolved, Sequence)
    assert resolved.transformations[0] is l1


def test_at_resolution_leaves_a_plain_transformation_unchanged() -> None:
    plain = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert _at_resolution(plain, target) is plain
