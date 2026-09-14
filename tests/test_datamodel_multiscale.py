"""Tests for the multiscale coordinate and displacement fields.

The placement used throughout is anisotropic and rotated on purpose. The
voxel-to-voxel normalization applied when a field is placed rescales the
field by the inverse of the linear part of the placement. On a
one-millimetre isotropic grid that rescaling is the identity, so an
isotropic test would pass even if the normalization were wrong. An
anisotropic and rotated placement makes the rescaling visible.
"""

import numpy as np
import pytest

from brainhops.datamodel import transformations as X
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Identity,
    MultiscaleCoordinatesField,
    MultiscaleDisplacementField,
    OmePlacementError,
    Scaling,
    Sequence,
)


def _rotated_anisotropic_affine() -> tuple:
    # A 2D placement that scales the axes by (2, 1/2), rotates them, and
    # shifts the origin. Its linear part is not a scalar multiple of the
    # identity, so the inverse-linear normalization is not invisible.
    theta = 0.37
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    linear = rot @ np.diag([2.0, 0.5])
    matrix = np.concatenate([linear, np.array([[5.0], [-3.0]])], axis=1)
    return Affine(matrix=matrix), linear


def _two_level_displacement(
    rng: np.random.Generator,
) -> MultiscaleDisplacementField:
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    return MultiscaleDisplacementField(
        levels=[fine, coarse],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )


# --- type and structure ------------------------------------------------


def test_multiscale_field_is_its_single_scale_type() -> None:
    field = _two_level_displacement(np.random.default_rng(0))
    assert isinstance(field, DisplacementField)
    coord = MultiscaleCoordinatesField(levels=[np.zeros((4, 4, 2))])
    assert isinstance(coord, X.CoordinatesField)


def test_default_level_is_the_finest() -> None:
    rng = np.random.default_rng(1)
    fine = rng.normal(size=(6, 5, 2))
    field = MultiscaleDisplacementField(levels=[fine, np.zeros((3, 3, 2))])
    assert field.level == 0
    np.testing.assert_allclose(np.asarray(field.field), fine)


def test_at_level_selects_a_coarser_level_and_keeps_the_pyramid() -> None:
    rng = np.random.default_rng(2)
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    field = MultiscaleDisplacementField(levels=[fine, coarse])
    coarser = field.at_level(1)
    assert isinstance(coarser, MultiscaleDisplacementField)
    assert coarser.level == 1
    np.testing.assert_allclose(np.asarray(coarser.field), coarse)
    # The original is unchanged.
    np.testing.assert_allclose(np.asarray(field.field), fine)


def test_replace_follows_the_active_level_rather_than_freezing_it() -> None:
    from bagof.magic import replace

    rng = np.random.default_rng(20)
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    field = MultiscaleDisplacementField(levels=[fine, coarse])
    switched = replace(field, level=1)
    assert switched.level == 1
    np.testing.assert_allclose(np.asarray(switched.field), coarse)
    # The finest array was not frozen onto the rebuilt field.
    np.testing.assert_allclose(np.asarray(field.field), fine)


def test_to_selects_a_level_without_freezing_the_array() -> None:
    rng = np.random.default_rng(21)
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    field = MultiscaleDisplacementField(levels=[fine, coarse])
    switched = field.to(level=1)
    np.testing.assert_allclose(np.asarray(switched.field), coarse)
    assert len(switched.levels) == 2


def test_inverse_of_a_multiscale_field_keeps_the_pyramid() -> None:
    rng = np.random.default_rng(22)
    fine = rng.normal(size=(6, 5, 2))
    coarse = rng.normal(size=(3, 3, 2))
    field = MultiscaleDisplacementField(levels=[fine, coarse])
    inverse = field.inverse()
    assert isinstance(inverse, MultiscaleDisplacementField)
    assert len(inverse.levels) == 2


# --- type transparency in composition ----------------------------------


def test_composes_like_its_active_single_scale_level() -> None:
    rng = np.random.default_rng(3)
    fine = rng.normal(size=(6, 5, 2))
    multiscale = MultiscaleDisplacementField(
        levels=[fine, np.zeros((3, 3, 2))]
    )
    plain = DisplacementField(field=fine)
    outer = Scaling(scale=[3.0, 4.0])
    from_multiscale = (outer @ multiscale).compute()
    from_plain = (outer @ plain).compute()
    assert type(from_multiscale) is type(from_plain)
    np.testing.assert_allclose(
        np.asarray(from_multiscale.field), np.asarray(from_plain.field)
    )


# --- level selection ---------------------------------------------------


def test_level_scales_read_the_stored_cross_level_transforms() -> None:
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    scales = field.level_scales
    assert scales[0] is None
    np.testing.assert_allclose(np.asarray(scales[1]), [2.0, 2.0])


def test_multiscale_level_for_matches_target_resolution() -> None:
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    assert X.multiscale_level_for(field, 1.0) == 0
    assert X.multiscale_level_for(field, 2.0) == 1
    # Closest in log-scale: a target of 1.6 is nearer to 2 than to 1.
    assert X.multiscale_level_for(field, 1.6) == 1


def test_select_level_accepts_a_transformation_target() -> None:
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    selected = field.select_level(Scaling(scale=[2.0, 2.0]))
    assert selected.level == 1


# --- the (b) sandwich and voxel-to-voxel normalization -----------------


def test_place_ome_field_builds_the_sandwich() -> None:
    placement, _ = _rotated_anisotropic_affine()
    field = MultiscaleDisplacementField(levels=[np.zeros((5, 4, 2))])
    sandwich = X.place_ome_field(field, placement)
    assert isinstance(sandwich, Sequence)
    left, middle, right = sandwich.transformations
    assert isinstance(middle, MultiscaleDisplacementField)
    assert right is placement
    # The left affine is the inverse of the placement.
    np.testing.assert_allclose(
        np.asarray(left.matrix),
        np.asarray(placement.inverse().matrix),
    )


def test_displacement_normalization_uses_the_inverse_linear_part() -> None:
    placement, linear = _rotated_anisotropic_affine()
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(6, 5, 2))
    normalized = X.normalize_ome_displacement(raw, placement)
    expected = raw @ np.linalg.inv(linear).T
    np.testing.assert_allclose(np.asarray(normalized), expected)


def test_displacement_normalization_is_invisible_on_isotropic_grid() -> None:
    # The exact trap the anisotropic test guards against: on a unit
    # isotropic placement the normalization is the identity.
    placement = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(6, 5, 2))
    normalized = X.normalize_ome_displacement(raw, placement)
    np.testing.assert_allclose(np.asarray(normalized), raw)


def test_coordinate_normalization_uses_the_inverse_placement() -> None:
    placement, _ = _rotated_anisotropic_affine()
    rng = np.random.default_rng(6)
    raw = rng.normal(size=(6, 5, 2))
    normalized = X.normalize_ome_coordinates(raw, placement)
    inverse = placement.inverse().matrix
    expected = raw @ inverse[:, :-1].T + inverse[:, -1]
    np.testing.assert_allclose(np.asarray(normalized), expected)


def test_placed_field_preserves_the_world_displacement() -> None:
    # A ground-truth check, not a restatement of the normalization
    # expression: the voxel-space field mapped back through the linear
    # part of the placement recovers the original world-unit displacement.
    placement, linear = _rotated_anisotropic_affine()
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(6, 5, 2))
    voxel_field = X.normalize_ome_displacement(raw, placement)
    recovered = np.asarray(voxel_field) @ linear.T
    np.testing.assert_allclose(recovered, raw)


def test_sandwich_moves_world_points_by_the_world_field() -> None:
    # End to end: placing a displacement field and applying it to the
    # world coordinates of its own voxels shifts each point by exactly the
    # world-unit displacement stored for that voxel.
    placement, linear = _rotated_anisotropic_affine()
    rng = np.random.default_rng(10)
    raw = rng.normal(size=(6, 5, 2))
    voxel_field = X.normalize_ome_displacement(raw, placement)
    field = MultiscaleDisplacementField(levels=[voxel_field])
    sandwich = X.place_ome_field(field, placement)
    grid = np.stack(
        np.meshgrid(np.arange(6), np.arange(5), indexing="ij"), axis=-1
    ).astype(float)
    world = grid @ linear.T + placement.matrix[:, -1]
    moved = (sandwich @ X.CoordinatesField(field=world)).compute()
    np.testing.assert_allclose(np.asarray(moved.field), world + raw, atol=1e-6)


def test_undimensioned_placement_takes_its_size_from_the_field() -> None:
    # An identity placement carries no size; `place_ome_field` reads the
    # dimensionality from the field's array rather than crashing later.
    field = MultiscaleDisplacementField(levels=[np.zeros((4, 3, 2))])
    sandwich = X.place_ome_field(field, Identity())
    right = sandwich.transformations[2]
    assert np.asarray(X._affine_scale(right)).shape == (2,)


# --- refusals ----------------------------------------------------------


def test_mixed_displacement_and_coordinate_axes_are_refused() -> None:
    # A field that names both a displacement axis and a coordinate axis is
    # the true mixed case: its vectors cannot be read as one kind.
    axes = [
        Axis(type="space"),
        Axis(type="displacement"),
        Axis(type="coordinate"),
    ]
    with pytest.raises(OmePlacementError):
        X.check_ome_axes(axes)


def test_pure_axis_kinds_are_classified() -> None:
    # A spec-conformant field lists its input space axes and exactly one
    # displacement or coordinate axis for the vector components.
    displacement = [
        Axis(type="space"),
        Axis(type="space"),
        Axis(type="displacement"),
    ]
    coordinate = [
        Axis(type="space"),
        Axis(type="space"),
        Axis(type="coordinate"),
    ]
    assert X.check_ome_axes(displacement) == "displacement"
    assert X.check_ome_axes(coordinate) == "coordinate"
    assert X.ome_vector_axis(displacement) == 2
    assert X.ome_vector_axis(coordinate) == 2


def test_axes_without_a_vector_axis_are_malformed() -> None:
    # A field must carry one vector axis; a purely spatial axes list names
    # none and cannot be read.
    with pytest.raises(OmePlacementError):
        X.check_ome_axes([Axis(type="space"), Axis(type="space")])


def test_vector_axis_need_not_be_last() -> None:
    axes = [
        Axis(type="displacement"),
        Axis(type="space"),
        Axis(type="space"),
    ]
    assert X.check_ome_axes(axes) == "displacement"
    assert X.ome_vector_axis(axes) == 0


def test_nonlinear_placed_displacement_is_refused() -> None:
    placement = DisplacementField(field=np.zeros((4, 4, 2)))
    with pytest.raises(OmePlacementError):
        X.check_ome_displacement_placement(placement)


def test_affine_placed_displacement_is_accepted() -> None:
    placement, _ = _rotated_anisotropic_affine()
    # Does not raise.
    X.check_ome_displacement_placement(placement)


# --- reslice wiring ----------------------------------------------------


def test_resolve_multiscale_level_leaves_a_bare_field_unchanged() -> None:
    # Without a placement the target resolution cannot be expressed
    # relative to the finest grid, so a bare field is returned unchanged
    # on its finest level, which reslices correctly at any resolution.
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    coarse_target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert X.resolve_multiscale_level(field, coarse_target) is field


def test_resolve_multiscale_level_selects_inside_a_sandwich() -> None:
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    placement = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    sandwich = X.place_ome_field(field, placement)
    coarse_target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    resolved = X.resolve_multiscale_level(sandwich, coarse_target)
    middle = resolved.transformations[1]
    assert middle.level == 1
    # The placement was adjusted to the coarse level's grid.
    np.testing.assert_allclose(
        np.asarray(X._affine_scale(resolved.transformations[2])), [2.0, 2.0]
    )


def test_resolve_multiscale_level_leaves_plain_transformations() -> None:
    plain = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert X.resolve_multiscale_level(plain, target) is plain
