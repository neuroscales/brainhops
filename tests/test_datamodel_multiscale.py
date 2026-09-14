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


# --- refusals ----------------------------------------------------------


def test_mixed_displacement_and_coordinate_axes_are_refused() -> None:
    axes = [Axis(type="displacement"), Axis(type="space")]
    with pytest.raises(OmePlacementError):
        X.check_ome_axes(axes)


def test_pure_axis_kinds_are_classified() -> None:
    assert X.check_ome_axes([Axis(type="displacement")] * 2) == "displacement"
    assert X.check_ome_axes([Axis(type="space")] * 2) == "coordinate"


def test_nonlinear_placed_displacement_is_refused() -> None:
    placement = DisplacementField(field=np.zeros((4, 4, 2)))
    with pytest.raises(OmePlacementError):
        X.check_ome_displacement_placement(placement)


def test_affine_placed_displacement_is_accepted() -> None:
    placement, _ = _rotated_anisotropic_affine()
    # Does not raise.
    X.check_ome_displacement_placement(placement)


# --- reslice wiring ----------------------------------------------------


def test_resolve_multiscale_level_selects_from_a_bare_field() -> None:
    field = MultiscaleDisplacementField(
        levels=[np.zeros((8, 8, 2)), np.zeros((4, 4, 2))],
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
    )
    coarse_target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    resolved = X.resolve_multiscale_level(field, coarse_target)
    assert resolved.level == 1
    fine_target = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert X.resolve_multiscale_level(field, fine_target).level == 0


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
