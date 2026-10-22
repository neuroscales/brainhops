"""
The parts that a field format stores, taken from any transformation.

A field format stores an affine from the world to a grid and the
displacement or the position of each node of that grid. These tests
check that `displacement_parts`, `coordinates_parts` and
`sample_on_grid` give the same map as the transformation inside the
grid, whatever kind of field it holds, and that they report a loss when
the field is interpolated otherwise or sampled on another grid.
"""

import numpy as np
import pytest

from brainhops.datamodel import transformations as xforms
from brainhops.datamodel.systems import LPSmm, RASmm, VoxelCoordinateSystem
from brainhops.errors import ConversionError, LossyConversionError
from brainhops.io.transformations.base._conversions import (
    FieldParts,
    coordinates_parts,
    displacement_parts,
    held,
    sample_on_grid,
)
from brainhops.io.transformations.base.affines import (
    RASToVoxel,
    VoxelToRAS,
)

SHAPE = (4, 5, 6)
"""Grid shape: small, and no two axes of the same length."""

VOX2RAS = np.array(
    [
        [0.0, -2.0, 0.0, 10.0],
        [1.5, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.5, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

VOXEL = VoxelCoordinateSystem()


class Format:
    """A stand-in for a format, which only names the errors."""


def _displacements() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(*SHAPE, 3))


def _chain(degree: int = 1) -> xforms.Sequence:
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=np.linalg.inv(VOX2RAS)[:-1]),
            xforms.DisplacementField(
                field=_displacements(),
                input=VOXEL,
                output=VOXEL,
                degree=degree,
            ),
            VoxelToRAS(matrix=VOX2RAS[:-1]),
        ]
    )


def _inside() -> np.ndarray:
    """RAS points inside the grid, on and between its nodes."""
    voxels = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.25, 2.75], [2.3, 3.7, 4.1]]
    )
    return voxels @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]


def _apply(xform: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    points = xforms.CoordinatesField(field=points[:, None, None, :])
    out = xforms.Sequence(transformations=[points, xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)[:, 0, 0, :]


def _sandwich(parts: FieldParts) -> xforms.Sequence:
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=parts.world2grid[:-1]),
            parts.field,
            VoxelToRAS(matrix=parts.grid2world[:-1]),
        ]
    )


def _coordinates(parts: FieldParts) -> xforms.Sequence:
    return xforms.Sequence(
        transformations=[
            RASToVoxel(matrix=parts.world2grid[:-1]),
            parts.field,
        ]
    )


def test_a_displacement_sandwich_is_kept_as_it_is() -> None:
    chain = _chain()
    parts = displacement_parts(chain, RASmm(), Format)
    assert parts.exact
    np.testing.assert_allclose(parts.world2grid, np.linalg.inv(VOX2RAS))
    np.testing.assert_array_equal(
        np.asarray(parts.field.values), _displacements()
    )


def test_coordinates_become_displacements() -> None:
    chain = _chain()
    coordinates = coordinates_parts(chain, RASmm(), RASmm(), Format)
    assert coordinates.exact
    again = displacement_parts(_coordinates(coordinates), RASmm(), Format)
    assert again.exact
    np.testing.assert_allclose(
        np.asarray(again.field.values), _displacements(), atol=1e-12
    )
    points = _inside()
    expected = _apply(chain, points)
    np.testing.assert_allclose(
        _apply(_coordinates(coordinates), points), expected, atol=1e-10
    )
    np.testing.assert_allclose(
        _apply(_sandwich(again), points), expected, atol=1e-10
    )


def test_lps_is_bridged_into_ras() -> None:
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    vox2lps = flip @ VOX2RAS
    chain = xforms.Sequence(
        transformations=[
            xforms.Affine(
                np.linalg.inv(vox2lps)[:-1], input=LPSmm(), output=VOXEL
            ),
            xforms.DisplacementField(
                field=_displacements(), input=VOXEL, output=VOXEL
            ),
            xforms.Affine(vox2lps[:-1], input=VOXEL, output=LPSmm()),
        ]
    )
    parts = displacement_parts(chain, RASmm(), Format)
    np.testing.assert_allclose(parts.world2grid, np.linalg.inv(VOX2RAS))
    points = _inside()
    lps = points * [-1.0, -1.0, 1.0]
    np.testing.assert_allclose(
        _apply(_sandwich(parts), points),
        _apply(chain, lps) * [-1.0, -1.0, 1.0],
        atol=1e-10,
    )


def test_a_cubic_field_is_sampled_with_a_loss() -> None:
    chain = _chain(degree=3)
    parts = displacement_parts(chain, RASmm(), Format)
    assert not parts.exact
    assert "degree 3" in parts.loss
    with pytest.raises(LossyConversionError, match="degree 3") as info:
        held("result", parts)
    assert info.value.result == "result"


def test_a_velocity_is_integrated_unless_it_is_kept() -> None:
    velocity = xforms.DisplacementField(
        data=0.1 * _displacements(), log=True, input=VOXEL, output=VOXEL
    )
    kept = displacement_parts(velocity, VOXEL, Format, keep_log=True)
    assert kept.field.log
    integrated = displacement_parts(velocity, VOXEL, Format)
    assert not integrated.field.log
    expected = velocity.to(xforms.DisplacementField, log=False).values
    np.testing.assert_allclose(
        np.asarray(integrated.field.values), np.asarray(expected)
    )


def test_two_fields_are_refused() -> None:
    field = xforms.DisplacementField(field=_displacements())
    chain = xforms.Sequence(transformations=[field, field])
    with pytest.raises(ConversionError, match="holds 2 fields"):
        displacement_parts(chain, RASmm(), Format)


def test_sampling_on_the_grid_of_the_field_is_exact() -> None:
    chain = _chain()
    parts = sample_on_grid(
        chain, np.linalg.inv(VOX2RAS), SHAPE, RASmm(), RASmm(), Format
    )
    assert parts.exact


def test_sampling_on_another_grid_is_a_loss() -> None:
    chain = _chain()
    world2grid = np.diag([2.0, 2.0, 2.0, 1.0]) @ np.linalg.inv(VOX2RAS)
    shape = (8, 10, 12)
    parts = sample_on_grid(chain, world2grid, shape, RASmm(), RASmm(), Format)
    assert not parts.exact
    # The nodes of the new grid are mapped as the transformation maps them.
    nodes = np.array([[2.0, 4.0, 6.0], [4.0, 0.0, 10.0]])
    grid2world = np.linalg.inv(world2grid)
    points = nodes @ grid2world[:3, :3].T + grid2world[:3, 3]
    values = np.asarray(parts.field.values)
    np.testing.assert_allclose(
        values[tuple(nodes.astype(int).T)], _apply(chain, points), atol=1e-10
    )
