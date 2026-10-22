"""
Tests of the affines that a field format stores beside its field.

A transformation `A o (id + d) o B` holds a field between two affines.
A format that has a place for an affine before or after its field, such
as the header extensions of a NiftyReg grid or another transform of a
chain, stores the affines there, and the field keeps the grid on which
it is placed.
"""

from pathlib import Path

import nibabel as nb
import numpy as np

from brainhops import io
from brainhops.datamodel import systems
from brainhops.datamodel import transformations as xforms
from brainhops.datamodel._transformations.compute.adaptors import bridge
from brainhops.datamodel.enums import StoreEnum
from brainhops.io.transformations.base._conversions import (
    field_slots,
    world_pieces,
)
from brainhops.io.transformations.base.affines import (
    RASToRAS,
    RASToVoxel,
    VoxelToRAS,
)
from brainhops.io.transformations.niftyreg import (
    NiftyRegControlPointGrid,
    NiftyRegVelocityGrid,
)

RAS = systems.RASmm()
VOXEL = systems.VoxelCoordinateSystem()

GRID = np.array(
    [[2.0, 0, 0, -10], [0, 2.0, 0, -10], [0, 0, 2.0, -10], [0, 0, 0, 1]]
)
"""The placement of the grid of the field in the RAS world."""

AFTER = np.array(
    [[1.0, 0.1, 0, 1], [0, 0.9, 0, 2], [0, 0, 1.1, 3], [0, 0, 0, 1]]
)
"""The affine applied after the field."""

BEFORE = np.array(
    [[0.95, 0, 0.05, -2], [0, 1.0, 0, 0.5], [0, 0, 1.0, 1], [0, 0, 0, 1]]
)
"""The affine applied before the field."""


def _chain(log: bool = False) -> xforms.Sequence:
    # `AFTER o (id + d) o BEFORE`, with a cubic field on `GRID`.
    rng = np.random.default_rng(0)
    field = xforms.DisplacementField(
        data=rng.normal(size=(10, 10, 10, 3)) * 0.3,
        degree=3,
        store=StoreEnum.coefficients,
        bound="nearest",
        log=log,
        input=VOXEL,
        output=VOXEL,
    )
    return xforms.Sequence(
        transformations=[
            RASToRAS(matrix=BEFORE[:-1]),
            RASToVoxel(matrix=np.linalg.inv(GRID)[:-1]),
            field,
            VoxelToRAS(matrix=GRID[:-1]),
            RASToRAS(matrix=AFTER[:-1]),
        ]
    )


def _apply(t: xforms.Transformation, points: np.ndarray) -> np.ndarray:
    # The map of `t` on RAS points, written in RAS. A format between LPS
    # spaces, such as elastix, is bridged to RAS on both sides.
    points = xforms.CoordinatesField(field=points[:, None, None, :])
    chain = [points, t]
    if t.input is not None and t.input != RAS:
        chain = [
            points,
            bridge(RAS, t.input, allow_type_grouped_positional=True),
            t,
            bridge(t.output, RAS, allow_type_grouped_positional=True),
        ]
    out = xforms.Sequence(transformations=chain).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)[:, 0, 0, :]


def _points() -> np.ndarray:
    return np.random.default_rng(1).uniform(-5, 5, size=(20, 3))


def test_world_pieces_keep_the_grid_of_the_field() -> None:
    # The affines on either side of the field are pieces of their own.
    pieces = world_pieces(_chain(), RAS, NiftyRegControlPointGrid)
    assert len(pieces) == 3
    first, field, last = pieces
    np.testing.assert_allclose(first.homogeneous_matrix, BEFORE)
    np.testing.assert_allclose(last.homogeneous_matrix, AFTER)
    np.testing.assert_allclose(
        field.transformations[0].homogeneous_matrix, np.linalg.inv(GRID)
    )


def test_field_slots_return_the_affines_around_the_field() -> None:
    before, field, after = field_slots(_chain(), RAS, NiftyRegControlPointGrid)
    np.testing.assert_allclose(before, BEFORE)
    np.testing.assert_allclose(after, AFTER)
    np.testing.assert_allclose(
        field.transformations[-1].homogeneous_matrix, GRID
    )


def test_a_grid_stores_both_affines_in_its_extensions(tmp_path: Path) -> None:
    # NiftyReg applies the first extension before the spline and composes
    # the second one after it.
    t = _chain()
    grid = t.to(NiftyRegControlPointGrid)
    assert len(grid) == 5
    np.testing.assert_allclose(grid.affine.homogeneous_matrix, BEFORE)
    np.testing.assert_allclose(grid.affine_after.homogeneous_matrix, AFTER)
    np.testing.assert_allclose(grid.voxel2ras.homogeneous_matrix, GRID)
    points = _points()
    np.testing.assert_allclose(_apply(grid, points), _apply(t, points))

    path = tmp_path / "cpp.nii.gz"
    io.save(grid, path)
    assert len(nb.load(path).header.extensions) == 2
    back = io.load(path)
    assert isinstance(back, NiftyRegControlPointGrid)
    np.testing.assert_allclose(
        _apply(back, points), _apply(t, points), atol=1e-5
    )


def test_a_grid_with_an_affine_after_writes_an_identity_first(
    tmp_path: Path,
) -> None:
    # NiftyReg reads the affine after the spline from the second extension,
    # so the first one holds the identity.
    t = _chain()
    t = xforms.Sequence(transformations=list(t)[1:])
    path = tmp_path / "cpp.nii.gz"
    io.save(t.to(NiftyRegControlPointGrid), path)
    extensions = nb.load(path).header.extensions
    assert len(extensions) == 2
    first = np.frombuffer(extensions[0].get_content()[:64], "<f4")
    np.testing.assert_allclose(first.reshape(4, 4), np.eye(4))
    back = io.load(path)
    points = _points()
    np.testing.assert_allclose(
        _apply(back, points), _apply(t, points), atol=1e-5
    )


def test_a_velocity_grid_stores_its_affine_after_the_flow(
    tmp_path: Path,
) -> None:
    # A change of coordinates carries the flow of a velocity, so the
    # affines around it are composed after it, in the second extension.
    t = _chain(log=True)
    grid = t.to(NiftyRegVelocityGrid)
    points = _points()
    np.testing.assert_allclose(
        _apply(grid, points), _apply(t, points), atol=1e-6
    )
    path = tmp_path / "vel.nii.gz"
    io.save(grid, path)
    extensions = nb.load(path).header.extensions
    assert len(extensions) == 2
    first = np.frombuffer(extensions[0].get_content()[:64], "<f4")
    np.testing.assert_allclose(first.reshape(4, 4), np.eye(4))
    back = io.load(path)
    assert isinstance(back, NiftyRegVelocityGrid)
    np.testing.assert_allclose(
        _apply(back, points), _apply(t, points), atol=1e-5
    )


def test_elastix_chains_the_affines_around_its_bspline() -> None:
    from brainhops.io.transformations.elastix import (
        ElastixParameterTransform,
    )

    t = _chain()
    result = t.to(ElastixParameterTransform)
    # Each file of the chain is the initial transform of the next one.
    files = 0
    link = result
    while link is not None:
        files += 1
        link = link.initial
    assert files == 3
    points = _points()
    np.testing.assert_allclose(
        _apply(result, points), _apply(t, points), atol=1e-6
    )


def test_a_chain_format_stores_the_affines_as_transforms() -> None:
    from brainhops.io.transformations.x5 import X5Transform

    t = _chain()
    result = t.to(X5Transform)
    assert len(result.transformations) == 3
    points = _points()
    np.testing.assert_allclose(
        _apply(result, points), _apply(t, points), atol=1e-6
    )
