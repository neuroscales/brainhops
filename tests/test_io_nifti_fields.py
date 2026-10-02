"""
RAS fields stored in NIfTI files: displacements and coordinates.

The NIfTI-1 standard reserves `DISPVECT` (1006) "specifically for
displacements" and `VECTOR` (1007) "for any other type of vector". These
tests pin down that a standard `DISPVECT` file is read as RAS
displacements in millimetres, that a field of coordinates is written as
`VECTOR`, and that both kinds round-trip.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
)

DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR

SHAPE = (4, 5, 6)
"""Grid shape: small, and no two axes of the same length."""

# A voxel-to-RAS affine with a permutation, a flip, anisotropic spacing
# and an offset, so that displacements left in voxel units, or rotated
# the wrong way, cannot pass for millimetres.
VOX2RAS = np.array(
    [
        [0.0, -3.0, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _ramp() -> np.ndarray:
    """An `(X, Y, Z, 3)` RAS displacement whose entries name their voxel."""
    i, j, k = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    return np.stack(
        [1.0 + 0.1 * i, 2.0 + 0.2 * j, 3.0 + 0.3 * k], axis=-1
    ).astype("float32")


def _grid_points() -> np.ndarray:
    """The RAS coordinates of every voxel, as an `(X, Y, Z, 3)` array."""
    ijk = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), axis=-1)
    return ijk @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]


def _write(path, vectors: np.ndarray, intent: int):  # noqa: ANN001, ANN202
    """Write `(X, Y, Z, 3)` vectors in the `(X, Y, Z, 1, 3)` layout."""
    img = nb.Nifti1Image(vectors[:, :, :, None, :], VOX2RAS)
    img.header.set_intent(intent)
    nb.save(img, str(path))
    return path


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through a RAS-to-RAS transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points, float))
    out = xforms.Sequence(transformations=[points, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _apply_coordinates(field, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through a voxel-to-RAS coordinates field."""
    # The reader keeps the singleton axis of the `(X, Y, Z, 1, 3)` layout
    # in its field, so the grid is rebuilt from the squeezed values.
    field = xforms.CoordinatesField(field=np.asarray(field.field)[:, :, :, 0])
    ras2vox = np.linalg.inv(VOX2RAS)
    voxels = np.asarray(points, float) @ ras2vox[:3, :3].T + ras2vox[:3, 3]
    voxels = xforms.CoordinatesField(field=voxels)
    out = xforms.Sequence(transformations=[voxels, field]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


@pytest.fixture
def standard_warp(tmp_path):  # noqa: ANN001, ANN201
    """A `DISPVECT` file built to the standard: RAS displacements in mm."""
    return _write(tmp_path / "warp.nii.gz", _ramp(), DISPVECT)


# ----------------------------------------------------------------------
#   READING A STANDARD DISPVECT FILE
# ----------------------------------------------------------------------


def test_a_dispvect_file_maps_ras_to_ras(standard_warp) -> None:  # noqa: ANN001
    field = io.transformations.load(standard_warp)
    assert type(field) is NiftiRASDisplacementField
    assert field.input.name == "RAS"
    assert field.output.name == "RAS"
    assert len(field) == 3


def test_a_dispvect_file_moves_every_node_by_its_vector(
    standard_warp,  # noqa: ANN001
) -> None:
    """
    Each point at a voxel centre moves by that voxel's stored vector, in
    RAS millimetres: `x -> x + u(x)`.
    """
    field = io.transformations.load(standard_warp)
    points = _grid_points().reshape(-1, 3)
    moved = _apply(field, points)
    np.testing.assert_allclose(
        moved, points + _ramp().reshape(-1, 3), atol=1e-4
    )


def test_a_constant_displacement_is_a_translation_in_mm(tmp_path) -> None:  # noqa: ANN001
    """Off the grid nodes too, a constant field is a plain translation."""
    vectors = np.broadcast_to(
        np.array([1.0, -2.0, 3.0], "float32"), (*SHAPE, 3)
    ).copy()
    field = NiftiRASDisplacementField.from_file(
        _write(tmp_path / "shift.nii", vectors, DISPVECT)
    )
    points = np.array([[10.0, -19.0, 31.0], [9.5, -17.0, 40.0]])
    np.testing.assert_allclose(
        _apply(field, points) - points, [[1.0, -2.0, 3.0]] * 2, atol=1e-5
    )


def test_displacements_are_stored_in_voxel_units(standard_warp) -> None:  # noqa: ANN001
    """The middle slot adds its values on the grid, in voxels."""
    field = io.transformations.load(standard_warp)
    voxels = np.asarray(field.displacement.field)
    assert voxels.shape == (*SHAPE, 3)
    np.testing.assert_allclose(
        voxels @ VOX2RAS[:3, :3].T, _ramp(), rtol=1e-6, atol=1e-6
    )


def test_a_coordinates_field_and_its_displacements_agree(tmp_path) -> None:  # noqa: ANN001
    """
    A field of RAS positions (`VECTOR`) and the matching field of RAS
    displacements (`DISPVECT`) are the same map.
    """
    coords = _write(
        tmp_path / "coords.nii.gz",
        (_grid_points() + _ramp()).astype("float32"),
        VECTOR,
    )
    disp = _write(tmp_path / "disp.nii.gz", _ramp(), DISPVECT)
    coords = io.transformations.load(coords)
    disp = io.transformations.load(disp)
    assert type(coords) is NiftiRASCoordinatesField
    assert type(disp) is NiftiRASDisplacementField
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply_coordinates(coords, points), _apply(disp, points), atol=1e-4
    )


# ----------------------------------------------------------------------
#   ROUND TRIPS
# ----------------------------------------------------------------------


def test_a_displacement_field_round_trips(standard_warp, tmp_path) -> None:  # noqa: ANN001
    field = io.transformations.load(standard_warp)
    target = tmp_path / "out.nii.gz"
    field.save(target)

    written = nb.load(str(target))
    assert int(written.header["intent_code"]) == DISPVECT
    assert written.shape == (*SHAPE, 1, 3)
    np.testing.assert_allclose(written.affine, VOX2RAS)
    np.testing.assert_allclose(
        np.asarray(written.dataobj)[:, :, :, 0], _ramp(), atol=1e-5
    )

    reloaded = io.transformations.load(target)
    assert type(reloaded) is NiftiRASDisplacementField
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(reloaded, points), _apply(field, points), atol=1e-4
    )


def test_a_displacement_field_built_from_its_slots_is_written(
    tmp_path,  # noqa: ANN001
) -> None:
    """A field that was never read from a file is written from its chain."""
    from brainhops.io.transformations.base.affines import (
        RASToVoxel,
        VoxelToRAS,
    )

    voxels = (_ramp() @ np.linalg.inv(VOX2RAS[:3, :3]).T).astype("float32")
    field = NiftiRASDisplacementField(
        transformations=(
            RASToVoxel(matrix=np.linalg.inv(VOX2RAS)[:3]),
            xforms.DisplacementField(field=voxels),
            VoxelToRAS(matrix=VOX2RAS[:3]),
        )
    )
    target = tmp_path / "built.nii"
    field.save(target)

    reloaded = io.transformations.load(target)
    assert type(reloaded) is NiftiRASDisplacementField
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(reloaded, points),
        points + _ramp().reshape(-1, 3),
        atol=1e-4,
    )


def test_a_coordinates_field_round_trips_as_vector(tmp_path) -> None:  # noqa: ANN001
    """
    A field of coordinates is written as `VECTOR` (with SPM's intent
    name, "Mapping"), never as `DISPVECT`, and reads back as coordinates.
    """
    values = (_grid_points() + _ramp()).astype("float32")
    source = _write(tmp_path / "coords.nii.gz", values, VECTOR)
    field = io.transformations.load(source)
    target = tmp_path / "out.nii.gz"
    field.save(target)

    header = nb.load(str(target)).header
    code, _, name = header.get_intent()
    assert int(header["intent_code"]) == VECTOR
    assert code == "vector"
    assert name == "Mapping"

    reloaded = io.transformations.load(target)
    assert type(reloaded) is NiftiRASCoordinatesField
    np.testing.assert_array_equal(
        np.asarray(reloaded.field)[:, :, :, 0], values
    )


def test_a_legacy_coordinates_file_is_read_through_a_hint(tmp_path) -> None:  # noqa: ANN001
    """
    Older brainhops wrote coordinates as `DISPVECT`. Read with an explicit
    hint and saved again, such a file comes back as `VECTOR`.
    """
    values = (_grid_points() + _ramp()).astype("float32")
    legacy = _write(tmp_path / "legacy.nii.gz", values, DISPVECT)
    assert type(io.transformations.load(legacy)) is NiftiRASDisplacementField

    field = io.transformations.load(legacy, hint="nifti.coordinates")
    assert type(field) is NiftiRASCoordinatesField
    target = tmp_path / "migrated.nii.gz"
    field.save(target)
    assert int(nb.load(str(target)).header["intent_code"]) == VECTOR
    assert type(io.transformations.load(target)) is NiftiRASCoordinatesField


# ----------------------------------------------------------------------
#   ERRORS
# ----------------------------------------------------------------------


def test_a_two_component_field_is_refused(tmp_path) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros((4, 5, 1, 1, 2), "float32"), VOX2RAS)
    img.header.set_intent(DISPVECT)
    path = tmp_path / "flat.nii"
    nb.save(img, str(path))
    field = NiftiRASDisplacementField.from_file(path)
    with pytest.raises(Exception, match="three-dimensional"):
        field.transformations  # noqa: B018


def test_a_spline_field_is_not_written(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.base.parsers import WriterError
    from brainhops.io.transformations.base.affines import (
        RASToVoxel,
        VoxelToRAS,
    )

    field = NiftiRASDisplacementField(
        transformations=(
            RASToVoxel(matrix=np.eye(4)[:3]),
            xforms.DisplacementField(
                field=np.zeros((*SHAPE, 3), "float32"), coeff=True
            ),
            VoxelToRAS(matrix=np.eye(4)[:3]),
        )
    )
    with pytest.raises(WriterError, match="coefficients"):
        field.save(tmp_path / "spline.nii")
