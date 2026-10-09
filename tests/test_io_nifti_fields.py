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
from brainhops.backends import available_backends, backend  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.datamodel._transformations import (  # noqa: E402
    tangents as _tangents,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SpmCoordinatesField,
)

DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR
MAPPING = "Mapping"  # the intent name of a coordinates field

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
    # A field of coordinates is named as SPM and brainhops name it.
    img.header.set_intent(intent, name=MAPPING if intent == VECTOR else "")
    nb.save(img, str(path))
    return path


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through a RAS-to-RAS transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points, float))
    out = xforms.Sequence(transformations=[points, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _apply_coordinates(field, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through a voxel-to-RAS coordinates field."""
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


def test_a_coordinates_field_drops_its_singleton_axis(tmp_path) -> None:  # noqa: ANN001
    """
    NIfTI stores a vector field as `(X, Y, Z, 1, 3)`. The singleton axis
    is dropped on read, or the field would be sampled as a 4-D grid.
    """
    values = (_grid_points() + _ramp()).astype("float32")
    field = io.transformations.load(_write(tmp_path / "c.nii", values, VECTOR))
    assert type(field) is NiftiRASCoordinatesField
    assert np.asarray(field.field).shape == (*SHAPE, 3)
    np.testing.assert_array_equal(np.asarray(field.field), values)


@pytest.mark.parametrize(
    "array_backend",
    [
        "numpy",
        pytest.param(
            "dask",
            marks=pytest.mark.skipif(
                "dask" not in available_backends(),
                reason="dask is not installed",
            ),
        ),
    ],
)
def test_an_spm_deformation_is_sampled_in_ras(
    tmp_path,  # noqa: ANN001
    array_backend: str,
) -> None:
    """
    An SPM `y_` deformation, stored as `(X, Y, Z, 1, 3)`, maps each node
    of its grid to the RAS position it holds, and every other point --
    inside the grid or outside of it -- to a position that is the same on
    every call and on every backend.

    Its field once kept the singleton axis, and was sampled as a 4-D grid
    of 3-vectors: values of the order of 1e24 and 1e-22.
    """
    values = (_grid_points() + _ramp()).astype("float32")
    path = _write(tmp_path / "y_sub01.nii.gz", values, VECTOR)
    nodes = _grid_points().reshape(-1, 3)
    outside = np.stack(
        np.meshgrid(*[np.linspace(-60.0, 60.0, 5)] * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    with backend("numpy"):
        reference = _apply(io.transformations.load(path), outside)
    with backend(array_backend):
        field = io.transformations.load(path)
        assert type(field) is SpmCoordinatesField
        assert np.asarray(field.rasfield.field).shape == (*SHAPE, 3)
        np.testing.assert_allclose(
            _apply(field, nodes), values.reshape(-1, 3), atol=1e-4
        )
        for _ in range(3):
            np.testing.assert_allclose(
                _apply(field, outside), reference, atol=1e-4
            )
    # A deformation of this grid moves no point further than its own
    # extent, so a sample far outside of that range is not a position.
    low, high = values.reshape(-1, 3).min(0), values.reshape(-1, 3).max(0)
    assert np.all(reference >= low - 1e-3)
    assert np.all(reference <= high + 1e-3)


def test_points_map_through_a_loaded_coordinates_field(tmp_path) -> None:  # noqa: ANN001
    """
    Points map through a loaded field of RAS coordinates, and through the
    `rasfield` slot of an SPM `y_` field, to one RAS position each.

    The field is a linear function of RAS, so trilinear sampling between
    the nodes is exact. Its singleton axis was once kept, which mapped a
    single point to a `(4, 1, 3)` array of wrong positions.
    """
    matrix = np.array([[1.1, 0.1, 0.0], [0.0, 0.9, -0.2], [0.1, 0.0, 1.05]])
    shift = np.array([1.5, -2.0, 3.0])
    values = (_grid_points() @ matrix.T + shift).astype("float32")
    # Points inside the grid, off its nodes.
    ras2vox = np.linalg.inv(VOX2RAS)
    voxels = np.array([[1.3, 2.6, 4.1], [0.5, 0.5, 0.5], [2.9, 3.2, 1.7]])
    points = voxels @ VOX2RAS[:3, :3].T + VOX2RAS[:3, 3]
    np.testing.assert_allclose(
        points @ ras2vox[:3, :3].T + ras2vox[:3, 3], voxels, atol=1e-12
    )
    expected = points @ matrix.T + shift

    coords = io.transformations.load(
        _write(tmp_path / "coords.nii.gz", values, VECTOR), "coordinates"
    )
    assert type(coords) is NiftiRASCoordinatesField
    spm = io.transformations.load(
        _write(tmp_path / "y_sub01.nii.gz", values, VECTOR)
    )
    assert type(spm) is SpmCoordinatesField
    for field in (coords, spm.rasfield):
        assert np.asarray(field.field).shape == (*SHAPE, 3)
        one = _apply_coordinates(field, points[:1])
        assert one.shape == (1, 3)
        np.testing.assert_allclose(one, expected[:1], atol=1e-4)
        many = _apply_coordinates(field, points)
        assert many.shape == points.shape
        np.testing.assert_allclose(many, expected, atol=1e-4)
    mapped = _apply(spm, points)
    assert mapped.shape == points.shape
    np.testing.assert_allclose(mapped, expected, atol=1e-4)


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
    assert name == MAPPING

    reloaded = io.transformations.load(target)
    assert type(reloaded) is NiftiRASCoordinatesField
    np.testing.assert_array_equal(np.asarray(reloaded.field), values)


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


def test_a_spline_field_is_written_as_its_values(tmp_path) -> None:  # noqa: ANN001
    """A field of coefficients is decoded: NIfTI stores sampled values."""
    from brainhops._core.bsplines import value2coeff_field
    from brainhops.datamodel import systems
    from brainhops.io.transformations.base.affines import (
        RASToVoxel,
        VoxelToRAS,
    )

    values = np.random.default_rng(0).normal(size=(*SHAPE, 3))
    coefficients = value2coeff_field(values, degree=3, bound="nearest")
    voxel = systems.VoxelCoordinateSystem()
    field = NiftiRASDisplacementField(
        transformations=(
            RASToVoxel(matrix=np.eye(4)[:3]),
            xforms.DisplacementField(
                data=coefficients,
                degree=3,
                store="coefficients",
                input=voxel,
                output=voxel,
            ),
            VoxelToRAS(matrix=np.eye(4)[:3]),
        )
    )
    target = tmp_path / "spline.nii.gz"
    field.save(target)

    written = np.asarray(nb.load(str(target)).dataobj)[:, :, :, 0]
    np.testing.assert_allclose(written, values, atol=1e-10)
    reloaded = io.transformations.load(target)
    assert type(reloaded) is NiftiRASDisplacementField
    assert reloaded.displacement.store == "values"
    np.testing.assert_allclose(
        np.asarray(reloaded.displacement.field), values, atol=1e-10
    )


def test_assigning_data_refreshes_the_field_view() -> None:
    """Its `data` is the array the parser holds, so the property is
    hand-written -- and it still has to clear the views keyed on it, or
    `field` would keep serving the array that was there before."""
    grid = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), -1)
    coords = NiftiRASCoordinatesField(data=grid)
    assert coords.field is coords.field  # cached
    coords.data = grid + 10.0
    np.testing.assert_allclose(np.asarray(coords.field), grid + 10.0)


# ----------------------------------------------------------------------
#   IMMUTABILITY
# ----------------------------------------------------------------------


def test_a_displacement_field_refuses_in_place_edits(standard_warp) -> None:  # noqa: ANN001
    """Its slots name fixed positions in the chain, so the chain is a
    tuple and the field cannot be edited in place."""
    from bagof.magic import replace

    field = io.transformations.load(standard_warp)
    assert isinstance(field, xforms.ImmutableSequence)
    assert isinstance(field.transformations, tuple)
    ras2voxel = field.ras2voxel
    with pytest.raises(TypeError):
        field[0] = ras2voxel
    with pytest.raises(TypeError):
        del field[0]
    with pytest.raises(AttributeError):
        field.insert(0, ras2voxel)
    assert len(field) == 3
    assert field.ras2voxel is ras2voxel

    # A chain given as a list is frozen too.
    built = NiftiRASDisplacementField(transformations=list(field))
    assert isinstance(built.transformations, tuple)

    # Rebuilding is how a field with other slots is made: here, one whose
    # way back to RAS is shifted by 1 mm along x.
    from brainhops.io.transformations.base.affines import VoxelToRAS

    matrix = np.asarray(field.voxel2ras.matrix).copy()
    matrix[0, 3] += 1.0
    shifted = replace(
        field,
        transformations=(
            field.ras2voxel,
            field.displacement,
            VoxelToRAS(matrix=matrix),
        ),
    )
    assert type(shifted) is NiftiRASDisplacementField
    assert isinstance(shifted.transformations, tuple)
    assert field.ras2voxel is ras2voxel  # the original is untouched
    points = _grid_points().reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(shifted, points),
        _apply(field, points) + [1.0, 0.0, 0.0],
        atol=1e-4,
    )


# ----------------------------------------------------------------------
#   VELOCITIES (`log`)
# ----------------------------------------------------------------------

# A constant RAS velocity: its flow is the translation by it.
VELOCITY = np.array([1.5, -2.0, 0.5], dtype="float32")


@pytest.fixture
def velocity_warp(tmp_path):  # noqa: ANN001, ANN201
    """A `DISPVECT` file that holds a constant velocity, in mm."""
    vectors = np.zeros((*SHAPE, 3), dtype="float32") + VELOCITY
    return _write(tmp_path / "vel.nii.gz", vectors, DISPVECT)


@pytest.mark.parametrize(
    # `steps` is the count the field integrates with: the one the spec
    # names, or the one the default rule picks when it names none.
    "spec, steps",
    [
        ("{}|svf", None),
        ("{}|svf|steps:6", 6),
        ("{}|displacements|log:true", None),
        ("{}|displacements|log:TRUE|steps:3", 3),
    ],
)
def test_a_velocity_is_read_with_the_log_option(
    velocity_warp,  # noqa: ANN001
    spec: str,
    steps: object,
) -> None:
    from brainhops.io.base import TransformationSpec

    spec = TransformationSpec.from_arg(spec.format(velocity_warp))
    field = io.transformations.load(spec)
    assert type(field) is NiftiRASDisplacementField and field.log
    velocity = field.displacement
    assert type(velocity) is xforms.StationaryVelocityField
    if steps is None:
        # The file names no count, so the default rule picks one.
        assert velocity.steps is None
        steps = _tangents._squaring_steps(np.asarray(velocity.values))
    assert velocity._compute_steps == steps
    points = _grid_points()[1:3, 1:3, 1:3].reshape(-1, 3)
    np.testing.assert_allclose(
        _apply(field, points), points + VELOCITY, atol=1e-4
    )


@pytest.mark.parametrize("spec", ["{}", "{}|displacements|log:false"])
def test_log_false_reads_a_displacement(velocity_warp, spec: str) -> None:  # noqa: ANN001
    from brainhops.io.base import TransformationSpec

    field = io.transformations.load(
        TransformationSpec.from_arg(spec.format(velocity_warp))
    )
    assert not field.log
    assert type(field.displacement) is xforms.DisplacementField


def test_a_velocity_is_read_with_a_keyword(velocity_warp) -> None:  # noqa: ANN001
    field = io.transformations.load(velocity_warp, log=True, steps=4)
    assert type(field.displacement) is xforms.StationaryVelocityField
    assert field.displacement.steps == 4


def test_steps_need_a_velocity(velocity_warp) -> None:  # noqa: ANN001
    from brainhops.io.base import TransformationSpec

    spec = TransformationSpec.from_arg(
        f"{velocity_warp}|displacements|steps:3"
    )
    with pytest.raises(Exception, match="squaring steps"):
        io.transformations.load(spec)


def test_a_velocity_is_written_in_the_encoding_of_the_format(
    velocity_warp,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    velocity = io.transformations.load(velocity_warp, log=True)
    chain = tuple(velocity.transformations)
    # As a displacement, the default: the velocity is integrated.
    written = NiftiRASDisplacementField(transformations=chain).to_nibabel()
    np.testing.assert_allclose(
        np.asarray(written.dataobj)[:, :, :, 0, :],
        np.zeros((*SHAPE, 3)) + VELOCITY,
        atol=1e-4,
    )
    # As a velocity, with `log`: the velocity as it is.
    stored = NiftiRASDisplacementField(
        transformations=chain, log=True
    ).to_nibabel()
    np.testing.assert_allclose(
        np.asarray(stored.dataobj)[:, :, :, 0, :],
        np.zeros((*SHAPE, 3)) + VELOCITY,
        atol=1e-6,
    )
    # A linear velocity tells the two apart.
    sheared = chain[1].to(data=chain[1].data * _ramp()[..., :1])
    chain = (chain[0], sheared, chain[2])
    integrated = NiftiRASDisplacementField(transformations=chain)
    kept = NiftiRASDisplacementField(transformations=chain, log=True)
    assert not np.allclose(
        np.asarray(integrated.to_nibabel().dataobj),
        np.asarray(kept.to_nibabel().dataobj),
    )
