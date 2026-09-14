"""
Tests for the FSL transformation readers.

FSL expresses transformations in *scaled-mm* coordinates: voxel indices
scaled by pixel size, with the x-axis flipped when the voxel-to-world
affine has a positive determinant. These tests pin down the scaled-mm
affine, the direction of the FLIRT `.mat` adaptation, and the
relative/absolute handling of a FNIRT warp field.

The numeric expectations here were checked against fslpy in the session
that wrote this module (see the design note), and are embedded so the
tests do not depend on fslpy at run time.
"""

# stdlib
from pathlib import Path

# dependencies
import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.io.transformations.fsl import (  # noqa: E402
    FLIRTTransform,
    ImageGeometry,
)
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    VoxelToScaledMM,
)
from brainhops.io.transformations.fsl.fnirt import (  # noqa: E402
    FNIRTCoefficientField,
    FNIRTDeformationField,
)

data_dir = Path(__file__).parent / "data"

# A "neurological" reference (positive determinant, so FSL flips x) and a
# "radiological" moving image (negative determinant, no flip).
REF_AFFINE = np.array(
    [[2, 0, 0, -30], [0, 2, 0, -40], [0, 0, 2.5, -20], [0, 0, 0, 1]], float
)
MOV_AFFINE = np.array(
    [[-1.5, 0, 0, 60], [0, 1.5, 0, -10], [0, 0, 3, -25], [0, 0, 0, 1]], float
)
REF_SHAPE = (6, 7, 5)
MOV_SHAPE = (9, 8, 4)


def _image(shape, affine):  # noqa: ANN001, ANN202
    return nb.Nifti1Image(np.zeros(shape, np.float32), affine)


def _homogeneous(affine_xform):  # noqa: ANN001, ANN202
    matrix = np.eye(4)
    matrix[:3, :] = affine_xform.matrix
    return matrix


def _compose(sequence):  # noqa: ANN001, ANN202
    """The `(4, 4)` matrix of a sequence of affines, applied in order."""
    matrix = np.eye(4)
    for xform in sequence:
        matrix = _homogeneous(xform) @ matrix
    return matrix


def _apply(matrix, coords):  # noqa: ANN001, ANN202
    coords = np.asarray(coords, float)
    homog = np.concatenate([coords, np.ones(coords.shape[:-1] + (1,))], -1)
    return (homog @ matrix.T)[..., :3]


def _fsl_vox2scaled(affine, shape, pixdim):  # noqa: ANN001, ANN202
    """The FSL scaled-mm affine, written out from FSL's documented rule.

    An independent oracle for the reader: voxel indices scaled by pixel
    size, with the x-axis flipped when the voxel-to-world affine has a
    positive determinant.
    """
    matrix = np.diag(list(pixdim) + [1.0])
    if np.linalg.det(affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (shape[0] - 1) * pixdim[0]
        matrix = flip @ matrix
    return matrix


# ----------------------------------------------------------------------
#   SCALED-MM AFFINE
# ----------------------------------------------------------------------


def test_scaled_mm_flips_x_for_a_neurological_image() -> None:
    """A positive determinant flips x, with a shape-dependent offset."""
    geom = ImageGeometry(_image(REF_SHAPE, REF_AFFINE))
    assert geom.is_neurological
    expected = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    assert np.allclose(geom.vox2fsl, expected)
    # The flip offset uses (nx - 1) * pixdim_x.
    assert geom.vox2fsl[0, 0] == -2.0
    assert geom.vox2fsl[0, 3] == (REF_SHAPE[0] - 1) * 2.0


def test_scaled_mm_does_not_flip_a_radiological_image() -> None:
    """A negative determinant leaves the x-axis a plain pixdim scaling."""
    geom = ImageGeometry(_image(MOV_SHAPE, MOV_AFFINE))
    assert not geom.is_neurological
    assert np.allclose(geom.vox2fsl, np.diag([1.5, 1.5, 3.0, 1.0]))


def test_scaled_mm_affine_object_matches_geometry() -> None:
    geom = ImageGeometry(_image(REF_SHAPE, REF_AFFINE))
    xform = VoxelToScaledMM(matrix=geom.vox2fsl[:-1])
    assert np.allclose(xform.matrix, geom.vox2fsl[:-1])


# ----------------------------------------------------------------------
#   FLIRT
# ----------------------------------------------------------------------

FLIRT_MATRIX = np.array(
    [
        [0.98, 0.03, -0.02, 1.1],
        [-0.01, 0.99, 0.05, -2.0],
        [0.02, -0.04, 1.01, 0.7],
        [0, 0, 0, 1],
    ],
    float,
)

# Reference -> moving RAS matrix produced by the reader for the geometry
# and FLIRT matrix above, checked against fslpy `fromFlirt`.
EXPECTED_FLIRT_REF2MOV = np.array(
    [
        [1.019659, 0.030023, -0.021678, 82.357399],
        [-0.011297, 1.007752, -0.049665, 31.128688],
        [0.019744, 0.040505, 0.987703, -3.819509],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_flirt_adapts_to_reference_to_moving_ras() -> None:
    """The FLIRT chain maps reference RAS to moving RAS (known answer)."""
    flirt = FLIRTTransform(
        matrix=FLIRT_MATRIX,
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert np.allclose(
        _compose(flirt.transformations), EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_flirt_chain_is_the_documented_composition() -> None:
    """The chain equals ``mov.fsl2ras @ inv(M) @ ref.ras2fsl``."""
    ref_v2f = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    mov_v2f = _fsl_vox2scaled(MOV_AFFINE, MOV_SHAPE, [1.5, 1.5, 3.0])
    ref_ras2fsl = ref_v2f @ np.linalg.inv(REF_AFFINE)
    mov_fsl2ras = MOV_AFFINE @ np.linalg.inv(mov_v2f)
    oracle = mov_fsl2ras @ np.linalg.inv(FLIRT_MATRIX) @ ref_ras2fsl

    flirt = FLIRTTransform(
        matrix=FLIRT_MATRIX,
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert np.allclose(_compose(flirt.transformations), oracle)


def test_flirt_chain_steps_expose_both_images() -> None:
    """The five affines pass through both images' voxel/scaled-mm."""
    flirt = FLIRTTransform(
        matrix=FLIRT_MATRIX,
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    names = [type(t).__name__ for t in flirt.transformations]
    assert names == [
        "RASToVoxel",
        "VoxelToScaledMM",
        "ScaledMMToScaledMM",
        "ScaledMMToVoxel",
        "VoxelToRAS",
    ]


def test_flirt_requires_both_images() -> None:
    flirt = FLIRTTransform(matrix=FLIRT_MATRIX)
    with pytest.raises(ValueError, match="reference and the moving image"):
        _ = flirt.transformations


def test_flirt_repr_and_inspection_do_not_raise() -> None:
    """A FLIRT matrix with no images is printable and inspectable."""
    flirt = FLIRTTransform(matrix=FLIRT_MATRIX)
    assert "FLIRTTransform" in repr(flirt)
    assert list(flirt) == []
    assert len(flirt) == 0


def test_flirt_from_lines_accepts_an_array_moving() -> None:
    """An array-like `moving=` does not trip an ambiguous truth value."""
    lines = ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]
    moving = np.eye(4)
    flirt = FLIRTTransform.from_lines(lines, moving=moving)
    assert flirt.moving is moving
    # The `src` alias is resolved the same way.
    other = FLIRTTransform.from_lines(lines, src=moving)
    assert other.moving is moving


def test_flirt_is_dispatched_from_a_mat_file(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "src2ref.mat"
    np.savetxt(str(path), FLIRT_MATRIX, fmt="%.8g")
    assert io.transformations.sniff(path) is FLIRTTransform
    loaded = io.transformations.load(
        path,
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert type(loaded) is FLIRTTransform
    assert np.allclose(loaded.matrix, FLIRT_MATRIX)
    assert np.allclose(
        _compose(loaded.transformations), EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_a_matlab_like_text_is_not_claimed_as_flirt() -> None:
    """A non-affine block of numbers is not a FLIRT matrix."""
    assert FLIRTTransform.sniff_lines(["1 2 3", "4 5 6"]) == 0.0


# ----------------------------------------------------------------------
#   FNIRT WARP FIELD
# ----------------------------------------------------------------------


def _fnirt_setup():  # noqa: ANN202
    """Build a synthetic absolute warp and its relative counterpart.

    The absolute field stores, per reference voxel, a moving scaled-mm
    coordinate. The relative field stores the displacement from the
    reference scaled-mm coordinate of the voxel. Both must yield the same
    world-space mapping.
    """
    ref_v2f = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in REF_SHAPE], indexing="ij"), -1
    ).astype(float)
    ref_fsl = _apply(ref_v2f, grid)
    absolute = ref_fsl + 3.0 * np.sin(grid / 3.0) + 0.5
    relative = absolute - ref_fsl
    return grid, absolute, relative


def _warp(field, intent=2006):  # noqa: ANN001, ANN202
    img = nb.Nifti1Image(field.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = intent
    warp = FNIRTDeformationField.from_nibabel(img)
    warp.moving = _image(MOV_SHAPE, MOV_AFFINE)
    return warp


def _chain_at_voxels(warp, grid):  # noqa: ANN001, ANN202
    """Evaluate the warp chain at each reference voxel centre."""
    sequence = warp.transformations
    field = np.asarray(sequence[1].field)
    voxels = np.rint(grid).astype(int)
    looked_up = field[voxels[..., 0], voxels[..., 1], voxels[..., 2]]
    tail = _homogeneous(sequence[3]) @ _homogeneous(sequence[2])
    return _apply(tail, looked_up)


def test_fnirt_absolute_and_relative_agree() -> None:
    """A relative warp and its absolute form map to the same moving RAS."""
    grid, absolute, relative = _fnirt_setup()
    out_abs = _chain_at_voxels(_warp(absolute), grid)
    out_rel = _chain_at_voxels(_warp(relative), grid)
    assert np.allclose(out_abs, out_rel, atol=1e-3)


def test_fnirt_maps_to_moving_ras() -> None:
    """The stored coordinates convert to moving RAS via moving scaled-mm."""
    grid, absolute, _ = _fnirt_setup()
    mov_v2f = _fsl_vox2scaled(MOV_AFFINE, MOV_SHAPE, [1.5, 1.5, 3.0])
    mov_fsl2ras = MOV_AFFINE @ np.linalg.inv(mov_v2f)
    expected = _apply(mov_fsl2ras, absolute)
    out = _chain_at_voxels(_warp(absolute), grid)
    assert np.allclose(out, expected, atol=1e-3)


def test_fnirt_detects_absolute_and_relative() -> None:
    """The storage type is inferred from the spread of the data."""
    from brainhops.backends import get_array_backend
    from brainhops.io.transformations.fsl.fnirt._warp import (
        _detect_deformation_type,
        _voxel_grid_in_scaled_mm,
    )

    _, absolute, relative = _fnirt_setup()
    ref_v2f = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    backend = get_array_backend()
    ref_scaled = _voxel_grid_in_scaled_mm(REF_SHAPE, ref_v2f, backend)
    assert (
        _detect_deformation_type(backend.asarray(absolute), ref_scaled)
        == "absolute"
    )
    assert (
        _detect_deformation_type(backend.asarray(relative), ref_scaled)
        == "relative"
    )


def test_fnirt_deformation_type_override() -> None:
    """A caller can force the interpretation of a warp field."""
    grid, absolute, relative = _fnirt_setup()
    forced_abs = _warp(absolute)
    forced_abs.deformation_type = "absolute"
    forced_rel = _warp(relative)
    forced_rel.deformation_type = "relative"
    assert np.allclose(
        _chain_at_voxels(forced_abs, grid),
        _chain_at_voxels(forced_rel, grid),
        atol=1e-3,
    )


def test_fnirt_requires_a_moving_image() -> None:
    _, absolute, _ = _fnirt_setup()
    img = nb.Nifti1Image(absolute.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = 2006
    warp = FNIRTDeformationField.from_nibabel(img)
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = warp.transformations


def test_fnirt_warp_repr_and_inspection_do_not_raise() -> None:
    """A warp loaded without `moving=` is printable and inspectable."""
    _, absolute, _ = _fnirt_setup()
    img = nb.Nifti1Image(absolute.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = 2006
    warp = FNIRTDeformationField.from_nibabel(img)
    assert "FNIRTDeformationField" in repr(warp)
    assert list(warp) == []
    assert len(warp) == 0


def test_fnirt_deformation_type_change_is_not_cached() -> None:
    """Changing `deformation_type` after a first access changes the result."""
    _, absolute, _ = _fnirt_setup()
    warp = _warp(absolute)
    warp.deformation_type = "absolute"
    first = np.asarray(warp.transformations[1].field).copy()
    warp.deformation_type = "relative"
    second = np.asarray(warp.transformations[1].field)
    assert not np.allclose(first, second)


def test_fnirt_field_is_dispatched() -> None:
    path = data_dir / "fsl_field.nii.gz"
    assert io.transformations.sniff(path) is FNIRTDeformationField
    assert io.sniff(path) is FNIRTDeformationField
    assert type(io.transformations.load(path)) is FNIRTDeformationField


def test_fnirt_field_fixture_builds_a_chain() -> None:
    """The real dense-field fixture reads into a four-step world chain."""
    warp = io.transformations.load(data_dir / "fsl_field.nii.gz")
    warp.moving = _image((30, 30, 30), MOV_AFFINE)
    names = [type(t).__name__ for t in warp.transformations]
    assert names == [
        "RASToVoxel",
        "ScaledMMCoordinatesField",
        "ScaledMMToVoxel",
        "VoxelToRAS",
    ]


# ----------------------------------------------------------------------
#   FNIRT COEFFICIENT FIELD
# ----------------------------------------------------------------------


def test_coefficient_field_is_dispatched() -> None:
    path = data_dir / "fsl_coef.nii.gz"
    assert io.transformations.sniff(path) is FNIRTCoefficientField
    assert type(io.transformations.load(path)) is FNIRTCoefficientField


def test_coefficient_field_exposes_parameters() -> None:
    coef = io.transformations.load(data_dir / "fsl_coef.nii.gz")
    assert coef.spline_order == 3  # cubic
    assert np.allclose(coef.knot_spacing, [10.0, 10.0, 10.0])
    assert np.allclose(coef.reference_pixdim, [2.0, 2.0, 2.0])
    assert coef.initial_affine.shape == (4, 4)


def test_coefficient_field_transformations_is_deferred() -> None:
    coef = io.transformations.load(data_dir / "fsl_coef.nii.gz")
    with pytest.raises(NotImplementedError, match="B-spline basis"):
        _ = coef.transformations


def test_coefficient_field_repr_and_inspection_do_not_raise() -> None:
    """A coefficient field is printable and inspectable without resolving."""
    coef = io.transformations.load(data_dir / "fsl_coef.nii.gz")
    assert "FNIRTCoefficientField" in repr(coef)
    assert list(coef) == []
    assert len(coef) == 0
