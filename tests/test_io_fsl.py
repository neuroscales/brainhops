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
from brainhops.datamodel import transformations as _xforms  # noqa: E402
from brainhops.io.transformations.fsl import FLIRTTransform  # noqa: E402
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    ImageGeometry,
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


def _world_field(warp, reference=None):  # noqa: ANN001, ANN202
    """The moving-RAS coordinate each reference voxel maps to.

    The warp is a lazy sequence that maps reference RAS to moving RAS.
    This feeds it the reference RAS coordinate of every voxel and computes
    the sequence, which evaluates the spline basis of the warp field.
    """
    if reference is None:
        reference = _image(REF_SHAPE, REF_AFFINE)
    ref = ImageGeometry(reference)
    shape = ref.shape
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1
    ).astype(float)
    ras = _apply(ref.vox2ras, grid)
    sequence = _xforms.Sequence(
        transformations=[_xforms.CoordinatesField(field=ras)]
        + list(warp.transformations)
    )
    return np.asarray(sequence.compute().field)


def test_fnirt_absolute_and_relative_agree() -> None:
    """A relative warp and its absolute form map to the same moving RAS."""
    _, absolute, relative = _fnirt_setup()
    out_abs = _world_field(_warp(absolute))
    out_rel = _world_field(_warp(relative))
    assert np.allclose(out_abs, out_rel, atol=1e-3)


def test_fnirt_maps_to_moving_ras() -> None:
    """The stored coordinates convert to moving RAS via moving scaled-mm."""
    _, absolute, _ = _fnirt_setup()
    mov_v2f = _fsl_vox2scaled(MOV_AFFINE, MOV_SHAPE, [1.5, 1.5, 3.0])
    mov_fsl2ras = MOV_AFFINE @ np.linalg.inv(mov_v2f)
    expected = _apply(mov_fsl2ras, absolute)
    out = _world_field(_warp(absolute))
    assert np.allclose(out, expected, atol=1e-3)


def test_fnirt_deformation_field_is_a_displacement_field() -> None:
    """A dense warp reads as a first-order displacement field on its grid."""
    _, absolute, _ = _fnirt_setup()
    field = _warp(absolute).transformations[1]
    assert type(field) is _xforms.DisplacementField
    assert field.order == 1
    assert field.coeff is False
    assert np.asarray(field.field).shape == REF_SHAPE + (3,)


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
    _, absolute, relative = _fnirt_setup()
    forced_abs = _warp(absolute)
    forced_abs.deformation_type = "absolute"
    forced_rel = _warp(relative)
    forced_rel.deformation_type = "relative"
    assert np.allclose(
        _world_field(forced_abs),
        _world_field(forced_rel),
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
    """The real dense-field fixture reads into a three-step world chain."""
    warp = io.transformations.load(data_dir / "fsl_field.nii.gz")
    warp.moving = _image((30, 30, 30), MOV_AFFINE)
    names = [type(t).__name__ for t in warp.transformations]
    assert names == [
        "RASToWarpField",
        "DisplacementField",
        "WarpFieldToRAS",
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


# The reference geometry the coefficient fixture was generated for: a
# 2 mm reference (its pixel size, stored in the intent parameters, is 2).
COEF_REF_AFFINE = np.array(
    [[-2, 0, 0, 40], [0, 2, 0, -40], [0, 0, 2, -40], [0, 0, 0, 1]], float
)
COEF_REF_SHAPE = (40, 40, 40)


def test_coefficient_field_needs_both_images() -> None:
    """Placing a coefficient field needs the reference and moving images."""
    coef = io.transformations.load(data_dir / "fsl_coef.nii.gz")
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = coef.transformations
    coef.moving = _image((30, 30, 30), MOV_AFFINE)
    with pytest.raises(ValueError, match="reference image is needed"):
        _ = coef.transformations


def test_coefficient_field_resolves_to_a_usable_transform() -> None:
    """A cubic coefficient field reads into a spline displacement chain."""
    coef = io.transformations.load(
        data_dir / "fsl_coef.nii.gz",
        reference=_image(COEF_REF_SHAPE, COEF_REF_AFFINE),
        moving=_image((30, 30, 30), MOV_AFFINE),
    )
    chain = coef.transformations
    names = [type(t).__name__ for t in chain]
    assert names == ["RASToWarpField", "DisplacementField", "WarpFieldToRAS"]
    field = chain[1]
    # The coefficients stay on the coarse knot grid, evaluated as a
    # third-order spline of coefficients when the chain is computed.
    assert type(field) is _xforms.DisplacementField
    assert field.order == 3
    assert field.coeff is True
    assert np.asarray(field.field).shape == (11, 11, 11, 3)
    # The chain computes to a finite moving-RAS field over the reference.
    world = _world_field(
        coef, reference=_image(COEF_REF_SHAPE, COEF_REF_AFFINE)
    )
    assert np.all(np.isfinite(world))


def test_coefficient_field_matches_fnirtfileutils() -> None:
    """Evaluating the spline reproduces fslpy's expanded deformation."""
    fsl_image = pytest.importorskip("fsl.data.image")
    fsl_fnirt = pytest.importorskip("fsl.transform.fnirt")

    ref_img = _image(COEF_REF_SHAPE, COEF_REF_AFFINE)
    mov_img = _image((30, 30, 30), MOV_AFFINE)
    coef = io.transformations.load(
        data_dir / "fsl_coef.nii.gz", reference=ref_img, moving=mov_img
    )
    out = _world_field(coef, reference=ref_img)

    # fslpy's fnirtfileutils-equivalent expansion, taken to moving RAS.
    fref = fsl_image.Image(
        np.zeros(COEF_REF_SHAPE, np.float32), xform=COEF_REF_AFFINE
    )
    fsrc = fsl_image.Image(
        np.zeros((30, 30, 30), np.float32), xform=MOV_AFFINE
    )
    field = fsl_fnirt.readFnirt(
        str(data_dir / "fsl_coef.nii.gz"), src=fsrc, ref=fref
    )
    absolute = field.asDeformationField(defType="absolute", premat=True)
    src_fsl = np.asarray(absolute.data).reshape(-1, 3)
    mov = ImageGeometry(mov_img)
    oracle = _apply(mov.fsl2ras, src_fsl).reshape(COEF_REF_SHAPE + (3,))
    assert np.allclose(out, oracle, atol=1e-4)


def test_dct_coefficient_field_is_refused() -> None:
    """A discrete-cosine coefficient field is recognized but refused."""
    img = nb.load(str(data_dir / "fsl_coef.nii.gz"))
    img = nb.Nifti1Image(
        np.asarray(img.dataobj, np.float32), img.affine, img.header
    )
    img.header["intent_code"] = 2008
    coef = FNIRTCoefficientField.from_nibabel(img)
    coef.reference = _image(COEF_REF_SHAPE, COEF_REF_AFFINE)
    coef.moving = _image((30, 30, 30), MOV_AFFINE)
    # Recognized as a coefficient field, and inspectable without raising.
    assert type(coef) is FNIRTCoefficientField
    assert list(coef) == []
    with pytest.raises(NotImplementedError, match="discrete-cosine"):
        _ = coef.transformations


def test_coefficient_field_repr_and_inspection_do_not_raise() -> None:
    """A coefficient field is printable and inspectable without resolving."""
    coef = io.transformations.load(data_dir / "fsl_coef.nii.gz")
    assert "FNIRTCoefficientField" in repr(coef)
    assert list(coef) == []
    assert len(coef) == 0
