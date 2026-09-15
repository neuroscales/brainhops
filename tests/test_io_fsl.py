"""
Tests for the FSL transformation readers.

FSL expresses transformations in *scaled-mm* coordinates: voxel indices
scaled by pixel size, with the x-axis flipped when the voxel-to-world
affine has a positive determinant. These tests pin down the scaled-mm
affine, the FLIRT `.mat` affine, and the FNIRT warp readers.

The FNIRT numeric tests use the real `fslpy` test fixtures in
`data/fsl/`, and check the reader against `fslpy` itself where it is
installed. See `data/fsl/ATTRIBUTION.md` for the provenance and license
of those fixtures.
"""

# stdlib
from pathlib import Path

# dependencies
import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import transformations as _xforms  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    CompositionError,
    _compose,
)
from brainhops.io.transformations.fsl import FLIRTTransform  # noqa: E402
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    VoxelToScaledMM,
    _ImageGeometry,
)
from brainhops.io.transformations.fsl.fnirt import FNIRTWarpField  # noqa: E402
from brainhops.io.transformations.fsl.fnirt._base import (  # noqa: E402
    _anchor_offsets,
    _detect_deformation_type,
    _voxel_grid_in_scaled_mm,
)

data_dir = Path(__file__).parent / "data"
fsl_dir = data_dir / "fsl"

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


def _apply(matrix, coords):  # noqa: ANN001, ANN202
    coords = np.asarray(coords, float)
    homog = np.concatenate([coords, np.ones(coords.shape[:-1] + (1,))], -1)
    return (homog @ matrix.T)[..., :3]


def _fsl_vox2scaled(affine, shape, pixdim):  # noqa: ANN001, ANN202
    """The FSL scaled-mm affine, written out from FSL's documented rule."""
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
    geom = _ImageGeometry(_image(REF_SHAPE, REF_AFFINE))
    assert geom.is_neurological
    expected = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    assert np.allclose(geom.vox2fsl, expected)
    assert geom.vox2fsl[0, 0] == -2.0
    assert geom.vox2fsl[0, 3] == (REF_SHAPE[0] - 1) * 2.0


def test_scaled_mm_does_not_flip_a_radiological_image() -> None:
    geom = _ImageGeometry(_image(MOV_SHAPE, MOV_AFFINE))
    assert not geom.is_neurological
    assert np.allclose(geom.vox2fsl, np.diag([1.5, 1.5, 3.0, 1.0]))


def test_scaled_mm_affine_object_matches_geometry() -> None:
    geom = _ImageGeometry(_image(REF_SHAPE, REF_AFFINE))
    xform = VoxelToScaledMM(matrix=geom.vox2fsl[:-1])
    assert np.allclose(xform.matrix, geom.vox2fsl[:-1])


class _StubHeader:
    """A header-like object with degenerate geometry fields."""

    def __init__(self, affine, zooms, shape) -> None:  # noqa: ANN001
        self._affine, self._zooms, self._shape = affine, zooms, shape

    def get_best_affine(self) -> np.ndarray:
        return self._affine

    def get_zooms(self) -> tuple:
        return self._zooms

    def get_data_shape(self) -> tuple:
        return self._shape


def test_degenerate_header_does_not_crash() -> None:
    """A non-finite affine and missing zooms still yield a finite geometry."""
    header = _StubHeader(np.full((4, 4), np.nan), (0.0, 0.0, 0.0), (4, 5, 6))
    geom = _ImageGeometry(header)
    assert np.all(np.isfinite(geom.vox2fsl))
    assert np.all(np.isfinite(geom.fsl2ras))
    # Missing zooms are replaced with one, not left at zero.
    assert np.all(geom.pixdim == 1.0)


def test_missing_zoom_axis_is_replaced_with_one() -> None:
    """A header with fewer than three zooms is padded, not truncated."""
    header = _StubHeader(np.eye(4), (2.0,), (4, 5, 6))
    geom = _ImageGeometry(header)
    assert np.allclose(geom.pixdim, [2.0, 1.0, 1.0])


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

# Reference -> moving RAS matrix produced by the reader, checked against
# fslpy `fromFlirt`.
EXPECTED_FLIRT_REF2MOV = np.array(
    [
        [1.019659, 0.030023, -0.021678, 82.357399],
        [-0.011297, 1.007752, -0.049665, 31.128688],
        [0.019744, 0.040505, 0.987703, -3.819509],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _flirt(**kwargs):  # noqa: ANN003, ANN202
    return FLIRTTransform(flirt_matrix=FLIRT_MATRIX, **kwargs)


def test_flirt_is_an_affine() -> None:
    """A FLIRT transform is an affine, not a sequence."""
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert isinstance(flirt, _xforms.Affine)


def test_flirt_matrix_is_reference_to_moving_ras() -> None:
    """The computed matrix maps reference RAS to moving RAS (known answer)."""
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert np.allclose(
        flirt.homogeneous_matrix, EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_flirt_matrix_is_the_documented_composition() -> None:
    """The matrix equals ``mov.fsl2ras @ inv(M) @ ref.ras2fsl``."""
    ref_v2f = _fsl_vox2scaled(REF_AFFINE, REF_SHAPE, [2.0, 2.0, 2.5])
    mov_v2f = _fsl_vox2scaled(MOV_AFFINE, MOV_SHAPE, [1.5, 1.5, 3.0])
    ref_ras2fsl = ref_v2f @ np.linalg.inv(REF_AFFINE)
    mov_fsl2ras = MOV_AFFINE @ np.linalg.inv(mov_v2f)
    oracle = mov_fsl2ras @ np.linalg.inv(FLIRT_MATRIX) @ ref_ras2fsl
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert np.allclose(flirt.homogeneous_matrix, oracle)


def test_flirt_requires_both_images() -> None:
    flirt = _flirt()
    with pytest.raises(ValueError, match="reference and the moving image"):
        _ = flirt.matrix


def test_flirt_repr_does_not_raise() -> None:
    flirt = _flirt()
    assert "FLIRTTransform" in repr(flirt)


def test_flirt_from_lines_accepts_an_array_moving() -> None:
    lines = ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]
    moving = np.eye(4)
    flirt = FLIRTTransform.from_lines(lines, moving=moving)
    assert flirt.moving is moving
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
    assert np.allclose(loaded.flirt_matrix, FLIRT_MATRIX)
    assert np.allclose(
        loaded.homogeneous_matrix, EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_a_matlab_like_text_is_not_claimed_as_flirt() -> None:
    assert FLIRTTransform.sniff_lines(["1 2 3", "4 5 6"]) == 0.0


# ----------------------------------------------------------------------
#   FNIRT (synthetic deformation field)
# ----------------------------------------------------------------------


def _fnirt_setup():  # noqa: ANN202
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
    warp = FNIRTWarpField.from_nibabel(img)
    warp.moving = _image(MOV_SHAPE, MOV_AFFINE)
    return warp


def _world_field(warp, reference):  # noqa: ANN001, ANN202
    ref = _ImageGeometry(reference)
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
    _, absolute, relative = _fnirt_setup()
    ref = _image(REF_SHAPE, REF_AFFINE)
    out_abs = _world_field(_warp(absolute), ref)
    out_rel = _world_field(_warp(relative), ref)
    assert np.allclose(out_abs, out_rel, atol=1e-3)


def test_fnirt_maps_to_moving_ras() -> None:
    _, absolute, _ = _fnirt_setup()
    mov_v2f = _fsl_vox2scaled(MOV_AFFINE, MOV_SHAPE, [1.5, 1.5, 3.0])
    mov_fsl2ras = MOV_AFFINE @ np.linalg.inv(mov_v2f)
    expected = _apply(mov_fsl2ras, absolute)
    out = _world_field(_warp(absolute), _image(REF_SHAPE, REF_AFFINE))
    assert np.allclose(out, expected, atol=1e-3)


def test_fnirt_deformation_field_is_a_first_order_displacement() -> None:
    _, absolute, _ = _fnirt_setup()
    warp = _warp(absolute)
    assert warp.order == 1
    assert warp.coeff is False
    field = warp.transformations[1]
    assert type(field) is _xforms.DisplacementField
    assert field.order == 1
    assert field.coeff is False
    assert np.asarray(field.field).shape == REF_SHAPE + (3,)


def test_fnirt_detects_absolute_and_relative() -> None:
    from brainhops.backends import get_array_backend

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
    _, absolute, relative = _fnirt_setup()
    ref = _image(REF_SHAPE, REF_AFFINE)
    forced_abs = _warp(absolute)
    forced_abs.deformation_type = "absolute"
    forced_rel = _warp(relative)
    forced_rel.deformation_type = "relative"
    assert np.allclose(
        _world_field(forced_abs, ref), _world_field(forced_rel, ref), atol=1e-3
    )


def test_fnirt_deformation_type_change_is_reflected() -> None:
    """Changing `deformation_type` rebuilds the cached chain."""
    _, absolute, _ = _fnirt_setup()
    warp = _warp(absolute)
    warp.deformation_type = "absolute"
    first = np.asarray(warp.transformations[1].field).copy()
    warp.deformation_type = "relative"
    second = np.asarray(warp.transformations[1].field)
    assert not np.allclose(first, second)


def test_fnirt_requires_a_moving_image() -> None:
    _, absolute, _ = _fnirt_setup()
    img = nb.Nifti1Image(absolute.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = 2006
    warp = FNIRTWarpField.from_nibabel(img)
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = warp.transformations


def test_fnirt_repr_and_inspection_do_not_raise() -> None:
    _, absolute, _ = _fnirt_setup()
    img = nb.Nifti1Image(absolute.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = 2006
    warp = FNIRTWarpField.from_nibabel(img)
    assert "FNIRTWarpField" in repr(warp)
    assert list(warp) == []
    assert len(warp) == 0


# ----------------------------------------------------------------------
#   FNIRT (real fixtures)
# ----------------------------------------------------------------------


def _real_ref():  # noqa: ANN202
    return nb.load(str(fsl_dir / "ref.nii.gz"))


def _real_src():  # noqa: ANN202
    return nb.load(str(fsl_dir / "src.nii.gz"))


def test_both_fnirt_fixtures_dispatch_to_one_reader() -> None:
    """Deformation (2006) and coefficient (2007) both read as one class."""
    for name in ("displacementfield.nii.gz", "coefficientfield.nii.gz"):
        path = fsl_dir / name
        assert io.transformations.sniff(path) is FNIRTWarpField
        assert type(io.transformations.load(path)) is FNIRTWarpField


def test_generic_reader_does_not_claim_fsl_intents() -> None:
    """The generic RAS-coordinates reader no longer sniffs FSL intents."""
    from brainhops.io.transformations.nifti.fields import (
        NiftiRASCoordinatesField,
    )

    img = nb.load(str(fsl_dir / "coefficientfield.nii.gz"))
    # A CERTAIN score would mean it is still claiming the FSL intent.
    assert NiftiRASCoordinatesField._score_nibabel(img.header) < 1.0


def test_coefficient_field_exposes_order_and_coeff() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    assert coef.order == 3  # cubic
    assert coef.coeff is True
    # The stored knot spacing and reference pixel sizes are read from the
    # header for the chain, in reference voxels.
    assert np.allclose(coef._stored_knot_spacing(), [5.0, 5.0, 5.0])
    assert np.allclose(coef._reference_pixdim(), [2.0, 2.0, 2.0])


def test_deformation_field_exposes_order_and_coeff() -> None:
    warp = io.transformations.load(fsl_dir / "displacementfield.nii.gz")
    assert warp.order == 1
    assert warp.coeff is False


def test_coefficient_field_needs_both_images() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = coef.transformations
    coef.moving = _real_src()
    with pytest.raises(ValueError, match="reference image is needed"):
        _ = coef.transformations


def test_coefficient_field_chain_shape() -> None:
    coef = io.transformations.load(
        fsl_dir / "coefficientfield.nii.gz",
        reference=_real_ref(),
        moving=_real_src(),
    )
    chain = coef.transformations
    names = [type(t).__name__ for t in chain]
    assert names == ["RASToWarpField", "DisplacementField", "WarpFieldToRAS"]
    field = chain[1]
    assert type(field) is _xforms.DisplacementField
    assert field.order == 3
    assert field.coeff is True
    # The coefficients stay on the coarse knot grid.
    assert np.asarray(field.field).shape == (6, 13, 7, 3)
    world = _world_field(coef, _real_ref())
    assert np.all(np.isfinite(world))


@pytest.mark.parametrize(
    "name", ["coefficientfield.nii.gz", "displacementfield.nii.gz"]
)
def test_fnirt_fixture_matches_fslpy(name) -> None:  # noqa: ANN001
    """The reader reproduces fslpy's world-to-world FNIRT deformation."""
    fsl_image = pytest.importorskip("fsl.data.image")
    fsl_fnirt = pytest.importorskip("fsl.transform.fnirt")
    nonlinear = pytest.importorskip("fsl.transform.nonlinear")

    ref_img, src_img = _real_ref(), _real_src()
    warp = io.transformations.load(
        fsl_dir / name, reference=ref_img, moving=src_img
    )
    out = _world_field(warp, ref_img)

    fref = fsl_image.Image(str(fsl_dir / "ref.nii.gz"))
    fsrc = fsl_image.Image(str(fsl_dir / "src.nii.gz"))
    field = fsl_fnirt.readFnirt(str(fsl_dir / name), src=fsrc, ref=fref)
    world = fsl_fnirt.fromFnirt(field, "world", "world")
    oracle = np.asarray(
        nonlinear.convertDeformationType(world, "absolute")
    ).reshape(out.shape)
    assert np.allclose(out, oracle, atol=1e-4)


def test_dct_coefficient_field_is_refused() -> None:
    img = nb.load(str(fsl_dir / "coefficientfield.nii.gz"))
    img = nb.Nifti1Image(
        np.asarray(img.dataobj, np.float32), img.affine, img.header
    )
    img.header["intent_code"] = 2008
    coef = FNIRTWarpField.from_nibabel(img)
    coef.reference = _real_ref()
    coef.moving = _real_src()
    assert type(coef) is FNIRTWarpField
    assert list(coef) == []
    with pytest.raises(NotImplementedError, match="discrete-cosine"):
        _ = coef.transformations


def test_coefficient_field_repr_and_inspection_do_not_raise() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    assert "FNIRTWarpField" in repr(coef)
    assert list(coef) == []
    assert len(coef) == 0


# ----------------------------------------------------------------------
#   C1 -- spline anchor offset
# ----------------------------------------------------------------------


def test_anchor_offset_is_floor_order_over_two() -> None:
    """The knot offset is `order // 2`, the same for cubic and quadratic."""
    assert np.allclose(_anchor_offsets(3, [5, 5, 5]), [1, 1, 1])
    assert np.allclose(_anchor_offsets(2, [5, 5, 5]), [1, 1, 1])
    # A dense field (order 1, spacing 1) has no offset.
    assert np.allclose(_anchor_offsets(1, [1, 1, 1]), [0, 0, 0])


def test_anchor_offset_is_zero_where_knot_spacing_is_one() -> None:
    """An axis with unit knot spacing has no coarse grid, so no offset."""
    assert np.allclose(_anchor_offsets(2, [1, 5, 5]), [0, 1, 1])
    assert np.allclose(_anchor_offsets(3, [5, 1, 5]), [1, 0, 1])


# ----------------------------------------------------------------------
#   M1 -- constant boundary maps to grid-constant
# ----------------------------------------------------------------------


def test_constant_boundary_maps_to_grid_constant() -> None:
    """The constant condition uses scipy's zero-padding grid-constant mode."""
    from brainhops._core.bsplines import _scipy_boundary

    assert _scipy_boundary("constant") == ("grid-constant", 0.0)
    assert _scipy_boundary(0.0) == ("grid-constant", 0.0)
    # A numeric fill value is carried through as the constant.
    assert _scipy_boundary(3.5) == ("grid-constant", 3.5)
    # Other named conditions are unchanged.
    assert _scipy_boundary("nearest") == ("nearest", 0.0)
    assert _scipy_boundary("mirror") == ("mirror", 0.0)


# ----------------------------------------------------------------------
#   C2 -- composing an affine with a warp field
# ----------------------------------------------------------------------


def test_affine_does_not_fold_into_a_coefficient_field() -> None:
    """A coefficient field stays uncomposed rather than being corrupted."""
    coef = io.transformations.load(
        fsl_dir / "coefficientfield.nii.gz",
        reference=_real_ref(),
        moving=_real_src(),
    )
    _, disp, post = coef.transformations
    assert disp.coeff is True
    with pytest.raises(CompositionError, match="coefficient"):
        _ = _compose(post, disp)
    # compute() therefore keeps the three-step chain intact.
    computed = _xforms.Sequence(
        transformations=list(coef.transformations)
    ).compute()
    names = [type(t).__name__ for t in computed.transformations]
    assert names == ["RASToWarpField", "DisplacementField", "WarpFieldToRAS"]


def test_affine_folds_into_a_dense_field_and_propagates_attributes() -> None:
    """Folding a dense field keeps its order, bound and coeff flag."""
    warp = io.transformations.load(
        fsl_dir / "displacementfield.nii.gz",
        reference=_real_ref(),
        moving=_real_src(),
    )
    _, disp, post = warp.transformations
    folded = _compose(post, disp)
    assert type(folded) is _xforms.DisplacementField
    assert folded.order == disp.order
    assert folded.bound == disp.bound
    assert folded.coeff is False
