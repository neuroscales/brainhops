"""Tests for the FSL transformation readers.

FSL works in scaled millimetres: voxel indices times the pixel size, with
x flipped when the voxel-to-world matrix has a positive determinant. The
FNIRT tests use the fslpy fixtures in data/fsl/ (see ATTRIBUTION.md there)
and compare with fslpy when it is installed.
"""

from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.backends import available_backends, backend  # noqa: E402
from brainhops.datamodel import transformations as _xforms  # noqa: E402
from brainhops.io.transformations.fsl import FlirtTransform  # noqa: E402
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    VoxelToScaledMm,
    _ImageGeometry,
)
from brainhops.io.transformations.fsl.fnirt import FnirtWarpField  # noqa: E402
from brainhops.io.transformations.fsl.fnirt._base import (  # noqa: E402
    _anchor_offsets,
    _detect_deformation_type,
    _voxel_grid_in_scaled_mm,
)

data_dir = Path(__file__).parent / "data"
fsl_dir = data_dir / "fsl"

# A neurological reference (FSL flips x) and a radiological moving image.
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
    """Compute the FSL scaled-mm affine by the documented rule."""
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
    xform = VoxelToScaledMm(matrix=geom.vox2fsl[:-1])
    assert np.allclose(xform.matrix, geom.vox2fsl[:-1])


class _StubHeader:
    """Header stub with degenerate geometry fields."""

    def __init__(self, affine, zooms, shape) -> None:  # noqa: ANN001
        self._affine, self._zooms, self._shape = affine, zooms, shape

    def get_best_affine(self) -> np.ndarray:
        return self._affine

    def get_zooms(self) -> tuple:
        return self._zooms

    def get_data_shape(self) -> tuple:
        return self._shape


def test_degenerate_header_does_not_crash() -> None:
    """A non-finite affine and zero zooms still give a finite geometry."""
    header = _StubHeader(np.full((4, 4), np.nan), (0.0, 0.0, 0.0), (4, 5, 6))
    geom = _ImageGeometry(header)
    assert np.all(np.isfinite(geom.vox2fsl))
    assert np.all(np.isfinite(geom.fsl2ras))
    # Missing zooms are replaced with one, not zero.
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

# Reference-to-moving RAS matrix, checked against fslpy fromFlirt.
EXPECTED_FLIRT_REF2MOV = np.array(
    [
        [1.019659, 0.030023, -0.021678, 82.357399],
        [-0.011297, 1.007752, -0.049665, 31.128688],
        [0.019744, 0.040505, 0.987703, -3.819509],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _flirt(**kwargs):  # noqa: ANN003, ANN202
    return FlirtTransform(flirt_matrix=FLIRT_MATRIX, **kwargs)


def test_flirt_is_an_affine() -> None:
    """A FLIRT transform is an affine, not a sequence."""
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert isinstance(flirt, _xforms.Affine)


def test_flirt_matrix_is_reference_to_moving_ras() -> None:
    """The matrix maps reference RAS to moving RAS (known answer)."""
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert np.allclose(
        flirt.homogeneous_matrix, EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_flirt_matrix_is_the_documented_composition() -> None:
    """The matrix equals mov.fsl2ras @ inv(M) @ ref.ras2fsl."""
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


def test_a_sequence_holding_flirt_inverts() -> None:
    """`Sequence.inverse` passes compute to the FLIRT inverse."""
    flirt = _flirt(
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    inverse = _xforms.Sequence([flirt]).inverse(compute=True)
    assert len(inverse) == 1
    assert np.allclose(
        inverse[0].homogeneous_matrix,
        np.linalg.inv(EXPECTED_FLIRT_REF2MOV),
        atol=1e-4,
    )


def test_flirt_requires_both_images() -> None:
    flirt = _flirt()
    with pytest.raises(ValueError, match="reference and the moving image"):
        _ = flirt.matrix


def test_flirt_repr_does_not_raise() -> None:
    flirt = _flirt()
    assert "FlirtTransform" in repr(flirt)


def test_flirt_from_lines_accepts_an_array_moving() -> None:
    lines = ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]
    moving = np.eye(4)
    flirt = FlirtTransform.from_lines(lines, moving=moving)
    assert flirt.moving is moving
    other = FlirtTransform.from_lines(lines, src=moving)
    assert other.moving is moving


@pytest.mark.parametrize("keyword", ["moving", "mov", "src"])
def test_flirt_mat_file_accepts_every_moving_alias(
    tmp_path,  # noqa: ANN001
    keyword,  # noqa: ANN001
) -> None:
    """A `.mat` file reader accepts each alias of `moving`."""
    path = tmp_path / "src2ref.mat"
    np.savetxt(str(path), FLIRT_MATRIX, fmt="%.8g")
    moving = _image(MOV_SHAPE, MOV_AFFINE)
    loaded = io.transformations.load(
        path, ref=_image(REF_SHAPE, REF_AFFINE), **{keyword: moving}
    )
    assert loaded.moving is moving
    assert np.allclose(
        loaded.homogeneous_matrix, EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_flirt_from_lines_rejects_two_moving_aliases() -> None:
    """Giving the moving image twice is an error, as in the constructor."""
    lines = ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]
    with pytest.raises(TypeError, match="multiple values"):
        FlirtTransform.from_lines(lines, moving=np.eye(4), mov=np.eye(4))


def test_flirt_is_dispatched_from_a_mat_file(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "src2ref.mat"
    np.savetxt(str(path), FLIRT_MATRIX, fmt="%.8g")
    assert io.transformations.sniff(path) is FlirtTransform
    loaded = io.transformations.load(
        path,
        reference=_image(REF_SHAPE, REF_AFFINE),
        moving=_image(MOV_SHAPE, MOV_AFFINE),
    )
    assert type(loaded) is FlirtTransform
    assert np.allclose(loaded.flirt_matrix, FLIRT_MATRIX)
    assert np.allclose(
        loaded.homogeneous_matrix, EXPECTED_FLIRT_REF2MOV, atol=1e-4
    )


def test_a_matlab_like_text_is_not_claimed_as_flirt() -> None:
    assert FlirtTransform.sniff_lines(["1 2 3", "4 5 6"]) == 0.0


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
    warp = FnirtWarpField.from_nibabel(img)
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


def test_fnirt_deformation_field_is_a_first_degree_displacement() -> None:
    _, absolute, _ = _fnirt_setup()
    warp = _warp(absolute)
    assert warp.degree == 1
    assert warp.store == "values"
    field = warp.transformations[1]
    assert type(field) is _xforms.DisplacementField
    assert field.degree == 1
    assert field.store == "values"
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
    """Changing deformation_type rebuilds the cached chain."""
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
    warp = FnirtWarpField.from_nibabel(img)
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = warp.transformations


def test_fnirt_repr_and_inspection_do_not_raise() -> None:
    _, absolute, _ = _fnirt_setup()
    img = nb.Nifti1Image(absolute.astype(np.float32), REF_AFFINE)
    img.header["intent_code"] = 2006
    warp = FnirtWarpField.from_nibabel(img)
    assert "FnirtWarpField" in repr(warp)
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
    """Deformation and coefficient fixtures are read by one class."""
    for name in ("displacementfield.nii.gz", "coefficientfield.nii.gz"):
        path = fsl_dir / name
        assert io.transformations.sniff(path) is FnirtWarpField
        assert type(io.transformations.load(path)) is FnirtWarpField


def test_generic_reader_does_not_claim_fsl_intents() -> None:
    """The generic RAS coordinates reader does not claim FSL intents."""
    from brainhops.io.transformations.nifti.fields import (
        NiftiRASCoordinatesField,
    )

    img = nb.load(str(fsl_dir / "coefficientfield.nii.gz"))
    # A score of 1.0 would mean that the generic reader claims the file.
    assert NiftiRASCoordinatesField._score_nibabel(img.header) < 1.0


def test_coefficient_field_exposes_degree_and_coeff() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    assert coef.degree == 3
    assert coef.store == "coefficients"
    # Knot spacing and reference pixel sizes come from the header.
    assert np.allclose(coef._stored_knot_spacing(), [5.0, 5.0, 5.0])
    assert np.allclose(coef._reference_pixdim(), [2.0, 2.0, 2.0])


def test_deformation_field_exposes_degree_and_store() -> None:
    warp = io.transformations.load(fsl_dir / "displacementfield.nii.gz")
    assert warp.degree == 1
    assert warp.store == "values"


def test_coefficient_field_needs_both_images() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    with pytest.raises(ValueError, match="moving image is needed"):
        _ = coef.transformations
    coef.moving = _real_src()
    with pytest.raises(ValueError, match="reference image is needed"):
        _ = coef.transformations


@pytest.mark.parametrize("keyword", ["moving", "mov", "src"])
def test_fnirt_reader_accepts_every_moving_alias(keyword) -> None:  # noqa: ANN001
    """The FNIRT reader accepts each alias of `moving`."""
    moving = _real_src()
    coef = io.transformations.load(
        fsl_dir / "coefficientfield.nii.gz",
        ref=_real_ref(),
        **{keyword: moving},
    )
    assert coef.moving is moving
    assert len(coef) == 3


def test_coefficient_field_ignores_an_invalid_deformation_type() -> None:
    """An invalid `deformation_type` does not hide the chain."""
    coef = io.transformations.load(
        fsl_dir / "coefficientfield.nii.gz",
        reference=_real_ref(),
        moving=_real_src(),
    )
    coef.deformation_type = "bogus"
    assert len(coef) == len(coef.transformations) == 3


def test_deformation_field_with_an_invalid_deformation_type_is_empty() -> None:
    """A deformation field with an invalid type has no chain to iterate."""
    warp = io.transformations.load(
        fsl_dir / "displacementfield.nii.gz", moving=_real_src()
    )
    assert len(warp) > 0
    warp.deformation_type = "bogus"
    assert len(warp) == 0


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
    assert field.degree == 3
    assert field.store == "coefficients"
    # The coefficients stay on the coarse knot grid.
    assert np.asarray(field.field).shape == (6, 13, 7, 3)
    world = _world_field(coef, _real_ref())
    assert np.all(np.isfinite(world))


def _fnirt_world_oracle(name, shape):  # noqa: ANN001, ANN202
    """Compute the fslpy world-to-world deformation, or skip without fslpy."""
    fsl_image = pytest.importorskip("fsl.data.image")
    fsl_fnirt = pytest.importorskip("fsl.transform.fnirt")
    nonlinear = pytest.importorskip("fsl.transform.nonlinear")

    fref = fsl_image.Image(str(fsl_dir / "ref.nii.gz"))
    fsrc = fsl_image.Image(str(fsl_dir / "src.nii.gz"))
    field = fsl_fnirt.readFnirt(str(fsl_dir / name), src=fsrc, ref=fref)
    world = fsl_fnirt.fromFnirt(field, "world", "world")
    return np.asarray(
        nonlinear.convertDeformationType(world, "absolute")
    ).reshape(shape)


@pytest.mark.parametrize(
    "name", ["coefficientfield.nii.gz", "displacementfield.nii.gz"]
)
def test_fnirt_fixture_matches_fslpy(name) -> None:  # noqa: ANN001
    """The reader reproduces the fslpy world-to-world deformation."""
    ref_img, src_img = _real_ref(), _real_src()
    warp = io.transformations.load(
        fsl_dir / name, reference=ref_img, moving=src_img
    )
    out = _world_field(warp, ref_img)
    oracle = _fnirt_world_oracle(name, out.shape)
    assert np.allclose(out, oracle, atol=1e-4)


def test_dct_coefficient_field_is_refused() -> None:
    img = nb.load(str(fsl_dir / "coefficientfield.nii.gz"))
    img = nb.Nifti1Image(
        np.asarray(img.dataobj, np.float32), img.affine, img.header
    )
    img.header["intent_code"] = 2008
    coef = FnirtWarpField.from_nibabel(img)
    coef.reference = _real_ref()
    coef.moving = _real_src()
    assert type(coef) is FnirtWarpField
    assert list(coef) == []
    with pytest.raises(NotImplementedError, match="discrete-cosine"):
        _ = coef.transformations


def test_coefficient_field_repr_and_inspection_do_not_raise() -> None:
    coef = io.transformations.load(fsl_dir / "coefficientfield.nii.gz")
    assert "FnirtWarpField" in repr(coef)
    assert list(coef) == []
    assert len(coef) == 0


# ----------------------------------------------------------------------
#   C1 -- spline anchor offset
# ----------------------------------------------------------------------


def test_anchor_offset_is_floor_degree_over_two() -> None:
    """The knot offset is degree // 2 for cubic and quadratic splines."""
    assert np.allclose(_anchor_offsets(3, [5, 5, 5]), [1, 1, 1])
    assert np.allclose(_anchor_offsets(2, [5, 5, 5]), [1, 1, 1])
    # A dense field (degree 1, spacing 1) has no offset.
    assert np.allclose(_anchor_offsets(1, [1, 1, 1]), [0, 0, 0])


def test_anchor_offset_is_zero_where_knot_spacing_is_one() -> None:
    """An axis with a knot spacing of one has no offset."""
    assert np.allclose(_anchor_offsets(2, [1, 5, 5]), [0, 1, 1])
    assert np.allclose(_anchor_offsets(3, [5, 1, 5]), [1, 0, 1])


# ----------------------------------------------------------------------
#   M1 -- constant boundary maps to grid-constant
# ----------------------------------------------------------------------


def test_constant_boundary_maps_to_grid_constant() -> None:
    """A constant boundary maps to the grid-constant mode of scipy."""
    from brainhops._core.bsplines import _scipy_boundary

    assert _scipy_boundary("constant") == ("grid-constant", 0.0)
    assert _scipy_boundary(0.0) == ("grid-constant", 0.0)
    # A numeric fill value is carried through as the constant.
    assert _scipy_boundary(3.5) == ("grid-constant", 3.5)
    # Other named conditions are unchanged.
    assert _scipy_boundary("nearest") == ("nearest", 0.0)
    assert _scipy_boundary("mirror") == ("mirror", 0.0)


# ----------------------------------------------------------------------
#   C2 -- folding an affine into a warp field
# ----------------------------------------------------------------------


ARRAY_BACKENDS = [
    "numpy",
    pytest.param(
        "dask",
        marks=pytest.mark.skipif(
            "dask" not in available_backends(),
            reason="dask is not installed",
        ),
    ),
]
"""Array backends on which folding is checked."""


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
def test_affine_folds_into_coefficient_field_warp_stays_correct(
    array_backend: str,
) -> None:
    """Folding the trailing affine into a coefficient field keeps the warp."""
    with backend(array_backend):
        coef = io.transformations.load(
            fsl_dir / "coefficientfield.nii.gz",
            reference=_real_ref(),
            moving=_real_src(),
        )
        _, disp, post = coef.transformations
        assert disp.store == "coefficients"

        # The affine folds into a displacement field that keeps coefficients.
        folded = post(disp).compute()
        assert type(folded) is _xforms.DisplacementField
        assert folded.store == "coefficients"

        # `compute()` folds the trailing affine, so two steps replace three.
        computed = _xforms.Sequence(
            transformations=list(coef.transformations)
        ).compute()
        names = [type(t).__name__ for t in computed.transformations]
        assert names == ["RASToWarpField", "DisplacementField"]

        # Led by a sampling grid, the full warp reproduces fslpy.
        out = _world_field(coef, _real_ref())
        oracle = _fnirt_world_oracle("coefficientfield.nii.gz", out.shape)
        assert np.allclose(out, oracle, atol=1e-4)


def test_affine_folds_into_a_dense_field_and_warp_stays_correct() -> None:
    """Folding the trailing affine into a dense field is exact."""
    warp = io.transformations.load(
        fsl_dir / "displacementfield.nii.gz",
        reference=_real_ref(),
        moving=_real_src(),
    )
    _, disp, post = warp.transformations
    assert disp.store == "values"

    folded = post(disp).compute()
    assert type(folded) is _xforms.DisplacementField
    assert folded.store == "values"

    computed = _xforms.Sequence(
        transformations=list(warp.transformations)
    ).compute()
    names = [type(t).__name__ for t in computed.transformations]
    assert names == ["RASToWarpField", "DisplacementField"]

    # Led by a sampling grid, both chains reproduce fslpy.
    out_full = _world_field(warp, _real_ref())
    out_folded = _world_field(computed, _real_ref())
    oracle = _fnirt_world_oracle("displacementfield.nii.gz", out_full.shape)
    assert np.allclose(out_full, oracle, atol=1e-4)
    assert np.allclose(out_folded, oracle, atol=1e-4)
