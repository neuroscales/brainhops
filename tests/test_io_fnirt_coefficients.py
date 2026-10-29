"""Tests for FNIRT spline coefficient files, which FNIRT writes with --cout.

A coefficient file holds the coefficients of a quadratic or cubic
B-spline on a coarse grid of knots over the reference image, and the
header holds what is needed to evaluate them. The tests build such files
by hand, as `common_coef_construction` in
`src/warpfns/fnirt_file_writer.cpp` of FSL 5.0.9 writes them, and compare
the reader with an evaluation written here from the formulas of FSL
(https://github.com/neurodebian/fsl at commit 513f6a0).
"""

import gzip

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from brainhops.datamodel import transformations as _xforms  # noqa: E402
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    ImageGeometry,
)
from brainhops.io.transformations.fsl.fnirt import FnirtWarpField  # noqa: E402

CUBIC = 2007
QUADRATIC = 2009

REF_SHAPE = (11, 9, 8)
REF_PIXDIM = (2.0, 2.5, 3.0)
MOV_AFFINE = np.diag([-1.7, 1.9, 2.1, 1.0])

# An initial affine that is not the identity, as FNIRT stores it in the
# sform of the coefficient file.
PREMAT = np.array(
    [
        [1.05, 0.02, 0.0, 1.5],
        [-0.03, 0.97, 0.05, -2.0],
        [0.0, 0.01, 1.02, 0.7],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# ----------------------------------------------------------------------
#   EVALUATION BY THE FORMULAS OF FSL
# ----------------------------------------------------------------------


def _kernel(degree, x):  # noqa: ANN001, ANN202
    """Return the centred B-spline of FSL (`Spline1D::kernel_value`)."""
    ax = np.abs(x)
    if degree == 3:
        inner = 2.0 / 3.0 + ax * ax * (ax / 2.0 - 1.0)
        outer = (2.0 - ax) ** 3 / 6.0
        return np.where(ax <= 1.0, inner, np.where(ax < 2.0, outer, 0.0))
    inner = 0.75 - ax * ax
    outer = 0.5 * (1.5 - ax) ** 2
    return np.where(ax <= 0.5, inner, np.where(ax < 1.5, outer, 0.0))


def _centre(index, ksp, degree):  # noqa: ANN001, ANN202
    """Return the voxel of a coefficient (`Spline1D::OffsetIntoField`)."""
    if ksp == 1:
        return index * ksp
    return (index - degree // 2) * ksp


def _kernel_size(degree, ksp):  # noqa: ANN001, ANN202
    """Return the support of a spline in voxels (`Spline1D::spline_size`)."""
    if degree % 2 == 0 and ksp % 2 == 1:
        return (degree + 1) * ksp
    return (degree + 1) * ksp - 1


def _ncoef(size, ksp, degree):  # noqa: ANN001, ANN202
    """Return the number of coefficients of an axis (`Spline1D::NCoef`)."""
    if ksp == 1:
        return size
    if degree == 3:
        return int(np.ceil((size + 1) / ksp)) + 2
    count = size // ksp + degree // 2
    half = _kernel_size(degree, ksp) // 2
    while _centre(count, ksp, degree) - half < size - 1:
        count += 1
    return count


def _fsl_displacements(coef, ksp, degree, voxels):  # noqa: ANN001, ANN202
    """Evaluate the field at voxels of the reference as FSL holds it.

    Every coefficient of the grid contributes, and no other, which is the
    sum of `splinefield::Peek` in `src/basisfield/splinefield.cpp`.
    """
    weights = []
    for axis in range(3):
        index = np.arange(coef.shape[axis])
        centre = _centre(index, ksp[axis], degree)
        offset = voxels[:, axis, None] - centre[None, :]
        weights.append(_kernel(degree, offset / ksp[axis]))
    return np.einsum("ni,nj,nk,ijkd->nd", *weights, coef)


def _fsl_world(coef, ksp, degree, premat, reference, stored_pixdim):  # noqa: ANN001, ANN202
    """Map every voxel of the reference to the moving RAS space as FSL.

    FSL holds the reference in the radiological order, so a voxel in
    memory is the scaled-mm position divided by the voxel size of the
    reference that FNIRT registered to. The displacement is added to the
    position moved by the inverse of the initial affine
    (`FnirtFileReader::add_affine_part` in
    `src/warpfns/fnirt_file_reader.cpp`).
    """
    ref = ImageGeometry(reference)
    mov = ImageGeometry(_image(MOV_AFFINE, (10, 10, 10)))
    voxels = _grid(ref.shape)
    scaled = _apply(ref.vox2fsl, voxels)
    disp = _fsl_displacements(
        coef, ksp, degree, scaled / np.asarray(stored_pixdim)
    )
    source = _apply(np.linalg.inv(premat), scaled) + disp
    return _apply(mov.fsl2ras, source)


# ----------------------------------------------------------------------
#   FILES AND HELPERS
# ----------------------------------------------------------------------


def _image(affine, shape):  # noqa: ANN001, ANN202
    return nb.Nifti1Image(np.zeros(shape, np.float32), affine)


def _reference(neurological, pixdim=REF_PIXDIM, shape=REF_SHAPE):  # noqa: ANN001, ANN202
    affine = np.diag(list(pixdim) + [1.0])
    affine[:3, 3] = [-10.0, -12.0, -9.0]
    if not neurological:
        affine[0, 0] = -affine[0, 0]
    return _image(affine, shape)


def _grid(shape):  # noqa: ANN001, ANN202
    axes = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    return np.stack(axes, -1).reshape(-1, 3).astype(float)


def _apply(matrix, points):  # noqa: ANN001, ANN202
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _coefficients(degree, ksp, seed=0):  # noqa: ANN001, ANN202
    shape = tuple(_ncoef(size, k, degree) for size, k in zip(REF_SHAPE, ksp))
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape + (3,)).astype(np.float32)


def _coefficient_file(coef, degree, ksp, premat=None):  # noqa: ANN001, ANN202
    """Build a coefficient file with the header that FNIRT writes.

    The intent gives the degree, its parameters the voxel sizes of the
    reference, the pixdims the knot spacing, the offsets of the qform the
    shape of the reference, and the sform the initial affine.
    """
    image = nb.Nifti1Image(coef, None)
    header = image.header
    header["intent_code"] = CUBIC if degree == 3 else QUADRATIC
    header["intent_p1"], header["intent_p2"], header["intent_p3"] = REF_PIXDIM
    header.set_zooms(tuple(float(k) for k in ksp) + (1.0,))
    qform = np.diag([float(k) for k in ksp] + [1.0])
    qform[:3, 3] = REF_SHAPE
    image.set_qform(qform, 1)
    image.set_sform(np.eye(4) if premat is None else premat, 1)
    return image


def _world(warp, reference):  # noqa: ANN001, ANN202
    """Map every voxel of the reference to the moving RAS space."""
    ref = ImageGeometry(reference)
    ras = _apply(ref.vox2ras, _grid(ref.shape)).reshape(ref.shape + (3,))
    sequence = _xforms.Sequence(
        transformations=[_xforms.CoordinateField(field=ras)]
        + list(warp.transformations)
    )
    return np.asarray(sequence.compute().field).reshape(-1, 3)


def _warp(image, reference):  # noqa: ANN001, ANN202
    warp = FnirtWarpField.from_nibabel(image)
    warp.reference = reference
    warp.moving = _image(MOV_AFFINE, (10, 10, 10))
    return warp


# ----------------------------------------------------------------------
#   EVALUATION
# ----------------------------------------------------------------------


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("ksp", [(3, 4, 2), (1, 1, 1), (1, 3, 2)])
@pytest.mark.parametrize("neurological", [False, True])
@pytest.mark.parametrize("initial", ["identity", "oblique"])
def test_the_field_is_evaluated_as_fsl_evaluates_it(
    degree: int,
    ksp: tuple,
    neurological: bool,
    initial: str,
) -> None:
    """The chain gives the positions that the formulas of FSL give."""
    premat = np.eye(4) if initial == "identity" else PREMAT
    coef = _coefficients(degree, ksp)
    reference = _reference(neurological)
    warp = _warp(_coefficient_file(coef, degree, ksp, premat), reference)
    assert warp.degree == degree
    # The sform holds the initial affine in single precision.
    stored = np.asarray(warp.metadata.raw.header.get_sform(), float)
    expected = _fsl_world(
        coef.astype(float), ksp, degree, stored, reference, REF_PIXDIM
    )
    assert np.allclose(_world(warp, reference), expected, atol=1e-5)


def test_the_knots_follow_the_radiological_order_of_fsl() -> None:
    """A neurological reference reads the grid of knots from its right.

    FSL flips such a reference to the radiological order before it
    registers, and writes the coefficients without flipping them back, so
    the first coefficient lies at the last voxel of the file along x.
    """
    ksp = (3, 4, 2)
    coef = _coefficients(3, ksp)
    radiological = _warp(_coefficient_file(coef, 3, ksp), _reference(False))
    neurological = _warp(_coefficient_file(coef, 3, ksp), _reference(True))
    first = radiological.transformations
    second = neurological.transformations
    # Both references give the same displacements on the same grid, and
    # only the place of the grid in the reference differs.
    assert np.array_equal(
        np.asarray(first[1].field), np.asarray(second[1].field)
    )
    corner = np.array([[-10.0, -12.0, -9.0]])
    ras_to_grid = np.vstack([second[0].matrix, [0.0, 0.0, 0.0, 1.0]])
    assert _apply(ras_to_grid, corner)[0, 0] == pytest.approx(
        (REF_SHAPE[0] - 1) / ksp[0] + 1
    )


def test_a_reference_of_another_resolution_keeps_the_knot_offset() -> None:
    """The knots stay where FSL put them for a coarser reference.

    The knot spacing of two voxels becomes one voxel of a reference whose
    voxels are twice as large, and the offset of the knots must still be
    applied, because it depends on the knot spacing that FNIRT used.
    """
    ksp = (2, 2, 2)
    coef = _coefficients(3, ksp)
    pixdim = tuple(2 * p for p in REF_PIXDIM)
    shape = tuple(s // 2 for s in REF_SHAPE)
    reference = _reference(False, pixdim, shape)
    warp = _warp(_coefficient_file(coef, 3, ksp), reference)
    expected = _fsl_world(
        coef.astype(float), ksp, 3, np.eye(4), reference, REF_PIXDIM
    )
    assert np.allclose(_world(warp, reference), expected, atol=1e-5)


def test_the_coefficients_stay_on_their_grid() -> None:
    """The field of the chain holds the coefficients, unevaluated."""
    ksp = (3, 4, 2)
    coef = _coefficients(3, ksp)
    warp = _warp(_coefficient_file(coef, 3, ksp), _reference(False))
    field = warp.transformations[1]
    assert type(field) is _xforms.DisplacementField
    assert field.store == "coefficients"
    assert field.degree == 3
    assert np.asarray(field.field).shape == coef.shape


def test_the_knot_spacing_is_rounded_to_whole_voxels() -> None:
    """FSL rounds the pixdims of a coefficient file to whole voxels."""
    ksp = (3, 4, 2)
    coef = _coefficients(3, ksp)
    reference = _reference(False)
    image = _coefficient_file(coef, 3, ksp)
    image.header.set_zooms((2.8, 4.3, 1.6, 1.0))
    warp = _warp(image, reference)
    expected = _fsl_world(
        coef.astype(float), ksp, 3, np.eye(4), reference, REF_PIXDIM
    )
    assert np.allclose(_world(warp, reference), expected, atol=1e-5)


@pytest.mark.parametrize("name", ["intent_p2", "pixdim"])
def test_a_header_without_spline_parameters_has_no_chain(name) -> None:  # noqa: ANN001
    """A knot spacing or a voxel size that is not positive is refused."""
    ksp = (3, 4, 2)
    image = _coefficient_file(_coefficients(3, ksp), 3, ksp)
    if name == "pixdim":
        image.header["pixdim"][2] = 0.0
    else:
        image.header[name] = 0.0
    warp = _warp(image, _reference(False))
    assert len(warp) == 0
    with pytest.raises(ValueError, match="not positive"):
        _ = warp.transformations


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_an_untouched_coefficient_file_is_saved_byte_for_byte(
    tmp_path,  # noqa: ANN001
) -> None:
    ksp = (3, 4, 2)
    path = tmp_path / "coef.nii.gz"
    nb.save(_coefficient_file(_coefficients(3, ksp), 3, ksp, PREMAT), path)
    copy = tmp_path / "copy.nii.gz"
    FnirtWarpField.load(path).save(copy)
    assert gzip.open(copy).read() == gzip.open(path).read()


def test_new_coefficients_keep_the_header_of_the_spline(tmp_path) -> None:  # noqa: ANN001
    """Assigned coefficients are written under the header of the file."""
    ksp = (3, 4, 2)
    coef = _coefficients(3, ksp)
    path = tmp_path / "coef.nii"
    nb.save(_coefficient_file(coef, 3, ksp, PREMAT), path)
    warp = FnirtWarpField.load(path)
    warp.data = 2 * coef
    copy = tmp_path / "copy.nii"
    warp.save(copy)
    before = nb.load(path).header
    after = nb.load(copy).header
    for key in ("intent_code", "intent_p1", "intent_p2", "intent_p3"):
        assert after[key] == before[key]
    assert np.array_equal(after["pixdim"], before["pixdim"])
    assert np.array_equal(after.get_qform(), before.get_qform())
    assert np.array_equal(after.get_sform(), before.get_sform())
    assert np.array_equal(np.asarray(nb.load(copy).dataobj), 2 * coef)
