"""Tests for the default grid of a NIfTI file whose codes are both zero.

When the sform code and the qform code of a NIfTI file are both zero,
the header does not say where the voxels lie. brainhops then uses the
default grid of nibabel's `get_best_affine()`, chosen in issue #631, for
the image and for every transformation stored in a NIfTI file. The
default grid scales each axis by its voxel size, centres the image with
the translation `-zoom * (shape - 1) / 2`, and flips the first axis.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.common.nifti._header import best_affine  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.afni._geometry import (  # noqa: E402
    nifti_matrix,
)
from brainhops.io.transformations.fsl._affines import (  # noqa: E402
    ImageGeometry,
)
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.matrix._conventions import (  # noqa: E402
    image_to_world,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASDisplacementField,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm import (  # noqa: E402
    SpmCoordinateField,
)

DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR

SHAPE = (4, 5, 1)
"""A grid whose last axis holds a single slice."""

ZOOMS = (2.0, 3.0, 4.0)
"""Voxel sizes that differ on every axis."""


def _default(shape: tuple, zooms: tuple) -> np.ndarray:
    """Return the default grid of #631, written out from its definition."""
    full_shape = np.ones(3)
    full_zooms = np.ones(3)
    n = min(3, len(shape))
    full_shape[:n] = shape[:n]
    full_zooms[:n] = zooms[:n]
    full_zooms[0] = -full_zooms[0]
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(full_zooms)
    matrix[:3, 3] = -full_zooms * (full_shape - 1) / 2
    return matrix


DEFAULT = _default(SHAPE, ZOOMS)
"""The default grid of a file of shape `SHAPE` and voxel size `ZOOMS`."""


def _uncoded(data: np.ndarray, zooms: tuple = ZOOMS) -> nb.Nifti1Image:
    """Return a nibabel image whose sform and qform codes are both zero."""
    nii = nb.Nifti1Image(data, None)
    nii.header.set_zooms(zooms[: data.ndim])
    nii.header.set_sform(None, code=0)
    nii.header.set_qform(None, code=0)
    return nii


def _save(path, nii: nb.Nifti1Image) -> str:  # noqa: ANN001
    nb.save(nii, str(path))
    return str(path)


# ----------------------------------------------------------------------
#   THE RULE OF NIBABEL
# ----------------------------------------------------------------------

_GRIDS = pytest.mark.parametrize(
    ("shape", "zooms"),
    [
        ((4, 5, 1), (2.0, 3.0, 4.0)),
        ((4, 5), (2.0, 3.0)),
        ((4, 5, 6, 7, 3), (1.0, 2.0, 3.0, 4.0, 5.0)),
        ((3, 4, 5), (0.0, 2.0, -3.0)),
    ],
    ids=["single-slice", "2d", "5d", "zero-and-negative-zooms"],
)


@_GRIDS
@pytest.mark.parametrize("cls", [nb.Nifti1Header, nb.Nifti2Header])
def test_nibabel_decodes_a_header_without_codes_to_the_default_grid(
    cls: type, shape: tuple, zooms: tuple
) -> None:
    """The default grid is centred and flips the first axis.

    brainhops relies on nibabel's `get_best_affine()` for the default
    grid, so this test pins the rule that nibabel follows.
    """
    header = cls()
    header.set_data_shape(shape)
    header["pixdim"][1 : len(zooms) + 1] = zooms
    assert np.array_equal(header.get_best_affine(), _default(shape, zooms))
    matrix = best_affine(header)
    assert matrix.dtype == np.float64
    assert np.array_equal(matrix, _default(shape, zooms))


@_GRIDS
@pytest.mark.parametrize("cls", [nb.Nifti1Image, nb.Nifti2Image])
def test_a_nifti_affine_reads_the_grid_of_nibabel(
    tmp_path,  # noqa: ANN001
    cls: type,
    shape: tuple,
    zooms: tuple,
) -> None:
    """The affine stored in a file of any shape is nibabel's grid."""
    nii = cls(np.zeros(shape, dtype="float32"), None)
    nii.header["pixdim"][1 : len(zooms) + 1] = zooms
    nii.header.set_sform(None, code=0)
    nii.header.set_qform(None, code=0)
    path = _save(tmp_path / "image.nii", nii)
    expected = nb.load(path).header.get_best_affine()
    affine = NiftiVoxelToRAS.from_filename(path)
    assert np.array_equal(affine.matrix, expected[:3])


def test_the_sform_comes_before_the_qform() -> None:
    header = nb.Nifti1Header()
    header.set_data_shape(SHAPE)
    sform = np.diag([1.0, 2.0, 3.0, 1.0])
    sform[:3, 3] = [4.0, 5.0, 6.0]
    header.set_qform(np.diag([7.0, 7.0, 7.0, 1.0]), code=1)
    assert np.allclose(best_affine(header), header.get_qform())
    header.set_sform(sform, code=2)
    assert np.array_equal(best_affine(header), sform)


def test_a_single_slice_is_centred_on_its_voxel(tmp_path) -> None:  # noqa: ANN001
    """The centre of the image is the world origin."""
    data = np.zeros(SHAPE, dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    centre = (np.asarray(SHAPE) - 1) / 2
    matrix = np.asarray(image.transformation.homogeneous_matrix)
    world = matrix @ [*centre, 1.0]
    assert np.allclose(world[:3], 0.0)
    assert centre[2] == 0.0


def test_the_helper_reads_a_record_or_falls_back_on_the_affine(
    tmp_path,  # noqa: ANN001
) -> None:
    """The helper reads a record, an affine, or refuses the object.

    A `NiftiImage` is read through the header of its record, an object
    without a header is read through its affine, and an object with
    neither is refused.
    """
    data = np.zeros(SHAPE, dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    assert np.array_equal(best_affine(image), DEFAULT)

    class _WithAffine:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

    assert np.array_equal(best_affine(_WithAffine()), _WithAffine.affine)
    with pytest.raises(ValueError):
        best_affine(object())


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def test_an_image_without_codes_has_the_default_grid(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros(SHAPE, dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    assert isinstance(image, NiftiImage)
    names = [xform.output.name for xform in image.transformations]
    assert names == ["physical", "unknown"]
    xform = image.transformation
    assert isinstance(xform, Affine)
    assert np.array_equal(xform.homogeneous_matrix, DEFAULT)
    assert xform.output.name == "unknown"
    orientations = [axis.orientation.value for axis in xform.output.axes]
    assert orientations == [
        "left-to-right",
        "posterior-to-anterior",
        "inferior-to-superior",
    ]


def test_a_2d_image_without_codes_has_the_default_grid(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros(SHAPE[:2], dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    expected = _default(SHAPE[:2], ZOOMS[:2])
    matrix = np.asarray(image.transformation.homogeneous_matrix)
    assert matrix.shape == (3, 3)
    assert np.array_equal(matrix, expected[[0, 1, 3]][:, [0, 1, 3]])


def test_a_4d_image_without_codes_has_the_default_spatial_grid(
    tmp_path,  # noqa: ANN001
) -> None:
    data = np.zeros((*SHAPE, 2), dtype="float32")
    nii = _uncoded(data, (*ZOOMS, 1.5))
    image = io.load(_save(tmp_path / "image.nii", nii))
    matrix = np.asarray(image.transformation.to(Affine).homogeneous_matrix)
    assert np.allclose(matrix[:3, :3], DEFAULT[:3, :3])
    assert np.allclose(matrix[:3, 4], DEFAULT[:3, 3])
    assert matrix[3, 3] == 1.5


@pytest.mark.parametrize("read", [False, True])
def test_an_image_without_codes_is_saved_byte_for_byte(
    tmp_path,  # noqa: ANN001
    read: bool,
) -> None:
    """Reading the default grid does not write it into the file."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    source = _save(tmp_path / "image.nii", _uncoded(data))
    image = io.load(source)
    if read:
        assert image.transformations
    image.save(tmp_path / "copy.nii")
    original = (tmp_path / "image.nii").read_bytes()
    assert (tmp_path / "copy.nii").read_bytes() == original


def test_new_data_of_the_same_shape_keep_the_codes_zero(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros(SHAPE, dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    assert image.transformations
    image.data = np.ones(SHAPE, dtype="float32")
    image.save(tmp_path / "copy.nii")
    header = nb.load(str(tmp_path / "copy.nii")).header
    assert int(header["sform_code"]) == 0
    assert int(header["qform_code"]) == 0
    copy = io.load(str(tmp_path / "copy.nii"))
    assert np.array_equal(copy.transformation.homogeneous_matrix, DEFAULT)


def test_a_changed_grid_is_written_into_the_sform(tmp_path) -> None:  # noqa: ANN001
    data = np.zeros(SHAPE, dtype="float32")
    image = io.load(_save(tmp_path / "image.nii", _uncoded(data)))
    xform = image.transformation
    matrix = np.array(xform.matrix, copy=True)
    matrix[:, 3] += 1.0
    image.transformations = [Affine(matrix=matrix, input=xform.input)]
    image.save(tmp_path / "copy.nii")
    header = nb.load(str(tmp_path / "copy.nii")).header
    assert int(header["sform_code"]) != 0
    assert np.allclose(header.get_sform()[:3], matrix)


# ----------------------------------------------------------------------
#   TRANSFORMATIONS STORED IN A NIFTI FILE
# ----------------------------------------------------------------------


def _vectors(intent: int, name: str = "") -> nb.Nifti1Image:
    """Return a field of shape `(*SHAPE, 1, 3)` without codes."""
    data = np.zeros((*SHAPE, 1, 3), dtype="float32")
    nii = _uncoded(data, (*ZOOMS, 1.0, 1.0))
    nii.header.set_intent(intent, name=name)
    return nii


def test_a_nifti_affine_without_codes_has_the_default_grid(
    tmp_path,  # noqa: ANN001
) -> None:
    data = np.zeros(SHAPE, dtype="float32")
    path = _save(tmp_path / "image.nii", _uncoded(data))
    affine = NiftiVoxelToRAS.from_filename(path)
    assert np.array_equal(affine.matrix, DEFAULT[:3])
    inverse = affine.inverse()
    assert np.allclose(inverse.inverse().matrix, DEFAULT[:3])


def test_a_nifti_displacement_field_without_codes_has_the_default_grid(
    tmp_path,  # noqa: ANN001
) -> None:
    path = _save(tmp_path / "warp.nii", _vectors(DISPVECT))
    field = NiftiRASDisplacementField.from_filename(path)
    assert np.array_equal(field.voxel2ras.matrix, DEFAULT[:3])
    image = field.to_image()
    assert np.array_equal(
        np.asarray(image.transformation.to(Affine).homogeneous_matrix)[
            [0, 1, 2, -1]
        ][:, [0, 1, 2, -1]],
        DEFAULT,
    )


def test_an_spm_field_without_codes_has_the_default_grid(tmp_path) -> None:  # noqa: ANN001
    path = _save(tmp_path / "y_warp.nii", _vectors(VECTOR, "Mapping"))
    field = SpmCoordinateField.from_filename(path)
    assert np.allclose(field.ras2voxel.matrix, np.linalg.inv(DEFAULT)[:3])


def test_an_itk_field_without_codes_has_the_default_grid(tmp_path) -> None:  # noqa: ANN001
    path = _save(tmp_path / "warp.nii.gz", _vectors(VECTOR))
    field = ItkNiftiDisplacementField.from_filename(path)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ DEFAULT
    assert np.array_equal(field.voxel2lps.matrix, lps[:3])


def test_the_other_nifti_readers_use_the_default_grid() -> None:
    """AFNI, FSL and the matrix conventions read the same default grid."""
    nii = _uncoded(np.zeros(SHAPE, dtype="float32"))
    assert np.array_equal(nifti_matrix(nii.header), DEFAULT)
    assert np.array_equal(nifti_matrix(nii), DEFAULT)
    assert np.array_equal(best_affine(nii), DEFAULT)
    assert np.array_equal(best_affine(nii.header), DEFAULT)
    assert np.array_equal(ImageGeometry(nii).vox2ras, DEFAULT)
    assert np.array_equal(ImageGeometry(nii.header).vox2ras, DEFAULT)
    assert np.array_equal(image_to_world(nii)[0], DEFAULT)
    assert np.array_equal(image_to_world(nii.header)[0], DEFAULT)


def test_a_header_of_another_format_keeps_its_own_affine() -> None:
    """An MGH header is decoded by nibabel, under its own rules."""
    mgh = nb.MGHImage(np.zeros(SHAPE, dtype="float32"), np.eye(4))
    expected = mgh.header.get_best_affine()
    assert np.array_equal(best_affine(mgh), expected)
    assert np.array_equal(ImageGeometry(mgh).vox2ras, expected)
    assert np.array_equal(image_to_world(mgh)[0], expected)


def test_a_field_without_codes_is_saved_byte_for_byte(tmp_path) -> None:  # noqa: ANN001
    path = _save(tmp_path / "warp.nii", _vectors(DISPVECT))
    field = NiftiRASDisplacementField.from_filename(path)
    assert np.array_equal(field.voxel2ras.matrix, DEFAULT[:3])
    field.save(tmp_path / "copy.nii")
    original = (tmp_path / "warp.nii").read_bytes()
    assert (tmp_path / "copy.nii").read_bytes() == original
