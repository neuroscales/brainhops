"""Integration tests of NIfTI dispatch.

A NIfTI file can be an image, an affine and sometimes a deformation field at
once. These tests pin how the intent code, the data shape and the file name
decide between the readers.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm import (  # noqa: E402
    SpmCoordinatesField,
)

DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR
MAPPING = "Mapping"  # intent name of a coordinates field
NONE = 0  # NIFTI_INTENT_NONE


def _write(tmp_path, name: str, shape: tuple, intent: int):  # noqa: ANN001, ANN202
    img = nb.Nifti1Image(np.zeros(shape, "float32"), np.eye(4))
    img.header["intent_code"] = intent
    if intent == VECTOR:
        # Coordinates fields are named as SPM and brainhops name them.
        img.header["intent_name"] = MAPPING
    target = tmp_path / name
    nb.save(img, str(target))
    return target


# ----------------------------------------------------------------------
#   INTENT-DRIVEN DISPATCH
# ----------------------------------------------------------------------


def test_a_plain_nifti_loads_as_its_affine(tmp_path) -> None:  # noqa: ANN001
    """A plain image has no transformation besides its vox-to-RAS affine."""
    path = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    assert type(io.transformations.load(path)) is NiftiVoxelToRAS


def test_a_displacement_intent_loads_as_a_displacement_field(
    tmp_path,  # noqa: ANN001
) -> None:
    """Per NIfTI-1, DISPVECT means displacements, which outrank the affine."""
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert io.transformations.sniff(path) is NiftiRASDisplacementField
    assert type(io.transformations.load(path)) is NiftiRASDisplacementField
    assert type(io.load(path)) is NiftiRASDisplacementField


def test_a_vector_intent_loads_as_a_coordinates_field(tmp_path) -> None:  # noqa: ANN001
    """VECTOR is the intent that brainhops and SPM write for coordinates."""
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), VECTOR)
    assert io.transformations.sniff(path) is NiftiRASCoordinatesField
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


def test_each_field_reader_declines_the_other_intent(tmp_path) -> None:  # noqa: ANN001
    """Neither field reader claims the intent of the other on content."""
    disp = _write(tmp_path, "disp.nii", (4, 5, 6, 1, 3), DISPVECT)
    coords = _write(tmp_path, "coords.nii", (4, 5, 6, 1, 3), VECTOR)
    assert NiftiRASCoordinatesField.sniff(disp) == 0
    assert NiftiRASDisplacementField.sniff(coords) == 0
    assert SpmCoordinatesField.sniff(disp) == 0


def test_a_field_without_an_intent_is_not_read_as_displacements(
    tmp_path,  # noqa: ANN001
) -> None:
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), NONE)
    assert NiftiRASDisplacementField.sniff(path) == 0


@pytest.mark.parametrize("hint", ["nifti.coordinates", "coordinates"])
def test_a_hint_reads_a_dispvect_file_as_coordinates(tmp_path, hint) -> None:  # noqa: ANN001
    """A hint reads the DISPVECT coordinates that older brainhops wrote."""
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    loaded = io.transformations.load(path, hint=hint)
    assert type(loaded) is NiftiRASCoordinatesField


@pytest.mark.parametrize("hint", ["nifti.displacements", "displacements"])
def test_a_hint_selects_the_displacement_reader(tmp_path, hint) -> None:  # noqa: ANN001
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    loaded = io.transformations.load(path, hint=hint)
    assert type(loaded) is NiftiRASDisplacementField


def test_a_field_shape_is_recognized_without_an_intent_code(
    tmp_path,  # noqa: ANN001
) -> None:
    """The intent is often unset, so a trailing axis of length 3 decides."""
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), NONE)
    assert io.transformations.sniff(path) is NiftiRASCoordinatesField
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


def test_the_spm_prefix_wins_an_otherwise_exact_tie(tmp_path) -> None:  # noqa: ANN001
    """The SPM and generic fields are byte-identical, so y_ breaks the tie."""
    path = _write(tmp_path, "y_sub01.nii", (4, 5, 6, 1, 3), VECTOR)
    assert type(io.transformations.load(path)) is SpmCoordinatesField


def test_the_spm_prefix_does_not_override_a_displacement_intent(
    tmp_path,  # noqa: ANN001
) -> None:
    """The file name is weaker evidence than a DISPVECT intent."""
    path = _write(tmp_path, "y_sub01.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is NiftiRASDisplacementField


def test_the_spm_prefix_works_without_an_intent_code(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path, "iy_sub01.nii", (4, 5, 6, 1, 3), NONE)
    assert type(io.transformations.load(path)) is SpmCoordinatesField


def test_the_spm_reader_does_not_claim_files_it_is_not_named_for(
    tmp_path,  # noqa: ANN001
) -> None:
    path = _write(tmp_path, "sub01.nii", (4, 5, 6, 1, 3), VECTOR)
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


# ----------------------------------------------------------------------
#   COMPRESSION
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_dispatch_is_the_same_compressed_or_not(tmp_path, suffix) -> None:  # noqa: ANN001
    """nibabel decompresses by name, so gzip is detected by its magic bytes."""
    path = _write(tmp_path, "field" + suffix, (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is NiftiRASDisplacementField


def test_a_compressed_file_is_actually_decompressed(tmp_path) -> None:  # noqa: ANN001
    """The image data is decompressed, not only the header."""
    data = np.arange(4 * 5 * 6 * 3, dtype="float32").reshape(4, 5, 6, 1, 3)
    img = nb.Nifti1Image(data, np.eye(4))
    img.header.set_intent(VECTOR)
    target = tmp_path / "field.nii.gz"
    nb.save(img, str(target))
    obj = NiftiRASCoordinatesField.from_file(target)
    assert obj.raw is not None
    assert np.asarray(obj.raw).sum() == pytest.approx(data.sum())


# ----------------------------------------------------------------------
#   SCORES
# ----------------------------------------------------------------------


def test_a_non_nifti_is_not_identified(tmp_path) -> None:  # noqa: ANN001
    junk = tmp_path / "junk.nii"
    junk.write_bytes(b"not a nifti at all")
    assert io.transformations.sniff(junk) is None


def test_a_field_scores_above_a_plain_image(tmp_path) -> None:  # noqa: ANN001
    field = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), VECTOR)
    plain = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    assert NiftiRASCoordinatesField.sniff(
        field
    ) > NiftiRASCoordinatesField.sniff(plain)


def test_sniff_identifies_the_format_without_parsing(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.nifti import NiftiImage

    plain = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    field = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert io.sniff(plain) is NiftiImage
    assert io.sniff(field) is NiftiRASDisplacementField
    assert type(io.load(plain)) is io.sniff(plain)
    assert type(io.load(field)) is io.sniff(field)


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def test_a_plain_nifti_loads_as_an_image(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.images.nifti import NiftiImage

    path = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    assert type(io.load(path)) is NiftiImage
    assert type(io.images.load(path)) is NiftiImage


def test_a_field_shaped_volume_is_not_claimed_as_an_image(
    tmp_path,  # noqa: ANN001
) -> None:
    """A field-shaped volume is a valid image but loads as the field."""
    vector = _write(tmp_path, "coords.nii", (4, 5, 6, 1, 3), VECTOR)
    dispvect = _write(tmp_path, "disp.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.load(vector)) is NiftiRASCoordinatesField
    assert type(io.load(dispvect)) is NiftiRASDisplacementField


def test_image_data_survives_the_reader_closing_the_file(
    tmp_path,  # noqa: ANN001
) -> None:
    """nibabel reads the voxels lazily, so the file must stay readable."""
    data = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
    img = nb.Nifti1Image(data, np.eye(4))
    target = tmp_path / "scan.nii"
    nb.save(img, str(target))

    loaded = io.images.load(target)
    assert loaded.shape == (4, 5, 6)
    assert loaded.dtype == np.dtype("float32")
    assert np.asarray(loaded).sum() == pytest.approx(data.sum())


def test_transformations_are_derived_from_the_header(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    loaded = io.images.load(path)
    names = [getattr(t.output, "name", None) for t in loaded.transformations]
    assert "physical" in names
    assert loaded.transformations, "transformations were not derived"


def test_an_unset_qform_is_skipped_rather_than_crashing(tmp_path) -> None:  # noqa: ANN001
    """get_qform(coded=True) returns None when the qform code is 0."""
    img = nb.Nifti1Image(np.zeros((4, 5, 6), "float32"), np.eye(4))
    img.header.set_qform(None, code=0)
    target = tmp_path / "noqform.nii"
    nb.save(img, str(target))
    loaded = io.images.load(target)
    assert loaded.transformations


def test_an_image_can_be_built_without_data(tmp_path) -> None:  # noqa: ANN001
    """Image data is optional, so that readers can derive it lazily."""
    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.nifti import NiftiImage

    assert SingleScaleImage().data is None
    assert NiftiImage().data is None
