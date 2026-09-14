"""
Integration tests for NIfTI dispatch.

A NIfTI file is legitimately an image, a set of affines and sometimes a
deformation field, all at once. These tests pin down how the intent
code, the data shape and the filename decide which one you get.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SPMCoordinatesField,
)

DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
NONE = 0  # NIFTI_INTENT_NONE


def _write(tmp_path, name: str, shape: tuple, intent: int):  # noqa: ANN001, ANN202
    """Write a NIfTI file with a given shape and intent code."""
    img = nb.Nifti1Image(np.zeros(shape, "float32"), np.eye(4))
    img.header["intent_code"] = intent
    target = tmp_path / name
    nb.save(img, str(target))
    return target


# ----------------------------------------------------------------------
#   INTENT-DRIVEN DISPATCH
# ----------------------------------------------------------------------


def test_a_plain_nifti_loads_as_its_affine(tmp_path) -> None:  # noqa: ANN001
    """
    Every NIfTI carries a voxel-to-RAS affine, and a plain image carries
    nothing else a transformation could be built from.
    """
    path = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    assert type(io.transformations.load(path)) is NiftiVoxelToRAS


def test_a_displacement_intent_loads_as_a_field(tmp_path) -> None:  # noqa: ANN001
    """The intent code outranks the affine that every NIfTI also has."""
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


def test_a_field_shape_is_recognized_without_an_intent_code(
    tmp_path,  # noqa: ANN001
) -> None:
    """
    Intent codes are often left unset, so the trailing axis of length 3
    has to carry the decision on its own.
    """
    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), NONE)
    assert io.transformations.sniff(path) is NiftiRASCoordinatesField
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


def test_the_spm_prefix_wins_an_otherwise_exact_tie(tmp_path) -> None:  # noqa: ANN001
    """
    An SPM field and a generic one are byte-identical; the `y_` prefix
    is the only evidence, and it decides.
    """
    path = _write(tmp_path, "y_sub01.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is SPMCoordinatesField


def test_the_spm_prefix_works_without_an_intent_code(tmp_path) -> None:  # noqa: ANN001
    """SPM writes no intent code, so this is the realistic case."""
    path = _write(tmp_path, "iy_sub01.nii", (4, 5, 6, 1, 3), NONE)
    assert type(io.transformations.load(path)) is SPMCoordinatesField


def test_the_spm_reader_does_not_claim_files_it_is_not_named_for(
    tmp_path,  # noqa: ANN001
) -> None:
    path = _write(tmp_path, "sub01.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


# ----------------------------------------------------------------------
#   COMPRESSION
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_dispatch_is_the_same_compressed_or_not(tmp_path, suffix) -> None:  # noqa: ANN001
    """
    `nibabel` decides whether to decompress from the file *name*, so a
    gzipped stream has to be detected from its magic bytes instead.
    """
    path = _write(tmp_path, "field" + suffix, (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField


def test_a_compressed_file_is_actually_decompressed(tmp_path) -> None:  # noqa: ANN001
    """The image data must survive, not just the header."""
    data = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
    img = nb.Nifti1Image(data, np.eye(4))
    target = tmp_path / "scan.nii.gz"
    nb.save(img, str(target))
    obj = NiftiVoxelToRAS.from_file(target)
    assert obj.image is not None
    assert np.asarray(obj.image.dataobj).sum() == pytest.approx(data.sum())


# ----------------------------------------------------------------------
#   SCORES
# ----------------------------------------------------------------------


def test_a_non_nifti_is_not_identified(tmp_path) -> None:  # noqa: ANN001
    junk = tmp_path / "junk.nii"
    junk.write_bytes(b"not a nifti at all")
    assert io.transformations.sniff(junk) is None


def test_a_field_scores_above_a_plain_image(tmp_path) -> None:  # noqa: ANN001
    """The ranking these scores produce is what drives dispatch."""
    field = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    plain = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    assert NiftiRASCoordinatesField.sniff(
        field
    ) > NiftiRASCoordinatesField.sniff(plain)


def test_sniff_identifies_the_format_without_parsing(tmp_path) -> None:  # noqa: ANN001
    """Asking what a file is should not require building the object."""
    from brainhops.io.images.nifti import NiftiImage

    plain = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    field = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert io.sniff(plain) is NiftiImage
    assert io.sniff(field) is NiftiRASCoordinatesField
    # and it agrees with what `load` actually returns
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
    """
    Reading a deformation field as a plain image is technically valid
    and almost never what was wanted, so it must lose to the field.
    """
    from brainhops.io.transformations.nifti import NiftiRASCoordinatesField

    path = _write(tmp_path, "field.nii", (4, 5, 6, 1, 3), DISPVECT)
    assert type(io.load(path)) is NiftiRASCoordinatesField


def test_image_data_survives_the_reader_closing_the_file(
    tmp_path,  # noqa: ANN001
) -> None:
    """
    `nibabel` reads voxels lazily, long after `load` returns. Opening the
    stream ourselves and closing it would leave the array proxy pointing
    at a closed file.
    """
    data = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
    img = nb.Nifti1Image(data, np.eye(4))
    target = tmp_path / "scan.nii"
    nb.save(img, str(target))

    loaded = io.images.load(target)
    assert loaded.shape == (4, 5, 6)
    assert loaded.dtype == np.dtype("float32")
    assert np.asarray(loaded).sum() == pytest.approx(data.sum())


def test_transformations_are_derived_from_the_header(tmp_path) -> None:  # noqa: ANN001
    """
    The field defaults to an empty tuple, so the lazy derivation has to
    treat "empty" as "not set" or it never runs.
    """
    path = _write(tmp_path, "plain.nii", (4, 5, 6), NONE)
    loaded = io.images.load(path)
    names = [getattr(t.output, "name", None) for t in loaded.transformations]
    assert "physical" in names
    assert loaded.transformations, "transformations were not derived"


def test_an_unset_qform_is_skipped_rather_than_crashing(tmp_path) -> None:  # noqa: ANN001
    """`get_qform(coded=True)` returns `None` when the code is 0."""
    img = nb.Nifti1Image(np.zeros((4, 5, 6), "float32"), np.eye(4))
    img.header.set_qform(None, code=0)
    target = tmp_path / "noqform.nii"
    nb.save(img, str(target))
    loaded = io.images.load(target)
    assert loaded.transformations


def test_an_image_can_be_built_without_data(tmp_path) -> None:  # noqa: ANN001
    """
    `SingleScaleImage.data` is optional so that format readers can derive
    it lazily from the file they hold, rather than supplying it upfront.
    """
    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.nifti import NiftiImage

    assert SingleScaleImage().data is None
    assert NiftiImage().data is None
