"""
Tests for writing images and transformations back to NIfTI.

A NIfTI file that this package reads should be writable again without
drifting: the data array, the sform and qform matrices, and their codes
should all survive a save-and-reload round trip. A transformation that
NIfTI cannot represent should be refused rather than silently resampled.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from bagof.magic import replace  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import (  # noqa: E402
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
    Scaling,
    Sequence,
)
from brainhops.io.base.parsers import (  # noqa: E402
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)

# A rigid rotation, an anisotropic zoom and a translation, with no shear,
# so the qform can hold it exactly.
AFFINE = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write_image(tmp_path, name, data, scode=2, qcode=1):  # noqa: ANN001, ANN202
    """Write a NIfTI image fixture with both an sform and a qform set."""
    img = nb.Nifti1Image(data, AFFINE)
    img.header.set_sform(AFFINE, code=scode)
    img.header.set_qform(AFFINE, code=qcode)
    target = tmp_path / name
    nb.save(img, str(target))
    return target


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_an_image_round_trips_through_save(tmp_path, suffix) -> None:  # noqa: ANN001
    """The data and the geometry survive a save and a reload unchanged."""
    data = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
    source = _write_image(tmp_path, "source.nii", data)

    loaded = io.images.load(source)
    assert isinstance(loaded, NiftiImage)

    target = tmp_path / ("out" + suffix)
    loaded.save(target)
    assert target.exists()

    reloaded = io.images.load(target)
    assert np.array_equal(np.asarray(reloaded), data)

    before, after = nb.load(str(source)).header, nb.load(str(target)).header
    assert np.allclose(before.get_sform(), after.get_sform())
    assert np.allclose(before.get_qform(), after.get_qform())
    assert int(before["sform_code"]) == int(after["sform_code"])
    assert int(before["qform_code"]) == int(after["qform_code"])
    assert np.allclose(
        nb.load(str(source)).affine, nb.load(str(target)).affine
    )


def test_the_sform_code_follows_the_world_space(tmp_path) -> None:  # noqa: ANN001
    """A scanner-coded sform is written back as a scanner-coded sform."""
    data = np.zeros((3, 4, 5), dtype="float32")
    source = _write_image(tmp_path, "scanner.nii", data, scode=1, qcode=1)

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    assert int(nb.load(str(target)).header["sform_code"]) == 1


def test_a_qform_is_written_even_without_a_rigid_edge(tmp_path) -> None:  # noqa: ANN001
    """
    When the source sets only an sform, the writer stores the rigid part
    of the sform as the qform rather than leaving the qform unset.
    """
    data = np.zeros((3, 4, 5), dtype="float32")
    img = nb.Nifti1Image(data, AFFINE)
    img.header.set_sform(AFFINE, code=2)
    img.header.set_qform(None, code=0)
    source = tmp_path / "sform_only.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    header = nb.load(str(target)).header
    assert int(header["qform_code"]) != 0
    assert np.allclose(header.get_qform(), AFFINE, atol=1e-5)


def test_a_scaling_geometry_is_representable(tmp_path) -> None:  # noqa: ANN001
    """A non-affine transformation that reduces to an affine is written."""
    image = NiftiImage(
        data=np.zeros((4, 5, 6), dtype="float32"),
        transformations=[Scaling(scale=[2.0, 3.0, 4.0])],
    )
    target = tmp_path / "scaled.nii"
    image.save(target)

    assert np.allclose(np.diag(nb.load(str(target)).affine), [2, 3, 4, 1])


def test_a_sequence_of_affines_writes_the_composed_sform(tmp_path) -> None:  # noqa: ANN001
    """
    The preferred transform may be a sequence of affines, which composes
    to a single affine. That composition is written as the sform.
    """
    first = Affine(
        matrix=np.array([[2.0, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3]])
    )
    second = Affine(
        matrix=np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[Sequence([first, second])],
    )
    target = tmp_path / "seq.nii"
    image.save(target)

    first_h = np.vstack([first.matrix, [0, 0, 0, 1]])
    second_h = np.vstack([second.matrix, [0, 0, 0, 1]])
    assert np.allclose(nb.load(str(target)).affine, second_h @ first_h)


def test_the_qform_matrix_and_code_come_from_one_edge(tmp_path) -> None:  # noqa: ANN001
    """
    The qform matrix and its code must describe the same world space. With
    a scanner edge and an mni edge, the mni becomes the preferred sform,
    and the scanner supplies both the qform matrix and its code.
    """
    ras, voxel = RASCoordinateSystem(), VoxelCoordinateSystem()
    scanner = Affine(
        matrix=np.eye(3, 4),
        input=voxel,
        output=replace(ras, name="scanner"),
    )
    mni = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3],
        input=voxel,
        output=replace(ras, name="mni"),
    )
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="float32"),
        transformations=[scanner, mni],
    )
    target = tmp_path / "coded.nii"
    image.save(target)

    header = nb.load(str(target)).header
    assert int(header["sform_code"]) == 4  # mni
    assert int(header["qform_code"]) == 1  # scanner
    assert np.allclose(header.get_qform(), np.eye(4))


def test_a_2d_nifti_round_trips(tmp_path) -> None:  # noqa: ANN001
    """
    A 2D image yields a `(3, 3)` matrix, which is embedded in the `(4, 4)`
    matrix NIfTI stores.
    """
    data = np.arange(4 * 5, dtype="float32").reshape(4, 5)
    img = nb.Nifti1Image(data, np.diag([2.0, 3.0, 1.0, 1.0]))
    source = tmp_path / "plane.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    reloaded = io.images.load(target)
    assert np.array_equal(np.asarray(reloaded), data)
    assert np.allclose(
        nb.load(str(source)).affine, nb.load(str(target)).affine
    )


def test_the_units_survive_the_round_trip(tmp_path) -> None:  # noqa: ANN001
    """The spatial and temporal units are read off the axes and written."""
    img = nb.Nifti1Image(np.zeros((4, 5, 6, 2), dtype="float32"), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    source = tmp_path / "units.nii"
    nb.save(img, str(source))

    target = tmp_path / "out.nii"
    io.images.load(source).save(target)

    assert nb.load(str(target)).header.get_xyzt_units() == ("mm", "sec")


def test_int64_data_is_reported_as_a_writer_error(tmp_path) -> None:  # noqa: ANN001
    """NIfTI-1 cannot store 64-bit integers, and the writer says so."""
    image = NiftiImage(
        data=np.zeros((3, 4, 5), dtype="int64"),
        transformations=[Affine(matrix=np.eye(3, 4))],
    )
    with pytest.raises(WriterError):
        image.save(tmp_path / "big.nii")


def test_an_image_without_data_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(WriterError):
        NiftiImage().save(tmp_path / "empty.nii")


# ----------------------------------------------------------------------
#   TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_an_affine_round_trips_through_save(tmp_path) -> None:  # noqa: ANN001
    """The voxel-to-RAS matrix is preserved across a save and a reload."""
    data = np.zeros((4, 5, 6), dtype="float32")
    source = _write_image(tmp_path, "source.nii", data)

    affine = NiftiVoxelToRAS.from_file(source)
    target = tmp_path / "affine.nii"
    affine.save(target)

    reloaded = NiftiVoxelToRAS.from_file(target)
    assert np.allclose(affine.matrix, reloaded.matrix)
    # The affine is written over a minimal placeholder volume, not the
    # full source data.
    assert nb.load(str(target)).shape == (1, 1, 1)


def test_a_field_round_trips_with_its_intent_code(tmp_path) -> None:  # noqa: ANN001
    """
    A field is written with a displacement-vector intent code, so it is
    read back as a field rather than as a plain image.
    """
    field = np.zeros((4, 5, 6, 1, 3), dtype="float32")
    field[..., 0] = 1.0
    img = nb.Nifti1Image(field, np.eye(4))
    img.header["intent_code"] = 1006
    source = tmp_path / "field.nii"
    nb.save(img, str(source))

    loaded = io.transformations.load(source)
    assert isinstance(loaded, NiftiRASCoordinatesField)

    target = tmp_path / "out.nii"
    loaded.save(target)

    reloaded = io.transformations.load(target)
    assert isinstance(reloaded, NiftiRASCoordinatesField)
    assert np.array_equal(np.asarray(reloaded.field), field)
    assert int(nb.load(str(target)).header["intent_code"]) == 1006


def test_a_4d_field_is_written_5d(tmp_path) -> None:  # noqa: ANN001
    """
    A field shaped `(X, Y, Z, C)` is written as `(X, Y, Z, 1, C)`, so the
    component axis is not mistaken for a time axis on reload.
    """
    field = np.zeros((4, 5, 6, 3), dtype="float32")
    field[..., 0] = 1.0
    source = NiftiRASCoordinatesField(field=field)

    target = tmp_path / "field.nii"
    source.save(target)

    assert nb.load(str(target)).shape == (4, 5, 6, 1, 3)
    assert isinstance(
        io.transformations.load(target), NiftiRASCoordinatesField
    )


# ----------------------------------------------------------------------
#   UNREPRESENTABLE TRANSFORMATIONS
# ----------------------------------------------------------------------


def test_a_nonaffine_geometry_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    """
    A displacement field is not affine geometry, so writing an image whose
    preferred transformation is one is refused rather than resampled.
    """
    image = NiftiImage(
        data=np.zeros((4, 5, 6), dtype="float32"),
        transformations=[
            DisplacementField(field=np.zeros((4, 5, 6, 3), dtype="float32"))
        ],
    )
    with pytest.raises(UnrepresentableTransformationError) as info:
        image.save(tmp_path / "bad.nii")
    assert "DisplacementField" in str(info.value)


def test_the_writer_is_registered_for_the_image_kind() -> None:
    """`NiftiImage` opts into the writable image registry."""
    from brainhops.io.images.base import WritableFileBasedImage

    assert NiftiImage in WritableFileBasedImage._REGISTRY
    assert issubclass(NiftiImage, SingleScaleImage)
