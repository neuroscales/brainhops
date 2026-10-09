"""
Tests for `from_any` on file-based data models.

A file-based image or transformation can be built from the file that
stores it, so its `from_any` reads a path, an open file, bytes or a
structured source with `load`, and leaves everything else to the data
model's own `from_any`.
"""

from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.base import ImageSpec  # noqa: E402
from brainhops.io.base.parsers import ParserError  # noqa: E402
from brainhops.io.images import FileBasedImage  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations import FileBasedTransformation  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASDisplacementField,
    NiftiVoxelToRAS,
)

DATA = np.arange(24, dtype="float32").reshape(2, 3, 4)


def _write_image(path: Path) -> Path:
    nb.save(nb.Nifti1Image(DATA, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))
    return path


def _write_field(path: Path) -> Path:
    field = np.zeros((2, 3, 4, 1, 3), dtype="float32")
    image = nb.Nifti1Image(field, np.eye(4))
    image.header["intent_code"] = 1006
    nb.save(image, str(path))
    return path


# ----------------------------------------------------------------------
#   FILES ARE READ
# ----------------------------------------------------------------------


@pytest.mark.parametrize("as_type", [str, Path], ids=["str", "path"])
def test_a_concrete_format_reads_a_path(tmp_path, as_type) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii")
    image = NiftiImage.from_any(as_type(source))
    assert isinstance(image, NiftiImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_a_dispatcher_reads_a_path_in_the_format_it_finds(tmp_path) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii.gz")
    image = FileBasedImage.from_any(source)
    assert isinstance(image, NiftiImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_a_transformation_dispatcher_reads_a_path(tmp_path) -> None:  # noqa: ANN001
    source = _write_field(tmp_path / "field.nii")
    # Intent code 1006 (displacement vector) is read as a displacement.
    field = FileBasedTransformation.from_any(source)
    assert isinstance(field, NiftiRASDisplacementField)


def test_an_open_file_is_read(tmp_path) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii")
    with open(source, "rb") as f:
        image = NiftiImage.from_any(f)
        # The array proxy reads the voxels lazily, from the open file.
        assert np.array_equal(np.asarray(image.data), DATA)


def test_bytes_are_read(tmp_path) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii")
    image = NiftiImage.from_any(source.read_bytes())
    assert np.array_equal(np.asarray(image.data), DATA)


def test_a_structured_source_is_read_with_its_hints(tmp_path) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii")
    spec = ImageSpec(path=source, hints=("nifti",))
    image = FileBasedImage.from_any(spec)
    assert isinstance(image, NiftiImage)
    # A hint naming no image format leaves nothing to read it with.
    with pytest.raises(ParserError):
        FileBasedImage.from_any(ImageSpec(path=source, hints=("zarr",)))


def test_keyword_options_reach_the_reader(tmp_path) -> None:  # noqa: ANN001
    # `mmap` is a `nibabel.load` option: it reaching the reader shows the
    # keywords are passed to `load` rather than to the constructor.
    source = _write_image(tmp_path / "image.nii")
    image = NiftiImage.from_any(source, mmap=False)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_positional_arguments_with_a_file_are_refused(tmp_path) -> None:  # noqa: ANN001
    source = _write_image(tmp_path / "image.nii")
    with pytest.raises(TypeError, match="keyword options only"):
        NiftiImage.from_any(source, DATA)


def test_a_missing_file_fails_to_read_rather_than_build(tmp_path) -> None:  # noqa: ANN001
    # Before, a string was handed to the constructor as the image data.
    with pytest.raises(ParserError):
        NiftiImage.from_any(str(tmp_path / "missing.nii"))
    with pytest.raises(FileNotFoundError):
        NiftiImage.from_any(tmp_path / "missing.nii")


# ----------------------------------------------------------------------
#   EVERYTHING ELSE IS LEFT TO THE DATA MODEL
# ----------------------------------------------------------------------


def test_a_mapping_is_read_field_by_field() -> None:
    image = NiftiImage.from_any({"data": DATA})
    assert isinstance(image, NiftiImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_a_mapping_with_an_unknown_key_is_refused() -> None:
    with pytest.raises(TypeError, match="no field named"):
        NiftiImage.from_any({"data": DATA, "dtaa": None})


def test_an_instance_of_the_data_model_is_copied_into_the_format() -> None:
    affine = Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])
    nifti = NiftiVoxelToRAS.from_any(affine)
    assert isinstance(nifti, NiftiVoxelToRAS)
    np.testing.assert_allclose(nifti.matrix, affine.matrix)

    image = NiftiImage.from_any(SingleScaleImage(data=DATA))
    assert isinstance(image, NiftiImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_an_instance_of_the_format_keeps_what_was_set_on_it(tmp_path) -> None:  # noqa: ANN001
    # A file-backed object serves its data through a property backed by
    # private state. Copying it must carry what was set, not re-read the
    # file.
    image = NiftiImage.load(_write_image(tmp_path / "image.nii"))
    image.data = np.zeros_like(DATA)
    copy = NiftiImage.from_any(image)
    assert copy is not image
    assert not np.asarray(copy.data).any()


# ----------------------------------------------------------------------
#   EVERY FILE-BASED CLASS READS FILES
# ----------------------------------------------------------------------


def _file_based_classes() -> list:
    from brainhops.io.base import FileBasedObject

    found = set(FileBasedObject._REGISTRY)
    found |= {FileBasedImage, FileBasedTransformation}
    return sorted(found, key=lambda cls: cls.__qualname__)


@pytest.mark.parametrize(
    "cls", _file_based_classes(), ids=lambda cls: cls.__qualname__
)
def test_every_file_based_class_reads_files_in_from_other(cls: type) -> None:
    # The file branch must come before the data model's own `from_any`
    # in the MRO of every format, whatever the order of its bases.
    owner = next(base for base in cls.__mro__ if "from_any" in vars(base))
    assert owner.__name__ == "_FileBasedModelMixin"
