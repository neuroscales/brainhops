"""Saves over the files that a lazy image reads, and reader options.

An image read from a file holds a lazy array that reads the file by name
whenever the data are needed. Writing the image over that file must not
lose the data: the writer reads the old file while it writes the new
one, and the lazy array then reads the new file, which holds the same
values, even when the new file stores them in another way (#426). The
same images refuse reader options that they do not know, instead of
ignoring them.
"""

import numpy as np
import pytest

import brainhops.io as io
from brainhops._core.backends import backend
from brainhops.io.images.afni import AfniImage, AfniMetadata
from brainhops.io.images.mrtrix import MrtrixImage, MrtrixMetadata
from brainhops.io.images.nrrd import (
    AttachedNrrdImage,
    DetachedNrrdImage,
    NrrdMetadata,
)

try:
    import nibabel as nb

    from brainhops.io.images.freesurfer.mgh import MghImage, MghMetadata
    from brainhops.io.images.nifti import NiftiImage, NiftiMetadata
    from brainhops.io.transformations.nifti import NiftiRASCoordinateField
except ImportError:
    nb = None

DATA = np.arange(24, dtype="float32").reshape(2, 3, 4)

# Each case names a format, the name of its file, and the writer option
# that stores the voxels in another way than the file that is read.
CASES = [
    pytest.param(AttachedNrrdImage, "a.nrrd", {"datatype": "int16"}),
    pytest.param(AttachedNrrdImage, "b.nrrd", {"encoding": "raw"}),
    pytest.param(DetachedNrrdImage, "a.nhdr", {"encoding": "raw"}),
    pytest.param(MrtrixImage, "a.mif", {"datatype": "int16"}),
    pytest.param(MrtrixImage, "a.mih", {"datatype": "int16"}),
    pytest.param(AfniImage, "a+orig.HEAD", {"datatype": "short"}),
]
if nb is not None:
    CASES += [
        pytest.param(NiftiImage, "a.nii", {}, id="NiftiImage-untouched"),
        pytest.param(NiftiImage, "a.nii", {"dtype": "float64"}),
        pytest.param(NiftiImage, "a.nii.gz", {"dtype": "float64"}),
        pytest.param(MghImage, "a.mgh", {"dtype": "int16"}),
        pytest.param(MghImage, "a.mgz", {"dtype": "int16"}),
    ]


@pytest.mark.parametrize("name", ["numpy", "dask"])
@pytest.mark.parametrize("cls, filename, options", CASES)
def test_an_image_saved_over_itself_keeps_its_data(
    tmp_path,  # noqa: ANN001
    cls: type,
    filename: str,
    options: dict,
    name: str,
) -> None:
    if name == "dask":
        pytest.importorskip("dask")
    target = tmp_path / filename
    cls(data=DATA).save(target)
    with backend(name):
        image = cls.load(target)
        # The data are the lazy array of the file, which reads the file
        # by name, so only the values that the caller reads outlive it.
        cached = image.data.to_numpy()
        image.save(target, **options)
        assert np.array_equal(np.asarray(image.data), DATA)
        assert np.array_equal(cached, DATA)
        image.save(target)
        assert np.array_equal(np.asarray(cls.load(target).data), DATA)


@pytest.mark.skipif(nb is None, reason="needs nibabel")
@pytest.mark.parametrize("name", ["numpy", "dask"])
@pytest.mark.parametrize("filename", ["y.nii", "y.nii.gz"])
def test_a_nifti_field_saved_over_itself_keeps_its_values(
    tmp_path,  # noqa: ANN001
    filename: str,
    name: str,
) -> None:
    if name == "dask":
        pytest.importorskip("dask")
    values = np.arange(72, dtype="float32").reshape(2, 3, 4, 3)
    target = tmp_path / filename
    NiftiRASCoordinateField(field=values).save(target)
    with backend(name):
        field = NiftiRASCoordinateField.load(target)
        cached = field.data.to_numpy()
        field.save(target, dtype="float64")
        assert np.array_equal(np.asarray(field.data), values)
        assert np.array_equal(cached, values)
    back = NiftiRASCoordinateField.load(target)
    assert np.array_equal(np.asarray(back.data), values)


# Each case names a format, its metadata, the name of its file, and the
# reader options that it accepts.
READERS = [
    pytest.param(AttachedNrrdImage, NrrdMetadata, "a.nrrd", ("mmap",)),
    pytest.param(MrtrixImage, MrtrixMetadata, "a.mif", ("mmap",)),
    pytest.param(AfniImage, AfniMetadata, "a+orig.HEAD", ("mmap",)),
]
if nb is not None:
    READERS += [
        pytest.param(
            NiftiImage, NiftiMetadata, "a.nii", ("mmap", "keep_file_open")
        ),
        pytest.param(
            MghImage, MghMetadata, "a.mgh", ("mmap", "keep_file_open")
        ),
    ]


@pytest.mark.parametrize("cls, metadata, filename, accepted", READERS)
def test_unknown_reader_options_are_refused(
    tmp_path,  # noqa: ANN001
    cls: type,
    metadata: type,
    filename: str,
    accepted: tuple,
) -> None:
    target = tmp_path / filename
    cls(data=DATA).save(target)
    for option in ({"foo": 1}, {"transformations": []}):
        with pytest.raises(TypeError, match="option"):
            cls.load(target, **option)
        with pytest.raises(TypeError, match="option"):
            metadata.load(target, **option)
    for option in accepted:
        image = cls.load(target, **{option: False})
        assert np.array_equal(np.asarray(image.data), DATA)
    with open(target, "rb") as f:
        with pytest.raises(TypeError, match="option"):
            cls.from_fileobj(f, foo=1)
        if cls is not AfniImage:
            image = cls.from_fileobj(f, mmap=False)
            assert np.array_equal(np.asarray(image.data), DATA)
    assert isinstance(io.load(target), cls)
