"""Tests of `data`, which is the lazy array of a file read from disk."""

import warnings

import numpy as np
import pytest

import brainhops.io as io
from brainhops._core.backends import backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import DisplacementField

da = pytest.importorskip("dask.array")
nb = pytest.importorskip("nibabel")

from brainhops.io.common.nifti._data import NiftiProxy  # noqa: E402

VOLUME = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list:
    # Count the reads of NIfTI proxies through the lazy arrays.
    log = []
    original = NiftiProxy._read

    def _read(self, index):  # noqa: ANN001, ANN202
        log.append(index)
        return original(self, index)

    monkeypatch.setattr(NiftiProxy, "_read", _read)
    return log


def _nifti(tmp_path, values=VOLUME):  # noqa: ANN001, ANN202
    path = tmp_path / "image.nii.gz"
    nb.save(nb.Nifti1Image(values, np.eye(4)), path)
    return io.images.load(path)


def test_an_array_holds_itself_as_data() -> None:
    # An image or a field built from an array has that array as data.
    assert SingleScaleImage(VOLUME).data is VOLUME
    field = np.zeros((4, 5, 6, 3))
    assert DisplacementField(field).data is field


def test_a_file_is_described_without_reading(tmp_path, reads) -> None:  # noqa: ANN001
    # data, shape, dtype and geometry read nothing, and indexing data
    # reads only the selected box when it is converted.
    image = _nifti(tmp_path)
    assert isinstance(image.data, NiftiProxy)
    assert image.shape == VOLUME.shape
    assert image.dtype == VOLUME.dtype
    assert image.geometry is not None
    part = image.data[1:3, :, 0]
    assert reads == []
    np.testing.assert_array_equal(np.asarray(part), VOLUME[1:3, :, 0])
    assert len(reads) == 1


def test_data_is_the_lazy_array_of_the_file(tmp_path, reads) -> None:  # noqa: ANN001
    # data returns the lazy array of the file, which reads only what an
    # index selects, and reads again at each computation.
    image = _nifti(tmp_path)
    assert isinstance(image.data, NiftiProxy)
    assert image.data.shape == VOLUME.shape
    part = image.data[1:3, :, 0]
    assert reads == []
    np.testing.assert_array_equal(part, VOLUME[1:3, :, 0])
    assert len(reads) == 1
    assert float(image.data.sum()) == float(VOLUME.sum())
    assert float(image.data.sum()) == float(VOLUME.sum())
    assert len(reads) == 3


def test_as_numpy_inplace_keeps_the_values(tmp_path, reads) -> None:  # noqa: ANN001
    # as_numpy(inplace=True) reads the file once and keeps its values.
    image = _nifti(tmp_path)
    image.as_numpy(inplace=True)
    assert isinstance(image.data, np.ndarray)
    np.testing.assert_array_equal(image.data, VOLUME)
    np.testing.assert_array_equal(image.data, VOLUME)
    assert len(reads) == 1


def test_a_computation_under_dask_reads_nothing(tmp_path, reads) -> None:  # noqa: ANN001
    # Under the Dask setting, data is still the lazy array of the file, and
    # a computation on it gives a Dask array without reading the file.
    image = _nifti(tmp_path)
    with backend("dask"):
        assert isinstance(image.data, NiftiProxy)
        lazy = image.data + 1
    assert isinstance(lazy, da.Array)
    assert reads == []
    np.testing.assert_array_equal(lazy.compute(), VOLUME + 1)


def test_an_index_of_an_image_reads_nothing(tmp_path, reads) -> None:  # noqa: ANN001
    # Indexing an image indexes the lazy array of its file.
    image = _nifti(tmp_path)
    part = image[1:3, :, 0]
    assert reads == []
    assert part.shape == (2, 5)
    np.testing.assert_array_equal(part.data, VOLUME[1:3, :, 0])
    assert len(reads) == 1


def test_assigned_data_replaces_the_lazy_array(tmp_path) -> None:  # noqa: ANN001
    # An array assigned as data replaces the lazy array of the file, and
    # assigning None gives the lazy array again.
    image = _nifti(tmp_path)
    values = VOLUME + 1
    image.data = values
    assert image.data is values
    image.data = None
    assert isinstance(image.data, NiftiProxy)


def test_a_swapped_layout_is_kept_lazy(tmp_path) -> None:  # noqa: ANN001
    # A PNG stores rows first, and data presents the columns first,
    # without reading the file.
    pytest.importorskip("PIL")
    pixels = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6)
    io.save(SingleScaleImage(pixels), tmp_path / "image.png")
    image = io.images.load(tmp_path / "image.png")
    assert not isinstance(image.data, np.ndarray)
    assert image.data.shape == pixels.shape
    np.testing.assert_array_equal(np.asarray(image.data), pixels)


def test_an_itk_field_drops_its_singleton_axis_lazily(tmp_path) -> None:  # noqa: ANN001
    # An ITK displacement field stores (X, Y, Z, 1, 3), and its data is
    # the lazy (X, Y, Z, 3) array.
    values = np.linspace(0, 1, 4 * 5 * 6 * 3, dtype=np.float32)
    values = values.reshape(4, 5, 6, 1, 3)
    header = nb.Nifti1Image(values, np.eye(4))
    header.header.set_intent("vector")
    nb.save(header, tmp_path / "field.nii.gz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = io.transformations.load(tmp_path / "field.nii.gz", hint="itk")
    assert isinstance(field.data.root, NiftiProxy)
    assert field.data.shape == (4, 5, 6, 3)
    np.testing.assert_array_equal(
        np.asarray(field.data), values[:, :, :, 0, :]
    )


def test_a_compressed_tiff_is_decoded_at_each_read(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A compressed TIFF level must be decoded in full, which describing
    # the image or reading its data does not do. Each computation on
    # data decodes it, and as_numpy(inplace=True) decodes it once.
    tifffile = pytest.importorskip("tifffile")
    from brainhops.io.images.tiff import _data as tiff_data

    pixels = np.arange(64 * 32, dtype=np.uint8).reshape(64, 32)
    tifffile.imwrite(tmp_path / "image.tif", pixels, compression="zlib")
    decodes = []
    original = tiff_data._TiffProxy._level_array

    def _level_array(self):  # noqa: ANN001, ANN202
        decodes.append(self.level)
        return original(self)

    monkeypatch.setattr(tiff_data._TiffProxy, "_level_array", _level_array)
    image = io.images.load(tmp_path / "image.tif")
    assert image.shape == (32, 64)
    assert image.dtype == np.uint8
    assert image.data.shape == (32, 64)
    assert decodes == []
    np.testing.assert_array_equal(image.data, pixels.T)
    assert len(decodes) == 1
    image.as_numpy(inplace=True)
    np.testing.assert_array_equal(image.data, pixels.T)
    np.testing.assert_array_equal(image.data, pixels.T)
    assert len(decodes) == 2


def test_the_lazy_array_is_kept_until_the_file_changes(tmp_path) -> None:  # noqa: ANN001
    # The lazy array of the file is built once, and built again when the
    # image is given another lazy array of a file.
    image = _nifti(tmp_path)
    data = image.data
    assert image.data is data
    (tmp_path / "other").mkdir()
    other = _nifti(tmp_path / "other", VOLUME + 1)
    image.rawdata = other.rawdata
    assert image.data is not data
    np.testing.assert_array_equal(image.data, VOLUME + 1)
