"""Tests of `dataobj`, the lazy array of an object in the layout of data."""

import warnings

import numpy as np
import pytest

import brainhops.io as io
from brainhops.backends import backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import DisplacementField
from brainhops.io.base._utils_proxies import NibabelProxy

da = pytest.importorskip("dask.array")
nb = pytest.importorskip("nibabel")

VOLUME = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list:
    # Count the reads of nibabel proxies through the lazy arrays.
    log = []
    original = NibabelProxy._read

    def _read(self, index):  # noqa: ANN001, ANN202
        log.append(index)
        return original(self, index)

    monkeypatch.setattr(NibabelProxy, "_read", _read)
    return log


def _nifti(tmp_path, values=VOLUME):  # noqa: ANN001, ANN202
    path = tmp_path / "image.nii.gz"
    nb.save(nb.Nifti1Image(values, np.eye(4)), path)
    return io.images.load(path)


def test_an_array_holds_itself_as_dataobj() -> None:
    # An image or a field built from an array has that array as dataobj.
    assert SingleScaleImage(VOLUME).dataobj is VOLUME
    field = np.zeros((4, 5, 6, 3))
    assert DisplacementField(field).dataobj is field


def test_a_file_is_described_without_reading(tmp_path, reads) -> None:  # noqa: ANN001
    # dataobj, shape, dtype and geometry read nothing, and indexing dataobj
    # reads only the selected box when it is converted.
    image = _nifti(tmp_path)
    assert isinstance(image.dataobj, NibabelProxy)
    assert image.shape == VOLUME.shape
    assert image.dtype == VOLUME.dtype
    assert image.geometry is not None
    part = image.dataobj[1:3, :, 0]
    assert reads == []
    np.testing.assert_array_equal(np.asarray(part), VOLUME[1:3, :, 0])
    assert len(reads) == 1


def test_data_reads_dataobj_once(tmp_path, reads) -> None:  # noqa: ANN001
    # The first read of data reads the file and caches the values.
    image = _nifti(tmp_path)
    data = image.data
    np.testing.assert_array_equal(data, VOLUME)
    assert not data.flags.writeable
    assert image.data is data
    assert len(reads) == 1


def test_a_first_read_under_dask_reads_nothing(tmp_path, reads) -> None:  # noqa: ANN001
    # Under the Dask setting, the first read keeps the file lazy and
    # caches the Dask array, which later reads return.
    image = _nifti(tmp_path)
    with backend("dask"):
        lazy = image.data
    assert isinstance(lazy, da.Array)
    assert reads == []
    assert image.data is lazy
    np.testing.assert_array_equal(lazy.compute(), VOLUME)


def test_assigned_data_is_the_dataobj(tmp_path) -> None:  # noqa: ANN001
    # An array assigned as data replaces the lazy array of the file.
    image = _nifti(tmp_path)
    values = VOLUME + 1
    image.data = values
    assert image.dataobj is values
    assert image.data is values


def test_a_swapped_layout_is_kept_lazy(tmp_path) -> None:  # noqa: ANN001
    # A PNG stores rows first, and dataobj presents the columns first, as
    # data does, without reading the file.
    pytest.importorskip("PIL")
    pixels = np.arange(5 * 6, dtype=np.uint8).reshape(5, 6)
    io.save(SingleScaleImage(pixels), tmp_path / "image.png")
    image = io.images.load(tmp_path / "image.png")
    assert not isinstance(image.dataobj, np.ndarray)
    assert image.dataobj.shape == pixels.shape
    np.testing.assert_array_equal(np.asarray(image.dataobj), pixels)
    np.testing.assert_array_equal(image.data, pixels)


def test_an_itk_field_drops_its_singleton_axis_lazily(tmp_path) -> None:  # noqa: ANN001
    # An ITK displacement field stores (X, Y, Z, 1, 3), and its dataobj is
    # the lazy (X, Y, Z, 3) array.
    values = np.linspace(0, 1, 4 * 5 * 6 * 3, dtype=np.float32)
    values = values.reshape(4, 5, 6, 1, 3)
    header = nb.Nifti1Image(values, np.eye(4))
    header.header.set_intent("vector")
    nb.save(header, tmp_path / "field.nii.gz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = io.transformations.load(tmp_path / "field.nii.gz", hint="itk")
    assert isinstance(field.dataobj, NibabelProxy)
    assert field.dataobj.shape == (4, 5, 6, 3)
    np.testing.assert_array_equal(
        np.asarray(field.dataobj), values[:, :, :, 0, :]
    )


def test_a_compressed_tiff_is_decoded_once(
    tmp_path,  # noqa: ANN001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A compressed TIFF level must be decoded in full, which describing
    # the image or reading its dataobj does not do, and which the first
    # read of data does once.
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
    assert image.dataobj.shape == (32, 64)
    assert decodes == []
    np.testing.assert_array_equal(image.data, pixels.T)
    np.testing.assert_array_equal(image.data, pixels.T)
    assert len(decodes) == 1
