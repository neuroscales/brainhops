"""Tests of the NIfTI and MGH proxies against what nibabel reads.

The NIfTI and MGH readers present their voxels through `StridedProxy`,
wrapped in a `ScaledProxy` when a NIfTI header scales them. nibabel
still parses the headers, and these tests check that the values, their
type and their shape are those that nibabel's own proxy gives, for whole
reads, partial reads and Dask blocks.
"""

import gzip
import io as _io
import shutil

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

from brainhops._core.proxy import ScaledProxy  # noqa: E402
from brainhops.io.common.mgh._data import MghProxy  # noqa: E402
from brainhops.io.common.mgh._raw import read_mgh  # noqa: E402
from brainhops.io.common.nifti._data import (  # noqa: E402
    NiftiProxy,
    whole_file,
)
from brainhops.io.images.freesurfer.mgh import MghImage  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402


def read_nifti(file, **kwargs):  # noqa: ANN001, ANN003, ANN201
    """Return the record and the lazy array that a NIfTI file is read as."""
    image = NiftiImage.load(file, **kwargs)
    return image.metadata.raw, image.rawdata


SHAPE = (5, 4, 3, 2)
"""The shape of the test arrays, with a fourth axis of volumes."""

INDICES = [
    (slice(None),),
    (1,),
    (slice(1, 4), slice(None, None, 2), 0),
    (Ellipsis, 1),
    (slice(None, None, -1), 2, slice(None), slice(1, 2)),
    (0, 0, 0, 0),
]
"""Selections that a partial read is checked with."""

SCALINGS = [
    (1.0, 0.0),
    (2.0, 1.0),
    (1.0, 3.0),
    (0.5, -7.25),
    (0.0, 5.0),
    (float("nan"), 5.0),
    (float("inf"), 5.0),
]
"""Slopes and intercepts, including those that nibabel ignores."""

DTYPES = ["u1", "i1", "i2", "u2", "i4", "u4", "i8", "f4", "f8", "c8", "c16"]
"""The stored types that NIfTI files can hold."""


def _values(dtype: str, shape: tuple = SHAPE) -> np.ndarray:
    values = np.arange(int(np.prod(shape))).reshape(shape) % 23 - 5
    if np.dtype(dtype).kind == "u":
        values = np.abs(values)
    if np.dtype(dtype).kind == "c":
        values = values + 1j * values[::-1]
    return values.astype(dtype)


def _nifti_bytes(
    values: np.ndarray,
    slope: float = float("nan"),
    inter: float = 0.0,
    version: int = 1,
    endian: str = "<",
    extension: bytes = b"",
) -> bytes:
    """Return a single NIfTI file, built field by field.

    The header is written with its slope and intercept as given, which
    nibabel would otherwise change when it writes an image.
    """
    header_class = nb.Nifti1Header if version == 1 else nb.Nifti2Header
    header = header_class(endianness=endian)
    header.set_data_dtype(values.dtype)
    header.set_data_shape(values.shape)
    header["scl_slope"] = slope
    header["scl_inter"] = inter
    if extension:
        header.extensions.append(nb.nifti1.Nifti1Extension(6, extension))
    stream = _io.BytesIO()
    header.write_to(stream)
    offset = header.get_data_offset()
    stream.write(b"\0" * (offset - stream.tell()))
    stored = values.astype(header.get_data_dtype())
    stream.write(stored.tobytes(order="F"))
    return stream.getvalue()


def _write(path, content: bytes):  # noqa: ANN001, ANN202
    if str(path).endswith(".gz"):
        content = gzip.compress(content)
    path.write_bytes(content)
    return path


def _check(proxy, reference) -> None:  # noqa: ANN001
    """Check a proxy against nibabel's proxy of the same voxels.

    nibabel scales the values in float64. brainhops keeps a stored
    floating-point type and scales an integer type in the floating-point
    type that NumPy gives for it and a float32. The values are therefore
    compared to the precision of that type when the file is scaled.
    """
    expected = np.asanyarray(reference)
    stored = np.dtype(reference.dtype)
    scaled = expected.dtype != stored
    want = np.result_type(stored, np.float32) if scaled else stored
    rtol = np.finfo(want).eps * 4 if scaled else 0

    def compare(values, target) -> None:  # noqa: ANN001
        if scaled:
            np.testing.assert_allclose(values, target, rtol=rtol)
        else:
            np.testing.assert_array_equal(values, target)

    assert tuple(proxy.shape) == tuple(reference.shape)
    assert proxy.dtype == want
    values = np.asarray(proxy)
    assert values.dtype == want
    compare(values, expected)
    for index in INDICES:
        if len([i for i in index if i is not Ellipsis]) > proxy.ndim:
            continue
        part = np.asarray(proxy[index])
        assert part.dtype.newbyteorder("=") == want.newbyteorder("=")
        compare(part, expected[index])


# ----------------------------------------------------------------------
#   NIFTI
# ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scaling", SCALINGS, ids=str)
@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_a_nifti_file_reads_what_nibabel_reads(
    tmp_path,  # noqa: ANN001
    dtype: str,
    scaling: tuple,
    suffix: str,
) -> None:
    content = _nifti_bytes(_values(dtype), *scaling)
    path = _write(tmp_path / f"image{suffix}", content)
    reference = nb.load(str(path))
    _, proxy = read_nifti(path)
    _check(proxy, reference.dataobj)
    if np.dtype(dtype).kind != "c":
        # get_fdata drops the imaginary part of complex values.
        expected = reference.get_fdata()
        np.testing.assert_array_equal(np.asarray(proxy), expected)
    scaled = (reference.dataobj.slope, reference.dataobj.inter) != (1, 0)
    assert isinstance(proxy, ScaledProxy) == scaled
    assert isinstance(proxy.root, NiftiProxy)
    assert whole_file(proxy) is proxy.root


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("version", [1, 2])
def test_byte_order_versions_and_extensions_are_read(
    tmp_path,  # noqa: ANN001
    endian: str,
    version: int,
) -> None:
    content = _nifti_bytes(
        _values("i2"), 2.0, 1.0, version, endian, extension=b"a comment"
    )
    path = _write(tmp_path / "image.nii", content)
    reference = nb.load(str(path))
    assert reference.dataobj.offset > 352
    raw, proxy = read_nifti(path)
    assert type(raw.header) is type(reference.header)
    _check(proxy, reference.dataobj)
    assert proxy.root.dtype == np.dtype(f"{endian}i2")


def test_an_infinite_intercept_is_refused_as_nibabel_refuses_it(
    tmp_path,  # noqa: ANN001
) -> None:
    content = _nifti_bytes(_values("i2"), 2.0, float("inf"))
    path = _write(tmp_path / "image.nii", content)
    with pytest.raises(nb.spatialimages.HeaderDataError):
        nb.load(str(path)).dataobj  # noqa: B018
    with pytest.raises(nb.spatialimages.HeaderDataError):
        read_nifti(path)


def test_rgb_voxels_are_read(tmp_path) -> None:  # noqa: ANN001
    rgb = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    values = np.zeros(SHAPE[:3], rgb)
    values["R"] = _values("u1", SHAPE[:3])
    values["B"] = 255 - values["R"]
    path = tmp_path / "rgb.nii"
    nb.save(nb.Nifti1Image(values, np.eye(4)), str(path))
    reference = nb.load(str(path))
    _, proxy = read_nifti(path)
    _check(proxy, reference.dataobj)


@pytest.mark.parametrize("suffix", [".hdr", ".hdr.gz"])
def test_the_image_file_of_a_pair_is_read(tmp_path, suffix: str) -> None:  # noqa: ANN001
    values = _values("i2")
    image = nb.Nifti1Pair(values, np.eye(4))
    image.header.set_slope_inter(0.25, 4.0)
    nb.save(image, str(tmp_path / f"pair{suffix}"))
    reference = nb.load(str(tmp_path / f"pair{suffix}"))
    compressed = suffix.endswith(".gz")
    opener = gzip.open if compressed else open
    with opener(tmp_path / f"pair{suffix}", "rb") as f:
        header = nb.Nifti1Pair.header_class.from_fileobj(f)
    data = tmp_path / ("pair.img.gz" if compressed else "pair.img")
    proxy = NiftiProxy.from_header(data, header)
    assert isinstance(proxy, ScaledProxy)
    _check(proxy, reference.dataobj)


@pytest.mark.parametrize("compress", [False, True])
def test_a_stream_of_the_caller_is_read_from_its_position(
    compress: bool,
) -> None:
    content = _nifti_bytes(_values("i2"), 2.0, 1.0)
    if compress:
        content = gzip.compress(content)
    reference = nb.Nifti1Image.from_bytes(_nifti_bytes(_values("i2"), 2, 1))
    stream = _io.BytesIO(b"prefix" + content + b"suffix")
    stream.seek(len(b"prefix"))
    _, proxy = read_nifti(stream)
    assert stream.tell() == len(b"prefix")
    stream.seek(0)
    _check(proxy, reference.dataobj)
    assert stream.tell() == 0


def test_a_stream_that_cannot_seek_is_read_into_memory() -> None:
    content = _nifti_bytes(_values("f4"))

    class _Pipe(_io.RawIOBase):
        def __init__(self) -> None:
            self.inner = _io.BytesIO(content)

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, buffer) -> int:  # noqa: ANN001
            data = self.inner.read(len(buffer))
            buffer[: len(data)] = data
            return len(data)

    _, proxy = read_nifti(_Pipe())
    np.testing.assert_array_equal(np.asarray(proxy), _values("f4"))


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_dask_blocks_hold_what_nibabel_reads(tmp_path, suffix: str) -> None:  # noqa: ANN001
    dask = pytest.importorskip("dask")
    content = _nifti_bytes(_values("i2"), 0.5, 2.0)
    path = _write(tmp_path / f"image{suffix}", content)
    reference = nb.load(str(path))
    slab_bytes = int(np.prod(SHAPE[:3])) * 2
    with dask.config.set({"array.chunk-size": f"{slab_bytes}B"}):
        _, proxy = read_nifti(path)
        lazy = proxy.to_dask()
    assert lazy.chunks[-1] == (1, 1)
    assert lazy.dtype == np.float32
    np.testing.assert_array_equal(lazy.compute(), reference.get_fdata())
    np.testing.assert_array_equal(
        lazy[1:4, :, ::-1, 1].compute(), reference.get_fdata()[1:4, :, ::-1, 1]
    )


def test_a_local_file_is_memory_mapped_unless_asked(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "image.nii", _nifti_bytes(_values("f4")))
    _, proxy = read_nifti(path)
    assert isinstance(np.asanyarray(proxy), np.memmap)
    _, proxy = read_nifti(path, mmap=False)
    assert not isinstance(np.asanyarray(proxy), np.memmap)
    with pytest.raises(ValueError, match="mmap"):
        read_nifti(path, mmap="w")


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_a_nibabel_proxy_is_replaced_by_the_same_lazy_array(
    tmp_path,  # noqa: ANN001
    suffix: str,
) -> None:
    path = _write(tmp_path / f"image{suffix}", _nifti_bytes(_values("i2"), 3))
    reference = nb.load(str(path))
    image = NiftiImage.from_nibabel(reference)
    assert isinstance(image.rawdata, ScaledProxy)
    assert whole_file(image.rawdata) is not None
    _check(image.rawdata, reference.dataobj)


def test_a_scaled_file_saved_over_itself_keeps_its_bytes(tmp_path) -> None:  # noqa: ANN001
    content = _nifti_bytes(_values("i2"), 0.5, 2.0)
    path = _write(tmp_path / "image.nii", content)
    copy = tmp_path / "copy.nii"
    shutil.copy(path, copy)
    image = NiftiImage.load(path)
    image.save(path)
    assert path.read_bytes() == content
    expected = nb.load(str(copy)).get_fdata()
    np.testing.assert_array_equal(np.asarray(image.data), expected)
    # Changed values are written over the file that the image reads.
    image = NiftiImage.load(path)
    image.data = np.asarray(image.data) + 1
    image.save(path)
    # nibabel stores them as int16 with a new scaling.
    written = nb.load(str(path)).get_fdata()
    np.testing.assert_allclose(written, expected + 1, atol=1e-3)


def test_the_type_of_scaled_values_survives_indexing(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "image.nii", _nifti_bytes(_values("u1"), 2, 1))
    _, proxy = read_nifti(path)
    assert proxy.dtype == np.float32
    assert proxy[1:3].dtype == np.float32
    assert proxy.T.dtype == np.float32
    assert np.asarray(proxy.T[0]).dtype == np.float32


# ----------------------------------------------------------------------
#   MGH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["u1", "i2", "i4", "f4"])
@pytest.mark.parametrize("shape", [SHAPE, SHAPE[:3]])
@pytest.mark.parametrize("suffix", [".mgh", ".mgz"])
def test_an_mgh_file_reads_what_nibabel_reads(
    tmp_path,  # noqa: ANN001
    dtype: str,
    shape: tuple,
    suffix: str,
) -> None:
    path = tmp_path / f"volume{suffix}"
    nb.save(nb.MGHImage(_values(dtype, shape), np.eye(4)), str(path))
    reference = nb.load(str(path))
    _, proxy = read_mgh(path)
    assert isinstance(proxy, MghProxy)
    assert proxy.dtype == reference.header.get_data_dtype()
    _check(proxy, reference.dataobj)


def test_an_mgh_stream_and_a_misnamed_file_are_read(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "volume.mgz"
    nb.save(nb.MGHImage(_values("f4"), np.eye(4)), str(path))
    reference = nb.load(str(path))
    stream = _io.BytesIO(b"prefix" + path.read_bytes())
    stream.seek(len(b"prefix"))
    _, proxy = read_mgh(stream)
    _check(proxy, reference.dataobj)
    misnamed = tmp_path / "volume.mgh"
    shutil.copy(path, misnamed)
    _, proxy = read_mgh(misnamed)
    assert proxy.source == str(misnamed)
    _check(proxy, reference.dataobj)


def test_mgh_dask_blocks_hold_what_nibabel_reads(tmp_path) -> None:  # noqa: ANN001
    dask = pytest.importorskip("dask")
    path = tmp_path / "volume.mgz"
    nb.save(nb.MGHImage(_values("f4"), np.eye(4)), str(path))
    reference = nb.load(str(path))
    slab_bytes = int(np.prod(SHAPE[:3])) * 4
    with dask.config.set({"array.chunk-size": f"{slab_bytes}B"}):
        lazy = read_mgh(path)[1].to_dask()
    assert lazy.chunks[-1] == (1, 1)
    np.testing.assert_array_equal(lazy.compute(), reference.get_fdata())


def test_an_mgh_nibabel_proxy_is_replaced(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "volume.mgz"
    nb.save(nb.MGHImage(_values("i2"), np.eye(4)), str(path))
    reference = nb.load(str(path))
    image = MghImage.from_nibabel(reference)
    assert isinstance(image.rawdata, MghProxy)
    _check(image.rawdata, reference.dataobj)


def test_an_mgh_file_saved_over_itself_keeps_its_bytes(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "volume.mgh"
    nb.save(nb.MGHImage(_values("i2"), np.eye(4)), str(path))
    content = path.read_bytes()
    image = MghImage.load(path)
    image.save(path)
    assert path.read_bytes() == content
    image = MghImage.load(path)
    image.data = np.asarray(image.data) * 2
    image.save(path)
    np.testing.assert_array_equal(
        np.asarray(nb.load(str(path)).dataobj), _values("i2") * 2
    )
