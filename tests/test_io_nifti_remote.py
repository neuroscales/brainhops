"""
Tests for reading and writing NIfTI files at remote paths, without
touching a network.

`nibabel` opens a file it is handed by name as a local file, so a remote
path must be opened through its own backend and `nibabel` handed the
open file. A local path must still go to `nibabel` by name, so that it
memory-maps the voxels.

Two kinds of remote path stand in for cloud storage: a minimal in-memory
path object, which needs no optional dependency, and fsspec's `memory://`
file system, which needs universal-pathlib.
"""

import gzip
import io as _io
import os

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

from bagof.paths import Path  # noqa: E402

import brainhops.io as io  # noqa: E402
from brainhops.io.base import nifti as nifti_base  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.nifti import NiftiVoxelToRAS  # noqa: E402

AFFINE = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
DATA = np.arange(24, dtype="float32").reshape(2, 3, 4)
EXTENSIONS = [".nii", ".nii.gz"]


def _encode(ext: str, data: np.ndarray = DATA) -> bytes:
    """The bytes of a NIfTI-1 file, gzipped for a `.gz` name."""
    image = nb.Nifti1Image(data, AFFINE)
    image.header.set_sform(AFFINE, code=2)
    image.header.set_qform(AFFINE, code=1)
    raw = image.to_bytes()
    return gzip.compress(raw) if ext.endswith(".gz") else raw


def _decode(raw: bytes) -> nb.Nifti1Image:
    """A NIfTI-1 image from its bytes, gzipped or not."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nb.Nifti1Image.from_bytes(raw)


# ----------------------------------------------------------------------
#   A REMOTE PATH WITH NO BACKEND
# ----------------------------------------------------------------------


class RemotePath(os.PathLike):
    """
    A path to remote storage, held in memory.

    Like a cloud path, it gives its URL as `str()` and cannot be turned
    into a local path: `os.fspath` raises.
    """

    store: tx.ClassVar[tx.Dict[str, bytes]] = {}

    def __init__(self, url: str) -> None:
        self.url = url

    def __fspath__(self) -> str:
        raise NotImplementedError("a remote path has no local path")

    def __str__(self) -> str:
        return self.url

    def exists(self) -> bool:
        return self.url in self.store

    def open(self, mode: str = "rb", **kwargs) -> tx.BinaryIO:  # noqa: A003
        if "w" not in mode:
            return _io.BytesIO(self.store[self.url])
        url, store = self.url, self.store

        class _Writer(_io.BytesIO):
            def close(self) -> None:
                store[url] = self.getvalue()
                super().close()

        return _Writer()


@pytest.fixture
def remote() -> tx.Iterator[tx.Callable[[str], RemotePath]]:
    """Make an in-memory remote path, its store emptied after the test."""
    RemotePath.store = {}
    yield RemotePath
    RemotePath.store = {}


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_an_image_is_read_from_a_remote_path(remote, ext: str) -> None:  # noqa: ANN001
    url = f"s3://bucket/dir/image{ext}"
    remote.store[url] = _encode(ext)
    image = NiftiImage.load(remote(url))
    assert np.array_equal(np.asarray(image.data), DATA)
    assert np.allclose(image.header.get_best_affine(), AFFINE)


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_an_affine_is_read_from_a_remote_path(remote, ext: str) -> None:  # noqa: ANN001
    url = f"https://host/dir/affine{ext}"
    remote.store[url] = _encode(ext)
    affine = NiftiVoxelToRAS.load(remote(url))
    assert np.allclose(affine.matrix, AFFINE[:3])


@pytest.mark.parametrize("ext", EXTENSIONS)
@pytest.mark.parametrize("query", ["", "?sig=a/b.nii"])
def test_an_image_is_written_to_a_remote_path(
    remote,  # noqa: ANN001
    ext: str,
    query: str,
) -> None:
    # Only the path of the URL says whether to compress, not its query.
    source = f"https://host/in{ext}"
    target = f"https://host/out{ext}{query}"
    remote.store[source] = _encode(ext)
    NiftiImage.load(remote(source)).save(remote(target))

    written = remote.store[target]
    assert (written[:2] == b"\x1f\x8b") == ext.endswith(".gz")
    back = _decode(written)
    assert np.array_equal(np.asarray(back.dataobj), DATA)
    assert np.allclose(back.affine, AFFINE)


def test_a_remote_header_template_is_read(remote) -> None:  # noqa: ANN001
    url = "s3://bucket/template.nii.gz"
    remote.store[url] = _encode(".nii.gz")
    header = nifti_base._like_header(remote(url))
    assert np.allclose(header.get_best_affine(), AFFINE)


def test_a_nifti2_stream_is_read(remote) -> None:  # noqa: ANN001
    url = "s3://bucket/image2.nii"
    remote.store[url] = nb.Nifti2Image(DATA, AFFINE).to_bytes()
    image = NiftiImage.load(remote(url))
    assert isinstance(image.image, nb.Nifti2Image)
    assert np.array_equal(np.asarray(image.data), DATA)
    header = nifti_base._like_header(remote(url))
    assert isinstance(header, nb.Nifti2Header)


# ----------------------------------------------------------------------
#   A LOCAL PATH STILL GOES TO NIBABEL BY NAME
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_a_local_path_is_loaded_by_name(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
    ext: str,
) -> None:
    target = tmp_path / f"image{ext}"
    target.write_bytes(_encode(ext))

    from_filename = nb.Nifti1Image.from_filename
    calls = []

    def spy(klass, filename, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls.append(filename)
        return from_filename.__func__(klass, filename, *args, **kwargs)

    def no_stream(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("a local path was read as a stream")

    monkeypatch.setattr(nb.Nifti1Image, "from_filename", classmethod(spy))
    monkeypatch.setattr(nifti_base, "_nifti_from_stream", no_stream)

    for file in (str(target), target, Path(target), f"file://{target}"):
        calls.clear()
        image = NiftiImage.load(file)
        assert calls == [str(target)]
        assert np.array_equal(np.asarray(image.data), DATA)


def test_a_local_uncompressed_image_is_memory_mapped(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "image.nii"
    target.write_bytes(_encode(".nii"))
    image = NiftiImage.load(target)
    assert isinstance(image.image.dataobj.get_unscaled(), np.memmap)


# ----------------------------------------------------------------------
#   FSSPEC'S IN-MEMORY FILE SYSTEM
# ----------------------------------------------------------------------


@pytest.fixture
def memory() -> tx.Iterator[str]:
    """A fresh directory in fsspec's in-memory file system, for paths
    that universal-pathlib opens."""
    pytest.importorskip("upath")
    fsspec = pytest.importorskip("fsspec")
    fs = fsspec.filesystem("memory")
    root = "memory://brainhops-tests/nifti"
    yield root
    if fs.exists(root):
        fs.rm(root, recursive=True)


def _put(url: str, raw: bytes) -> None:
    with Path(url).open("wb") as f:
        f.write(raw)


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_io_load_reads_a_memory_url(memory: str, ext: str) -> None:
    url = f"{memory}/image{ext}"
    _put(url, _encode(ext))
    image = io.load(url)
    assert type(image) is NiftiImage
    assert np.array_equal(np.asarray(image.data), DATA)


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_concrete_readers_read_a_memory_path(memory: str, ext: str) -> None:
    url = f"{memory}/image{ext}"
    _put(url, _encode(ext))
    for file in (url, Path(url)):
        image = NiftiImage.load(file)
        assert np.array_equal(np.asarray(image.data), DATA)
        affine = NiftiVoxelToRAS.load(file)
        assert np.allclose(affine.matrix, AFFINE[:3])


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_a_memory_path_is_sniffed(memory: str, ext: str) -> None:
    url = f"{memory}/image{ext}"
    _put(url, _encode(ext))
    assert io.sniff(url) is NiftiImage
    assert NiftiImage.sniff(Path(url))


@pytest.mark.parametrize("ext", EXTENSIONS)
def test_an_image_round_trips_through_a_memory_path(
    memory: str, ext: str
) -> None:
    source, target = f"{memory}/in{ext}", Path(f"{memory}/out{ext}")
    _put(source, _encode(ext))
    NiftiImage.load(source).save(target)
    assert (target.read_bytes()[:2] == b"\x1f\x8b") == ext.endswith(".gz")

    back = NiftiImage.load(target)
    assert np.array_equal(np.asarray(back.data), DATA)
    assert np.allclose(back.header.get_best_affine(), AFFINE)
