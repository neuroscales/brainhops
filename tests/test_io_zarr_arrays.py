"""Tests for plain Zarr arrays: their record, lazy array and writer.

A plain Zarr array is read as a `ZarrImage`, which holds the document of
the array as a `ZarrRaw` record and the array as a lazy proxy. These tests
cover what the conformance harness does not: both versions of the Zarr
format, stores written by a driver, the bytes of the documents, atomic
saves of directories, remote stores, and the lazy array itself.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import replace

from brainhops._core.dependencies import has_abczarr_driver
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, Identity
from brainhops.errors import UnrepresentableTransformationError
from brainhops.io.base.parsers import WriterError

abczarr = pytest.importorskip("abczarr")

from abczarr.errors import UnsupportedZarrOperation  # noqa: E402

from brainhops.io.common.zarr import ZarrMetadata, ZarrRaw  # noqa: E402
from brainhops.io.common.zarr import _data as zarr_data  # noqa: E402
from brainhops.io.common.zarr._data import ZarrProxy  # noqa: E402
from brainhops.io.images.zarr import ZarrImage  # noqa: E402

pytestmark = pytest.mark.skipif(
    not has_abczarr_driver(), reason="needs abczarr and a driver"
)

DATA = np.arange(64, dtype="int16").reshape(4, 4, 4)


def _tree(path: str) -> dict:
    """Return the bytes of every file of a directory, by relative name."""
    tree = {}
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            tree[os.path.relpath(full, path)] = open(full, "rb").read()
    return tree


def _native(path: str, version: int, attributes: dict = None) -> str:
    """Write a store with the driver of abczarr, not with brainhops."""
    abczarr.create(
        path,
        abczarr.ArrayConfig(zarr_version=version, chunks=(2, 2, 2)),
        data=DATA,
    )
    if attributes:
        abczarr.open(path, mode="r+").update_attributes(attributes)
    return path


# ---- versions and bytes ----------------------------------------------


@pytest.mark.parametrize("version", [2, 3])
def test_a_store_of_a_driver_is_saved_byte_for_byte(
    tmp_path: Path, version: int
) -> None:
    source = _native(str(tmp_path / "src.zarr"), version, {"a": 1})
    image = ZarrImage.load(source)
    assert image.metadata.raw.zarr_version == version
    image.save(str(tmp_path / "copy.zarr"))
    assert _tree(str(tmp_path / "copy.zarr")) == _tree(source)


@pytest.mark.parametrize("version", [2, 3])
def test_a_store_keeps_its_version_when_its_data_changes(
    tmp_path: Path, version: int
) -> None:
    source = _native(str(tmp_path / "src.zarr"), version)
    image = ZarrImage.load(source)
    image.data = DATA + 1
    image.save(str(tmp_path / "changed.zarr"))
    back = ZarrImage.load(str(tmp_path / "changed.zarr"))
    assert back.metadata.raw.zarr_version == version
    assert back.metadata.raw.chunks == (2, 2, 2)
    assert np.array_equal(np.asarray(back.data), DATA + 1)


def test_a_new_store_is_written_in_version_3(tmp_path: Path) -> None:
    ZarrImage(data=DATA).save(str(tmp_path / "new.zarr"))
    assert os.path.exists(str(tmp_path / "new.zarr" / "zarr.json"))
    back = ZarrImage.load(str(tmp_path / "new.zarr"))
    assert back.metadata.raw.zarr_version == 3
    # The option writes another version.
    ZarrImage(data=DATA).save(str(tmp_path / "old.zarr"), zarr_version=2)
    assert (
        ZarrImage.load(str(tmp_path / "old.zarr")).metadata.raw.zarr_version
        == 2
    )


@pytest.mark.parametrize("version", [2, 3])
def test_an_edited_document_alone_is_written_again(
    tmp_path: Path, version: int
) -> None:
    source = _native(str(tmp_path / "src.zarr"), version, {"a": 1})
    image = ZarrImage.load(source)
    record = image.metadata.raw
    # A whole number that becomes a float is a change of the document.
    edited = record.document.update_attributes({"a": 1.0})
    image.metadata = ZarrMetadata.from_raw(replace(record, document=edited))
    image.save(str(tmp_path / "edited.zarr"))
    before, after = _tree(source), _tree(str(tmp_path / "edited.zarr"))
    changed = sorted(key for key in before if before[key] != after.get(key))
    assert changed == (["zarr.json"] if version == 3 else [".zattrs"])
    back = ZarrImage.load(str(tmp_path / "edited.zarr"))
    assert json.dumps(back.metadata.raw.attributes) == '{"a": 1.0}'


def test_an_equal_document_keeps_its_bytes() -> None:
    document = abczarr.ArrayConfig(shape=(2,), dtype="f4").to_metadata()
    content = b'{ "zarr_format": 3, "node_type": "array", "shape": [2] }'
    record = ZarrRaw(document=document, stored=(("zarr.json", content),))
    # The stored bytes do not give the document, so it is written again.
    assert record.documents()["zarr.json"] != content
    stored = json.dumps(document.to_json()).encode()
    record = ZarrRaw(document=document, stored=(("zarr.json", stored),))
    assert record.documents()["zarr.json"] == stored


def test_the_record_describes_the_array(tmp_path: Path) -> None:
    source = _native(str(tmp_path / "src.zarr"), 3, {"a": 1})
    record = ZarrMetadata.load(source).raw
    assert record.shape == (4, 4, 4)
    assert record.dtype == np.dtype("int16")
    assert record.chunks == (2, 2, 2)
    assert record.attributes == {"a": 1}
    assert record.zarr_version == 3


# ---- consolidation ----------------------------------------------------


def test_options_describe_the_store_again(tmp_path: Path) -> None:
    source = _native(str(tmp_path / "src.zarr"), 3, {"a": 1})
    image = ZarrImage.load(source)
    once = image.consolidate(chunks=(4, 4, 4), dtype="float32")
    assert once.metadata.raw.chunks == (4, 4, 4)
    assert once.metadata.raw.dtype == np.dtype("float32")
    # The attributes of the record are kept.
    assert once.metadata.raw.attributes == {"a": 1}
    once.save(str(tmp_path / "rechunked.zarr"))
    back = ZarrImage.load(str(tmp_path / "rechunked.zarr"))
    assert back.metadata.raw == once.metadata.raw
    assert np.array_equal(np.asarray(back.data), DATA.astype("float32"))


def test_an_unknown_option_is_refused(tmp_path: Path) -> None:
    ZarrImage(data=DATA).save(str(tmp_path / "a.zarr"))
    image = ZarrImage.load(str(tmp_path / "a.zarr"))
    with pytest.raises(TypeError):
        image.consolidate(level=1)
    with pytest.raises(TypeError):
        image.save(str(tmp_path / "b.zarr"), level=1)
    with pytest.raises(TypeError):
        ZarrImage.load(str(tmp_path / "a.zarr"), level=1)


def test_a_geometry_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    ZarrImage(data=DATA).save(str(tmp_path / "a.zarr"))
    image = ZarrImage.load(
        str(tmp_path / "a.zarr"),
        transformation=Affine(np.diag([2.0, 3.0, 4.0, 1.0])[:3]),
    )
    with pytest.raises(UnrepresentableTransformationError):
        image.consolidate()
    with pytest.raises(UnrepresentableTransformationError):
        image.save(str(tmp_path / "b.zarr"))
    assert not (tmp_path / "b.zarr").exists()
    # An identity stores no geometry, so it is written.
    image.transformations = [Identity()]
    image.save(str(tmp_path / "b.zarr"))
    assert _tree(str(tmp_path / "b.zarr")) == _tree(str(tmp_path / "a.zarr"))


# ---- directories ------------------------------------------------------


def test_a_failed_save_leaves_the_store_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ZarrImage(data=DATA).save(str(tmp_path / "a.zarr"))
    before = _tree(str(tmp_path / "a.zarr"))
    image = ZarrImage(data=DATA + 1)

    def fail(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(zarr_data, "_store", fail)
    with pytest.raises(OSError):
        image.save(str(tmp_path / "a.zarr"))
    assert _tree(str(tmp_path / "a.zarr")) == before
    assert sorted(os.listdir(str(tmp_path))) == ["a.zarr"]


def test_a_save_over_its_own_store_reads_the_new_store(tmp_path: Path) -> None:
    path = str(tmp_path / "a.zarr")
    ZarrImage(data=DATA).save(path)
    image = ZarrImage.load(path)
    image.data = DATA * 2
    image.save(path, chunks=(4, 4, 4))
    assert isinstance(image.rawdata, ZarrProxy)
    assert image._data is None
    assert image.metadata.raw.chunks == (4, 4, 4)
    assert image.metadata.raw == ZarrImage.load(path).metadata.raw
    assert np.array_equal(np.asarray(image.data), DATA * 2)


def test_a_remote_save_over_its_own_store_is_refused() -> None:
    pytest.importorskip("upath")
    from bagof.paths import Path

    store = Path("memory://brainhops-tests/arrays/own.zarr")
    try:
        ZarrImage(data=DATA).save(store)
    except UnsupportedZarrOperation as error:
        # The TensorStore driver, the only Zarr driver available on
        # Python 3.8, refuses memory:// locations, because the memory
        # store of TensorStore is not the memory file system of fsspec.
        pytest.skip(f"the Zarr driver cannot write to memory://: {error}")
    image = ZarrImage.load(store)
    image.data = DATA + 1
    with pytest.raises(WriterError, match="remote store"):
        image.save(store)
    assert np.array_equal(np.asarray(ZarrImage.load(store).data), DATA)
    # Another remote store is written in place.
    other = Path("memory://brainhops-tests/arrays/other.zarr")
    image.save(other)
    assert np.array_equal(np.asarray(ZarrImage.load(other).data), DATA + 1)


def test_a_store_taken_as_remote_is_written_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without a remote file system at hand, local paths are taken as
    # remote, so that the writer follows the branch of remote stores.
    from brainhops.io.images.zarr import _image as zarr_image

    monkeypatch.setattr(zarr_image, "is_local", lambda location: False)
    monkeypatch.setattr(zarr_data, "_local", lambda location: False)
    path = str(tmp_path / "own.zarr")
    ZarrImage(data=DATA).save(path)
    image = ZarrImage.load(path)
    image.data = DATA + 1
    with pytest.raises(WriterError, match="remote store"):
        image.save(path)
    assert np.array_equal(np.asarray(ZarrImage.load(path).data), DATA)
    image.save(str(tmp_path / "other.zarr"))
    back = ZarrImage.load(str(tmp_path / "other.zarr"))
    assert np.array_equal(np.asarray(back.data), DATA + 1)
    assert back.metadata.raw == image.consolidate().metadata.raw


# ---- the lazy array ---------------------------------------------------


class _Refusing:
    """An array that fails when it is read, with the shape of `DATA`."""

    shape = DATA.shape
    dtype = DATA.dtype
    chunks = (2, 2, 2)

    def __getitem__(self, index: object) -> None:
        raise AssertionError("the array was read")


def test_an_empty_selection_reads_nothing() -> None:
    proxy = ZarrProxy(_Refusing(), "nowhere")
    assert np.asarray(proxy[0:0]).shape == (0, 4, 4)
    assert np.asarray(proxy[..., 2:2]).shape == (4, 4, 0)
    with pytest.raises(AssertionError):
        np.asarray(proxy[0])


def test_a_permuted_proxy_reads_the_transposed_array(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    node = abczarr.create(str(tmp_path / "p.zarr"), data=data)
    perm = (2, 0, 1)
    proxy = ZarrProxy(node, str(tmp_path / "p.zarr"), perm)
    expected = np.transpose(data, perm)
    assert proxy.shape == expected.shape
    assert np.array_equal(np.asarray(proxy), expected)
    for index in [
        (slice(1, 3), 0),
        (Ellipsis, 1),
        (0, slice(None), slice(0, 2)),
        (slice(None, None, -1),),
        (np.array([0, 2]),),
    ]:
        assert np.array_equal(proxy[index], expected[index]), index


def test_the_dask_backend_keeps_the_array_lazy(tmp_path: Path) -> None:
    da = pytest.importorskip("dask.array")
    from brainhops._core.backends import get_backend, set_backend

    path = str(tmp_path / "a.zarr")
    _native(path, 3)
    previous = get_backend()
    set_backend("dask")
    try:
        data = ZarrImage.load(path).to_backend()
    finally:
        set_backend(previous)
    assert isinstance(data, da.Array)
    assert data.chunksize == (2, 2, 2)
    assert np.array_equal(data.compute(), DATA)


def test_an_image_from_a_data_model_is_written(tmp_path: Path) -> None:
    image = ZarrImage.from_instance(SingleScaleImage(data=DATA))
    assert image.metadata is None
    image.save(str(tmp_path / "a.zarr"))
    assert np.array_equal(
        np.asarray(ZarrImage.load(str(tmp_path / "a.zarr")).data), DATA
    )
