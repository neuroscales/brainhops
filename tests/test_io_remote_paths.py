"""Tests of file names taken from remote paths, without a network.

Dispatch matches the name against format extensions when reading and
writing. The name comes from the path text alone, so a remote path is never
made local, and the name of a URL ends before its query.
"""

import io as _io
import os

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops._core.dependencies import has_abczarr_driver
from brainhops.datamodel.base import DataModelBase
from brainhops.io.base import Format
from brainhops.io.base._base import (
    register_format,
)
from brainhops.io.base._dispatch import Source, _to_filename
from brainhops.io.base.parsers import (
    ParserError,
    TextFileWriter,
)


class RemotePath(os.PathLike):
    """An in-memory remote path whose os.fspath raises and counts its calls."""

    store: tx.ClassVar[tx.Dict[str, str]] = {}
    fspath_calls: tx.ClassVar[int] = 0

    def __init__(self, url: str) -> None:
        self.url = url

    def __fspath__(self) -> str:
        type(self).fspath_calls += 1
        raise NotImplementedError("a remote path has no local path")

    def __str__(self) -> str:
        return self.url

    def exists(self) -> bool:
        return self.url in self.store

    def open(self, mode: str = "r") -> tx.IO:  # noqa: A003
        if "w" not in mode:
            return _io.StringIO(self.store[self.url])
        url, store = self.url, self.store

        class _Writer(_io.StringIO):
            def close(self) -> None:
                store[url] = self.getvalue()
                super().close()

        return _Writer()


@pytest.fixture(autouse=True)
def _fresh_store() -> tx.Iterator[None]:
    RemotePath.store = {}
    RemotePath.fspath_calls = 0
    yield


# ----------------------------------------------------------------------
#   READING THE NAME
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "location, name",
    [
        ("s3://bucket/dir/x.nii.gz", "x.nii.gz"),
        ("gs://bucket/dir/x.nii.gz", "x.nii.gz"),
        ("https://host/dir/x.nii.gz?token=abc", "x.nii.gz"),
        # A query may contain its own slashes and dots.
        ("https://host/dir/x.nii.gz?sig=a/b.zip", "x.nii.gz"),
        ("https://host/dir/x.nii.gz#part", "x.nii.gz"),
        # A store is a directory, named with or without a trailing slash.
        ("s3://bucket/dir/store.zarr/", "store.zarr"),
        ("s3://bucket/dir/store.ome.zarr?version=2", "store.ome.zarr"),
        # The last link of an fsspec chain is the file.
        ("simplecache::s3://bucket/x.nii.gz", "x.nii.gz"),
        ("file:///tmp/x.nii.gz", "x.nii.gz"),
        ("/tmp/dir/x.nii.gz", "x.nii.gz"),
        ("/tmp/dir/store.zarr/", "store.zarr"),
        # A local name keeps what a URL would call a query.
        ("/tmp/dir/x?.nii", "x?.nii"),
    ],
)
def test_the_name_is_read_from_the_path(location: str, name: str) -> None:
    assert _to_filename(location) == name
    assert _to_filename(RemotePath(location)) == name


def test_a_remote_path_is_never_made_local() -> None:
    _to_filename(RemotePath("s3://bucket/x.nii.gz"))
    assert RemotePath.fspath_calls == 0


def test_local_paths_and_open_files_are_named(tmp_path) -> None:  # noqa: ANN001
    from pathlib import Path

    target = tmp_path / "x.nii.gz"
    target.write_bytes(b"")
    assert _to_filename(Path(target)) == "x.nii.gz"
    with os.scandir(tmp_path) as entries:
        assert [_to_filename(e) for e in entries] == ["x.nii.gz"]
    with open(target, "rb") as f:
        assert _to_filename(f) == "x.nii.gz"


def test_content_and_unnamed_streams_name_nothing() -> None:
    assert _to_filename(b"bytes") is None
    assert _to_filename(_io.BytesIO()) is None


def test_a_source_names_its_path_and_never_its_content() -> None:
    # A path is named from its text and never made local.
    assert Source("https://host/x.nii.gz?token=abc").name == "x.nii.gz"
    assert Source(RemotePath("s3://bucket/x.nii.gz")).name == "x.nii.gz"
    assert RemotePath.fspath_calls == 0
    # In-memory text names nothing, even when it looks like a path.
    assert Source.content("s3://bucket/x.nii.gz").name is None
    assert Source.content("x.nii.gz").name is None
    assert Source.content(b"x.nii.gz").name is None


# ----------------------------------------------------------------------
#   SAVING AND READING THROUGH A REMOTE PATH
# ----------------------------------------------------------------------


class Note(DataModelBase):
    """A one-field data model for a text format."""

    text: str = ""


@pytest.fixture
def note_formats() -> tx.Iterator[tx.Tuple[type, type]]:
    """Two text note formats, .n and .long.n, registered for one test."""

    def make(name: str, extensions: tx.Tuple[str, ...]) -> type:
        def to_lines(self, **kwargs) -> tx.Iterator[str]:  # noqa: ANN001
            yield f"{name}:{self.text}"

        namespace = {"to_lines": to_lines, "EXTENSIONS": extensions}
        bases = (Note, Format, TextFileWriter)
        return register_format(type(name, bases, namespace))

    made = (make("Short", (".n",)), make("Long", (".long.n",)))
    yield made
    for fmt in made:
        for base in fmt.__mro__:
            if "_REGISTRY" in base.__dict__:
                base._REGISTRY.discard(fmt)


def test_save_chooses_the_format_from_the_url_path(note_formats) -> None:  # noqa: ANN001
    # The query also ends in .n, but only the URL path decides.
    url = "https://host/dir/x.long.n?sig=a/b.n"
    io.save(Note(text="hi"), RemotePath(url))
    assert RemotePath.store[url].rstrip("\n") == "Long:hi"
    assert RemotePath.fspath_calls == 0


def test_from_other_reads_a_remote_path_rather_than_building() -> None:
    from brainhops.io.images import ImageFormat

    # Reading fails, but the path is neither made local nor taken as data.
    with pytest.raises(ParserError):
        ImageFormat.from_any(RemotePath("s3://bucket/missing.nii.gz"))
    assert RemotePath.fspath_calls == 0


@pytest.mark.skipif(not has_abczarr_driver(), reason="needs a zarr driver")
def test_a_zarr_store_round_trips_through_in_memory_storage() -> None:
    # fsspec memory storage stands in for the cloud.
    pytest.importorskip("upath")
    from bagof.paths import Path

    from brainhops.datamodel.images import SingleScaleImage
    from brainhops.io.images.zarr import ZarrImage

    data = np.arange(8, dtype="float32").reshape(2, 2, 2)
    store = Path("memory://brainhops-tests/remote.zarr")
    io.save(SingleScaleImage(data=data), store)
    back = io.images.ImageFormat.from_any(store)
    assert isinstance(back, ZarrImage)
    assert np.array_equal(np.asarray(back.data), data)
