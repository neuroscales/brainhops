"""Tests of the pyramids that the OME-Zarr writers build while they write.

A single scale saved to OME-Zarr with `io.save` is written one level at
a time, and each coarse level is computed from the levels already
written, read back from the store (#638). These tests check that the
store is the one that a pyramid built in memory, with
`to(OmeZarrImage, nscales=...)`, writes, that the source is read only
once, and that a pyramid built in memory is written as it is.
"""

# stdlib
import itertools
import threading
from pathlib import Path

# dependencies
import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import DisplacementField
from brainhops.errors import WriterError

da = pytest.importorskip("dask.array")
# The modules import abczarr themselves, so the skip must come first.
abczarr = pytest.importorskip("abczarr")

from brainhops.io.common.zarr._data import read_back  # noqa: E402
from brainhops.io.images.zarr import OmeZarrImage  # noqa: E402
from brainhops.io.transformations.zarr import OmeZarrField  # noqa: E402

_CASES = [
    ("mean", "exact"),
    ("mean", "edges"),
    ("gaussian", "centers"),
    ("binomial", "exact"),
    ("nearest", "exact"),
    ("nearest", "edges"),
    ("max", "exact"),
    ("mode", "exact"),
]
"""The modes and the alignments with which the pyramids are built."""


class _CountingArray:
    """An array that counts how many times each of its voxels is read."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.reads = np.zeros(values.shape, dtype=int)
        self.lock = threading.Lock()

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        with self.lock:
            self.reads[index] += 1
        return self.values[index]


def _image(dtype: str = "float32") -> SingleScaleImage:
    values = np.random.default_rng(0).random((40, 48, 36)) * 100
    data = da.from_array(values.astype(dtype), chunks=(16, 16, 16))
    return SingleScaleImage(data=data)


def _counted_image() -> tx.Tuple[SingleScaleImage, _CountingArray]:
    values = np.random.default_rng(2).random((40, 48, 36))
    source = _CountingArray(values.astype("float32"))
    data = da.from_array(source, chunks=(16, 16, 16), asarray=True)
    return SingleScaleImage(data=data), source


def _assert_same_store(new: str, old: str) -> None:
    count = len(abczarr.open_group(old).ome.multiscales[0].datasets)
    assert count >= 2
    for index in range(count):
        written = np.asarray(abczarr.open_group(new)[str(index)][...])
        expected = np.asarray(abczarr.open_group(old)[str(index)][...])
        assert written.dtype == expected.dtype
        np.testing.assert_allclose(written, expected, rtol=1e-6, atol=1e-4)
    for name in ["zarr.json"] + [f"{i}/zarr.json" for i in range(count)]:
        assert (Path(new) / name).read_text() == (Path(old) / name).read_text()


@pytest.mark.parametrize(("mode", "align"), _CASES)
def test_a_saved_image_is_written_as_its_pyramid_in_memory(
    tmp_path: Path, mode: str, align: str
) -> None:
    image = _image("int16" if mode == "mode" else "float32")
    options = dict(nscales=4, mode=mode, align=align, min_size=5)
    new, old = str(tmp_path / "new.ome.zarr"), str(tmp_path / "old.ome.zarr")

    io.save(image, new, **options)
    image.to(OmeZarrImage, **options).save(old)

    _assert_same_store(new, old)


def test_a_saved_image_builds_a_full_pyramid_by_default(
    tmp_path: Path,
) -> None:
    image = _image()
    new, old = str(tmp_path / "new.ome.zarr"), str(tmp_path / "old.ome.zarr")

    io.save(image, new)
    image.to(OmeZarrImage).save(old)

    _assert_same_store(new, old)
    io.save(image, str(tmp_path / "one.ome.zarr"), nscales=1)
    assert io.load(str(tmp_path / "one.ome.zarr")).nscales == 1


@pytest.mark.parametrize("mode", ["mean", "nearest"])
def test_a_saved_field_is_written_as_its_pyramid_in_memory(
    tmp_path: Path, mode: str
) -> None:
    values = np.random.default_rng(1).random((12, 40, 36, 3))
    field = DisplacementField(field=da.from_array(values, chunks=16))
    options = dict(nscales=3, mode=mode, min_size=5)
    new, old = str(tmp_path / "new.ome.zarr"), str(tmp_path / "old.ome.zarr")

    io.save(field, new, **options)
    field.to(OmeZarrField, **options).save(old)

    _assert_same_store(new, old)


@pytest.mark.parametrize("mode", ["mean", "nearest"])
def test_the_source_of_a_saved_pyramid_is_read_once(
    tmp_path: Path, mode: str
) -> None:
    # Each coarse level is computed from the store, so the source is
    # read once, to write the finest level, whatever the number of
    # levels and even in the mode "nearest".
    image, source = _counted_image()

    io.save(image, str(tmp_path / "pyramid.ome.zarr"), mode=mode, min_size=5)

    assert source.reads.min() == source.reads.max() == 1


def test_a_coarse_level_of_a_saved_pyramid_holds_no_task_of_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _image()
    graphs = []  # type: tx.List[tx.Any]
    store = abczarr.ZarrGroup.create_array

    def create_array(self: tx.Any, name: str, **kwargs: tx.Any) -> tx.Any:
        data = kwargs.get("data")
        if isinstance(data, da.Array):
            graphs.append(set(data.dask))
        return store(self, name, **kwargs)

    monkeypatch.setattr(abczarr.ZarrGroup, "create_array", create_array)
    io.save(image, str(tmp_path / "pyramid.ome.zarr"), min_size=5)

    assert len(graphs) > 2
    assert set(image.data.dask) <= graphs[0]
    for graph in graphs[1:]:
        assert not set(image.data.dask) & graph


def test_a_pyramid_in_memory_is_never_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, source = _counted_image()
    pyramid = image.to(OmeZarrImage, min_size=5)

    def refuse(*args: tx.Any) -> None:
        raise AssertionError("A level was read back from the store.")

    monkeypatch.setattr(
        "brainhops.io.images.zarr._multiscale.next_scale", refuse
    )
    pyramid.save(str(tmp_path / "pyramid.ome.zarr"))

    # Each level is computed from the source.
    assert source.reads.max() > 1


def test_a_pyramid_of_several_levels_refuses_the_options(
    tmp_path: Path,
) -> None:
    pyramid = _image().to(OmeZarrImage, nscales=2)

    with pytest.raises(WriterError, match="already holds 2 scales"):
        pyramid.save(str(tmp_path / "pyramid.ome.zarr"), nscales=3)


def test_a_saved_pyramid_reads_its_levels_back_one_chunk_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under the default backend, a level read back is a Dask array whose
    # blocks are the chunks of the store, so that a coarse level never
    # loads a whole level into memory.
    image = _image()
    blocks = []  # type: tx.List[tx.Tuple[tx.Any, tx.Any]]

    def counted(node: tx.Any, perm: tx.Any = None) -> tx.Any:
        level = read_back(node, perm)
        blocks.append((level, node))
        return level

    monkeypatch.setattr(
        "brainhops.io.images.zarr._multiscale.read_back", counted
    )
    io.save(image, str(tmp_path / "pyramid.ome.zarr"), min_size=5)

    assert blocks
    for level, node in blocks:
        assert isinstance(level, da.Array)
        largest = max(int(np.prod(block)) for block in _block_shapes(level))
        assert largest <= int(np.prod(node.chunks))
    finest = blocks[0][0]
    assert finest.numblocks != (1,) * finest.ndim


def _block_shapes(array: tx.Any) -> tx.List[tx.Tuple[int, ...]]:
    return list(itertools.product(*array.chunks))
