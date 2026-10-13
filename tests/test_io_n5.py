"""Tests of the N5 readers on small containers built by hand.

The containers are written as the specification of N5 describes them
(saalfeldlab/n5@c489d16, README.md, "File-system specification"),
without TensorStore, so that the reader is checked against the
specification: each block is a big-endian header followed by big-endian
samples, first dimension fastest, and the blocks at the upper edges are
smaller than `blockSize`, as the Java tools write them.
"""

import bz2
import gzip
import json
import lzma
import math
import os
import pickle
import struct
import warnings

import numpy as np
import pytest
import typing_extensions as tx

pytest.importorskip("tensorstore")

import brainhops.io as io  # noqa: E402
from brainhops.backends import backend  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    ParserContentError,
    ParserTypeError,
    WriterError,
)
from brainhops.io.images.n5 import (  # noqa: E402
    N5AttributeWarning,
    N5Image,
    N5Metadata,
    N5MultiScaleImage,
    N5Raw,
)
from brainhops.io.images.n5._data import _N5Proxy, node_location  # noqa: E402
from brainhops.io.metadata import MetadataFormat  # noqa: E402

_COMPRESS = {
    "raw": lambda data: data,
    "gzip": gzip.compress,
    "bzip2": bz2.compress,
    "xz": lzma.compress,
}


def write_dataset(
    directory: tx.Any,
    data: np.ndarray,
    block: tx.Sequence[int],
    compression: str = "gzip",
    attributes: tx.Optional[tx.Dict[str, tx.Any]] = None,
    skip: tx.Sequence[tx.Tuple[int, ...]] = (),
) -> None:
    """Write an N5 dataset whose `dimensions` are the shape of `data`.

    The array is indexed in the order of `dimensions`, first dimension
    fastest in a block. The blocks at the grid positions in `skip` are
    not written, since a dataset is sparse.
    """
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    content = {
        "dimensions": list(data.shape),
        "blockSize": list(block),
        "dataType": data.dtype.name,
        "compression": {"type": compression},
    }
    content.update(attributes or {})
    with open(os.path.join(directory, "attributes.json"), "w") as file:
        json.dump(content, file)
    grid = [math.ceil(n / b) for n, b in zip(data.shape, block)]
    for position in np.ndindex(*grid):
        if position in skip:
            continue
        begin = [g * b for g, b in zip(position, block)]
        values = data[tuple(slice(i, i + b) for i, b in zip(begin, block))]
        header = struct.pack(">HH", 0, values.ndim)
        header += struct.pack(f">{values.ndim}I", *values.shape)
        samples = values.astype(values.dtype.newbyteorder(">"))
        samples = _COMPRESS[compression](samples.tobytes(order="F"))
        name = os.path.join(directory, *map(str, position))
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with open(name, "wb") as file:
            file.write(header + samples)


def write_group(directory: tx.Any, attributes: tx.Dict[str, tx.Any]) -> None:
    """Write the attributes of a group."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "attributes.json"), "w") as file:
        json.dump(attributes, file)


def sample(
    shape: tx.Sequence[int] = (5, 3, 4), dtype: str = "uint16"
) -> np.ndarray:
    """Return an array whose values tell their position."""
    return np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)


def matrix(image: tx.Any) -> np.ndarray:
    """Return the voxel-to-physical matrix of an image."""
    return np.asarray(image.transformation.to(Affine).matrix, dtype=float)


def units(image: tx.Any) -> tx.List[tx.Any]:
    """Return the units of the physical axes of an image, as strings."""
    return [
        None if axis.unit is None else str(axis.unit)
        for axis in image.transformation.output.axes
    ]


# ----------------------------------------------------------------------
#   THE SPECIFICATION
# ----------------------------------------------------------------------


@pytest.mark.parametrize("compression", sorted(_COMPRESS))
def test_blocks_are_read_as_the_specification_writes_them(
    tmp_path: tx.Any, compression: str
) -> None:
    data = sample()
    write_dataset(tmp_path / "d", data, (2, 2, 3), compression)
    image = io.images.load(tmp_path / "d")
    assert isinstance(image, N5Image)
    assert isinstance(image.raw, _N5Proxy)
    assert image.shape == (5, 3, 4)
    assert image.raw.dtype == np.dtype("uint16")
    assert np.array_equal(np.asarray(image.raw), data)


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "int8", "int16", "uint32", "int64", "float32", "float64"],
)
def test_every_numeric_type_is_read(tmp_path: tx.Any, dtype: str) -> None:
    data = sample((3, 4), dtype) - (1 if dtype.startswith("int") else 0)
    write_dataset(tmp_path / "d", data, (2, 2), "raw")
    image = N5Image.load(tmp_path / "d")
    assert np.asarray(image.raw).dtype == np.dtype(dtype)
    assert np.array_equal(np.asarray(image.raw), data)


def test_a_missing_block_reads_as_zeros(tmp_path: tx.Any) -> None:
    data = sample()
    write_dataset(tmp_path / "d", data, (2, 2, 3), skip=[(1, 1, 0)])
    expected = data.copy()
    expected[2:4, 2:3, 0:3] = 0
    assert np.array_equal(
        np.asarray(N5Image.load(tmp_path / "d").raw), expected
    )


def test_a_dataset_is_named_by_its_attributes_file(tmp_path: tx.Any) -> None:
    data = sample()
    write_dataset(tmp_path / "d", data, (2, 2, 3))
    image = io.images.load(tmp_path / "d" / "attributes.json")
    assert isinstance(image, N5Image)
    assert np.array_equal(np.asarray(image.raw), data)
    with open(tmp_path / "d" / "attributes.json", "rb") as file:
        image = N5Image.from_fileobj(file)
    assert np.array_equal(np.asarray(image.raw), data)


def test_the_first_dimension_is_x(tmp_path: tx.Any) -> None:
    # The first dimension varies fastest in a block, so the first values
    # of the block (0, 0) run along it.
    data = sample((4, 2), "uint8")
    write_dataset(tmp_path / "d", data, (4, 2), "raw")
    with open(os.path.join(tmp_path, "d", "0", "0"), "rb") as file:
        content = file.read()
    assert content[4:12] == struct.pack(">II", 4, 2)
    assert list(content[12:16]) == [0, 2, 4, 6]
    image = N5Image.load(tmp_path / "d")
    assert [axis.name for axis in image.transformation.input.axes] == [
        "x",
        "y",
    ]
    assert np.array_equal(np.asarray(image.raw)[:, 0], [0, 2, 4, 6])


def test_a_block_of_variable_length_is_refused_when_read(
    tmp_path: tx.Any,
) -> None:
    write_dataset(tmp_path / "d", sample((2, 2)), (2, 2), "raw")
    with open(os.path.join(tmp_path, "d", "0", "0"), "r+b") as file:
        file.write(struct.pack(">H", 1))
    image = N5Image.load(tmp_path / "d")
    with pytest.raises(ParserContentError):
        np.asarray(image.raw)


@pytest.mark.parametrize(
    "attributes",
    [
        {"compression": {"type": "lz4"}},
        {"compression": {"type": "zstd"}},
        {"compression": None, "compressionType": "gzip"},
        {"dataType": "object"},
    ],
    ids=["lz4", "zstd", "version-0", "object"],
)
def test_unreadable_datasets_are_refused_when_loaded(
    tmp_path: tx.Any, attributes: tx.Dict[str, tx.Any]
) -> None:
    write_dataset(tmp_path / "d", sample((2, 2)), (2, 2), "raw")
    path = tmp_path / "d" / "attributes.json"
    content = {**json.loads(path.read_text()), **attributes}
    content = {key: value for key, value in content.items() if value}
    path.write_text(json.dumps(content))
    # The attributes are still read as metadata.
    assert isinstance(N5Metadata.load(tmp_path / "d").raw, N5Raw)
    with pytest.raises(ParserContentError):
        N5Image.load(tmp_path / "d")


# ----------------------------------------------------------------------
#   THE LAZY ARRAY
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    [
        (1, -1, 0),
        (slice(1, 4), Ellipsis),
        (Ellipsis, slice(None, None, 2)),
        (slice(-3, None), 0),
        (slice(None, None, -1),),
        ([0, 2],),
        (None, 1),
        np.s_[4:1],
    ],
)
def test_the_lazy_array_reads_numpy_indices(
    tmp_path: tx.Any, index: tx.Any
) -> None:
    data = sample()
    write_dataset(tmp_path / "d", data, (2, 2, 3))
    proxy = N5Image.load(tmp_path / "d").raw
    assert np.array_equal(proxy[index], data[index])


def test_an_empty_selection_reads_nothing(tmp_path: tx.Any) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    proxy = N5Image.load(tmp_path / "d").raw
    assert proxy[0:0].shape == (0, 3, 4)
    assert proxy._store is None
    proxy[0, 0, 0]
    assert proxy._store is not None
    # A pickled proxy keeps no open dataset, and reads again.
    clone = pickle.loads(pickle.dumps(proxy))
    assert clone._store is None
    assert np.array_equal(np.asarray(clone), sample())


def test_decoded_data_are_read_only(tmp_path: tx.Any) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    with backend("numpy"):
        data = N5Image.load(tmp_path / "d").data
        assert isinstance(data, np.ndarray)
        assert not data.flags.writeable
    with backend("dask"):
        data = N5Image.load(tmp_path / "d").data
        assert not isinstance(data, np.ndarray)
        assert np.array_equal(np.asarray(data), sample())


# ----------------------------------------------------------------------
#   THE RECORD AND THE METADATA
# ----------------------------------------------------------------------


def test_the_record_keeps_every_attribute(tmp_path: tx.Any) -> None:
    write_dataset(
        tmp_path / "d",
        sample(),
        (2, 2, 3),
        attributes={"custom": {"a": [1, 2]}, "compression": {"type": "gzip"}},
    )
    record = N5Metadata.load(tmp_path / "d").raw
    assert record.is_dataset
    assert record.dimensions == (5, 3, 4)
    assert record.block_size == (2, 2, 3)
    assert record.data_type == "uint16"
    assert record.compression == {"type": "gzip"}
    assert record.attributes["custom"] == {"a": [1, 2]}
    assert record.levels == ()
    copy = record.copy()
    assert copy == record
    copy.attributes["custom"]["a"].append(3)
    assert record.attributes["custom"] == {"a": [1, 2]}


def test_the_record_reads_version_0_compression() -> None:
    content = {
        "dimensions": [2],
        "blockSize": [2],
        "dataType": "uint8",
        "compressionType": "gzip",
    }
    record = N5Raw.from_bytes(json.dumps(content).encode())
    assert record.compression == {"type": "gzip"}


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"dimensions": [2]},
        {"dimensions": [2], "blockSize": [2, 2], "dataType": "uint8"},
        {"dimensions": [2], "blockSize": [0], "dataType": "uint8"},
        {"dimensions": [2], "blockSize": [2], "dataType": "complex64"},
        {"dimensions": [2], "blockSize": [2], "dataType": "uint8"},
    ],
)
def test_the_record_refuses_broken_dataset_attributes(content: tx.Any) -> None:
    content = json.dumps(content).encode()
    assert not N5Raw.sniff_bytes(content)
    with pytest.raises(ParserContentError):
        N5Raw.from_bytes(content)


def test_the_attributes_of_a_group_are_not_sniffed_but_are_read() -> None:
    content = json.dumps({"n5": "4.0.0"}).encode()
    assert not N5Raw.sniff_bytes(content)
    record = N5Raw.from_bytes(content)
    assert not record.is_dataset
    assert record.attributes == {"n5": "4.0.0"}


def test_the_metadata_never_reads_a_block(tmp_path: tx.Any) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    for name in ("0", "1", "2"):
        if (tmp_path / "d" / name).exists():
            for root, _, files in os.walk(tmp_path / "d" / name):
                for file in files:
                    with open(os.path.join(root, file), "wb") as stream:
                        stream.write(b"broken")
    metadata = MetadataFormat.load(tmp_path / "d")
    assert isinstance(metadata, N5Metadata)
    assert pickle.loads(pickle.dumps(metadata)).raw == metadata.raw
    image = N5Image.load(tmp_path / "d")
    assert image.shape == (5, 3, 4)
    with pytest.raises(ParserContentError):
        np.asarray(image.raw)


def test_options_are_refused(tmp_path: tx.Any) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    with pytest.raises(TypeError):
        N5Image.load(tmp_path / "d", mmap=True)
    with pytest.raises(TypeError):
        N5Metadata.load(tmp_path / "d", mmap=True)
    with pytest.raises(ParserContentError):
        N5Image.load(tmp_path / "d", level=1)


def test_a_remote_container_is_not_read() -> None:
    with pytest.raises(ParserTypeError):
        node_location("gs://bucket/container.n5/raw")


def test_a_directory_that_is_not_n5_is_not_recognized(
    tmp_path: tx.Any,
) -> None:
    write_group(tmp_path / "g", {"n5": "4.0.0"})
    assert not N5Image.sniff(tmp_path / "g")
    assert not N5MultiScaleImage.sniff(tmp_path / "g")
    assert not N5Metadata.sniff(tmp_path / "g")
    with pytest.raises(ParserContentError):
        N5Metadata.load(tmp_path / "g")


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def test_without_spatial_attributes_the_geometry_is_unknown(
    tmp_path: tx.Any,
) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    image = N5Image.load(tmp_path / "d")
    assert np.allclose(matrix(image), np.eye(4)[:3])
    assert units(image) == [None, None, None]
    assert image.transformation.output.name == "physical"


def test_a_pixel_resolution_object_gives_a_size_and_a_unit(
    tmp_path: tx.Any,
) -> None:
    attributes = {"pixelResolution": {"dimensions": [2, 3, 4], "unit": "um"}}
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    assert np.allclose(matrix(image), np.diag([2.0, 3.0, 4.0, 1.0])[:3])
    assert units(image) == ["micrometer"] * 3


def test_a_bare_pixel_resolution_has_no_unit(tmp_path: tx.Any) -> None:
    attributes = {"pixelResolution": [2, 3, 4], "resolution": [7, 7, 7]}
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    assert np.allclose(matrix(image), np.diag([2.0, 3.0, 4.0, 1.0])[:3])
    assert units(image) == [None, None, None]


def test_a_resolution_takes_the_unit_attribute(tmp_path: tx.Any) -> None:
    attributes = {"resolution": [4, 4, 40], "unit": "nm"}
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    assert np.allclose(matrix(image), np.diag([4.0, 4.0, 40.0, 1.0])[:3])
    assert units(image) == ["nanometer"] * 3


def test_a_pixel_unit_is_no_unit(tmp_path: tx.Any) -> None:
    attributes = {
        "pixelResolution": {"dimensions": [2, 2, 2], "unit": "pixel"}
    }
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert units(N5Image.load(tmp_path / "d")) == [None, None, None]


def test_downsampling_factors_shift_by_half_a_voxel(tmp_path: tx.Any) -> None:
    # n5-universe, N5SingleScaleMetadataParser.buildTransform: a voxel of a
    # level downsampled by f lies at r * (f * i + (f - 1) / 2).
    attributes = {
        "pixelResolution": {"dimensions": [2, 3, 4], "unit": "um"},
        "downsamplingFactors": [2, 4, 1],
    }
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    expected = [[4, 0, 0, 1], [0, 12, 0, 4.5], [0, 0, 4, 0]]
    assert np.allclose(matrix(N5Image.load(tmp_path / "d")), expected)


def test_the_offset_of_a_dataset_is_its_first_voxel(tmp_path: tx.Any) -> None:
    # n5-universe, N5GenericSingleScaleMetadataParser: the offset replaces
    # the translation of the transformation.
    attributes = {
        "resolution": [2, 3, 4],
        "downsamplingFactors": [2, 2, 2],
        "offset": [10, 20, 30],
    }
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    expected = [[4, 0, 0, 10], [0, 6, 0, 20], [0, 0, 8, 30]]
    assert np.allclose(matrix(N5Image.load(tmp_path / "d")), expected)


def test_attributes_cover_the_leading_axes(tmp_path: tx.Any) -> None:
    data = sample((3, 2, 2, 2))
    attributes = {"pixelResolution": {"dimensions": [2, 3, 4], "unit": "mm"}}
    write_dataset(tmp_path / "d", data, (2, 2, 2, 2), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    assert np.allclose(matrix(image), np.diag([2.0, 3.0, 4.0, 1.0, 1.0])[:4])
    assert units(image) == ["millimeter"] * 3 + [None]
    assert [axis.name for axis in image.transformation.input.axes] == [
        "x",
        "y",
        "z",
        "t",
    ]


@pytest.mark.parametrize(
    "attributes",
    [
        {"pixelResolution": [1, 2, 3, 4]},
        {"pixelResolution": [1, 0, 1]},
        {"resolution": "4,4,40"},
        {"downsamplingFactors": [2, None, 1]},
        {"pixelResolution": {"dimensions": [1, 1, 1], "unit": "blorps"}},
        {"resolution": [1, 1, 1], "unit": "s"},
    ],
)
def test_broken_spatial_attributes_are_ignored_with_a_warning(
    tmp_path: tx.Any, attributes: tx.Dict[str, tx.Any]
) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    with pytest.warns(N5AttributeWarning):
        transformation = image.transformation
    assert np.allclose(
        np.asarray(transformation.to(Affine).matrix), np.eye(4)[:3]
    )


# ----------------------------------------------------------------------
#   PYRAMIDS
# ----------------------------------------------------------------------


def write_pyramid(
    directory: tx.Any,
    group: tx.Dict[str, tx.Any],
    levels: tx.Sequence[tx.Dict[str, tx.Any]],
) -> np.ndarray:
    """Write a pyramid whose level `k` is downsampled by `2**k` in x and y."""
    data = sample((8, 4, 2))
    write_group(directory, group)
    for index, attributes in enumerate(levels):
        step = 2**index
        write_dataset(
            os.path.join(directory, f"s{index}"),
            np.ascontiguousarray(data[::step, ::step]),
            (2, 2, 2),
            attributes=attributes,
        )
    return data


def test_a_pyramid_is_read_whole_and_lazily(tmp_path: tx.Any) -> None:
    data = write_pyramid(tmp_path / "p", {}, [{}, {}, {}])
    pyramid = io.images.load(tmp_path / "p")
    assert isinstance(pyramid, N5MultiScaleImage)
    assert pyramid.nscales == 3
    assert pyramid.transformations == []
    for index, level in enumerate(pyramid.images):
        assert isinstance(level, N5Image)
        assert level.level == index
        assert level.metadata is pyramid.metadata
        assert isinstance(level.raw, _N5Proxy)
        assert level.raw._store is None
    step = 4
    expected = data[::step, ::step]
    assert np.array_equal(np.asarray(pyramid.images[2].raw), expected)
    assert np.array_equal(np.asarray(pyramid.data), data)


def test_one_level_of_a_pyramid_is_read_with_level(tmp_path: tx.Any) -> None:
    data = write_pyramid(tmp_path / "p", {}, [{}, {}])
    image = io.images.load(tmp_path / "p", level=-1)
    assert isinstance(image, N5Image)
    assert image.level == 1
    assert np.array_equal(np.asarray(image.raw), data[::2, ::2])
    assert len(image.metadata.raw.levels) == 2
    with pytest.raises(ParserContentError):
        N5Image.load(tmp_path / "p", level=2)


def test_a_pyramid_of_one_level_is_an_image(tmp_path: tx.Any) -> None:
    write_pyramid(tmp_path / "p", {}, [{}])
    image = io.images.load(tmp_path / "p")
    assert isinstance(image, N5Image)
    assert image.level == 0


def test_the_levels_stop_at_the_first_missing_one(tmp_path: tx.Any) -> None:
    write_pyramid(tmp_path / "p", {}, [{}, {}, {}])
    os.rename(tmp_path / "p" / "s1", tmp_path / "p" / "old")
    assert len(N5Metadata.load(tmp_path / "p").raw.levels) == 1


def test_paintera_puts_resolution_and_offset_on_the_group(
    tmp_path: tx.Any,
) -> None:
    # Paintera README: `resolution` and `offset` on the group, and
    # `downsamplingFactors` on each level from s1, whose own resolution and
    # offset are ignored. BigDataViewer writes the same for each time
    # point.
    group = {"multiScale": True, "resolution": [4, 4, 40], "offset": [8, 0, 0]}
    levels = [{}, {"downsamplingFactors": [2, 2, 1], "offset": [99, 99, 99]}]
    write_pyramid(tmp_path / "p", group, levels)
    pyramid = N5MultiScaleImage.load(tmp_path / "p")
    first, second = (matrix(level) for level in pyramid.images)
    assert np.allclose(first, [[4, 0, 0, 8], [0, 4, 0, 0], [0, 0, 40, 0]])
    assert np.allclose(second, [[8, 0, 0, 10], [0, 8, 0, 2], [0, 0, 40, 0]])
    # The voxels of both levels cover the same extent.
    for level in pyramid.images:
        edge = matrix(level) @ np.r_[np.array(level.shape) - 0.5, 1]
        assert np.allclose(edge, [8 + 4 * 7.5, 4 * 3.5, 40 * 1.5])


def test_the_deprecated_n5_viewer_layout_is_read(tmp_path: tx.Any) -> None:
    group = {
        "pixelResolution": {"dimensions": [1, 1, 2], "unit": "um"},
        "scales": [[1, 1, 1], [2, 2, 1]],
    }
    write_pyramid(tmp_path / "p", group, [{}, {}])
    pyramid = N5MultiScaleImage.load(tmp_path / "p")
    second = matrix(pyramid.images[1])
    assert np.allclose(second, [[2, 0, 0, 0.5], [0, 2, 0, 0.5], [0, 0, 2, 0]])
    assert units(pyramid.images[1]) == ["micrometer"] * 3


def test_the_attributes_of_a_level_take_precedence(tmp_path: tx.Any) -> None:
    # The modern n5-viewer layout, as n5-ij writes it: each level gives
    # the resolution of the finest level and its own factors.
    group = {"resolution": [9, 9, 9], "scales": [[1, 1, 1], [4, 4, 4]]}
    level = {
        "pixelResolution": {"dimensions": [1, 1, 1], "unit": "um"},
        "downsamplingFactors": [2, 2, 1],
    }
    write_pyramid(tmp_path / "p", group, [{}, level])
    second = matrix(N5MultiScaleImage.load(tmp_path / "p").images[1])
    assert np.allclose(second, [[2, 0, 0, 0.5], [0, 2, 0, 0.5], [0, 0, 1, 0]])


def test_a_level_without_factors_is_not_given_any(tmp_path: tx.Any) -> None:
    # n5-universe would infer factors of 2**k from the name s<k>, which
    # brainhops does not do.
    write_pyramid(tmp_path / "p", {"resolution": [3, 3, 3]}, [{}, {}])
    second = matrix(N5MultiScaleImage.load(tmp_path / "p").images[1])
    assert np.allclose(second, np.diag([3.0, 3.0, 3.0, 1.0])[:3])


def test_a_dataset_read_on_its_own_ignores_its_group(
    tmp_path: tx.Any,
) -> None:
    write_pyramid(tmp_path / "p", {"resolution": [3, 3, 3]}, [{}, {}])
    image = N5Image.load(tmp_path / "p" / "s1")
    assert image.level is None
    assert np.allclose(matrix(image), np.eye(4)[:3])


def test_a_pyramid_consolidates_its_levels(tmp_path: tx.Any) -> None:
    write_pyramid(tmp_path / "p", {}, [{}, {}])
    pyramid = N5MultiScaleImage.load(tmp_path / "p")
    raws = [level.raw for level in pyramid.images]
    pyramid.images[1].data = np.zeros((4, 2, 2))
    once = pyramid.consolidate()
    assert once.images[0].raw is raws[0]
    assert isinstance(once.images[1].raw, np.ndarray)
    assert pyramid.images[1].raw is raws[1]
    assert pyramid.consolidate(inplace=True) is pyramid
    assert pyramid.images[1]._data is None


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_n5_is_not_written(tmp_path: tx.Any) -> None:
    write_dataset(tmp_path / "d", sample(), (2, 2, 3))
    image = N5Image.load(tmp_path / "d")
    assert not hasattr(image, "save")
    with pytest.raises(WriterError):
        io.save(image, tmp_path / "copy", hint="n5")


def test_an_n5_image_is_saved_to_another_format(tmp_path: tx.Any) -> None:
    pytest.importorskip("nibabel")
    attributes = {"pixelResolution": {"dimensions": [2, 3, 4], "unit": "mm"}}
    write_dataset(tmp_path / "d", sample(), (2, 2, 3), attributes=attributes)
    image = N5Image.load(tmp_path / "d")
    io.save(image, tmp_path / "copy.nii.gz")
    back = io.images.load(tmp_path / "copy.nii.gz")
    assert np.array_equal(np.asarray(back.data), sample())
    sizes = np.linalg.norm(matrix(back)[:, :3], axis=0)
    assert np.allclose(sizes, [2, 3, 4])
