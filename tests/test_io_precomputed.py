"""Tests for Neuroglancer precomputed volumes.

The raw volumes are built by hand from the specification
(`src/datasource/precomputed/volume.md` at
google/neuroglancer@abbdfc5ea97b81c0c0bf04a414f5c59f4d84d27f), so that the
reader is checked against the specification and not against the writer of
TensorStore. The sharded, JPEG and compressed segmentation volumes are
written with TensorStore, which brainhops does not use to write.
"""

import io as _io
import json
import math
import os
import pickle

import numpy as np
import pytest
from bagof.magic import replace
from bagof.paths import Path

ts = pytest.importorskip("tensorstore")

import brainhops.io as io  # noqa: E402
import brainhops.io.images.precomputed._data as _data  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    ParserExistsError,
    ParserTypeError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.images.precomputed import (  # noqa: E402
    PrecomputedImage,
    PrecomputedMetadata,
    PrecomputedMultiScaleImage,
    PrecomputedRaw,
    PrecomputedScale,
    PrecomputedSharding,
)
from brainhops.io.images.precomputed._data import (  # noqa: E402
    PrecomputedProxy,
    dataset_location,
    kvstore_spec,
    read_info,
)
from brainhops.io.metadata import MetadataFormat  # noqa: E402

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------


def _scale(key, size, resolution, offset=None, chunk=(2, 2, 2), **extra):  # noqa: ANN001, ANN003, ANN202
    scale = {
        "key": key,
        "size": list(size),
        "resolution": list(resolution),
        "chunk_sizes": [list(chunk)],
        "encoding": "raw",
    }
    if offset is not None:
        scale["voxel_offset"] = list(offset)
    scale.update(extra)
    return scale


def _info(scales, data_type="uint16", channels=1, **extra):  # noqa: ANN001, ANN003, ANN202
    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "image",
        "data_type": data_type,
        "num_channels": channels,
        "scales": scales,
    }
    info.update(extra)
    return info


def _write_raw(directory, info, arrays):  # noqa: ANN001, ANN202
    """Write a raw, unsharded volume, as `volume.md` describes it.

    Each array is `[x, y, z, channel]`, and each chunk is written in
    Fortran order as little-endian values, in a file named by the bounds
    of the chunk, which include the voxel offset.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "info"), "w") as file:
        json.dump(info, file)
    dtype = np.dtype(info["data_type"]).newbyteorder("<")
    for scale, values in zip(info["scales"], arrays):
        chunk = scale["chunk_sizes"][0]
        offset = scale.get("voxel_offset", [0, 0, 0])
        os.makedirs(os.path.join(directory, scale["key"]), exist_ok=True)
        grid = [math.ceil(n / c) for n, c in zip(scale["size"], chunk)]
        for position in np.ndindex(*grid):
            begin = [g * c for g, c in zip(position, chunk)]
            end = [
                min(b + c, n) for b, c, n in zip(begin, chunk, scale["size"])
            ]
            block = values[tuple(slice(b, e) for b, e in zip(begin, end))]
            name = "_".join(
                f"{b + o}-{e + o}" for b, e, o in zip(begin, end, offset)
            )
            path = os.path.join(directory, scale["key"], name)
            with open(path, "wb") as file:
                file.write(block.astype(dtype).tobytes(order="F"))


def _values(shape, channels=1, dtype="uint16"):  # noqa: ANN001, ANN202
    count = int(np.prod(shape)) * channels
    return np.arange(count).reshape(tuple(shape) + (channels,)).astype(dtype)


@pytest.fixture
def pyramid(tmp_path):  # noqa: ANN001, ANN201
    """A raw volume of two channels and two scales, with voxel offsets."""
    fine = _values((5, 4, 3), channels=2)
    coarse = fine[::2, ::2, ::2] + 1000
    info = _info(
        [
            _scale("fine", (5, 4, 3), (4, 4, 40), offset=(1, 2, 3)),
            _scale("coarse", coarse.shape[:3], (8, 8, 80), offset=(0, 1, 1)),
        ],
        channels=2,
    )
    directory = tmp_path / "volume"
    _write_raw(str(directory), info, [fine, coarse])
    return directory, fine, coarse


@pytest.fixture
def single(tmp_path):  # noqa: ANN001, ANN201
    """A raw volume of one channel and one scale."""
    values = _values((4, 3, 2), dtype="float32")
    info = _info([_scale("s", (4, 3, 2), (1, 2, 3))], data_type="float32")
    directory = tmp_path / "single"
    _write_raw(str(directory), info, [values])
    return directory, values[..., 0]


def _tensorstore_volume(directory, values, scale, **multiscale):  # noqa: ANN001, ANN003, ANN202
    """Write a volume of one scale with TensorStore."""
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(directory)},
            "multiscale_metadata": multiscale,
            "scale_metadata": scale,
            "create": True,
        }
    ).result()
    store[...] = values
    return directory


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


def test_the_record_holds_the_info_file() -> None:
    sharding = {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "identity",
        "minishard_bits": 1,
        "shard_bits": 2,
    }
    info = _info(
        [
            _scale("a", (5, 4, 3), (4, 4, 40), encoding="RAW"),
            _scale(
                "b",
                (3, 2, 2),
                (8, 8, 80),
                offset=(1, -2, 3),
                encoding="compressed_segmentation",
                compressed_segmentation_block_size=[8, 8, 8],
                sharding=sharding,
                hidden=True,
                jpeg_quality=75,
            ),
        ],
        data_type="UInt32",
        type="segmentation",
        mesh="mesh",
    )
    record = PrecomputedRaw.from_bytes(json.dumps(info).encode())
    assert record.volume_type == "segmentation"
    assert record.data_type == "uint32"
    assert record.dtype == np.dtype("uint32")
    assert record.mesh == "mesh"
    assert record.skeletons is None
    first, second = record.scales
    # The voxel offset is optional, and encodings are read without case.
    assert first.voxel_offset == (0, 0, 0)
    assert first.encoding == "raw"
    assert first.sharding is None
    assert first.chunk_size == (2, 2, 2)
    assert first.grid_shape == (3, 2, 2)
    assert second.voxel_offset == (1, -2, 3)
    assert second.resolution == (8.0, 8.0, 80.0)
    assert second.compressed_segmentation_block_size == (8, 8, 8)
    assert second.hidden
    assert second.sharding == PrecomputedSharding(
        preshift_bits=0, hash="identity", minishard_bits=1, shard_bits=2
    )
    assert second.sharding.data_encoding == "raw"
    # A record is frozen, and equal to a record of the same members.
    assert record == PrecomputedRaw.from_bytes(json.dumps(info).encode())
    with pytest.raises(AttributeError):
        record.mesh = "other"
    assert record.copy() == record


@pytest.mark.parametrize(
    "data_type",
    [
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "float32",
    ],
)
def test_every_data_type_of_neuroglancer_is_read(data_type: str) -> None:
    info = _info([_scale("s", (1, 1, 1), (1, 1, 1))], data_type=data_type)
    record = PrecomputedRaw.from_bytes(json.dumps(info).encode())
    assert record.dtype == np.dtype(data_type)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"@type": "neuroglancer_legacy_mesh"}, "neuroglancer_legacy_mesh"),
        ({"type": "mesh"}, "type"),
        ({"data_type": "float64"}, "data_type"),
        ({"num_channels": 0}, "num_channels"),
        ({"type": "segmentation", "num_channels": 2}, "one channel"),
        ({"scales": []}, "scales"),
        ({"scales": [_scale("s", (1, 1), (1, 1, 1))]}, "size"),
        ({"scales": [_scale("s", (1, 1, 1), (0, 1, 1))]}, "resolution"),
        (
            {"scales": [_scale("s", (1, 1, 1), (1, 1, 1), chunk=(0, 1, 1))]},
            "chunk",
        ),
        (
            {"scales": [_scale("s", (1, 1, 1), (1, 1, 1), encoding="gif")]},
            "gif",
        ),
        (
            {
                "scales": [
                    _scale(
                        "s",
                        (1, 1, 1),
                        (1, 1, 1),
                        encoding="compressed_segmentation",
                    )
                ]
            },
            "compressed_segmentation_block_size",
        ),
        (
            {
                "scales": [
                    _scale(
                        "s",
                        (1, 1, 1),
                        (1, 1, 1),
                        compressed_segmentation_block_size=[8, 8, 8],
                    )
                ]
            },
            "compressed_segmentation_block_size",
        ),
        (
            {
                "scales": [
                    _scale(
                        "s",
                        (1, 1, 1),
                        (1, 1, 1),
                        sharding={"@type": "neuroglancer_uint64_sharded_v2"},
                    )
                ]
            },
            "sharded_v1",
        ),
    ],
)
def test_an_info_file_that_breaks_the_specification_is_refused(
    change: dict, message: str
) -> None:
    info = _info([_scale("s", (1, 1, 1), (1, 1, 1))])
    info.update(change)
    content = json.dumps(info).encode()
    with pytest.raises(ParserContentError, match=message):
        PrecomputedRaw.from_bytes(content)
    assert PrecomputedRaw.sniff_bytes(content) == Confidence.NO
    with pytest.raises(SnifferContentError):
        PrecomputedRaw.sniff_bytes(content, error=True)


def test_a_sharded_scale_lists_a_single_chunk_size() -> None:
    sharding = {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "murmurhash3_x86_128",
        "minishard_bits": 0,
        "shard_bits": 0,
    }
    scale = _scale("s", (4, 4, 4), (1, 1, 1), sharding=sharding)
    scale["chunk_sizes"] = [[2, 2, 2], [4, 4, 4]]
    with pytest.raises(ParserContentError, match="single chunk size"):
        PrecomputedRaw.from_bytes(json.dumps(_info([scale])).encode())


def test_content_that_is_not_json_is_refused() -> None:
    assert PrecomputedRaw.sniff_bytes(b"\x5c\x01\x00\x00") == Confidence.NO
    with pytest.raises(ParserContentError, match="JSON"):
        PrecomputedRaw.from_bytes(b"{not json")
    with pytest.raises(TypeError):
        PrecomputedRaw.from_bytes(b"{}", mode="r")


# ----------------------------------------------------------------------
#   METADATA AND LOCATIONS
# ----------------------------------------------------------------------


def test_the_metadata_is_read_from_the_directory_or_the_info(pyramid) -> None:  # noqa: ANN001
    directory, _, _ = pyramid
    by_directory = PrecomputedMetadata.load(directory)
    by_info = PrecomputedMetadata.load(directory / "info")
    by_stream = PrecomputedMetadata.load(
        _io.BytesIO((directory / "info").read_bytes())
    )
    assert by_directory.raw == by_info.raw == by_stream.raw
    assert [s.key for s in by_directory.raw.scales] == ["fine", "coarse"]
    assert isinstance(MetadataFormat.load(directory), PrecomputedMetadata)
    assert MetadataFormat.sniff(directory) is PrecomputedMetadata
    clone = pickle.loads(pickle.dumps(by_directory))
    assert clone.raw == by_directory.raw
    with pytest.raises(TypeError):
        PrecomputedMetadata.load(directory, level=1)


def test_the_metadata_never_reads_a_chunk(pyramid, tmp_path) -> None:  # noqa: ANN001
    directory, _, _ = pyramid
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "info").write_bytes((directory / "info").read_bytes())
    assert (
        PrecomputedMetadata.load(bare).raw
        == PrecomputedMetadata.load(directory).raw
    )


def test_a_name_that_is_not_a_volume_is_refused(tmp_path) -> None:  # noqa: ANN001
    empty = tmp_path / "empty"
    empty.mkdir()
    other = tmp_path / "other.json"
    other.write_text("{}")
    for name in (empty, other, tmp_path / "missing"):
        assert PrecomputedMetadata.sniff(name) == Confidence.NO
        with pytest.raises(ParserExistsError):
            dataset_location(name)
    (empty / "info").write_text(
        json.dumps({"@type": "neuroglancer_skeletons"})
    )
    assert PrecomputedMetadata.sniff(empty) == Confidence.NO
    with pytest.raises(ParserContentError):
        PrecomputedMetadata.load(empty)


def test_remote_names_are_given_to_tensorstore() -> None:
    # The URL is not opened: no network is used by the tests.
    volume = Path("gs://bucket/volume")
    assert dataset_location("gs://bucket/volume/") == volume
    assert dataset_location("gs://bucket/volume/info") == volume
    https = Path("https://host/a/volume")
    assert dataset_location("https://host/a/volume") == https
    assert kvstore_spec(volume) == "gs://bucket/volume/"
    spec = ts.KvStore.Spec(kvstore_spec(https)).to_json()
    assert spec == {
        "driver": "http",
        "base_url": "https://host",
        "path": "/a/volume/",
    }


def test_a_remote_volume_is_read_through_tensorstore(  # noqa: ANN201
    pyramid,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
):
    # A `file://` URL that is taken for a remote one goes through the
    # key-value stores of TensorStore, as a `gs://` URL would.
    directory, fine, _ = pyramid
    expected = PrecomputedMetadata.load(directory).raw
    monkeypatch.setattr(_data, "_is_local_location", lambda location: False)
    location = Path("file://" + str(directory))
    record = PrecomputedRaw.from_bytes(read_info(location))
    assert record == expected
    proxy = PrecomputedProxy(location, record, record.scales[0])
    assert np.array_equal(np.asarray(proxy), fine)
    with pytest.raises(ParserExistsError):
        read_info(Path("file://" + str(directory / "fine")))


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def test_a_volume_of_several_scales_is_read_as_a_pyramid(pyramid) -> None:  # noqa: ANN001
    directory, fine, coarse = pyramid
    image = io.images.load(directory)
    assert isinstance(image, PrecomputedMultiScaleImage)
    assert image.transformations == []
    assert [level.level for level in image.images] == [0, 1]
    assert all(isinstance(level, PrecomputedImage) for level in image.images)
    # The levels share the metadata of the pyramid.
    assert all(level.metadata is image.metadata for level in image.images)
    with backend("numpy"):
        assert np.array_equal(image.images[0].data, fine)
        assert np.array_equal(image.images[1].data, coarse)
    assert image.images[1].shape == coarse.shape
    assert io.images.sniff(directory) is PrecomputedMultiScaleImage
    assert (
        type(io.images.load(directory / "info")) is PrecomputedMultiScaleImage
    )
    with pytest.raises(WriterError):
        io.save(image, directory.parent / "copy")


def test_one_scale_is_read_with_level(pyramid) -> None:  # noqa: ANN001
    directory, fine, coarse = pyramid
    image = io.images.load(directory, level=1)
    assert type(image) is PrecomputedImage
    assert image.level == 1
    with backend("numpy"):
        assert np.array_equal(image.data, coarse)
        assert np.array_equal(
            PrecomputedImage.load(directory, level=-1).data, coarse
        )
        assert np.array_equal(PrecomputedImage.load(directory).data, fine)
    assert io.images.sniff(directory, level=0) is PrecomputedImage
    with pytest.raises(ParserContentError, match="no scale 2"):
        PrecomputedImage.load(directory, level=2)
    with pytest.raises(TypeError):
        PrecomputedImage.load(directory, mmap=True)
    with pytest.raises(TypeError):
        PrecomputedMultiScaleImage.load(directory, level=0)


def test_a_volume_of_one_scale_is_read_as_an_image(single) -> None:  # noqa: ANN001
    directory, values = single
    image = io.images.load(directory)
    assert type(image) is PrecomputedImage
    # A single channel has no channel axis.
    assert image.shape == values.shape
    with backend("numpy"):
        assert np.array_equal(image.data, values)
    with pytest.raises(TypeError):
        image.data[(0,) * image.ndim] = 1
    pyramid = PrecomputedMultiScaleImage.load(directory)
    assert len(pyramid.images) == 1


def test_the_geometry_puts_voxel_centres_where_neuroglancer_does(
    pyramid,  # noqa: ANN001
) -> None:
    directory, _, _ = pyramid
    fine, coarse = io.images.load(directory).images
    matrix = fine.transformation.to(Affine).matrix
    # The corner of voxel `voxel_offset` lies at `voxel_offset * resolution`
    # nanometres, so the centre of the first voxel is half a voxel further.
    expected = np.zeros((4, 5))
    expected[:3, :3] = np.diag([4.0, 4.0, 40.0])
    expected[3, 3] = 1.0
    expected[:3, 4] = (np.array([1, 2, 3]) + 0.5) * [4.0, 4.0, 40.0]
    assert np.allclose(matrix, expected)
    output = fine.transformation.output
    assert output.name == "physical"
    assert [axis.name for axis in output.axes] == ["x", "y", "z", "c"]
    assert [str(axis.unit) for axis in output.axes[:3]] == ["nanometer"] * 3
    assert fine.transformation.input.name == "voxel"
    # The coarse voxel (j, k, l) covers the fine voxels of its block, so
    # its centre is the centre of that block in the physical space.
    corner = np.array([0, 1, 1]) * [8.0, 8.0, 80.0]
    centre = coarse.transformation.to(Affine).homogeneous_matrix @ [
        0,
        0,
        0,
        0,
        1,
    ]
    assert np.allclose(centre[:3], corner + [4.0, 4.0, 40.0])


def test_loading_reads_no_chunk(pyramid) -> None:  # noqa: ANN001
    directory, fine, _ = pyramid
    image = io.images.load(directory)
    level = image.images[0]
    assert isinstance(level.rawdata, PrecomputedProxy)
    assert level.rawdata._store is None
    assert "_cache_data" not in vars(level)
    # An empty selection, which Dask reads to learn the type of an array,
    # opens nothing either.
    assert level.rawdata[:0].shape == (0, 4, 3, 2)
    assert level.rawdata._store is None
    # A broken chunk is found only when the voxels are read.
    (directory / "coarse" / "0-2_1-3_1-3").write_bytes(b"short")
    broken = io.images.load(directory).images[1]
    with pytest.raises(ValueError, match="0-2_1-3_1-3"):
        np.asarray(broken.rawdata)
    assert np.array_equal(np.asarray(level.rawdata), fine)


@pytest.mark.parametrize(
    "index",
    [
        (1, slice(None), 2),
        (slice(-2, None), Ellipsis, 1),
        (slice(0, 5, 2), slice(1, 3)),
        (Ellipsis, -1),
        (slice(None, None, -1), 0),
        (None, 0),
        ([0, 2], 1),
        (np.int64(-1),),
    ],
)
def test_the_lazy_array_reads_selections_like_numpy(pyramid, index) -> None:  # noqa: ANN001
    directory, fine, _ = pyramid
    proxy = io.images.load(directory).images[0].rawdata
    assert np.array_equal(proxy[index], fine[index])


def test_the_lazy_array_pickles_without_its_store(pyramid) -> None:  # noqa: ANN001
    directory, fine, _ = pyramid
    proxy = io.images.load(directory).images[0].rawdata
    np.asarray(proxy)
    assert proxy._store is not None
    clone = pickle.loads(pickle.dumps(proxy))
    assert clone._store is None
    assert np.array_equal(np.asarray(clone), fine)


def test_dask_reads_the_volume_lazily(pyramid) -> None:  # noqa: ANN001
    pytest.importorskip("dask.array")
    directory, fine, _ = pyramid
    level = io.images.load(directory).images[0]
    with backend("dask"):
        data = level.to_backend()
    assert type(data).__module__.startswith("dask")
    assert level.rawdata._store is None
    assert np.array_equal(np.asarray(data), fine)


def test_a_changed_volume_is_refused_when_it_is_read(pyramid) -> None:  # noqa: ANN001
    directory, _, _ = pyramid
    image = io.images.load(directory, level=0)
    info = json.loads((directory / "info").read_text())
    info["scales"][0]["size"] = [6, 4, 3]
    (directory / "info").write_text(json.dumps(info))
    with pytest.raises(ParserContentError, match="shape"):
        np.asarray(image.rawdata)


def test_a_stream_needs_a_name(pyramid) -> None:  # noqa: ANN001
    directory, fine, _ = pyramid
    with open(directory / "info", "rb") as file:
        image = PrecomputedImage.load(file)
        pyramid_image = PrecomputedMultiScaleImage.load(file)
    assert np.array_equal(np.asarray(image.rawdata), fine)
    assert len(pyramid_image.images) == 2
    content = _io.BytesIO((directory / "info").read_bytes())
    with pytest.raises(ParserTypeError, match="file name"):
        PrecomputedImage.from_fileobj(content)
    with pytest.raises(ParserContentError, match="bytes"):
        PrecomputedImage.from_bytes(content.getvalue())
    with pytest.raises(ParserContentError, match="bytes"):
        PrecomputedMultiScaleImage.from_bytes(content.getvalue())
    assert PrecomputedImage.sniff_bytes(content.getvalue()) == Confidence.NO


# ----------------------------------------------------------------------
#   ENCODINGS AND SHARDING
# ----------------------------------------------------------------------


def test_a_sharded_volume_is_read(tmp_path) -> None:  # noqa: ANN001
    values = _values((9, 7, 5), channels=1, dtype="uint32")
    sharding = {
        "@type": "neuroglancer_uint64_sharded_v1",
        "preshift_bits": 0,
        "hash": "identity",
        "minishard_bits": 1,
        "shard_bits": 2,
        "minishard_index_encoding": "gzip",
        "data_encoding": "gzip",
    }
    directory = _tensorstore_volume(
        tmp_path / "sharded",
        values,
        {
            "size": [9, 7, 5],
            "encoding": "raw",
            "chunk_size": [4, 4, 4],
            "resolution": [1, 1, 1],
            "voxel_offset": [3, 0, 2],
            "sharding": sharding,
        },
        type="segmentation",
        data_type="uint32",
        num_channels=1,
    )
    image = io.images.load(directory)
    assert type(image) is PrecomputedImage
    assert image.metadata.raw.scales[0].sharding.shard_bits == 2
    # The shards are named by their number in hexadecimal.
    key = image.metadata.raw.scales[0].key
    assert all(name.endswith(".shard") for name in os.listdir(directory / key))
    assert np.array_equal(np.asarray(image.rawdata), values[..., 0])


def test_compressed_segmentation_is_read(tmp_path) -> None:  # noqa: ANN001
    values = (_values((9, 8, 5), dtype="uint64") % 5) * 1000
    directory = _tensorstore_volume(
        tmp_path / "segmentation",
        values,
        {
            "size": [9, 8, 5],
            "encoding": "compressed_segmentation",
            "compressed_segmentation_block_size": [4, 4, 2],
            "chunk_size": [4, 4, 4],
            "resolution": [1, 1, 1],
        },
        type="segmentation",
        data_type="uint64",
        num_channels=1,
    )
    image = io.images.load(directory)
    assert image.metadata.raw.scales[0].compressed_segmentation_block_size == (
        4,
        4,
        2,
    )
    assert np.array_equal(np.asarray(image.rawdata), values[..., 0])


def test_jpeg_is_read(tmp_path) -> None:  # noqa: ANN001
    # A smooth ramp keeps the error of the lossy encoding small.
    x, y, z = np.meshgrid(
        np.arange(8), np.arange(8), np.arange(4), indexing="ij"
    )
    values = (x * 8 + y * 4 + z * 2).astype("uint8")[..., None]
    values = np.concatenate([values, values, values], axis=-1)
    directory = _tensorstore_volume(
        tmp_path / "jpeg",
        values,
        {
            "size": [8, 8, 4],
            "encoding": "jpeg",
            "chunk_size": [8, 8, 4],
            "resolution": [1, 1, 1],
        },
        type="image",
        data_type="uint8",
        num_channels=3,
    )
    image = io.images.load(directory)
    data = np.asarray(image.rawdata)
    assert data.shape == (8, 8, 4, 3)
    assert np.abs(data.astype(int) - values.astype(int)).max() <= 8


@pytest.mark.parametrize("encoding", ["png", "compresso", "jxl", "crackle"])
def test_an_encoding_that_tensorstore_cannot_read_is_refused(
    tmp_path,  # noqa: ANN001
    encoding: str,
) -> None:
    info = _info(
        [_scale("s", (2, 2, 2), (1, 1, 1), encoding=encoding)],
        data_type="uint8",
    )
    directory = tmp_path / encoding
    directory.mkdir()
    (directory / "info").write_text(json.dumps(info))
    # The metadata is read, but the image is refused when it is loaded.
    assert (
        PrecomputedMetadata.load(directory).raw.scales[0].encoding == encoding
    )
    with pytest.raises(ParserContentError, match=encoding):
        PrecomputedImage.load(directory)
    with pytest.raises(ParserContentError, match=encoding):
        io.images.load(directory)


# ----------------------------------------------------------------------
#   READ ONLY
# ----------------------------------------------------------------------


def test_an_image_is_saved_to_another_format(single, tmp_path) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    directory, values = single
    image = io.images.load(directory)
    assert not hasattr(image, "save")
    target = tmp_path / "copy.nii.gz"
    io.save(image, target)
    saved = nb.load(str(target))
    assert np.array_equal(saved.get_fdata(), values)
    # The voxel sizes are in nanometres, and NIfTI stores them in
    # micrometres, the smallest unit that it has.
    assert saved.header.get_xyzt_units()[0] == "micron"
    assert np.allclose(saved.header.get_zooms(), [1e-3, 2e-3, 3e-3])


def test_a_pyramid_consolidates_the_data_of_its_levels(pyramid) -> None:  # noqa: ANN001
    directory, fine, _ = pyramid
    image = io.images.load(directory)
    untouched = image.consolidate()
    assert untouched is not image
    assert untouched.metadata is image.metadata
    assert [level.rawdata for level in untouched.images] == [
        level.rawdata for level in image.images
    ]
    first = image.images[0]
    first.data = np.zeros_like(fine)
    consolidated = image.consolidate()
    assert np.array_equal(consolidated.images[0].rawdata, np.zeros_like(fine))
    assert consolidated.images[0]._data is None
    assert first._data is not None
    assert image.consolidate(inplace=True) is image
    assert image.images[0]._data is None
    assert image.images[1].rawdata is consolidated.images[1].rawdata


def test_assigned_transformations_are_kept(single) -> None:  # noqa: ANN001
    directory, _ = single
    image = io.images.load(directory)
    edited = replace(image.metadata.raw, mesh="mesh")
    image.metadata = PrecomputedMetadata.from_raw(edited)
    assert image._transformations is None
    xform = Affine(matrix=np.eye(4)[:3])
    image.transformations = [xform]
    assert image.consolidate().transformations[0] is xform
    assert isinstance(image.metadata.raw.scales[0], PrecomputedScale)
