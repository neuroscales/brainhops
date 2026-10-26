"""Tests for Imaris (`.ims`) images.

Brainhops cannot write Imaris files, so the fixtures are written with
h5py in the layout of the published description of the format
("Imaris5.5 File Format Description (IMS)", `doc/Imaris5FileFormat.pdf`
of ImarisWriter): one `Data` dataset per level, time point and channel,
ordered z, y, x and padded to whole chunks, with every parameter stored
as an array of single characters.
"""

import io as _io
import pickle

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import brainhops.io as io  # noqa: E402
from brainhops._core.backends import backend  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.common._geometry import (  # noqa: E402
    same_transformations,
)
from brainhops.io.images.imaris import (  # noqa: E402
    ImarisImage,
    ImarisLevel,
    ImarisMetadata,
    ImarisMultiScaleImage,
    ImarisRaw,
)
from brainhops.io.images.imaris._data import (  # noqa: E402
    _ImarisProxy,
    read_imaris_level,
)
from brainhops.io.images.imaris._raw import attribute_text  # noqa: E402
from brainhops.io.metadata import MetadataFormat  # noqa: E402

EXT_MIN = (-3.0, 2.0, 10.0)
EXT_MAX = (5.0, 8.0, 16.0)
CHUNKS = (2, 4, 4)
PADDING = 255


def text(value: object) -> np.ndarray:
    """Encode a parameter as Imaris does, one character per element."""
    content = str(value).encode()
    if not content:
        return np.array([b""], dtype="S1")
    return np.frombuffer(content, dtype="S1")


def volume(times: int = 2, channels: int = 3, size: tuple = (8, 6, 3)):  # noqa: ANN201
    """Return voxels indexed `(t, c, z, y, x)` for an image of size x, y, z."""
    x, y, z = size
    shape = (times, channels, z, y, x)
    return np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) % 200


def downsample(stack: np.ndarray) -> np.ndarray:
    """Halve x and y, rounding down, as one coarser Imaris level does."""
    y, x = stack.shape[-2] // 2, stack.shape[-1] // 2
    return stack[..., : 2 * y : 2, : 2 * x : 2]


def write_ims(
    path: object,
    levels: list,
    *,
    unit: object = "um",
    image: dict = None,
    chunks: tuple = CHUNKS,
    version: str = "5.5.0",
) -> None:
    """Write an Imaris file from one array `(t, c, z, y, x)` per level.

    Each dataset is padded to whole chunks with the value 255, which never
    appears in the image, so that a reader that does not crop is caught.
    """
    with h5py.File(path, "w") as f:
        f.attrs["ImarisVersion"] = text(version)
        f.attrs["ImarisDataSet"] = text("ImarisDataSet")
        for r, stack in enumerate(levels):
            times, channels, z, y, x = stack.shape
            for t in range(times):
                for c in range(channels):
                    group = f.create_group(
                        f"DataSet/ResolutionLevel {r}/TimePoint {t}/"
                        f"Channel {c}"
                    )
                    group.attrs["ImageSizeX"] = text(x)
                    group.attrs["ImageSizeY"] = text(y)
                    group.attrs["ImageSizeZ"] = text(z)
                    group.attrs["HistogramMin"] = text("0.000")
                    group.attrs["HistogramMax"] = text("255.000")
                    padded = [
                        -(-n // k) * k for n, k in zip((z, y, x), chunks)
                    ]
                    data = np.full(padded, PADDING, dtype=stack.dtype)
                    data[:z, :y, :x] = stack[t, c]
                    group.create_dataset(
                        "Data", data=data, chunks=chunks, compression="gzip"
                    )
                    group.create_dataset(
                        "Histogram", data=np.zeros(256, dtype=np.uint64)
                    )
        params = {"Name": "cells", "X": levels[0].shape[4]}
        for i in range(3):
            params[f"ExtMin{i}"] = f"{EXT_MIN[i]:.3f}"
            params[f"ExtMax{i}"] = f"{EXT_MAX[i]:.3f}"
        if unit is not None:
            params["Unit"] = unit
        params.update(image or {})
        group = f.create_group("DataSetInfo/Image")
        for key, value in params.items():
            if value is not None:
                group.attrs[key] = text(value)
        for c in range(levels[0].shape[1]):
            channel = f.create_group(f"DataSetInfo/Channel {c}")
            channel.attrs["Name"] = text(f"channel {c}")
            channel.attrs["Color"] = text("1.000 0.000 0.000")
        times = f.create_group("DataSetInfo/TimeInfo")
        times.attrs["DatasetTimePoints"] = text(levels[0].shape[0])
        for t in range(levels[0].shape[0]):
            times.attrs[f"TimePoint{t + 1}"] = text(
                f"2020-01-01 00:00:0{t}.000"
            )


@pytest.fixture
def pyramid_file(tmp_path):  # noqa: ANN001, ANN201
    base = volume()
    levels = [base, downsample(base), downsample(downsample(base))]
    path = tmp_path / "cells.ims"
    write_ims(path, levels)
    return path, levels


@pytest.fixture
def single_file(tmp_path):  # noqa: ANN001, ANN201
    base = volume()
    path = tmp_path / "single.ims"
    write_ims(path, [base])
    return path, base


def expected(stack: np.ndarray) -> np.ndarray:
    """Return the data `(x, y, z, t, c)` of an image from `(t, c, z, y, x)`."""
    return np.transpose(stack, (4, 3, 2, 0, 1))


# ----------------------------------------------------------------------
#   RECORD
# ----------------------------------------------------------------------


def test_the_record_holds_the_layout(pyramid_file) -> None:  # noqa: ANN001
    path, levels = pyramid_file
    record = ImarisMetadata.load(path).raw
    assert isinstance(record, ImarisRaw)
    assert record.version == "5.5.0"
    assert record.timepoints == 2
    assert record.channels == 3
    assert record.levels[0] == ImarisLevel(
        size=(8, 6, 3), stored=(4, 8, 8), datatype="<u2"
    )
    assert [level.size for level in record.levels] == [
        (8, 6, 3),
        (4, 3, 3),
        (2, 1, 3),
    ]
    assert record.extent_min == EXT_MIN
    assert record.extent_max == EXT_MAX
    assert record.unit == "um"
    assert record.info["Channel 1"]["Name"] == "channel 1"
    assert record.info["TimeInfo"]["TimePoint2"] == "2020-01-01 00:00:01.000"


def test_parameters_are_read_as_text() -> None:
    assert attribute_text(text("128")) == "128"
    # A C string ends at its first null character, which ImarisWriter
    # may leave after the text.
    padded = np.array([b"3", b".", b"0", b""], dtype="S1")
    assert attribute_text(padded) == "3.0"
    assert attribute_text(np.array([b""], dtype="S1")) == ""
    assert attribute_text(np.bytes_(b"um")) == "um"
    assert attribute_text("um") == "um"
    assert attribute_text(np.array([1], dtype=np.uint32)) == "1"
    assert attribute_text(text("µm")) == "µm"
    assert attribute_text(np.frombuffer(b"\xb5m", dtype="S1")) == "µm"
    assert attribute_text(None) is None


def test_long_parameters_stored_as_datasets_are_read(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "long.ims"
    write_ims(path, [volume(1, 1)])
    with h5py.File(path, "a") as f:
        # ImarisWriter writes a parameter longer than 1024 characters as a
        # dataset, under a name in which "%" is "%p" and "/" is "%s".
        group = f.create_group("DataSetInfo/Log")
        group.create_dataset("a%sb%pc", data=text("x" * 2000))
    record = ImarisMetadata.load(path).raw
    assert record.info["Log"]["a/b%c"] == "x" * 2000


def test_a_copy_of_the_record_is_independent(pyramid_file) -> None:  # noqa: ANN001
    record = ImarisMetadata.load(pyramid_file[0]).raw
    copy = record.copy()
    assert copy == record
    copy.info["Image"]["Name"] = "edited"
    assert record.info["Image"]["Name"] == "cells"
    clone = pickle.loads(pickle.dumps(ImarisMetadata.from_raw(record)))
    assert clone.raw == record


@pytest.mark.parametrize(
    "change, message",
    [
        ({"ExtMax0": None}, "ExtMax0"),
        ({"ExtMin1": "abc"}, "ExtMin1"),
        ({"ExtMax2": "10.000"}, "positive size"),
    ],
)
def test_an_invalid_extent_is_refused(tmp_path, change, message) -> None:  # noqa: ANN001
    path = tmp_path / "bad.ims"
    write_ims(path, [volume(1, 1)], image=change)
    with pytest.raises(ParserContentError, match=message):
        ImarisMetadata.load(path)


def test_missing_image_sizes_are_refused(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "bad.ims"
    write_ims(path, [volume(1, 1)])
    with h5py.File(path, "a") as f:
        del f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"].attrs[
            "ImageSizeY"
        ]
    with pytest.raises(ParserContentError, match="ImageSize"):
        ImarisMetadata.load(path)


def test_levels_must_hold_every_time_point_and_channel(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "bad.ims"
    base = volume(2, 2)
    write_ims(path, [base, downsample(base)])
    with h5py.File(path, "a") as f:
        del f["DataSet/ResolutionLevel 1/TimePoint 1/Channel 1"]
    with pytest.raises(ParserContentError, match="channels"):
        ImarisMetadata.load(path)


def test_a_size_beyond_the_dataset_is_refused(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "bad.ims"
    write_ims(path, [volume(1, 1)])
    with h5py.File(path, "a") as f:
        group = f["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
        group.attrs["ImageSizeX"] = text(100)
    with pytest.raises(ParserContentError, match="does not fit"):
        ImarisMetadata.load(path)


# ----------------------------------------------------------------------
#   SNIFFING AND DISPATCH
# ----------------------------------------------------------------------


def test_a_pyramid_is_read_as_a_multiscale_image(pyramid_file) -> None:  # noqa: ANN001
    path, levels = pyramid_file
    pyramid = io.images.load(path)
    assert type(pyramid) is ImarisMultiScaleImage
    assert pyramid.nscales == 3
    assert pyramid.transformations == []
    metadata = pyramid.images[0].metadata
    for index, (level, stack) in enumerate(zip(pyramid.images, levels)):
        assert type(level) is ImarisImage
        assert level.level == index
        assert level.metadata is metadata
        assert np.array_equal(np.asarray(level.data), expected(stack))


def test_the_option_level_reads_one_level(pyramid_file) -> None:  # noqa: ANN001
    path, levels = pyramid_file
    image = io.images.load(path, level=2)
    assert type(image) is ImarisImage
    assert image.level == 2
    assert np.array_equal(np.asarray(image.data), expected(levels[2]))
    assert io.images.sniff(path, level=1) is ImarisImage
    assert io.images.sniff(path) is ImarisMultiScaleImage


def test_a_single_level_is_read_as_an_image(single_file) -> None:  # noqa: ANN001
    path, base = single_file
    image = io.images.load(path)
    assert type(image) is ImarisImage
    assert np.array_equal(np.asarray(image.data), expected(base))
    # The pyramid class still reads it when it is asked to.
    pyramid = ImarisMultiScaleImage.load(path)
    assert pyramid.nscales == 1


def test_sniffing_scores(pyramid_file, single_file) -> None:  # noqa: ANN001
    many, one = pyramid_file[0], single_file[0]
    assert ImarisImage.sniff(one) == Confidence.CERTAIN
    assert ImarisImage.sniff(many) == Confidence.LIKELY
    assert ImarisImage.sniff(many, level=1) == Confidence.CERTAIN
    assert ImarisMultiScaleImage.sniff(many) == Confidence.CERTAIN
    assert ImarisMultiScaleImage.sniff(one) == Confidence.NO
    assert ImarisMultiScaleImage.sniff(many, level=0) == Confidence.NO
    content = many.read_bytes()
    assert ImarisMultiScaleImage.sniff_bytes(content) == Confidence.CERTAIN
    assert ImarisImage.sniff_bytes(content, level=0) == Confidence.CERTAIN
    with pytest.raises(SnifferContentError):
        ImarisMultiScaleImage.sniff(one, error=True)


def test_other_hdf5_files_are_not_imaris(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("DataSet", data=np.zeros(3))
    for klass in (ImarisImage, ImarisMultiScaleImage, ImarisMetadata):
        assert klass.sniff(path) == Confidence.NO
    text_file = tmp_path / "notes.ims"
    text_file.write_text("not an HDF5 file")
    assert ImarisImage.sniff(text_file) == Confidence.NO
    with pytest.raises(ParserContentError):
        ImarisMetadata.from_h5(h5py.File(path, "r"))


def test_the_metadata_is_found_by_the_dispatcher(pyramid_file) -> None:  # noqa: ANN001
    metadata = MetadataFormat.load(pyramid_file[0])
    assert type(metadata) is ImarisMetadata


@pytest.mark.parametrize("hint", ["imaris", "ims"])
def test_hints(tmp_path, single_file, hint) -> None:  # noqa: ANN001
    renamed = tmp_path / "image.h5"
    renamed.write_bytes(single_file[0].read_bytes())
    assert type(io.images.load(renamed, hint=hint)) is ImarisImage


def test_unknown_reader_options_are_refused(pyramid_file) -> None:  # noqa: ANN001
    path = pyramid_file[0]
    with pytest.raises(TypeError):
        ImarisImage.load(path, keep_open=True)
    with pytest.raises(TypeError):
        ImarisMultiScaleImage.load(path, level=1)
    with pytest.raises(TypeError):
        ImarisMetadata.load(path, level=1)
    with pytest.raises(IndexError):
        ImarisImage.load(path, level=3)


# ----------------------------------------------------------------------
#   VOXELS
# ----------------------------------------------------------------------


def test_the_padding_is_cropped(single_file) -> None:  # noqa: ANN001
    path, base = single_file
    image = ImarisImage.load(path)
    assert image.shape == (8, 6, 3, 2, 3)
    assert image.rawdata.shape == (3, 2, 3, 6, 8)
    data = np.asarray(image.data)
    assert data.dtype == np.uint16
    assert PADDING not in data


def test_loading_reads_no_voxels(single_file) -> None:  # noqa: ANN001
    image = ImarisImage.load(single_file[0])
    assert isinstance(image.rawdata, _ImarisProxy)
    assert "_cache_data" not in vars(image)
    assert image.shape == (8, 6, 3, 2, 3)
    assert "_cache_data" not in vars(image)


@pytest.mark.parametrize(
    "index",
    [
        (0, 1),
        (-1, -2, 1),
        (slice(None), 1, slice(1, 3), slice(None, None, 2)),
        (Ellipsis, slice(5, 0, -2)),
        (Ellipsis, 2, slice(None, None, -1), 0),
        (slice(1, 3), slice(None), slice(None, None, -2)),
        (np.int64(2), Ellipsis),
        ([0, 2], 1),
        (None, 0),
        (0, slice(1, 1)),
    ],
)
def test_the_lazy_array_reads_selections(single_file, index) -> None:  # noqa: ANN001
    path, base = single_file
    raw = ImarisImage.load(path).rawdata
    reference = np.transpose(base, (1, 0, 2, 3, 4))
    assert np.array_equal(raw[index], reference[index])
    assert np.array_equal(np.asarray(raw), reference)


def test_a_selection_reads_only_its_datasets(tmp_path, single_file) -> None:  # noqa: ANN001
    path, base = single_file
    raw = ImarisImage.load(path).rawdata
    # The datasets of channel 1 are replaced by datasets whose values are
    # in an external file that does not exist, so reading them fails.
    with h5py.File(path, "a") as f:
        for t in range(2):
            name = f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel 1/Data"
            shape, dtype = f[name].shape, f[name].dtype
            del f[name]
            f.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                external=[(str(tmp_path / "missing"), 0, h5py.h5f.UNLIMITED)],
            )
    assert np.array_equal(raw[0, 1], base[1, 0])
    assert np.array_equal(raw[2, :, 1:], base[:, 2, 1:])
    with pytest.raises(OSError):
        np.asarray(raw[1])


def test_an_empty_selection_reads_nothing(tmp_path) -> None:  # noqa: ANN001
    record = ImarisRaw(
        levels=(
            ImarisLevel(size=(4, 3, 2), stored=(2, 4, 4), datatype="<u2"),
        ),
        timepoints=1,
        channels=1,
    )
    proxy = read_imaris_level(str(tmp_path / "missing.ims"), record, 0)
    assert proxy[0, 0, :0].shape == (0, 3, 4)
    with pytest.raises(IndexError):
        read_imaris_level(str(tmp_path / "missing.ims"), record, 1)


def test_the_decoded_data_are_read_only(single_file) -> None:  # noqa: ANN001
    with backend("numpy"):
        image = ImarisImage.load(single_file[0])
        data = image.data
        assert isinstance(data, np.ndarray)
        assert not data.flags.writeable


def test_dask_blocks_hold_whole_chunks(single_file) -> None:  # noqa: ANN001
    da = pytest.importorskip("dask.array")
    image = ImarisImage.load(single_file[0])
    assert image.rawdata.chunks == (1, 1, 2, 4, 4)
    with backend("dask"):
        data = ImarisImage.load(single_file[0]).data
    assert isinstance(data, da.Array)
    for axis, size in enumerate((4, 4, 2)):
        assert all(n % size == 0 for n in data.chunks[axis][:-1])
    assert np.array_equal(np.asarray(data), expected(single_file[1]))


@pytest.mark.parametrize("how", ["bytes", "stream", "h5py"])
def test_from_memory(single_file, how) -> None:  # noqa: ANN001
    path, base = single_file
    content = path.read_bytes()
    if how == "bytes":
        image = ImarisImage.from_bytes(content)
    elif how == "stream":
        stream = _io.BytesIO(b"")
        stream.write(content)
        stream.seek(0)
        image = ImarisImage.from_fileobj(stream)
        assert stream.tell() == 0
    else:
        # A file that h5py opened over a stream is kept open and read.
        opened = h5py.File(_io.BytesIO(content), "r")
        image = ImarisImage.from_file(opened)
        assert image.rawdata.source is opened
    assert np.array_equal(np.asarray(image.data), expected(base))
    pyramid = ImarisMultiScaleImage.from_bytes(content)
    assert np.array_equal(np.asarray(pyramid.data), expected(base))


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def _matrix(image: ImarisImage) -> np.ndarray:
    return np.asarray(image.transformation.to(Affine).matrix, dtype=float)


def test_the_extent_lies_on_the_border_of_the_voxels(pyramid_file) -> None:  # noqa: ANN001
    path, _ = pyramid_file
    image = io.images.load(path, level=0)
    matrix = _matrix(image)
    size = (np.array(EXT_MAX) - np.array(EXT_MIN)) / (8, 6, 3)
    assert np.allclose(np.diag(matrix)[:3], size)
    assert np.allclose(matrix[:3, -1], np.array(EXT_MIN) + size / 2)
    # The time and channel axes stay indices.
    assert np.allclose(matrix[3:, :-1], np.eye(5)[3:])
    assert np.allclose(matrix[3:, -1], 0)
    # The last voxel is centered half a voxel inside the far corner.
    corner = matrix[:3, :3] @ np.array([7, 5, 2]) + matrix[:3, -1]
    assert np.allclose(corner, np.array(EXT_MAX) - size / 2)


def test_every_level_spans_the_extent(pyramid_file) -> None:  # noqa: ANN001
    path, levels = pyramid_file
    pyramid = io.images.load(path)
    for level, stack in zip(pyramid.images, levels):
        z, y, x = stack.shape[2:]
        matrix = _matrix(level)
        size = (np.array(EXT_MAX) - np.array(EXT_MIN)) / (x, y, z)
        assert np.allclose(np.diag(matrix)[:3], size)
        assert np.allclose(matrix[:3, -1], np.array(EXT_MIN) + size / 2)
    # The pyramid gives the geometry of its finest level.
    assert np.allclose(
        pyramid.geometry.transformation.to(Affine).matrix,
        _matrix(pyramid.images[0]),
    )


def test_the_systems(single_file) -> None:  # noqa: ANN001
    xform = ImarisImage.load(single_file[0]).transformation
    assert xform.input.name == "voxel"
    assert [axis.name for axis in xform.input.axes] == list("xyztc")
    assert [axis.type for axis in xform.input.axes] == [
        "space",
        "space",
        "space",
        "time",
        "channel",
    ]
    assert xform.output.name == "physical"
    units = [axis.unit for axis in xform.output.axes]
    assert [str(u) for u in units[:3]] == ["micrometer"] * 3
    assert units[3:] == [None, None]


@pytest.mark.parametrize(
    "unit, expected_unit",
    [("mm", "millimeter"), ("nm", "nanometer"), (None, "micrometer")],
)
def test_the_unit(tmp_path, unit, expected_unit) -> None:  # noqa: ANN001
    path = tmp_path / "unit.ims"
    write_ims(path, [volume(1, 1)], unit=unit)
    xform = ImarisImage.load(path).transformation
    assert str(xform.output.axes[0].unit) == expected_unit


def test_an_unknown_unit_is_left_out(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "unit.ims"
    write_ims(path, [volume(1, 1)], unit="parsec-ish")
    xform = ImarisImage.load(path).transformation
    assert xform.output.axes[0].unit is None


def test_the_official_minimal_file_layout(tmp_path) -> None:  # noqa: ANN001
    # The attributes of `doc/MinimalImarisImageFile/Minimal_IMS_File.ims`
    # of ImarisWriter: a single slice of 128 by 128 voxels whose extent
    # ends with a null character, as ImarisWriter writes some of them.
    path = tmp_path / "minimal.ims"
    with h5py.File(path, "w") as f:
        f.attrs["ImarisVersion"] = text("5.5.0")
        group = f.create_group(
            "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"
        )
        for name, value in (("X", 128), ("Y", 128), ("Z", 1)):
            group.attrs["ImageSize" + name] = text(value)
        data = np.arange(128 * 128, dtype=np.uint16).reshape(1, 128, 128)
        group.create_dataset("Data", data=data, chunks=(1, 128, 128))
        image = f.create_group("DataSetInfo/Image")
        for name, value in (
            ("ExtMin0", b"3.00000000\x00"),
            ("ExtMin1", b"3.00000000\x00"),
            ("ExtMin2", b"0.0000000000"),
            ("ExtMax0", b"10.000000000"),
            ("ExtMax1", b"10.000000000"),
            ("ExtMax2", b"0.100000000\x00"),
        ):
            image.attrs[name] = np.frombuffer(value, dtype="S1")
    loaded = io.images.load(path)
    assert loaded.shape == (128, 128, 1, 1, 1)
    assert np.array_equal(np.asarray(loaded.data)[..., 0, 0, 0], data[0].T)
    matrix = _matrix(loaded)
    assert np.allclose(np.diag(matrix)[:3], [7 / 128, 7 / 128, 0.1])
    assert np.allclose(matrix[:3, -1], [3 + 7 / 256, 3 + 7 / 256, 0.05])


# ----------------------------------------------------------------------
#   WRITING AND CONVERSION
# ----------------------------------------------------------------------


def test_imaris_files_cannot_be_written(tmp_path, pyramid_file) -> None:  # noqa: ANN001
    path = pyramid_file[0]
    for obj in (io.images.load(path), io.images.load(path, level=1)):
        assert not hasattr(obj, "save")
        with pytest.raises(WriterError):
            io.save(obj, tmp_path / "copy.ims")
    with pytest.raises(WriterError):
        io.save(
            ImarisImage(data=np.zeros((2, 2, 2, 1, 1))), tmp_path / "x.ims"
        )


def test_a_level_converts_to_nifti(tmp_path, pyramid_file) -> None:  # noqa: ANN001
    pytest.importorskip("nibabel")
    from brainhops.io.images.nifti import NiftiImage

    path, levels = pyramid_file
    image = io.images.load(path, level=1)
    target = tmp_path / "level.nii.gz"
    io.save(image, target)
    back = NiftiImage.load(target)
    assert np.array_equal(np.asarray(back.data), expected(levels[1]))
    assert np.allclose(
        np.asarray(back.transformation.to(Affine).matrix)[:3, :3],
        _matrix(image)[:3, :3],
    )


def test_consolidating_moves_pending_data_only(single_file) -> None:  # noqa: ANN001
    path, base = single_file
    image = ImarisImage.load(path)
    transformations = image.transformations
    value = np.ones(image.shape, dtype=np.uint16)
    image.data = value
    copy = image.consolidate()
    assert copy._data is None
    assert copy.metadata is image.metadata
    # The copy decodes the same record again, since the decoded list lives
    # in the cache of the image.
    assert same_transformations(copy.transformations, transformations)
    assert np.array_equal(copy.rawdata, value.T)
    assert isinstance(image.rawdata, _ImarisProxy)
