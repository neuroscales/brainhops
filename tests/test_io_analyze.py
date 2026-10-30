"""Reading of Analyze 7.5 images (#633).

The files are written with nibabel, whose Analyze classes write the
header file and the image file of a pair. The SPM classes of nibabel
also write a `.mat` file when the affine of the image is not the default
one, and brainhops does not read that file, so the tests write their
headers without an affine, from the voxel sizes and the origin alone.
"""

import gzip
from io import BytesIO

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402
from brainhops.errors import (  # noqa: E402
    ParserContentError,
    ParserExistsError,
    WriterError,
)
from brainhops.io.base.parsers import Confidence  # noqa: E402
from brainhops.io.common.nifti._data import NiftiProxy  # noqa: E402
from brainhops.io.images.analyze import (  # noqa: E402
    AnalyzeImage,
    AnalyzeMetadata,
    AnalyzeRaw,
)
from brainhops.io.images.nifti import NiftiImage  # noqa: E402


def _write(
    filename,  # noqa: ANN001
    data: np.ndarray,
    zooms=None,  # noqa: ANN001
    origin=None,  # noqa: ANN001
    klass=None,  # noqa: ANN001
    **fields,  # noqa: ANN003
) -> None:
    """Write an Analyze pair whose header holds the given fields.

    nibabel chooses the scaling of the voxels when it writes them, so the
    other fields are set afterwards, in the header file that it wrote.
    """
    klass = klass or nb.Spm2AnalyzeImage
    header = klass.header_class()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    if zooms is not None:
        header.set_zooms(zooms)
    if origin is not None:
        header["origin"][: len(origin)] = origin
    klass(data, None, header).to_filename(str(filename))
    if not fields:
        return
    with open(filename, "rb") as stream:
        header = klass.header_class.from_fileobj(stream)
    for name, value in fields.items():
        header[name] = value
    with open(filename, "r+b") as stream:
        header.write_to(stream)


@pytest.fixture
def analyze(tmp_path):  # noqa: ANN001, ANN201
    filename = tmp_path / "brain.hdr"
    data = np.arange(3 * 5 * 7, dtype="int16").reshape(3, 5, 7)
    _write(filename, data, zooms=(3.0, 2.0, 1.0))
    return filename, data


# ----------------------------------------------------------------------
#   SNIFFING AND READING
# ----------------------------------------------------------------------


def test_the_generic_loader_reads_either_file_of_the_pair(analyze) -> None:  # noqa: ANN001
    filename, data = analyze
    for name in (filename, filename.with_suffix(".img")):
        assert io.images.sniff(name) is AnalyzeImage
        image = io.images.load(name)
        assert isinstance(image, AnalyzeImage)
        assert isinstance(image.metadata, AnalyzeMetadata)
        assert isinstance(image.metadata.raw, AnalyzeRaw)
        assert np.array_equal(np.asarray(image.data), data)
    assert isinstance(io.images.load(filename, hint="analyze"), AnalyzeImage)


def test_the_voxels_are_read_lazily(analyze) -> None:  # noqa: ANN001
    filename, data = analyze
    image = AnalyzeImage.load(filename)
    assert isinstance(image.rawdata, NiftiProxy)
    assert image.data is image.rawdata
    assert image.shape == data.shape
    assert image.data.dtype == np.dtype("int16")
    assert np.array_equal(
        np.asarray(image.data[1, 2:4, ::2]), data[1, 2:4, ::2]
    )


def test_every_class_of_nibabel_is_read(tmp_path) -> None:  # noqa: ANN001
    data = np.arange(24, dtype="float32").reshape(2, 3, 4)
    for klass in (nb.AnalyzeImage, nb.Spm99AnalyzeImage, nb.Spm2AnalyzeImage):
        filename = tmp_path / f"{klass.__name__}.hdr"
        _write(filename, data, klass=klass)
        image = io.images.load(filename)
        assert isinstance(image, AnalyzeImage)
        assert np.array_equal(np.asarray(image.data), data)


def test_a_gzipped_pair_is_read(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "brain.hdr.gz"
    data = np.arange(24, dtype="uint8").reshape(2, 3, 4)
    _write(filename, data)
    assert (tmp_path / "brain.img.gz").exists()
    image = io.images.load(filename)
    assert isinstance(image, AnalyzeImage)
    assert np.array_equal(np.asarray(image.data), data)


def test_a_named_stream_of_the_header_is_read(analyze) -> None:  # noqa: ANN001
    filename, data = analyze
    with open(filename, "rb") as stream:
        image = AnalyzeImage.from_fileobj(stream)
        assert stream.tell() == 0
    assert np.array_equal(np.asarray(image.data), data)


def test_bytes_and_unnamed_streams_are_refused(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    content = filename.read_bytes()
    with pytest.raises(ParserContentError):
        AnalyzeImage.from_bytes(content)
    with pytest.raises(ParserContentError):
        AnalyzeImage.from_fileobj(BytesIO(content))
    # The metadata needs the header alone, so it reads bytes.
    metadata = AnalyzeMetadata.from_bytes(content)
    assert metadata.raw.header.get_data_shape() == (3, 5, 7)


def test_a_missing_image_file_is_refused(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    filename.with_suffix(".img").unlink()
    with pytest.raises(ParserExistsError):
        AnalyzeImage.load(filename)


def test_unknown_reader_options_are_refused(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    with pytest.raises(TypeError):
        AnalyzeImage.load(filename, compress=True)


def test_metadata_reads_the_header_alone(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    filename.with_suffix(".img").unlink()
    metadata = AnalyzeMetadata.load(filename)
    assert metadata.raw.header.get_zooms() == (3.0, 2.0, 1.0)


# ----------------------------------------------------------------------
#   NIFTI AND ANALYZE HEADERS
# ----------------------------------------------------------------------


def test_a_nifti_pair_is_left_to_the_nifti_reader(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "pair.hdr"
    data = np.arange(24, dtype="int16").reshape(2, 3, 4)
    nb.Nifti1Pair(data, np.eye(4)).to_filename(str(filename))
    assert AnalyzeImage.sniff(filename) == Confidence.NO
    assert AnalyzeRaw.sniff(filename) == Confidence.NO
    assert NiftiImage.sniff(filename) > Confidence.NO
    assert io.images.sniff(filename) is NiftiImage
    with pytest.raises(ParserContentError):
        AnalyzeImage.load(filename)
    with pytest.raises(ParserContentError):
        AnalyzeMetadata.load(filename)


def test_an_analyze_header_is_left_to_the_analyze_reader(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    assert NiftiImage.sniff(filename) == Confidence.NO
    assert AnalyzeImage.sniff(filename) > Confidence.NO


def test_other_content_is_declined(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "noise.hdr"
    filename.write_bytes(bytes(range(256)) * 2)
    assert AnalyzeImage.sniff(filename) == Confidence.NO
    with pytest.raises(Exception):  # noqa: B017, PT011
        AnalyzeImage.sniff(filename, error=True)


# ----------------------------------------------------------------------
#   INTENSITY SCALING
# ----------------------------------------------------------------------


def test_the_spm_scale_factor_and_intercept_are_applied(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "scaled.hdr"
    data = np.arange(24, dtype="int16").reshape(2, 3, 4)
    _write(filename, data, scl_slope=2.0, scl_inter=0.5)
    image = io.images.load(filename)
    expected = np.asanyarray(nb.load(str(filename)).dataobj)
    assert np.allclose(expected, 2.0 * data + 0.5)
    assert np.allclose(np.asarray(image.data), expected)
    assert image.data.dtype.kind == "f"


def test_spm99_scales_without_an_intercept(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "scaled.hdr"
    data = np.arange(24, dtype="int16").reshape(2, 3, 4)
    _write(filename, data, klass=nb.Spm99AnalyzeImage, scl_slope=3.0)
    image = io.images.load(filename)
    assert np.allclose(np.asarray(image.data), 3.0 * data)


def test_a_zero_scale_factor_does_not_scale(analyze) -> None:  # noqa: ANN001
    filename, data = analyze
    image = io.images.load(filename)
    assert image.metadata.raw.header["scl_slope"] in (0.0, 1.0)
    assert image.data.dtype == np.dtype("int16")
    assert np.array_equal(np.asarray(image.data), data)


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def _matrix(image) -> np.ndarray:  # noqa: ANN001
    return np.asarray(image.transformation.to(Affine).matrix, dtype=float)


def test_without_origin_the_grid_is_centred_and_x_is_flipped(analyze) -> None:  # noqa: ANN001
    filename, _ = analyze
    image = io.images.load(filename)
    # The example of `Spm99AnalyzeHeader.get_origin_affine` in nibabel.
    expected = [[-3.0, 0, 0, 3], [0, 2, 0, -4], [0, 0, 1, -3]]
    assert np.allclose(_matrix(image), expected)
    header = image.metadata.raw.header
    assert np.allclose(_matrix(image), header.get_base_affine()[:3])
    # The centre of the grid is the origin of the world.
    centre = (np.asarray(image.shape) - 1) / 2
    assert np.allclose(_matrix(image) @ [*centre, 1], 0)
    world = image.transformation.output
    assert world.name == "undefined"
    assert [axis.orientation.value for axis in world.axes] == [
        "left-to-right",
        "posterior-to-anterior",
        "inferior-to-superior",
    ]
    physical = image.transformations[0]
    assert np.allclose(physical.to(Affine).matrix[:, :3], np.diag([3, 2, 1]))


def test_a_single_slice_axis_is_centred_on_its_voxel(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "slice.hdr"
    data = np.zeros((4, 6, 1), dtype="float32")
    _write(filename, data, zooms=(1.0, 1.0, 5.0))
    matrix = _matrix(io.images.load(filename))
    assert np.allclose(matrix[2], [0, 0, 5, 0])
    assert np.allclose(matrix[:, 3], [1.5, -2.5, 0])


def test_the_spm_origin_is_the_origin_of_the_world(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "origin.hdr"
    data = np.zeros((3, 5, 7), dtype="float32")
    origin = np.asarray([3, 4, 5])
    vox = np.asarray([3.0, 2.0, 1.0])
    _write(filename, data, zooms=vox, origin=origin)
    image = io.images.load(filename)
    matrix = _matrix(image)
    # The example of `Spm99AnalyzeHeader.get_origin_affine` in nibabel.
    expected = [[-3.0, 0, 0, 6], [0, 2, 0, -6], [0, 0, 1, -4]]
    assert np.allclose(matrix, expected)
    # SPM (`decode_qform0.m`) maps the one-based index of a voxel with
    # `off = -vox .* origin`, then flips the first axis. brainhops maps
    # the zero-based index of the same voxel to the same point.
    spm = np.eye(4)
    spm[:3, :3] = np.diag(vox)
    spm[:3, 3] = -vox * origin
    spm = np.diag([-1.0, 1, 1, 1]) @ spm
    for index in ([0, 0, 0], [2, 4, 6], [1, 3, 2]):
        one_based = np.asarray([*index, 0]) + 1
        assert np.allclose(matrix @ [*index, 1], (spm @ one_based)[:3])
    # The origin voxel, zero-based, is the origin of the world.
    assert np.allclose(matrix @ [*(origin - 1), 1], 0)


def test_a_time_axis_does_not_move_a_world_point(tmp_path) -> None:  # noqa: ANN001
    filename = tmp_path / "series.hdr"
    data = np.zeros((2, 3, 4, 5), dtype="float32")
    _write(filename, data, zooms=(2.0, 2.0, 2.0, 1.5))
    image = io.images.load(filename)
    assert [axis.name for axis in image.system.axes] == ["x", "y", "z", "t"]
    matrix = _matrix(image)
    assert matrix.shape == (3, 5)
    assert np.allclose(matrix[:, 3], 0)
    assert np.allclose(matrix[:, :3], np.diag([-2.0, 2, 2]))


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_analyze_files_are_not_written(analyze, tmp_path) -> None:  # noqa: ANN001
    filename, data = analyze
    image = io.images.load(filename)
    assert not hasattr(image, "save")
    with pytest.raises(WriterError):
        io.save(image, tmp_path / "copy.hdr")
    assert not (tmp_path / "copy.hdr").exists()


def test_an_analyze_image_is_saved_as_nifti(analyze, tmp_path) -> None:  # noqa: ANN001
    filename, data = analyze
    image = io.images.load(filename)
    target = tmp_path / "copy.nii.gz"
    io.save(image, target)
    nifti = io.images.load(target)
    assert isinstance(nifti, NiftiImage)
    assert np.array_equal(np.asarray(nifti.data), data)
    assert np.allclose(_matrix(nifti), _matrix(image))
    # The file is a real gzip stream.
    assert gzip.decompress(target.read_bytes())[344:347] == b"n+1"
