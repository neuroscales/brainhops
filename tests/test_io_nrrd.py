"""Tests for NRRD images (.nrrd, and .nhdr with data files).

Headers are written by hand from the specification, and the data is encoded
independently of the writer. pynrrd cross-checks both directions when it is
installed.
"""

import bz2
import gzip
import io as _io
import struct

import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel.transformations import Affine, Scaling
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserError,
)
from brainhops.io.common.nrrd import NrrdHeader
from brainhops.io.common.nrrd._codecs import dtype_to_nrrd, nrrd_dtype
from brainhops.io.images import FileBasedImage
from brainhops.io.images.nrrd import (
    AttachedNrrdImage,
    DetachedNrrdImage,
    NrrdImage,
)

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------


def _values(shape, dtype="int16"):  # noqa: ANN001, ANN202
    """An array in NRRD axis order with distinct values."""
    n = int(np.prod(shape))
    return np.arange(n).reshape(shape, order="F").astype(dtype)


def _pack(array, fmt):  # noqa: ANN001, ANN202
    flat = np.asarray(array).ravel(order="F")
    return b"".join(struct.pack(fmt, v.item()) for v in flat)


def _header(*lines):  # noqa: ANN002, ANN202
    return ("\n".join(("NRRD0004",) + lines) + "\n").encode("ascii")


LPS_LINES = (
    "# a comment",
    "type: short",
    "dimension: 3",
    "space: left-posterior-superior",
    "sizes: 2 3 4",
    "space directions: (2,0,0) (0,3,0) (0,0,4)",
    "kinds: domain domain domain",
    "endian: little",
    "encoding: raw",
    "space origin: (10,20,30)",
    "DWMRI_b-value:=1000",
    "note:=two\\nlines",
)


@pytest.fixture
def lps_file(tmp_path):  # noqa: ANN001, ANN201
    values = _values((2, 3, 4))
    name = tmp_path / "lps.nrrd"
    name.write_bytes(_header(*LPS_LINES) + b"\n" + _pack(values, "<h"))
    return name, values


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def test_types() -> None:
    assert nrrd_dtype("signed short int") == np.int16
    assert nrrd_dtype("unsigned  char") == np.uint8
    assert nrrd_dtype("double", "big") == np.dtype(">f8")
    assert nrrd_dtype("ulonglong") == np.uint64
    assert dtype_to_nrrd(np.float32) == "float"
    assert dtype_to_nrrd(bool) == "uint8"
    assert dtype_to_nrrd("short") == "int16"
    with pytest.raises(ParserError):
        nrrd_dtype("block")


def test_header_parse() -> None:
    lines = [
        "type: float",
        "dimension: 4",
        "space: RAS",
        "sizes: 3 2 2 2",
        "space directions: none (1,0,0) (0,1,0) (0,0,1)",
        "kinds: vector domain domain domain",
        "centerings: ??? cell node cell",
        "endian: big",
        "encoding: gz",
        "byteskip: 4",
        "lineskip: 1",
        "measurement frame: (1,0,0) (0,-1,0) (0,0,1)",
        "content: name: with:=colon",
        "a key:=a value",
    ]
    header = NrrdHeader.from_lines(lines)
    assert header.space == "right-anterior-superior"
    assert header.space_dimension == 3
    assert header.sizes == (3, 2, 2, 2)
    assert header.encoding == "gzip"
    assert header.dtype == np.dtype(">f4")
    assert header.byte_skip == 4 and header.line_skip == 1
    assert header.centers == [None, "cell", "node", "cell"]
    assert header.kinds[0] == "vector"
    assert header.space_directions[0] is None
    np.testing.assert_array_equal(header.space_directions[2], [0, 1, 0])
    assert header.fields["content"] == "name: with:=colon"
    assert header.keyvalue == {"a key": "a value"}
    np.testing.assert_array_equal(
        header.measurement_frame, np.diag([1.0, -1.0, 1.0])
    )


def test_header_errors() -> None:
    with pytest.raises(ParserContentError):
        NrrdHeader.from_lines(["type: short", "dimension: 1"])
    header = NrrdHeader.from_lines(
        ["type: short", "dimension: 2", "sizes: 1 2", "encoding: raw"]
    )
    with pytest.raises(ParserContentError, match="endian"):
        _ = header.dtype
    with pytest.raises(ParserContentError):
        NrrdHeader.from_fileobj(_io.BytesIO(b"NOTNRRD\n"))


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_read_attached_lps(lps_file) -> None:  # noqa: ANN001
    name, values = lps_file
    image = io.load(name)
    assert type(image) is AttachedNrrdImage
    assert isinstance(image.dataobj, np.memmap)
    np.testing.assert_array_equal(image.data, values)
    assert [a.name for a in image.system.axes] == ["x", "y", "z"]
    physical, world = image.transformations
    assert isinstance(physical, Scaling)
    np.testing.assert_allclose(physical.scale, [2, 3, 4])
    assert isinstance(world, Affine)
    assert world.output.name == "LPS"
    assert [a.orientation.value for a in world.output.axes] == [
        "right-to-left",
        "anterior-to-posterior",
        "inferior-to-superior",
    ]
    np.testing.assert_allclose(
        world.matrix, [[2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, 30]]
    )
    assert image.header.keyvalue["DWMRI_b-value"] == "1000"
    assert image.header.keyvalue["note"] == "two\nlines"


def test_read_no_mmap(lps_file) -> None:  # noqa: ANN001
    name, values = lps_file
    image = AttachedNrrdImage.load(name, mmap=False)
    assert not isinstance(image.dataobj, np.memmap)
    np.testing.assert_array_equal(image.data, values)


def test_read_from_bytes_and_fileobj(lps_file) -> None:  # noqa: ANN001
    name, values = lps_file
    content = name.read_bytes()
    np.testing.assert_array_equal(
        AttachedNrrdImage.from_bytes(content).data, values
    )
    np.testing.assert_array_equal(
        io.images.load(_io.BytesIO(content)).data, values
    )


def test_read_detached_gzip_ras_vector(tmp_path) -> None:  # noqa: ANN001
    values = _values((3, 2, 3, 4), "float32")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dwi.raw.gz").write_bytes(
        gzip.compress(_pack(values, ">f"))
    )
    (tmp_path / "dwi.nhdr").write_bytes(
        _header(
            "type: float",
            "dimension: 4",
            "space: right-anterior-superior",
            "sizes: 3 2 3 4",
            "space directions: none (0,1,0) (-1,0,0) (0,0,2)",
            "kinds: vector domain domain domain",
            "endian: big",
            "encoding: gzip",
            "space origin: (1,2,3)",
            "data file: data/dwi.raw.gz",
        )
    )
    image = io.load(tmp_path / "dwi.nhdr")
    assert type(image) is DetachedNrrdImage
    assert image.data.shape == (2, 3, 4, 3)
    np.testing.assert_array_equal(image.data, values.transpose(1, 2, 3, 0))
    assert [a.name for a in image.system.axes] == ["x", "y", "z", "c"]
    assert image.system.axes[3].type == "channel"
    world = image.transformation
    assert world.output.name == "RAS"
    np.testing.assert_allclose(
        world.matrix, [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 2, 3]]
    )


@pytest.mark.parametrize(
    "encoding,encode",
    [
        ("raw", lambda b: b),
        ("gzip", gzip.compress),
        ("bzip2", bz2.compress),
        ("hex", lambda b: b.hex().encode() + b"\n"),
    ],
)
def test_read_encodings(tmp_path, encoding, encode) -> None:  # noqa: ANN001
    values = _values((4, 3), "uint16")
    content = (
        _header(
            "type: ushort",
            "dimension: 2",
            "sizes: 4 3",
            "endian: big",
            f"encoding: {encoding}",
        )
        + b"\n"
        + encode(_pack(values, ">H"))
    )
    (tmp_path / "a.nrrd").write_bytes(content)
    image = io.load(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(image.data, values)
    # Without a space, the transformation is a unit-less identity scaling.
    (xform,) = image.transformations
    assert isinstance(xform, Scaling)
    np.testing.assert_array_equal(xform.scale, [1, 1])


def test_read_ascii(tmp_path) -> None:  # noqa: ANN001
    values = _values((3, 2), "float64") / 4
    text = "\n".join(" ".join(repr(float(v)) for v in row) for row in values.T)
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: double", "dimension: 2", "sizes: 3 2", "encoding: ascii"
        )
        + b"\n"
        + text.encode()
    )
    np.testing.assert_array_equal(io.load(tmp_path / "a.nrrd").data, values)


def test_read_data_file_list_and_pattern(tmp_path) -> None:  # noqa: ANN001
    values = _values((2, 2, 3), "uint8")
    for k in range(3):
        (tmp_path / f"slice{k + 1:02d}.raw").write_bytes(
            values[:, :, k].ravel(order="F").tobytes()
        )
    common = ("type: uchar", "dimension: 3", "sizes: 2 2 3", "encoding: raw")
    (tmp_path / "pattern.nhdr").write_bytes(
        _header(*common, "data file: slice%02d.raw 1 3 1 2")
    )
    (tmp_path / "list.nhdr").write_bytes(
        _header(
            *common,
            "data file: LIST",
            "slice01.raw",
            "slice02.raw",
            "slice03.raw",
        )
    )
    for name in ("pattern.nhdr", "list.nhdr"):
        image = io.load(tmp_path / name)
        assert type(image) is DetachedNrrdImage
        np.testing.assert_array_equal(image.data, values)


def test_read_skips(tmp_path) -> None:  # noqa: ANN001
    values = _values((2, 3), "int32")
    payload = _pack(values, "<i")
    (tmp_path / "a.raw").write_bytes(b"junk line\nXYZ" + payload)
    (tmp_path / "end.raw").write_bytes(b"\x00" * 7 + payload)
    (tmp_path / "z.raw.gz").write_bytes(
        b"line\n" + gzip.compress(b"\x01\x02" + payload)
    )
    common = ("type: int", "dimension: 2", "sizes: 2 3", "endian: little")
    (tmp_path / "a.nhdr").write_bytes(
        _header(
            *common,
            "encoding: raw",
            "line skip: 1",
            "byte skip: 3",
            "data file: a.raw",
        )
    )
    (tmp_path / "end.nhdr").write_bytes(
        _header(
            *common, "encoding: raw", "byte skip: -1", "data file: end.raw"
        )
    )
    (tmp_path / "z.nhdr").write_bytes(
        _header(
            *common,
            "encoding: gzip",
            "line skip: 1",
            "byte skip: 2",
            "data file: z.raw.gz",
        )
    )
    for name in ("a.nhdr", "end.nhdr", "z.nhdr"):
        for mmap in (True, False):
            image = DetachedNrrdImage.load(tmp_path / name, mmap=mmap)
            np.testing.assert_array_equal(image.data, values)


def test_read_kinds_and_roles(tmp_path) -> None:  # noqa: ANN001
    values = _values((2, 3, 4, 5), "uint8")
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: uint8",
            "dimension: 4",
            "sizes: 2 3 4 5",
            "kinds: RGB-color domain time domain",
            "spacings: nan 2 0.5 3",
            'units: "" "mm" "s" "mm"',
            "encoding: raw",
        )
        + b"\n"
        + values.ravel(order="F").tobytes()
    )
    image = io.load(tmp_path / "a.nrrd")
    axes = image.system.axes
    assert [(a.name, a.type) for a in axes] == [
        ("x", "space"),
        ("y", "space"),
        ("t", "time"),
        ("c", "channel"),
    ]
    np.testing.assert_array_equal(image.data, values.transpose(1, 3, 2, 0))
    (xform,) = image.transformations
    np.testing.assert_allclose(xform.scale, [2, 3, 0.5, 1])
    units = [getattr(a.unit, "name", None) for a in xform.output.axes]
    assert units == ["millimeter", "millimeter", "second", None]


@pytest.mark.parametrize(
    "center,scale,first",
    [
        # Cell centring: the samples fill [min, max].
        ("cell", [0.5, 2], [0.25, 0]),
        # Node centring: the first and last samples lie at min and max.
        ("node", [2.0 / 3, 2], [0, 1]),
    ],
)
def test_read_axis_mins_maxs(tmp_path, center, scale, first) -> None:  # noqa: ANN001
    # axis 0: 4 samples over [0, 2]; axis 1: spacing 2, max 5, 3 samples
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: uint8",
            "dimension: 2",
            "sizes: 4 3",
            f"centers: {center} {center}",
            "axis mins: 0 nan",
            "axis maxs: 2 5",
            "spacings: nan 2",
            "encoding: raw",
        )
        + b"\n"
        + bytes(12)
    )
    (xform,) = io.load(tmp_path / "a.nrrd").transformations
    assert isinstance(xform, Affine)
    np.testing.assert_allclose(np.diag(xform.matrix[:, :2]), scale)
    np.testing.assert_allclose(xform.matrix[:, -1], first)


def test_read_unoriented_spaces(tmp_path) -> None:  # noqa: ANN001
    common = (
        "type: uint8",
        "dimension: 2",
        "sizes: 2 2",
        "space directions: (1,0) (0,2)",
        "encoding: raw",
    )
    (tmp_path / "a.nrrd").write_bytes(
        _header(*common, "space dimension: 2") + b"\n" + bytes(4)
    )
    world = io.load(tmp_path / "a.nrrd").transformation
    assert world.output.name == "world"
    assert all(a.orientation is None for a in world.output.axes)
    lines = list(common)
    lines[3] = "space directions: (1,0,0) (0,2,0)"
    (tmp_path / "b.nrrd").write_bytes(
        _header(*lines, "space: scanner-xyz") + b"\n" + bytes(4)
    )
    world = io.load(tmp_path / "b.nrrd").transformation
    assert world.output.name == "scanner-xyz"
    assert world.matrix.shape == (3, 3)


def test_missing_data_file(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "a.nhdr").write_bytes(
        _header(
            "type: uint8",
            "dimension: 1",
            "sizes: 2",
            "encoding: raw",
            "data file: nowhere.raw",
        )
    )
    with pytest.raises(Exception, match="nowhere.raw"):
        DetachedNrrdImage.load(tmp_path / "a.nhdr")


# ----------------------------------------------------------------------
#   SNIFF / DISPATCH
# ----------------------------------------------------------------------


def test_sniff_and_hints(lps_file, tmp_path) -> None:  # noqa: ANN001
    name, _ = lps_file
    assert AttachedNrrdImage.sniff(name) == Confidence.LIKELY
    assert DetachedNrrdImage.sniff(name) == Confidence.NO
    assert FileBasedImage.sniff(name) is AttachedNrrdImage
    assert isinstance(io.load(name, hint="nrrd"), AttachedNrrdImage)
    # The content decides, not the name.
    renamed = tmp_path / "renamed.bin"
    renamed.write_bytes(name.read_bytes())
    assert type(io.load(renamed)) is AttachedNrrdImage
    assert AttachedNrrdImage.sniff_bytes(b"\x89PNG....") == Confidence.NO
    assert issubclass(AttachedNrrdImage, NrrdImage)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_round_trip_attached(lps_file, tmp_path) -> None:  # noqa: ANN001
    name, values = lps_file
    image = io.load(name)
    out = tmp_path / "out.nrrd"
    image.save(out)
    back = io.load(out)
    np.testing.assert_array_equal(back.data, values)
    assert back.header.space == "left-posterior-superior"
    assert back.header.encoding == "raw"
    assert back.header.keyvalue == image.header.keyvalue
    np.testing.assert_allclose(
        back.transformation.matrix, image.transformation.matrix
    )


@pytest.mark.parametrize("encoding", ["raw", "gzip", "bzip2", "ascii", "hex"])
def test_round_trip_detached(tmp_path, encoding) -> None:  # noqa: ANN001
    values = _values((3, 2, 3, 4), "float32") / 3
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: float",
            "dimension: 4",
            "space: LPS",
            "sizes: 3 2 3 4",
            "space directions: none (1,0,0) (0,1,0) (0,0,1)",
            "kinds: vector domain domain domain",
            "endian: little",
            "encoding: raw",
            "measurement frame: (1,0,0) (0,1,0) (0,0,1)",
        )
        + b"\n"
        + _pack(values, "<f")
    )
    image = io.load(tmp_path / "a.nrrd")
    image.save(tmp_path / "b.nhdr", encoding=encoding)
    back = io.load(tmp_path / "b.nhdr")
    assert type(back) is DetachedNrrdImage
    assert back.header.data_files[0].startswith("b.")
    np.testing.assert_array_equal(back.data, image.data)
    assert back.header.fields["kinds"] == "vector domain domain domain"
    assert back.header.fields["measurement frame"] == (
        "(1,0,0) (0,1,0) (0,0,1)"
    )


def test_write_space_conversion(lps_file, tmp_path) -> None:  # noqa: ANN001
    name, _ = lps_file
    image = io.load(name)
    image.save(tmp_path / "ras.nrrd", space="RAS")
    back = io.load(tmp_path / "ras.nrrd")
    assert back.header.space == "right-anterior-superior"
    np.testing.assert_allclose(
        back.transformation.matrix,
        [[-2, 0, 0, -10], [0, -3, 0, -20], [0, 0, 4, 30]],
    )


def test_write_options(lps_file, tmp_path) -> None:  # noqa: ANN001
    name, values = lps_file
    image = io.load(name)
    image.save(
        tmp_path / "o.nrrd",
        endian="big",
        datatype="float",
        keyvalue={"DWMRI_b-value": None, "new": "1"},
    )
    back = io.load(tmp_path / "o.nrrd")
    assert back.dataobj.dtype == np.dtype(">f4")
    np.testing.assert_array_equal(back.data, values)
    assert "DWMRI_b-value" not in back.header.keyvalue
    assert back.header.keyvalue["new"] == "1"
    with pytest.raises(TypeError):
        image.save(tmp_path / "x.nrrd", bogus=1)


def test_nifti_to_nrrd_and_back(tmp_path) -> None:  # noqa: ANN001
    nib = pytest.importorskip("nibabel")
    affine = np.array(
        [[0, -2, 0, 5], [3, 0, 0, -1], [0, 0, 1.5, 2], [0, 0, 0, 1]]
    )
    data = np.random.default_rng(0).random((3, 4, 5)).astype("float32")
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "a.nii.gz")
    image = io.images.load(tmp_path / "a.nii.gz")
    io.save(image, tmp_path / "a.nrrd")
    nrrd_image = io.load(tmp_path / "a.nrrd")
    assert nrrd_image.header.space == "right-anterior-superior"
    np.testing.assert_allclose(
        nrrd_image.transformation.homogeneous_matrix, affine
    )
    nrrd_image.save(tmp_path / "lps.nrrd", space="LPS")
    io.save(io.load(tmp_path / "lps.nrrd"), tmp_path / "b.nii.gz")
    back = nib.load(tmp_path / "b.nii.gz")
    np.testing.assert_allclose(back.affine, affine)
    np.testing.assert_array_equal(np.asarray(back.dataobj), data)


def test_write_no_space_round_trip(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: uint8",
            "dimension: 2",
            "sizes: 4 3",
            "spacings: 0.5 2",
            "axis mins: 1 nan",
            "encoding: raw",
        )
        + b"\n"
        + bytes(range(12))
    )
    image = io.load(tmp_path / "a.nrrd")
    image.save(tmp_path / "b.nrrd")
    back = io.load(tmp_path / "b.nrrd")
    assert "space" not in back.header.fields
    assert "space dimension" not in back.header.fields
    np.testing.assert_allclose(
        back.transformation.matrix, image.transformation.matrix
    )


# ----------------------------------------------------------------------
#   PYNRRD CROSS-CHECK
# ----------------------------------------------------------------------


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("encoding", ["raw", "gzip", "ascii"])
def test_pynrrd_cross_check(tmp_path, detached, encoding) -> None:  # noqa: ANN001
    nrrd = pytest.importorskip("nrrd")
    values = _values((3, 4, 5, 2), "int16")
    header = {
        "space": "left-posterior-superior",
        "space directions": np.array(
            [[np.nan] * 3, [0, 2, 0], [1, 0, 0], [0, 0, 3]]
        ),
        "kinds": ["vector", "domain", "domain", "domain"],
        "space origin": np.array([1.0, -2.0, 3.0]),
        "encoding": encoding,
        "DWMRI_b-value": "1000",
    }
    ext = ".nhdr" if detached else ".nrrd"
    name = str(tmp_path / ("a" + ext))
    nrrd.write(name, values, header, detached_header=detached)
    image = io.load(name)
    np.testing.assert_array_equal(image.data, values.transpose(1, 2, 3, 0))
    np.testing.assert_allclose(
        image.transformation.matrix,
        [[0, 1, 0, 1], [2, 0, 0, -2], [0, 0, 3, 3]],
    )
    assert image.header.keyvalue["DWMRI_b-value"] == "1000"

    out = str(tmp_path / ("b" + ext))
    image.save(out)
    data, back = nrrd.read(out)
    np.testing.assert_array_equal(data, values)
    assert back["space"] == "left-posterior-superior"
    np.testing.assert_allclose(
        back["space directions"][1:], header["space directions"][1:]
    )
    np.testing.assert_allclose(back["space origin"], header["space origin"])
    assert back["kinds"] == header["kinds"]
    assert back["DWMRI_b-value"] == "1000"


def test_write_measurement_frame_follows_space(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: uint8",
            "dimension: 3",
            "space: left-posterior-superior",
            "sizes: 2 2 2",
            "space directions: (1,0,0) (0,1,0) (0,0,1)",
            "measurement frame: (0,1,0) (1,0,0) (0,0,1)",
            "encoding: raw",
        )
        + b"\n"
        + bytes(8)
    )
    io.load(tmp_path / "a.nrrd").save(tmp_path / "b.nrrd", space="RAS")
    back = io.load(tmp_path / "b.nrrd")
    np.testing.assert_array_equal(
        back.header.measurement_frame,
        [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
    )
