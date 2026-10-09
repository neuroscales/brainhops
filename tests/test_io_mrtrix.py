"""Tests for the MRtrix format (.mif, .mif.gz, and .mih with .dat).

The fixtures are encoded independently of the reader, by placing each voxel
at the byte offset MRtrix computes (core/stride.h).
"""

import gzip
import io as _io
import math
import struct

import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine
from brainhops.io.base import Format
from brainhops.io.base.parsers import (
    Confidence,
    FileWriter,
    ParserContentError,
    ParserError,
)
from brainhops.io.common.mrtrix import MrtrixHeader
from brainhops.io.common.mrtrix._codecs import (
    dtype_to_mrtrix,
    mrtrix_dtype,
    parse_layout,
)
from brainhops.io.images import ImageFormat
from brainhops.io.images.mrtrix import MrtrixImage

# ----------------------------------------------------------------------
#   REFERENCE ENCODER
# ----------------------------------------------------------------------


def _actual_strides(dim, layout):  # noqa: ANN001, ANN202
    """MRtrix voxel strides from a symbolic layout; '-' negates the stride."""
    entries = layout.split(",")
    ranks = [int(e.lstrip("+-")) for e in entries]
    signs = [-1 if e.startswith("-") else 1 for e in entries]
    by_rank = sorted(range(len(dim)), key=lambda i: ranks[i])
    strides = [0] * len(dim)
    step = 1
    for axis in by_rank:
        strides[axis] = signs[axis] * step
        step *= dim[axis]
    return strides


def _reference_order(data, layout):  # noqa: ANN001, ANN202
    """Values in file order, placed one by one at the MRtrix offset."""
    dim = data.shape
    strides = _actual_strides(dim, layout)
    start = sum(-s * (n - 1) for s, n in zip(strides, dim) if s < 0)
    out = [None] * data.size
    for index in np.ndindex(*dim):
        out[start + sum(i * s for i, s in zip(index, strides))] = data[index]
    return out


def _reference_bytes(data, layout, fmt):  # noqa: ANN001, ANN202
    return b"".join(
        struct.pack(fmt, v.item() if hasattr(v, "item") else v)
        for v in _reference_order(data, layout)
    )


def _reference_bits(data, layout):  # noqa: ANN001, ANN202
    """Pack booleans 8 per byte, the first in the top bit (core/raw.h)."""
    values = _reference_order(data, layout)
    out = bytearray((len(values) + 7) // 8)
    for i, v in enumerate(values):
        if v:
            out[i // 8] |= 0x80 >> (i % 8)
    return bytes(out)


def _header_text(dim, layout, datatype, extra="", file=None):  # noqa: ANN001, ANN202
    lines = [
        "mrtrix image",
        "dim: " + ",".join(map(str, dim)),
        "vox: " + ",".join(["1"] * len(dim)),
        f"layout: {layout}",
        f"datatype: {datatype}",
    ]
    if extra:
        lines.append(extra)
    if file is not None:
        lines.append(f"file: {file}")
    return "\n".join(lines) + "\n"


def _single_file(head, payload, pad=0):  # noqa: ANN001, ANN202
    """Build a .mif file from header, offset line, END, padding and data."""
    offset = 0
    while True:
        text = (head + f"file: . {offset}\nEND\n").encode()
        needed = len(text) + pad
        if needed == offset:
            return text + b"\0" * pad + payload
        offset = needed


def _write(path, content):  # noqa: ANN001, ANN202
    path.write_bytes(content)
    return path


DATA = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)

ROTATED = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 1.0, 5.0],
    ]
)


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def test_the_header_keys_are_decoded() -> None:
    text = (
        "mrtrix image\n"
        "DIM: 4,5,6,7\n"
        "Vox: 1.5,2,2.5,nan  # a comment\n"
        "layout: -0,-1,+2,+3\n"
        "datatype: UInt16BE\n"
        "transform: 1,0,0,-1\n"
        "transform: 0,1,0,-2\n"
        "transform: 0,0,1,-3\n"
        "scaling: 0.5,2\n"
        "# a full-line comment\n"
        "dw_scheme: 0,0,1,0\n"
        "dw_scheme: 0,1,0,1000\n"
        "Comments: kept as spelled\n"
        "no colon, ignored\n"
        "file: . 1024\n"
        "END\n"
        "garbage after the end: ignored\n"
    )
    header = MrtrixHeader.from_text(text)
    assert header.dim == (4, 5, 6, 7)
    assert header.vox[:3] == (1.5, 2.0, 2.5)
    assert math.isnan(header.vox[3])
    assert header.layout == (-1, -2, 3, 4)
    assert header.datatype == "UInt16BE"
    assert header.dtype == np.dtype(">u2")
    assert np.array_equal(
        header.transform, [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3]]
    )
    assert header.scaling == (0.5, 2.0)
    assert header.file == (".", 1024)
    assert header.keyval == {
        "dw_scheme": "0,0,1,0\n0,1,0,1000",
        "Comments": "kept as spelled",
    }


def test_the_header_writes_back_as_it_was_read() -> None:
    text = (
        "mrtrix image\n"
        "dim: 4,5,6\n"
        "vox: 1.5,2,2.5\n"
        "layout: +2,-0,+1\n"
        "datatype: Float32LE\n"
        "transform: 0.5, 0.25, 0, -1\n"
        "transform: 0, 1, 0, -2\n"
        "transform: 0, 0, 1, -3.125\n"
        "command_history: mrconvert a.nii b.mif  (version=3.0.4)\n"
        "command_history: mrcalc b.mif 2 -mult c.mif  (version=3.0.4)\n"
        "file: . 512\n"
        "END\n"
    )
    header = MrtrixHeader.from_text(text)
    assert header.to_text() == text
    assert MrtrixHeader.from_text(header.to_text()) == header


@pytest.mark.parametrize(
    "text",
    [
        "not mrtrix\ndim: 1\nvox: 1\ndatatype: Bit\nEND\n",
        "mrtrix image\nvox: 1,1,1\ndatatype: Bit\nEND\n",
        "mrtrix image\ndim: 1,1,1\ndatatype: Bit\nEND\n",
        "mrtrix image\ndim: 1,1,1\nvox: 1,1,1\nEND\n",
        "mrtrix image\ndim: 1,1,1\nvox: 1,1\ndatatype: Bit\nEND\n",
        "mrtrix image\ndim: 1,0,1\nvox: 1,1,1\ndatatype: Bit\nEND\n",
        "mrtrix image\ndim: 1,1,1\nvox: 1,1,1\ndatatype: Float16\nEND\n",
        "mrtrix image\ndim: 1,1,1\nvox: 1,1,1\ndatatype: Bit\n",
        "mrtrix image\ndim: 2,2\nvox: 1,1\ndatatype: Bit\n"
        "transform: 1,0,0,0\nEND\n",
    ],
    ids=[
        "magic",
        "no-dim",
        "no-vox",
        "no-datatype",
        "short-vox",
        "zero-dim",
        "bad-datatype",
        "no-end",
        "short-transform",
    ],
)
def test_a_malformed_header_is_refused(text) -> None:  # noqa: ANN001
    with pytest.raises(ParserContentError):
        MrtrixHeader.from_text(text)


@pytest.mark.parametrize(
    "layout", ["+0,+1", "+0,+0,+1", "+0,+1,+3", "0,x,1", "+0,+1,+2,+3"]
)
def test_a_malformed_layout_is_refused(layout) -> None:  # noqa: ANN001
    with pytest.raises(ParserContentError):
        parse_layout(layout, 3)


def test_a_layout_keeps_the_sign_of_rank_zero() -> None:
    assert parse_layout("-0,+1,-2", 3) == (-1, 2, -3)
    assert parse_layout("1,0,2", 3) == (2, 1, 3)


@pytest.mark.parametrize(
    "spec, dtype",
    [
        ("Int8", "i1"),
        ("UInt8", "u1"),
        ("int16le", "<i2"),
        ("UInt16BE", ">u2"),
        ("Int32LE", "<i4"),
        ("UInt32BE", ">u4"),
        ("Int64BE", ">i8"),
        ("UInt64LE", "<u8"),
        ("Float32BE", ">f4"),
        ("Float64LE", "<f8"),
        ("CFloat32LE", "<c8"),
        ("CFloat64BE", ">c16"),
        ("Float32", "=f4"),
    ],
)
def test_data_types(spec, dtype) -> None:  # noqa: ANN001
    assert mrtrix_dtype(spec) == np.dtype(dtype)
    assert mrtrix_dtype(dtype_to_mrtrix(np.dtype(dtype))) == np.dtype(dtype)


def test_bit_and_unsupported_data_types() -> None:
    assert mrtrix_dtype("Bit") is None
    assert dtype_to_mrtrix(bool) == "Bit"
    assert dtype_to_mrtrix("float32").endswith(("LE", "BE"))
    with pytest.raises(ParserError):
        dtype_to_mrtrix("float16")


# ----------------------------------------------------------------------
#   LAYOUTS
# ----------------------------------------------------------------------

LAYOUTS = [
    "+0,+1,+2",
    "-0,-1,+2",
    "+2,+1,+0",
    "+1,+2,+0",
    "-2,+0,-1",
    "-0,+2,+1",
]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_layout_is_undone_on_read(tmp_path, layout) -> None:  # noqa: ANN001
    payload = _reference_bytes(DATA, layout, "<f")
    head = _header_text(DATA.shape, layout, "Float32LE")
    source = _write(tmp_path / "a.mif", _single_file(head, payload, pad=3))

    image = io.images.load(source)
    assert isinstance(image, MrtrixImage)
    assert image.data.shape == DATA.shape
    assert np.array_equal(np.asarray(image.data), DATA)
    # The data is a view of a memory map, not a copy.
    assert isinstance(image.data, np.memmap) or isinstance(
        image.data.base, np.memmap
    )


def test_a_4d_volume_fastest_layout(tmp_path) -> None:  # noqa: ANN001
    # MRtrix stores DWI data volume-contiguous.
    data = np.arange(3 * 2 * 2 * 5, dtype="<i2").reshape(3, 2, 2, 5)
    layout = "+1,+2,+3,+0"
    payload = _reference_bytes(data, layout, "<h")
    # The file starts with the five volumes of voxel 0.
    assert struct.unpack("<5h", payload[:10]) == tuple(data[0, 0, 0])
    head = _header_text(data.shape, layout, "Int16LE")
    source = _write(tmp_path / "dwi.mif", _single_file(head, payload))
    image = MrtrixImage.load(source)
    assert np.array_equal(np.asarray(image.data), data)
    assert [axis.name for axis in image.system.axes] == ["x", "y", "z", "dim3"]


@pytest.mark.parametrize(
    "datatype, fmt, dtype",
    [
        ("UInt16BE", ">H", "u2"),
        ("Int32LE", "<i", "i4"),
        ("Float64BE", ">d", "f8"),
        ("Int8", "b", "i1"),
        ("UInt64LE", "<Q", "u8"),
    ],
)
def test_byte_orders(tmp_path, datatype, fmt, dtype) -> None:  # noqa: ANN001
    data = np.arange(DATA.size).reshape(DATA.shape).astype(dtype)
    layout = "-0,+1,-2"
    payload = _reference_bytes(data, layout, fmt)
    head = _header_text(data.shape, layout, datatype)
    source = _write(tmp_path / "a.mif", _single_file(head, payload))
    image = MrtrixImage.load(source)
    assert np.array_equal(np.asarray(image.data), data)


def test_complex_data(tmp_path) -> None:  # noqa: ANN001
    data = (DATA + 1j * DATA[::-1]).astype("complex64")
    layout = "+0,+1,+2"
    payload = b"".join(
        struct.pack("<ff", v.real, v.imag)
        for v in _reference_order(data, layout)
    )
    head = _header_text(data.shape, layout, "CFloat32LE")
    source = _write(tmp_path / "a.mif", _single_file(head, payload))
    assert np.array_equal(np.asarray(MrtrixImage.load(source).data), data)


@pytest.mark.parametrize("layout", ["+0,+1,+2", "-0,+2,-1"])
def test_bit_data(tmp_path, layout) -> None:  # noqa: ANN001
    rng = np.random.default_rng(0)
    data = rng.random((3, 5, 2)) > 0.5
    payload = _reference_bits(data, layout)
    head = _header_text(data.shape, layout, "Bit")
    source = _write(tmp_path / "mask.mif", _single_file(head, payload))
    image = MrtrixImage.load(source)
    assert image.data.dtype == bool
    assert np.array_equal(np.asarray(image.data), data)

    target = tmp_path / "out.mif"
    image.save(target)
    assert b"datatype: Bit" in target.read_bytes()[:200]
    assert np.array_equal(np.asarray(MrtrixImage.load(target).data), data)


def test_intensity_scaling(tmp_path) -> None:  # noqa: ANN001
    stored = np.arange(DATA.size, dtype="<u2").reshape(DATA.shape)
    payload = _reference_bytes(stored, "+0,+1,+2", "<H")
    head = _header_text(
        stored.shape, "+0,+1,+2", "UInt16LE", extra="scaling: -1,0.5"
    )
    source = _write(tmp_path / "a.mif", _single_file(head, payload))
    image = MrtrixImage.load(source)
    assert np.array_equal(np.asarray(image.dataobj), stored)
    assert np.allclose(np.asarray(image.data), -1 + 0.5 * stored)


# ----------------------------------------------------------------------
#   FILE VARIANTS
# ----------------------------------------------------------------------


def test_a_header_with_a_separate_data_file(tmp_path) -> None:  # noqa: ANN001
    layout = "-0,-1,+2"
    payload = _reference_bytes(DATA, layout, ">f")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "values.dat").write_bytes(b"\xff" * 7 + payload)
    head = _header_text(
        DATA.shape, layout, "Float32BE", file="sub/values.dat 7"
    )
    source = _write(tmp_path / "a.mih", (head + "END\n").encode())

    image = io.load(source)
    assert isinstance(image, MrtrixImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_a_missing_data_file_is_reported(tmp_path) -> None:  # noqa: ANN001
    head = _header_text(DATA.shape, "+0,+1,+2", "Float32LE", file="no.dat")
    source = _write(tmp_path / "a.mih", (head + "END\n").encode())
    with pytest.raises(FileNotFoundError):
        MrtrixImage.load(source)


def test_a_gzipped_image(tmp_path) -> None:  # noqa: ANN001
    layout = "+1,-0,+2"
    payload = _reference_bytes(DATA, layout, "<f")
    head = _header_text(DATA.shape, layout, "Float32LE")
    content = gzip.compress(_single_file(head, payload))
    source = _write(tmp_path / "a.mif.gz", content)

    assert io.images.sniff(source) is MrtrixImage
    image = io.images.load(source)
    assert isinstance(image, MrtrixImage)
    assert np.array_equal(np.asarray(image.data), DATA)


def test_streams_and_bytes(tmp_path) -> None:  # noqa: ANN001
    head = _header_text(DATA.shape, "+0,+1,+2", "Float32LE")
    content = _single_file(head, _reference_bytes(DATA, "+0,+1,+2", "<f"))
    for source in (content, gzip.compress(content)):
        assert MrtrixImage.sniff(source) >= Confidence.LIKELY
        image = MrtrixImage.from_bytes(source)
        assert np.array_equal(np.asarray(image.data), DATA)
        stream = _io.BytesIO(source)
        image = MrtrixImage.from_fileobj(stream)
        assert np.array_equal(np.asarray(image.data), DATA)
        assert stream.tell() == 0


def test_other_content_is_not_mrtrix() -> None:
    assert MrtrixImage.sniff(b"\x00" * 400) == Confidence.NO
    assert MrtrixImage.sniff(b"mrtrix imagine\nEND\n") == Confidence.NO
    with pytest.raises(ParserError):
        MrtrixImage.from_bytes(b"not an image")


def test_the_format_is_registered() -> None:
    assert MrtrixImage in Format._REGISTRY
    assert MrtrixImage in ImageFormat._REGISTRY
    assert issubclass(MrtrixImage, FileWriter)


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------


def _with_geometry(tmp_path, vox, transform=None, dim=DATA.shape):  # noqa: ANN001, ANN202
    layout = ",".join(f"+{i}" for i in range(len(dim)))
    lines = [
        "mrtrix image",
        "dim: " + ",".join(map(str, dim)),
        "vox: " + ",".join(map(str, vox)),
        f"layout: {layout}",
        "datatype: Float32LE",
    ]
    if transform is not None:
        for row in transform:
            lines.append("transform: " + ",".join(repr(float(v)) for v in row))
    head = "\n".join(lines) + "\n"
    data = np.zeros(dim, dtype="float32")
    payload = _reference_bytes(data, layout, "<f")
    return _write(tmp_path / "geom.mif", _single_file(head, payload))


def test_voxel_to_scanner_multiplies_by_the_voxel_size(tmp_path) -> None:  # noqa: ANN001
    vox = (2.0, 3.0, 4.0)
    image = MrtrixImage.load(_with_geometry(tmp_path, vox, ROTATED))
    scanner = image.transformation
    assert isinstance(scanner, Affine)
    assert scanner.output.name == "scanner"
    assert [a.orientation.value for a in scanner.output.axes] == [
        "left-to-right",
        "posterior-to-anterior",
        "inferior-to-superior",
    ]
    for index in [(0, 0, 0), (1, 2, 3), (1, 0, 2)]:
        expected = ROTATED[:, :3] @ (np.multiply(vox, index)) + ROTATED[:, 3]
        got = scanner.homogeneous_matrix @ np.array([*index, 1.0])
        assert np.allclose(got[:3], expected)
    # Voxel (1, 0, 0) is 2 mm along +y, (0, 1, 0) is 3 mm along -x.
    assert np.allclose(
        scanner.homogeneous_matrix @ [1, 0, 0, 1], [10, -18, 5, 1]
    )
    assert np.allclose(
        scanner.homogeneous_matrix @ [0, 1, 0, 1], [7, -20, 5, 1]
    )

    physical = image.transformations[0]
    assert physical.output.name == "physical"
    assert np.allclose(np.asarray(physical.scale), vox)
    assert image.system.order == "F"


def test_non_unit_direction_cosines(tmp_path) -> None:  # noqa: ANN001
    # MRtrix moves the column length into the voxel size, which leaves
    # transform @ diag(vox) unchanged.
    transform = ROTATED.copy()
    transform[:, 0] *= 2.0
    image = MrtrixImage.load(_with_geometry(tmp_path, (1, 1, 1), transform))
    expected = np.eye(4)
    expected[:3] = transform
    assert np.allclose(image.transformation.homogeneous_matrix, expected)


def test_a_missing_transform_centres_the_field_of_view(tmp_path) -> None:  # noqa: ANN001
    vox = (2.0, 3.0, 4.0)
    image = MrtrixImage.load(_with_geometry(tmp_path, vox, None))
    matrix = image.transformation.homogeneous_matrix
    expected = np.diag([*vox, 1.0])
    expected[:3, 3] = [-0.5 * (n - 1) * v for n, v in zip(DATA.shape, vox)]
    assert np.allclose(matrix, expected)
    # The centre of the field of view is at the origin.
    centre = [(n - 1) / 2 for n in DATA.shape]
    assert np.allclose(matrix @ [*centre, 1.0], [0, 0, 0, 1])


def test_a_nan_volume_size(tmp_path) -> None:  # noqa: ANN001
    dim = (2, 2, 2, 3)
    source = _with_geometry(tmp_path, (1, 1, 1, "nan"), ROTATED, dim=dim)
    image = MrtrixImage.load(source)
    assert np.allclose(np.asarray(image.transformations[0].scale), 1)
    target = tmp_path / "out.mif"
    image.save(target)
    assert math.isnan(MrtrixImage.load(target).header.vox[3])


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _source(tmp_path, layout="-0,+2,+1", datatype="Float32BE", fmt=">f"):  # noqa: ANN001, ANN202
    head = (
        "mrtrix image\n"
        "dim: 2,3,4\n"
        "vox: 2,3,4\n"
        f"layout: {layout}\n"
        f"datatype: {datatype}\n"
        + "".join(
            "transform: " + ",".join(repr(float(v)) for v in row) + "\n"
            for row in ROTATED
        )
        + "dw_scheme: 0,0,1,0\n"
        "dw_scheme: 1,0,0,1000\n"
        "command_history: mrconvert in.nii out.mif\n"
    )
    payload = _reference_bytes(DATA, layout, fmt)
    return _write(tmp_path / "source.mif", _single_file(head, payload))


@pytest.mark.parametrize("name", ["out.mif", "out.mih", "out.mif.gz"])
def test_an_image_round_trips(tmp_path, name) -> None:  # noqa: ANN001
    image = io.images.load(_source(tmp_path))
    target = tmp_path / name
    image.save(target)
    reloaded = io.images.load(target)
    assert isinstance(reloaded, MrtrixImage)
    assert np.array_equal(np.asarray(reloaded.data), DATA)
    assert np.allclose(
        reloaded.transformation.homogeneous_matrix,
        image.transformation.homogeneous_matrix,
    )
    before, after = image.header, reloaded.header
    assert after.layout == before.layout
    assert after.datatype == before.datatype
    assert after.vox == before.vox
    assert np.allclose(after.transform, before.transform)
    assert after.keyval == before.keyval
    if name.endswith(".mih"):
        assert (tmp_path / "out.dat").exists()
        assert after.file == ("out.dat", 0)


def test_the_written_bytes_follow_the_layout(tmp_path) -> None:  # noqa: ANN001
    image = io.images.load(_source(tmp_path))
    target = tmp_path / "out.mif"
    image.save(target, layout="+2,-0,+1", datatype="Int16BE")
    content = target.read_bytes()
    header, _ = MrtrixHeader.from_fileobj(_io.BytesIO(content))
    assert header.datatype == "Int16BE"
    assert header.file[1] % 4 == 0
    expected = _reference_bytes(DATA.astype("i2"), "+2,-0,+1", ">h")
    assert content[header.file[1] :] == expected


def test_a_written_file_starts_as_mrtrix_writes_it(tmp_path) -> None:  # noqa: ANN001
    image = MrtrixImage.from_any(SingleScaleImage(data=DATA))
    content = image.to_bytes()
    lines = content.split(b"\n")
    assert lines[0] == b"mrtrix image"
    assert lines[1] == b"dim: 2,3,4"
    assert lines[3] == b"layout: +0,+1,+2"
    assert lines[4] in (b"datatype: Float32LE", b"datatype: Float32BE")
    end = content.index(b"\nEND\n") + len(b"\nEND\n")
    offset = int(content[:end].split(b"file: . ")[1].split(b"\n")[0])
    assert offset >= end and offset % 4 == 0
    assert len(content) == offset + DATA.nbytes


def test_header_keys_can_be_changed_when_writing(tmp_path) -> None:  # noqa: ANN001
    image = io.images.load(_source(tmp_path))
    target = tmp_path / "out.mif"
    image.save(target, keyval={"dw_scheme": None, "comments": "a\nb"})
    keyval = MrtrixImage.load(target).header.keyval
    assert "dw_scheme" not in keyval
    assert keyval["comments"] == "a\nb"
    assert keyval["command_history"] == "mrconvert in.nii out.mif"


def test_an_image_without_geometry_is_written_at_identity(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "out.mif"
    io.save(SingleScaleImage(data=DATA), target)
    image = MrtrixImage.load(target)
    assert np.array_equal(np.asarray(image.data), DATA)
    assert np.allclose(image.transformation.homogeneous_matrix, np.eye(4))


def test_an_lps_world_is_written_as_ras(tmp_path) -> None:  # noqa: ANN001
    voxel = CoordinateSystem(name="voxel", axes=["x", "y", "z"])
    lps = CoordinateSystem(
        name="LPS",
        axes=[
            {"name": "x", "orientation": "right-to-left"},
            {"name": "y", "orientation": "anterior-to-posterior"},
            {"name": "z", "orientation": "inferior-to-superior"},
        ],
    )
    matrix = np.array([[2.0, 0, 0, 1.0], [0, 3.0, 0, 2.0], [0, 0, 4.0, 3.0]])
    affine = Affine(input=voxel, output=lps, matrix=matrix)
    target = tmp_path / "out.mif"
    io.save(SingleScaleImage(data=DATA, transformations=[affine]), target)
    image = MrtrixImage.load(target)
    expected = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.vstack(
        [matrix, [0, 0, 0, 1]]
    )
    assert np.allclose(image.transformation.homogeneous_matrix, expected)
    assert image.header.vox == (2.0, 3.0, 4.0)
    assert np.allclose(
        image.header.transform[:, :3], np.diag([-1.0, -1.0, 1.0])
    )


def test_an_image_round_trips_through_nifti(tmp_path) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    image = io.images.load(_source(tmp_path))
    nifti = tmp_path / "out.nii"
    io.save(SingleScaleImage.from_instance(image), nifti)
    assert np.allclose(
        nb.load(str(nifti)).affine, image.transformation.homogeneous_matrix
    )
    back = tmp_path / "back.mif"
    io.save(SingleScaleImage.from_instance(io.images.load(nifti)), back)
    reloaded = MrtrixImage.load(back)
    assert np.array_equal(np.asarray(reloaded.data), DATA)
    assert np.allclose(
        reloaded.transformation.homogeneous_matrix,
        image.transformation.homogeneous_matrix,
    )
