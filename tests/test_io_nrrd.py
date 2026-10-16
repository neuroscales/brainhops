"""Tests for NRRD images (.nrrd, and .nhdr with data files).

Headers are written by hand from the specification
(https://teem.sourceforge.net/nrrd/format.html), and the data is encoded
independently of the writer.
"""

import bz2
import gzip
import io as _io
import pickle
import struct

import numpy as np
import pytest
from bagof.magic import replace

import brainhops.io as io
from brainhops.backends import backend
from brainhops.datamodel.transformations import Affine, Scaling
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    ParserError,
    WriterError,
)
from brainhops.io.common.nrrd import NrrdMetadata, NrrdRaw
from brainhops.io.common.nrrd._codecs import dtype_to_nrrd, nrrd_dtype
from brainhops.io.common.nrrd._data import _NrrdProxy
from brainhops.io.images import ImageFormat
from brainhops.io.images.nrrd import (
    AttachedNrrdImage,
    DetachedNrrdImage,
    NrrdImage,
)
from brainhops.io.metadata import MetadataFormat

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
    header = NrrdRaw.from_lines(lines)
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
        NrrdRaw.from_lines(["type: short", "dimension: 1"])
    header = NrrdRaw.from_lines(
        ["type: short", "dimension: 2", "sizes: 1 2", "encoding: raw"]
    )
    with pytest.raises(ParserContentError, match="endian"):
        _ = header.dtype
    with pytest.raises(ParserContentError):
        NrrdRaw.from_fileobj(_io.BytesIO(b"NOTNRRD\n"))


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_read_attached_lps(lps_file) -> None:  # noqa: ANN001
    name, values = lps_file
    image = io.load(name)
    assert type(image) is AttachedNrrdImage
    assert isinstance(image.rawdata, _NrrdProxy) and image.rawdata.mmap
    assert isinstance(image.rawdata.read_samples().base, np.memmap)
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
    assert image.metadata.raw.keyvalue["DWMRI_b-value"] == "1000"
    assert image.metadata.raw.keyvalue["note"] == "two\nlines"


def test_read_no_mmap(lps_file) -> None:  # noqa: ANN001
    name, values = lps_file
    image = AttachedNrrdImage.load(name, mmap=False)
    assert not image.rawdata.mmap
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
    assert ImageFormat.sniff(name) is AttachedNrrdImage
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
    assert back.metadata.raw.space == "left-posterior-superior"
    assert back.metadata.raw.encoding == "raw"
    assert back.metadata.raw.keyvalue == image.metadata.raw.keyvalue
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
    assert back.metadata.raw.data_files[0].startswith("b.")
    np.testing.assert_array_equal(back.data, image.data)
    assert back.metadata.raw.fields["kinds"] == "vector domain domain domain"
    assert back.metadata.raw.fields["measurement frame"] == (
        "(1,0,0) (0,1,0) (0,0,1)"
    )


def test_write_space_conversion(lps_file, tmp_path) -> None:  # noqa: ANN001
    name, _ = lps_file
    image = io.load(name)
    image.save(tmp_path / "ras.nrrd", space="RAS")
    back = io.load(tmp_path / "ras.nrrd")
    assert back.metadata.raw.space == "right-anterior-superior"
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
    assert back.rawdata.dtype == np.dtype(">f4")
    np.testing.assert_array_equal(back.data, values)
    assert "DWMRI_b-value" not in back.metadata.raw.keyvalue
    assert back.metadata.raw.keyvalue["new"] == "1"
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
    assert nrrd_image.metadata.raw.space == "right-anterior-superior"
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
    assert "space" not in back.metadata.raw.fields
    assert "space dimension" not in back.metadata.raw.fields
    np.testing.assert_allclose(
        back.transformation.matrix, image.transformation.matrix
    )


def test_an_image_whose_transformations_are_emptied_has_no_geometry(
    lps_file,  # noqa: ANN001
    tmp_path,  # noqa: ANN001
) -> None:
    """An empty list is honoured, and `None` decodes the record again.

    The writer encodes an image without transformations as it encodes an
    image built from data alone: the header keeps the other fields of the
    base, but neither its space nor the spacings and positions of its
    spatial axes.
    """
    name, _ = lps_file
    image = io.load(name)
    decoded = image.transformation.matrix
    image.transformations = []
    assert image.transformations == []
    image.save(tmp_path / "emptied.nrrd")
    fields = io.load(tmp_path / "emptied.nrrd").metadata.raw.fields
    for field in ("space", "space directions", "space origin", "spacings"):
        assert field not in fields
    assert fields["kinds"] == "domain domain domain"
    image.transformations = None
    np.testing.assert_allclose(image.transformation.matrix, decoded)


def test_an_unoriented_image_whose_transformations_are_emptied_has_no_spacings(
    tmp_path,  # noqa: ANN001
) -> None:
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
    image.transformations = []
    image.save(tmp_path / "b.nrrd")
    fields = io.load(tmp_path / "b.nrrd").metadata.raw.fields
    assert "spacings" not in fields
    assert "axis mins" not in fields


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
        back.metadata.raw.measurement_frame,
        [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
    )


# ----------------------------------------------------------------------
#   RECORD, METADATA AND LAZY ARRAY
# ----------------------------------------------------------------------

_ENCODERS = {
    "raw": lambda b: b,
    "gzip": gzip.compress,
    "bzip2": bz2.compress,
    "hex": lambda b: b.hex().encode() + b"\n",
}

# A header in version 3, with a comment, spellings that the writer would
# change, and fields that the data model has no place for.
_FOREIGN = (
    b"NRRD0003\n"
    b"# written by hand\n"
    b"type: short\n"
    b"dimension: 3\n"
    b"space: LPS\n"
    b"sizes: 2 3 4\n"
    b"space directions: (2.0,0,0) (0,3,0) (0,0,4)\n"
    b"kinds: domain domain domain\n"
    b"endian: little\n"
    b"encoding: %s\n"
    b"space origin: (10,20,30)\n"
    b"min: 0\n"
    b"max: 23\n"
    b"content: kept\n"
    b"b:=1\n"
)


def _foreign(tmp_path, encoding="raw", name="source.nrrd"):  # noqa: ANN001, ANN202
    values = _values((2, 3, 4))
    payload = _ENCODERS[encoding](_pack(values, "<h"))
    target = tmp_path / name
    target.write_bytes(_FOREIGN % encoding.encode() + b"\n" + payload)
    return target, values


def _detached(tmp_path, name="source", encoding="gzip"):  # noqa: ANN001, ANN202
    values = _values((2, 3, 4))
    data = name + {"gzip": ".raw.gz", "raw": ".raw"}[encoding]
    (tmp_path / data).write_bytes(_ENCODERS[encoding](_pack(values, "<h")))
    header = (_FOREIGN % encoding.encode()) + f"data file: {data}\n".encode()
    (tmp_path / (name + ".nhdr")).write_bytes(header)
    return tmp_path / (name + ".nhdr"), values


def test_the_record_keeps_its_bytes_and_compares_by_text() -> None:
    stored = (_FOREIGN % b"raw") + b"\n"
    record = NrrdRaw.from_bytes(stored + b"samples")
    assert record.stored == stored
    assert record.to_bytes() == stored
    assert record.fields["content"] == "kept"
    # A record formats to the same text whatever bytes it was read from.
    plain = NrrdRaw.from_lines(stored.decode().splitlines()[1:-1], version=3)
    assert plain == record and plain.stored is None
    assert plain.to_bytes() != stored
    # A changed copy is formatted, and the record is not changed.
    copy = record.copy()
    copy.keyvalue["x"] = "y"
    assert copy != record and record.keyvalue == {"b": "1"}
    assert b"x:=y" in copy.to_bytes() and b"# written" not in copy.to_bytes()


@pytest.mark.parametrize("detached", [False, True])
def test_metadata_reads_the_header_alone(tmp_path, detached) -> None:  # noqa: ANN001
    if detached:
        source, _ = _detached(tmp_path)
        (tmp_path / "source.raw.gz").unlink()
    else:
        source, _ = _foreign(tmp_path)
        cut = source.read_bytes()[: len(_FOREIGN % b"raw") + 1]
        source.write_bytes(cut)
    metadata = NrrdMetadata.load(source)
    assert metadata.raw.fields["content"] == "kept"
    assert metadata.to_bytes() == source.read_bytes()
    assert pickle.loads(pickle.dumps(metadata)).raw == metadata.raw
    assert metadata.raw.copy() == metadata.raw
    assert metadata.raw.copy() is not metadata.raw
    assert isinstance(MetadataFormat.load(source), NrrdMetadata)


@pytest.mark.parametrize("encoding", sorted(_ENCODERS))
def test_an_untouched_attached_image_keeps_its_bytes(
    tmp_path,  # noqa: ANN001
    encoding: str,
) -> None:
    source, values = _foreign(tmp_path, encoding)
    loaded = AttachedNrrdImage.load(source)
    assert loaded.consolidate().metadata.raw is loaded.metadata.raw
    writers = {
        "save": lambda target: loaded.save(target),
        "io.save": lambda target: io.save(
            AttachedNrrdImage.load(source), target
        ),
        "from_instance": lambda target: AttachedNrrdImage.from_instance(
            loaded
        ).save(target),
    }
    for label, write in writers.items():
        target = tmp_path / (label + ".nrrd")
        write(target)
        assert target.read_bytes() == source.read_bytes(), label
    assert AttachedNrrdImage.from_bytes(source.read_bytes()).to_bytes() == (
        source.read_bytes()
    )
    # A copy made with `replace` keeps the lazy array, so it is written as
    # the file stores it.
    target = _saved_as(replace(loaded), tmp_path / "replace.nrrd")
    assert target.read_bytes() == source.read_bytes()
    # Samples that are assigned are encoded again, with the fields of the
    # source except `min` and `max`.
    copy = replace(loaded)
    copy.data = np.array(values)
    back = io.load(_saved_as(copy, tmp_path / "assigned.nrrd"))
    np.testing.assert_array_equal(back.data, values)
    assert back.metadata.raw.fields["content"] == "kept"
    assert "min" not in back.metadata.raw.fields


def _saved_as(image, target):  # noqa: ANN001, ANN202
    image.save(target)
    return target


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_an_untouched_detached_image_keeps_both_files(
    tmp_path,  # noqa: ANN001
    encoding: str,
) -> None:
    source, values = _detached(tmp_path, encoding=encoding)
    loaded = DetachedNrrdImage.load(source)
    assert loaded.consolidate().metadata.raw is loaded.metadata.raw
    (tmp_path / "same").mkdir()
    loaded.save(tmp_path / "same" / source.name)
    names = sorted(p.name for p in (tmp_path / "same").iterdir())
    assert len(names) == 2
    for name in names:
        assert (tmp_path / "same" / name).read_bytes() == (
            tmp_path / name
        ).read_bytes(), name
    # Under another name, the header names its own data file, which holds
    # the same bytes.
    loaded.save(tmp_path / "other.nhdr")
    (data,) = loaded.metadata.raw.data_files
    other = tmp_path / ("other" + data[len("source") :])
    assert other.read_bytes() == (tmp_path / data).read_bytes()
    back = DetachedNrrdImage.load(tmp_path / "other.nhdr")
    assert back.metadata.raw.data_files == (other.name,)
    np.testing.assert_array_equal(back.data, values)


def test_skipped_bytes_are_copied_with_the_samples(tmp_path) -> None:  # noqa: ANN001
    values = _values((2, 3), "int32")
    payload = _pack(values, "<i")
    (tmp_path / "z.raw.gz").write_bytes(
        b"line\n" + gzip.compress(b"\x01\x02" + payload)
    )
    (tmp_path / "z.nhdr").write_bytes(
        _header(
            "type: int",
            "dimension: 2",
            "sizes: 2 3",
            "endian: little",
            "encoding: gzip",
            "line skip: 1",
            "byte skip: 2",
            "data file: z.raw.gz",
        )
    )
    loaded = DetachedNrrdImage.load(tmp_path / "z.nhdr")
    (tmp_path / "copy").mkdir()
    loaded.save(tmp_path / "copy" / "z.nhdr")
    for name in ("z.nhdr", "z.raw.gz"):
        assert (tmp_path / "copy" / name).read_bytes() == (
            tmp_path / name
        ).read_bytes()
    # The skips stay with the samples in an attached copy, and go when the
    # samples are encoded again.
    loaded.save(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(io.load(tmp_path / "a.nrrd").data, values)
    loaded.save(tmp_path / "b.nrrd", encoding="raw")
    back = io.load(tmp_path / "b.nrrd")
    assert "line skip" not in back.metadata.raw.fields
    np.testing.assert_array_equal(back.data, values)


def test_samples_in_several_files_are_written_in_one(tmp_path) -> None:  # noqa: ANN001
    values = _values((2, 2, 3), "uint8")
    for k in range(3):
        (tmp_path / f"s{k + 1:02d}.raw").write_bytes(
            values[:, :, k].ravel(order="F").tobytes()
        )
    (tmp_path / "p.nhdr").write_bytes(
        _header(
            "type: uchar",
            "dimension: 3",
            "sizes: 2 2 3",
            "encoding: raw",
            "data file: s%02d.raw 1 3 1 2",
        )
    )
    loaded = DetachedNrrdImage.load(tmp_path / "p.nhdr")
    assert not loaded.rawdata.copyable
    loaded.save(tmp_path / "q.nhdr")
    back = DetachedNrrdImage.load(tmp_path / "q.nhdr")
    assert back.metadata.raw.data_files == ("q.raw",)
    assert back.metadata.raw.fields["type"] == "uchar"
    np.testing.assert_array_equal(back.data, values)


def test_an_image_saved_over_itself_keeps_its_samples(tmp_path) -> None:  # noqa: ANN001
    source, values = _foreign(tmp_path)
    before = source.read_bytes()
    AttachedNrrdImage.load(source).save(source)
    assert source.read_bytes() == before
    # A longer header moves the samples. A memory map of the old file
    # keeps its content, and the lazy array reads the new file.
    loaded = AttachedNrrdImage.load(source)
    data = loaded.data
    loaded.transformations = [Affine(matrix=np.diag([5.0, 6.0, 7.0, 1.0])[:3])]
    loaded.save(source)
    assert source.read_bytes() != before
    np.testing.assert_array_equal(data, values)
    np.testing.assert_array_equal(np.asarray(loaded.rawdata), values)
    np.testing.assert_allclose(
        io.load(source).transformation.matrix,
        np.diag([5.0, 6.0, 7.0]) @ np.eye(3, 4),
    )


def test_a_detached_image_saved_over_itself_keeps_its_samples(
    tmp_path,  # noqa: ANN001
) -> None:
    source, values = _detached(tmp_path)
    loaded = DetachedNrrdImage.load(source)
    loaded.save(source, encoding="raw", datatype="int32")
    # The old data file is kept, the header names the new one, and the
    # lazy array reads the new files.
    assert (tmp_path / "source.raw.gz").exists()
    assert loaded.rawdata.record.data_files == ("source.raw",)
    np.testing.assert_array_equal(np.asarray(loaded.rawdata), values)
    assert np.asarray(loaded.rawdata).dtype == np.int32
    np.testing.assert_array_equal(io.load(source).data, values)


def test_a_file_written_again_elsewhere_is_refused(tmp_path) -> None:  # noqa: ANN001
    source, values = _foreign(tmp_path)
    loaded = AttachedNrrdImage.load(source)
    AttachedNrrdImage(values.astype("float32")).save(source)
    with pytest.raises(ParserContentError, match="no longer describes"):
        np.asarray(loaded.rawdata)


def test_consolidate_follows_each_knob(tmp_path) -> None:  # noqa: ANN001
    """Each public knob is either written or leaves the record as read."""
    source, values = _foreign(tmp_path, "gzip")

    def loaded():  # noqa: ANN202
        return AttachedNrrdImage.load(source)

    image = loaded()
    record = image.metadata.raw
    assert image.consolidate().metadata.raw is record
    # Options that name what the file stores keep the record and the
    # stored samples.
    same = dict(encoding="gz", endian="little", datatype="int16", space="LPS")
    assert image.consolidate(**same).metadata.raw is record
    assert image.to_bytes(**same) == source.read_bytes()
    # Other options are written, and the samples are encoded again.
    assert (
        image.consolidate(encoding="raw").metadata.raw.fields["encoding"]
        == "raw"
    )
    assert (
        image.consolidate(endian="big").metadata.raw.fields["endian"] == "big"
    )
    written = image.consolidate(datatype="float").metadata.raw
    assert written.fields["type"] == "float" and "min" not in written.fields
    written = image.consolidate(space="RAS").metadata.raw
    assert written.fields["space"] == "right-anterior-superior"
    assert written.fields["space origin"] == "(-10,-20,30)"
    back = io.load(_saved_as(image, tmp_path / "be.nrrd"))
    np.testing.assert_array_equal(back.data, values)
    # Key/value pairs are merged last, and a pair that would read back as
    # another is refused.
    written = image.consolidate(keyvalue={"b": None, "x": "y"}).metadata.raw
    assert written.keyvalue == {"x": "y"}
    for key in ("a:=b", "#a", ""):
        with pytest.raises(WriterError, match="key"):
            image.consolidate(keyvalue={key: "1"})
    with pytest.raises(WriterError, match="UTF-8"):
        image.consolidate(keyvalue={"a": "\ud800"})
    with pytest.raises(WriterError):
        image.consolidate(encoding="zip")
    with pytest.raises(WriterError):
        image.consolidate(endian="middle")
    with pytest.raises(WriterError):
        image.consolidate(space="scanner-xyz")
    with pytest.raises(TypeError):
        image.to_bytes(data_file="x.raw")

    # New samples are stored with their own type, and keep the geometry.
    image = loaded()
    image.data = np.asarray(image.data, dtype="float32") * 2
    written = image.consolidate().metadata.raw
    assert written is not record and written.fields["type"] == "float"
    assert (
        written.fields["space directions"]
        == (record.fields["space directions"])
    )
    # Assigned transformations give the geometry.
    image = loaded()
    image.transformations = [Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])]
    written = image.consolidate().metadata.raw
    assert written.fields["space directions"] == "(2,0,0) (0,3,0) (0,0,4)"
    assert written.fields["content"] == "kept"
    # Other metadata gives its record.
    image = loaded()
    edited = record.copy()
    edited.keyvalue["x"] = "y"
    image.metadata = NrrdMetadata.from_raw(edited)
    assert image.consolidate().metadata.raw is edited
    # The record of the metadata is never changed.
    assert record == AttachedNrrdImage.load(source).metadata.raw
    assert record.stored == AttachedNrrdImage.load(source).metadata.raw.stored
    # The index system is derived from the record and cannot be assigned.
    with pytest.raises(AttributeError):
        image.system = None


class _CountedReads:
    """Count the reads of the samples of every NRRD image."""

    def __init__(self, monkeypatch) -> None:  # noqa: ANN001
        self.count = 0
        samples = _NrrdProxy.read_samples
        stored = _NrrdProxy.read_stored

        def counted_samples(proxy, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            self.count += 1
            return samples(proxy, *args, **kwargs)

        def counted_stored(proxy, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            self.count += 1
            return stored(proxy, *args, **kwargs)

        monkeypatch.setattr(_NrrdProxy, "read_samples", counted_samples)
        monkeypatch.setattr(_NrrdProxy, "read_stored", counted_stored)


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_nothing_but_the_data_reads_the_samples(
    tmp_path,  # noqa: ANN001
    encoding: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, values = _foreign(tmp_path, encoding)
    reads = _CountedReads(monkeypatch)
    image = io.load(source)
    assert image.shape == values.shape
    assert image.ndim == 3
    assert image.rawdata.dtype == np.dtype("<i2")
    repr(image)
    assert image.transformations
    assert image.system is not None
    assert image.consolidate().metadata.raw is image.metadata.raw
    pickle.loads(pickle.dumps(image))
    AttachedNrrdImage.from_bytes(source.read_bytes())
    assert reads.count == 0
    assert "_cache_data" not in vars(image)
    with backend("numpy"):
        data = image.data
    assert not data.flags.writeable
    np.testing.assert_array_equal(data, values)
    assert reads.count == 1
    image.data = np.zeros((2, 3))
    assert image.shape == (2, 3)
    with pytest.raises(AttributeError):
        _ = NrrdImage().shape


def test_dask_reads_the_samples_once(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    pytest.importorskip("dask")
    source, values = _foreign(tmp_path, "gzip")
    reads = _CountedReads(monkeypatch)
    with backend("dask"):
        data = io.load(source).data
    assert reads.count == 0
    np.testing.assert_array_equal(np.asarray(data), values)
    assert reads.count == 1


def test_a_stream_stays_the_source_of_the_samples(tmp_path) -> None:  # noqa: ANN001
    source, values = _foreign(tmp_path, "gzip")
    with open(source, "rb") as f:
        f.seek(0)
        image = AttachedNrrdImage.from_fileobj(f)
        assert f.tell() == 0
        np.testing.assert_array_equal(image.data, values)
    detached, _ = _detached(tmp_path)
    with pytest.raises(ParserContentError, match="no file name"):
        DetachedNrrdImage.from_bytes(detached.read_bytes())


def test_a_space_is_refused_for_a_world_without_orientation(
    tmp_path,  # noqa: ANN001
) -> None:
    (tmp_path / "a.nrrd").write_bytes(
        _header(
            "type: uint8",
            "dimension: 3",
            "space dimension: 3",
            "sizes: 2 2 2",
            "space directions: (1,0,0) (0,1,0) (0,0,1)",
            "encoding: raw",
        )
        + b"\n"
        + bytes(8)
    )
    image = io.load(tmp_path / "a.nrrd")
    assert image.consolidate().metadata.raw is image.metadata.raw
    with pytest.raises(WriterError, match="anatomical orientation"):
        image.consolidate(space="LPS")
    with pytest.raises(WriterError, match="anatomical orientation"):
        AttachedNrrdImage(np.zeros((2, 2, 2))).consolidate(space="RAS")


def test_unknown_types_and_missing_directories_are_reported(
    tmp_path,  # noqa: ANN001
) -> None:
    source, _ = _foreign(tmp_path)
    image = AttachedNrrdImage.load(source)
    with pytest.raises(WriterError, match="type"):
        image.consolidate(datatype="no such type")
    with pytest.raises(FileNotFoundError) as error:
        image.save(tmp_path / "a.nhdr", data_file="missing/a.raw")
    assert error.value.filename == str(tmp_path / "missing" / "a.raw")
    # A new encoding keeps the spelling of the type of the source.
    assert (
        image.consolidate(encoding="gzip").metadata.raw.fields["type"]
        == "short"
    )


def test_a_unit_scaling_drops_the_spacings_of_the_base(
    tmp_path,  # noqa: ANN001
) -> None:
    # A file without geometry decodes to the unit scaling. Writing that
    # scaling over a record with spacings writes no geometry, rather than
    # the geometry of the record.
    data = np.arange(24, dtype="float32").reshape(2, 3, 4)
    AttachedNrrdImage(data).save(tmp_path / "bare.nrrd")
    placed = AttachedNrrdImage(data)
    placed.transformations = [Affine(matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3])]
    placed.save(tmp_path / "placed.nrrd")
    image = AttachedNrrdImage.load(tmp_path / "placed.nrrd")
    bare = AttachedNrrdImage.load(tmp_path / "bare.nrrd")
    image.transformations = list(bare.transformations)
    record = image.consolidate().metadata.raw
    assert "spacings" not in record.fields
    assert "axis mins" not in record.fields
    image.save(tmp_path / "copy.nrrd")
    back = AttachedNrrdImage.load(tmp_path / "copy.nrrd")
    np.testing.assert_array_equal(back.transformation.data, [1.0, 1.0, 1.0])
