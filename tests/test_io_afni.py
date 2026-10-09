"""Tests for AFNI datasets, a prefix+view.HEAD header with a .BRIK file.

The fixtures are built from the specification by an independent encoder:
the header attribute by attribute, as thd_writeatr.c writes it, and the
BRIK one voxel at a time with x fastest.
"""

import bz2
import gzip
import io as _io
import itertools
import os
import struct

import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling
from brainhops.io.base import FileBasedObject
from brainhops.io.base.parsers import (
    Confidence,
    FileWriter,
    ParserContentError,
    ParserError,
    ParserExistsError,
    WriterError,
)
from brainhops.io.base.specs import format_hints
from brainhops.io.common.afni import AfniHeader
from brainhops.io.common.afni._constants import DICOM_TO_RAS
from brainhops.io.common.afni._data import afni_dataset_files, brick_dtype
from brainhops.io.common.afni._geometry import (
    afni_cardinal_matrix,
    afni_geometry_from_matrix,
    afni_view,
)
from brainhops.io.images import FileBasedImage
from brainhops.io.images.afni import AfniImage

# ----------------------------------------------------------------------
#   REFERENCE ENCODER
# ----------------------------------------------------------------------

SHAPE = (4, 3, 2)


def _attribute(name, value):  # noqa: ANN001, ANN202
    """Encode one attribute as AFNI writes it."""
    if isinstance(value, str):
        chars = value.replace("~", "*").replace("\0", "~") + "~"
        return (
            f"\ntype = string-attribute\nname = {name}\n"
            f"count = {len(chars)}\n'{chars}\n"
        )
    if any(isinstance(v, float) for v in value):
        body = "".join(
            f" {v:14.7g}" + ("\n" if i % 5 == 4 and i < len(value) - 1 else "")
            for i, v in enumerate(value)
        )
        return (
            f"\ntype  = float-attribute\nname  = {name}\n"
            f"count = {len(value)}\n{body}\n"
        )
    body = "".join(
        f" {v}" + ("\n" if i % 5 == 4 and i < len(value) - 1 else "")
        for i, v in enumerate(value)
    )
    return (
        f"\ntype = integer-attribute\nname = {name}\n"
        f"count = {len(value)}\n{body}\n"
    )


def _header(  # noqa: ANN202
    shape: tuple = SHAPE,
    nvals: int = 1,
    orient: tuple = (0, 3, 4),
    origin: tuple = (-10.0, -20.0, -30.0),
    delta: tuple = (2.0, 3.0, 4.0),
    view: int = 0,
    types: object = None,
    **extra: object,
):
    """Encode a header; an attribute whose value is None is left out."""
    attrs = {
        "TYPESTRING": "3DIM_HEAD_ANAT",
        "SCENE_DATA": (view, 0, 0, -999, -999, -999, -999, -999),
        "ORIENT_SPECIFIC": tuple(orient),
        "ORIGIN": tuple(float(o) for o in origin),
        "DELTA": tuple(float(d) for d in delta),
        "DATASET_RANK": (3, nvals, 0, 0, 0, 0, 0, 0),
        "DATASET_DIMENSIONS": tuple(shape) + (0, 0),
        "BRICK_TYPES": tuple(types if types is not None else (3,) * nvals),
        "BYTEORDER_STRING": "LSB_FIRST",
    }
    attrs.update(extra)
    return "".join(
        _attribute(name, value)
        for name, value in attrs.items()
        if value is not None
    )


def _brik(bricks, fmts):  # noqa: ANN001, ANN202
    """Pack each sub-brick voxel by voxel, x fastest."""
    out = b""
    for brick, fmt in zip(bricks, fmts):
        nx, ny, nz = brick.shape
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    v = brick[i, j, k]
                    if np.iscomplexobj(v):
                        out += struct.pack(fmt, v.real, v.imag)
                    else:
                        out += struct.pack(fmt, v.item())
    return out


def _dataset(tmp_path, head, brik, name="dset+orig", suffix=""):  # noqa: ANN001, ANN202
    (tmp_path / f"{name}.HEAD").write_text(head)
    if brik is not None:
        if suffix == ".gz":
            brik = gzip.compress(brik)
        elif suffix == ".bz2":
            brik = bz2.compress(brik)
        (tmp_path / f"{name}.BRIK{suffix}").write_bytes(brik)
    return tmp_path / f"{name}.HEAD"


DATA = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) - 5.5


def _source(tmp_path, **kwargs):  # noqa: ANN001, ANN202
    return _dataset(tmp_path, _header(**kwargs), _brik([DATA], ["<f"]))


def _nibabel_data(name):  # noqa: ANN001, ANN202
    nb = pytest.importorskip("nibabel")
    root = os.path.join(os.path.dirname(nb.__file__), "tests", "data")
    path = os.path.join(root, name)
    if not os.path.exists(path):
        pytest.skip(f"nibabel test file {name} is not installed")
    return path


# ----------------------------------------------------------------------
#   HEADER
# ----------------------------------------------------------------------


def test_the_attributes_are_decoded() -> None:
    text = (
        "\ntype = integer-attribute\nname = ORIENT_SPECIFIC\ncount = 3\n"
        " 3 5 1\n"
        "\ntype  = float-attribute\nname  = IJK_TO_DICOM_REAL\ncount = 12\n"
        "  1 0 0 -10 0\n 1 0 -20 0 0\n 1 -30\n"
        "\ntype = string-attribute\nname = BRICK_LABS\ncount = 12\n"
        "'one~two~*hr~\n"
        "\ntype = integer-attribute\nname = EMPTY\ncount = 0\n"
        "type=string-attribute name=HISTORY_NOTE count=7 'a\\nb\nc~\n"
    )
    header = AfniHeader.from_text(text)
    assert header["ORIENT_SPECIFIC"] == (3, 5, 1)
    assert header["IJK_TO_DICOM_REAL"][3] == -10.0
    assert all(isinstance(v, float) for v in header["IJK_TO_DICOM_REAL"])
    assert header.labels == ["one", "two", "*hr"]
    assert "EMPTY" not in header
    # A string attribute counts its characters, newlines included.
    assert header["HISTORY_NOTE"] == "a\\nb\nc"


def test_the_header_writes_back_as_it_was_read() -> None:
    text = _header(
        nvals=2,
        BRICK_LABS="a\0b",
        BRICK_FLOAT_FACS=(0.5, 0.0),
        HISTORY_NOTE="made by\\n hand ~ tilde",
    )
    header = AfniHeader.from_text(text)
    again = AfniHeader.from_text(header.to_text())
    # A tilde in a string comes back as an asterisk, as in AFNI.
    expected = dict(header.attributes)
    expected["HISTORY_NOTE"] = "made by\\n hand * tilde"
    assert again.attributes == expected
    assert list(again.attributes) == list(header.attributes)


def test_written_attributes_follow_afni() -> None:
    header = AfniHeader(
        attributes={
            "ORIGIN": (1, 2, 3),
            "DATASET_RANK": (3, 1, 0, 0, 0, 0, 0, 0),
            "TYPESTRING": "3DIM_HEAD_ANAT",
        }
    )
    text = header.to_text()
    # A float attribute stays a float attribute when given integers.
    assert "type = float-attribute\nname = ORIGIN\ncount = 3\n 1 2 3\n" in text
    assert "count = 8\n 3 1 0 0 0\n 0 0 0\n" in text
    assert "count = 15\n'3DIM_HEAD_ANAT~\n" in text


@pytest.mark.parametrize(
    "text",
    [
        "type = vector-attribute\nname = X\ncount = 1\n 1\n",
        "type = string-attribute\nname = X\ncount = 3\nabc\n",
        "type = string-attribute\nname = X\ncount = 30\n'abc~\n",
        "type = integer-attribute\nname = X\ncount = 2\n 1 2.5\n",
        "type = float-attribute\nname = X\ncount = 3\n 1 2\n",
        "not a header at all",
    ],
)
def test_a_malformed_header_is_refused(text) -> None:  # noqa: ANN001
    with pytest.raises(ParserContentError):
        AfniHeader.from_text(text)


def test_reading_stops_at_trailing_garbage_as_in_afni() -> None:
    header = AfniHeader.from_text(_header() + "\n\x00\x00 garbage\n")
    assert header.shape == SHAPE


def test_a_niml_header_is_refused() -> None:
    with pytest.raises(ParserContentError, match="NIML"):
        AfniHeader.from_text("<AFNI_attributes ni_form='ni_group'>\n")


@pytest.mark.parametrize(
    "changes",
    [
        {"DATASET_RANK": None},
        {"DATASET_DIMENSIONS": None},
        {"ORIENT_SPECIFIC": (0, 1, 4)},
        {"ORIENT_SPECIFIC": (0, 3, 7)},
        {"DELTA": None},
        {"BRICK_TYPES": (6,)},
        {"DATASET_RANK": (3, 0, 0, 0, 0, 0, 0, 0)},
    ],
)
def test_an_invalid_header_is_refused(changes) -> None:  # noqa: ANN001
    header = AfniHeader.from_text(_header(**changes))
    with pytest.raises(ParserContentError):
        header.validate()


def test_types_byteorder_and_facs_defaults() -> None:
    header = AfniHeader.from_text(
        _header(nvals=3, types=None, BRICK_TYPES=None, BYTEORDER_STRING=None)
    )
    assert header.brick_types == (1, 1, 1)
    assert header.byteorder == "="
    assert header.float_facs == (0.0, 0.0, 0.0)
    assert header.view == "orig"
    assert header.taxis is None
    short = AfniHeader.from_text(_header(nvals=3, types=(0, 3)))
    assert short.brick_types == (0, 3, 3)  # The last code is repeated.


# ----------------------------------------------------------------------
#   GEOMETRY
# ----------------------------------------------------------------------

# Every assignment of three distinct DICOM axes, in either direction.
ORIENTS = [
    tuple(2 * row + flip for row, flip in zip(rows, flips))
    for rows in itertools.permutations(range(3))
    for flips in itertools.product((0, 1), repeat=3)
]

# Each orientation code maps to its DICOM axis and to a sign that is +1
# when the axis runs toward L, P or S, as named in the README.
_DIRECTION = {
    0: (0, +1),
    1: (0, -1),
    2: (1, -1),
    3: (1, +1),
    4: (2, +1),
    5: (2, -1),
}


@pytest.mark.parametrize("orient", ORIENTS)
def test_the_cardinal_grid_follows_the_readme(orient) -> None:  # noqa: ANN001
    sizes = (2.0, 3.0, 4.0)
    # DELTA is negative for an axis running toward R, A or I.
    delta = [s * _DIRECTION[o][1] for s, o in zip(sizes, orient)]
    origin = (5.0, -7.0, 11.0)
    matrix = afni_cardinal_matrix(orient, origin, delta)
    for index in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (3, 1, 1)]:
        expected = np.zeros(3)
        for i, o in enumerate(orient):
            axis, _ = _DIRECTION[o]
            # The voxel centre lies at ORIGIN + n * DELTA along the DICOM
            # axis of axis n.
            expected[axis] = origin[i] + index[i] * delta[i]
        assert np.allclose(matrix @ (*index, 1), (*expected, 1))
    back = afni_geometry_from_matrix(matrix)
    assert back[0] == tuple(orient)
    assert np.allclose(back[1], origin)
    assert np.allclose(back[2], delta)


def test_an_oblique_matrix_is_decomposed_as_afni_does() -> None:
    angle = np.deg2rad(10)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    matrix = np.eye(4)
    # x runs L2R (negative DICOM x), y runs I2S and z runs P2A.
    matrix[:3, :3] = rotation @ np.array(
        [[-2.0, 0, 0], [0, 0, -4.0], [0, 3.0, 0]]
    )
    matrix[:3, 3] = (30.0, -40.0, 50.0)
    orient, origin, delta = afni_geometry_from_matrix(matrix)
    assert orient == (1, 4, 2)
    # Sizes and origins are signed by orientation, as in THD_daxes_from_mat44.
    assert np.allclose(delta, (-2.0, 3.0, -4.0))
    columns = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
    projections = columns.T @ matrix[:3, 3]
    assert np.allclose(origin, projections * (-1, 1, -1))


def test_view_names() -> None:
    assert afni_view("anat+tlrc.HEAD") == "tlrc"
    assert afni_view("/data/x+acpc.BRIK.gz") == "acpc"
    assert afni_view("anat+orig") == "orig"
    assert afni_view("mni") == "tlrc"
    assert afni_view("talairach") == "tlrc"
    assert afni_view("orig-cardinal") == "orig"
    assert afni_view("scanner") is None
    assert afni_view("anat.nii.gz") is None
    assert afni_view(None) is None


def test_the_transformations(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path, orient=(1, 2, 4)))
    names = [x.output.name for x in image.transformations]
    assert names == ["physical", "orig-cardinal", "orig"]
    physical = image.transformations[0]
    assert isinstance(physical, Scaling)
    assert np.allclose(physical.scale, (2.0, 3.0, 4.0))
    world = image.transformation.output
    assert type(world).__name__ == "LPSmm"
    expected = np.array(
        [
            [2.0, 0, 0, -10.0],
            [0, 3.0, 0, -20.0],
            [0, 0, 4.0, -30.0],
        ]
    )
    assert np.allclose(image.transformation.matrix, expected)
    assert np.allclose(image.transformations[1].matrix, expected)


def test_an_oblique_dataset(tmp_path) -> None:  # noqa: ANN001
    real = np.array(
        [
            [1.9, 0.6, 0.0, -9.0],
            [-0.6, 2.9, 0.0, -21.0],
            [0.0, 0.0, 4.0, -30.0],
        ]
    )
    path = _source(tmp_path, IJK_TO_DICOM_REAL=tuple(real.ravel()))
    image = AfniImage.load(path)
    assert image.header.is_oblique
    assert np.allclose(image.transformation.matrix, real)
    cardinal = image.transformations[1]
    assert cardinal.output.name == "orig-cardinal"
    assert np.allclose(cardinal.matrix[:, :3], np.diag((2.0, 3.0, 4.0)))


def test_a_header_with_only_the_real_matrix(tmp_path) -> None:  # noqa: ANN001
    real = np.array([[-2.0, 0, 0, 9], [0, 0, 4.0, 8], [0, -3.0, 0, 7]])
    path = _source(
        tmp_path,
        ORIENT_SPECIFIC=None,
        ORIGIN=None,
        DELTA=None,
        IJK_TO_DICOM_REAL=tuple(real.ravel()),
    )
    image = AfniImage.load(path)
    assert image.header.orient == (1, 5, 3)
    assert np.allclose(image.transformation.matrix, real)
    assert np.allclose(image.transformations[1].matrix, real)


# ----------------------------------------------------------------------
#   DATA
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,fmt,dtype",
    [
        (0, "B", np.uint8),
        (1, "h", np.int16),
        (2, "i", np.int32),
        (3, "f", np.float32),
        (4, "d", np.float64),
        (5, "ff", np.complex64),
    ],
)
@pytest.mark.parametrize("order", ["LSB_FIRST", "MSB_FIRST"])
def test_data_types_and_byte_orders(tmp_path, code, fmt, dtype, order) -> None:  # noqa: ANN001
    data = (np.arange(np.prod(SHAPE)).reshape(SHAPE) % 100).astype(dtype)
    if code == 5:
        data = data + 1j * data[::-1]
    prefix = "<" if order == "LSB_FIRST" else ">"
    head = _header(types=(code,), BYTEORDER_STRING=order)
    path = _dataset(tmp_path, head, _brik([data], [prefix + fmt]))
    image = AfniImage.load(path)
    # The values are a view of the file in its byte order.
    assert image.data.dtype == np.dtype(dtype).newbyteorder(prefix)
    assert image.data.shape == SHAPE
    assert np.array_equal(image.data, data)


def test_sub_bricks_of_different_types(tmp_path) -> None:  # noqa: ANN001
    a = (np.arange(24).reshape(SHAPE) % 7).astype(np.int16)
    b = np.linspace(0, 1, 24, dtype=np.float32).reshape(SHAPE)
    head = _header(nvals=2, types=(1, 3))
    path = _dataset(tmp_path, head, _brik([a, b], ["<h", "<f"]))
    image = AfniImage.load(path)
    assert image.data.shape == SHAPE + (2,)
    assert np.array_equal(image.data[..., 0], a)
    assert np.array_equal(image.data[..., 1], b)


def test_brick_scaling_factors(tmp_path) -> None:  # noqa: ANN001
    a = (np.arange(24).reshape(SHAPE)).astype(np.int16)
    head = _header(nvals=3, types=(1,), BRICK_FLOAT_FACS=(0.5, 0.0, 2.0))
    path = _dataset(tmp_path, head, _brik([a, a, a], ["<h"] * 3))
    image = AfniImage.load(path)
    assert image.dataobj.dtype == np.int16
    assert image.data.dtype == np.float32
    # A factor of zero means that the sub-brick is not scaled.
    assert np.allclose(image.data[..., 0], 0.5 * a)
    assert np.allclose(image.data[..., 1], a)
    assert np.allclose(image.data[..., 2], 2.0 * a)


def test_an_uncompressed_brik_is_memory_mapped(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    assert isinstance(image.dataobj.base, np.memmap) or isinstance(
        image.dataobj, np.memmap
    )
    assert image.data is image.dataobj
    eager = AfniImage.load(_source(tmp_path), mmap=False)
    assert not isinstance(eager.dataobj, np.memmap)
    assert np.array_equal(eager.data, DATA)


@pytest.mark.parametrize("suffix", [".gz", ".bz2"])
def test_a_compressed_brik(tmp_path, suffix) -> None:  # noqa: ANN001
    path = _dataset(tmp_path, _header(), _brik([DATA], ["<f"]), suffix=suffix)
    for name in (path, f"{tmp_path}/dset+orig.BRIK{suffix}"):
        assert np.array_equal(AfniImage.load(name).data, DATA)


def test_an_unsupported_compression_is_reported(tmp_path) -> None:  # noqa: ANN001
    path = _dataset(tmp_path, _header(), b"\x1f\x9d...", suffix=".Z")
    with pytest.raises(ParserContentError, match="compressed"):
        AfniImage.load(path)


def test_a_missing_or_short_brik(tmp_path) -> None:  # noqa: ANN001
    path = _dataset(tmp_path, _header(), None)
    with pytest.raises(ParserExistsError):
        AfniImage.load(path)
    short = _dataset(tmp_path, _header(), b"\x00" * 10, name="short+orig")
    with pytest.raises(ParserContentError):
        AfniImage.load(short)


def test_a_time_series(tmp_path) -> None:  # noqa: ANN001
    head = _header(
        nvals=3,
        TAXIS_NUMS=(3, 0, 77001),
        TAXIS_FLOATS=(0.0, 2500.0, 0.0, 0.0, 0.0),
    )
    path = _dataset(tmp_path, head, _brik([DATA] * 3, ["<f"] * 3))
    image = AfniImage.load(path)
    assert image.data.shape == SHAPE + (3,)
    axes = image.system.axes
    assert axes[3].name == "t" and axes[3].type == "time"
    physical = image.transformations[0]
    assert np.allclose(physical.scale, (2.0, 3.0, 4.0, 2500.0))
    assert str(physical.output.axes[3].unit) == "millisecond"


def test_sub_bricks_that_are_not_a_time_series(tmp_path) -> None:  # noqa: ANN001
    path = _dataset(tmp_path, _header(nvals=2), _brik([DATA] * 2, ["<f"] * 2))
    image = AfniImage.load(path)
    assert image.system.axes[3].name == "brick"
    assert image.system.axes[3].type is None


# ----------------------------------------------------------------------
#   FILES AND DISPATCH
# ----------------------------------------------------------------------


def test_dataset_file_names(tmp_path) -> None:  # noqa: ANN001
    _dataset(tmp_path, _header(), _brik([DATA], ["<f"]), suffix=".gz")
    for name in ("dset+orig.HEAD", "dset+orig.BRIK.gz", "dset+orig"):
        head, brik, stem = afni_dataset_files(str(tmp_path / name))
        assert head.name == "dset+orig.HEAD"
        assert brik.name == "dset+orig.BRIK.gz"
        assert stem == "dset+orig"
    head, brik, _ = afni_dataset_files(str(tmp_path / "low+orig.head"))
    assert (head.name, brik.name) == ("low+orig.head", "low+orig.brik")


@pytest.mark.parametrize(
    "name", ["dset+orig.HEAD", "dset+orig.BRIK", "dset+orig"]
)
def test_every_name_of_a_dataset_loads(tmp_path, name) -> None:  # noqa: ANN001
    _source(tmp_path)
    assert np.array_equal(io.load(tmp_path / name, hint="afni").data, DATA)
    assert np.array_equal(AfniImage.load(tmp_path / name).data, DATA)


def test_dispatch_by_extension_and_hints(tmp_path) -> None:  # noqa: ANN001
    path = _source(tmp_path)
    assert isinstance(io.load(path), AfniImage)
    assert isinstance(
        io.images.load(str(tmp_path / "dset+orig.BRIK")), AfniImage
    )
    for hint in ("afni", "brik", "afni.brik"):
        assert isinstance(io.load(path, hint=hint), AfniImage)
    assert {"afni", "brik", "afni.brik"} <= set(format_hints(AfniImage))


def test_the_format_is_registered() -> None:
    assert AfniImage in FileBasedObject._REGISTRY
    assert AfniImage in FileBasedImage._REGISTRY
    assert issubclass(AfniImage, FileWriter)


def test_sniffing(tmp_path) -> None:  # noqa: ANN001
    path = _source(tmp_path)
    assert AfniImage.sniff(path) == Confidence.LIKELY
    assert AfniImage.sniff(tmp_path / "dset+orig.BRIK") == Confidence.LIKELY
    assert AfniImage.sniff(path.read_bytes()) == Confidence.LIKELY
    three = _dataset(
        tmp_path, _header(nvals=3), _brik([DATA] * 3, ["<f"] * 3), "w+tlrc"
    )
    assert AfniImage.sniff(three) == Confidence.WEAK
    assert AfniImage.sniff(b"\x00" * 400) == Confidence.NO
    assert AfniImage.sniff(b"type = nonsense") == Confidence.NO
    assert AfniImage.sniff(tmp_path / "missing+orig.HEAD") == Confidence.NO


def test_streams_and_bytes(tmp_path) -> None:  # noqa: ANN001
    path = _source(tmp_path)
    with open(path, "rb") as f:
        assert np.array_equal(AfniImage.load(f).data, DATA)
    with pytest.raises(ParserError):
        AfniImage.load(_io.BytesIO(path.read_bytes()))
    with pytest.raises(ParserError):
        AfniImage.from_bytes(path.read_bytes())


# ----------------------------------------------------------------------
#   NIBABEL'S TEST DATA
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["example4d+orig.HEAD", "example4d+orig.BRIK.gz", "scaled+tlrc.HEAD"],
)
def test_nibabel_reads_the_same(name) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    path = _nibabel_data(name)
    ref = nb.load(path)
    image = io.load(path)
    assert isinstance(image, AfniImage)
    assert np.allclose(np.squeeze(ref.get_fdata()), image.data)
    ras = DICOM_TO_RAS @ image.transformation.homogeneous_matrix
    assert np.allclose(ras, ref.affine)
    view = "tlrc" if "tlrc" in name else "orig"
    assert image.transformation.output.name == view


def test_nibabel_example_time_series() -> None:
    image = io.load(_nibabel_data("example4d+orig.HEAD"))
    assert image.data.shape == (33, 41, 25, 3)
    assert image.system.axes[3].type == "time"
    assert image.header.labels == ["#0", "#1", "#2"]
    assert np.allclose(image.transformations[0].scale, (3, 3, 3, 3))


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_an_image_round_trips(tmp_path) -> None:  # noqa: ANN001
    path = _source(
        tmp_path,
        orient=(1, 5, 2),
        delta=(-2.0, -3.0, -4.0),
        HISTORY_NOTE="hand made",
        BRICK_LABS="volume",
        IDCODE_STRING="XYZ_original",
    )
    image = AfniImage.load(path)
    out = tmp_path / "out+orig.HEAD"
    image.save(out)
    back = AfniImage.load(out)
    assert np.array_equal(back.data, DATA)
    assert back.header.orient == (1, 5, 2)
    assert np.allclose(back.header.origin, image.header.origin)
    assert np.allclose(back.header.delta, image.header.delta)
    assert np.allclose(back.transformation.matrix, image.transformation.matrix)
    assert back.header["HISTORY_NOTE"] == "hand made"
    assert back.header.labels == ["volume"]
    assert back.header["IDCODE_STRING"] != "XYZ_original"
    assert back.header.view == "orig"
    assert back.header.taxis is None
    assert back.header["BRICK_STATS"] == (float(DATA.min()), float(DATA.max()))
    assert (tmp_path / "out+orig.BRIK").exists()


def test_a_written_dataset_is_read_by_nibabel(tmp_path) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    image = AfniImage.load(_source(tmp_path, orient=(1, 2, 4)))
    image.save(tmp_path / "nib+orig.BRIK.gz")
    ref = nb.load(str(tmp_path / "nib+orig.HEAD"))
    assert np.allclose(np.squeeze(ref.get_fdata()), DATA)
    assert np.allclose(
        ref.affine, DICOM_TO_RAS @ image.transformation.homogeneous_matrix
    )


def test_a_written_header_starts_as_afni_writes_it(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    image.save(tmp_path / "out+orig.HEAD")
    text = (tmp_path / "out+orig.HEAD").read_text()
    assert text.startswith(
        "\ntype = string-attribute\nname = TYPESTRING\ncount = 15\n"
        "'3DIM_HEAD_ANAT~\n"
    )
    for name in (
        "SCENE_DATA",
        "ORIENT_SPECIFIC",
        "ORIGIN",
        "DELTA",
        "IJK_TO_DICOM",
        "IJK_TO_DICOM_REAL",
        "DATASET_RANK",
        "DATASET_DIMENSIONS",
        "BRICK_TYPES",
        "BRICK_STATS",
        "BRICK_FLOAT_FACS",
        "BYTEORDER_STRING",
        "IDCODE_STRING",
    ):
        assert f"name = {name}\n" in text


def test_writing_compresses_and_replaces_other_briks(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    (tmp_path / "out+orig.BRIK").write_bytes(b"stale")
    image.save(tmp_path / "out+orig.BRIK.gz")
    assert not (tmp_path / "out+orig.BRIK").exists()
    raw = gzip.decompress((tmp_path / "out+orig.BRIK.gz").read_bytes())
    assert raw == _brik([DATA], ["<f"])
    assert np.array_equal(AfniImage.load(tmp_path / "out+orig").data, DATA)
    image.save(tmp_path / "out+orig.BRIK.bz2")
    assert not (tmp_path / "out+orig.BRIK.gz").exists()
    assert np.array_equal(AfniImage.load(tmp_path / "out+orig").data, DATA)


def test_the_view_written(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    image.save(tmp_path / "a+tlrc.HEAD")
    assert AfniImage.load(tmp_path / "a+tlrc.HEAD").header.view == "tlrc"
    image.save(tmp_path / "b.HEAD", view="acpc")
    assert AfniImage.load(tmp_path / "b.HEAD").header.view == "acpc"
    image.save(tmp_path / "c.HEAD")
    assert AfniImage.load(tmp_path / "c.HEAD").header.view == "orig"
    with pytest.raises(WriterError):
        image.save(tmp_path / "d.HEAD", view="mni")


def test_the_view_of_a_template_world(tmp_path) -> None:  # noqa: ANN001
    voxel = CoordinateSystem(
        name="voxel", axes=[Axis(n, "space") for n in "xyz"], order="F"
    )
    world = io.load(_source(tmp_path)).transformation.output
    xform = Affine(
        input=voxel,
        output=CoordinateSystem(name="mni", axes=world.axes),
        matrix=np.eye(3, 4),
    )
    image = SingleScaleImage(data=DATA, transformations=[xform])
    io.save(image, tmp_path / "t.HEAD")
    assert AfniImage.load(tmp_path / "t.HEAD").header.view == "tlrc"


def test_the_stored_type(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    image.save(tmp_path / "s+orig.HEAD", datatype="short")
    back = AfniImage.load(tmp_path / "s+orig.HEAD")
    assert back.dataobj.dtype == np.int16
    assert np.array_equal(back.data, np.round(DATA))
    big = SingleScaleImage(data=DATA * 1e6, transformations=[])
    with pytest.raises(WriterError):
        io.save(big, tmp_path / "big+orig.HEAD", datatype="short")


@pytest.mark.parametrize(
    "dtype,stored",
    [
        (bool, np.uint8),
        (np.int8, np.int16),
        (np.uint16, np.int32),
        (np.int64, np.float64),
        (np.float16, np.float32),
        (np.complex128, np.complex64),
        (np.float64, np.float64),
    ],
)
def test_types_afni_has_not(dtype, stored) -> None:  # noqa: ANN001
    assert brick_dtype(dtype) == stored


def test_attributes_can_be_changed_when_writing(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path, HISTORY_NOTE="old"))
    image.save(
        tmp_path / "k+orig.HEAD",
        attributes={"HISTORY_NOTE": None, "DATASET_NAME": "mine"},
    )
    back = AfniImage.load(tmp_path / "k+orig.HEAD")
    assert "HISTORY_NOTE" not in back.header
    assert back.header["DATASET_NAME"] == "mine"


def test_per_brick_attributes_are_dropped_with_the_bricks(tmp_path) -> None:  # noqa: ANN001
    head = _header(
        nvals=2,
        BRICK_LABS="a\0b",
        TAXIS_NUMS=(2, 0, 77002),
        TAXIS_FLOATS=(0.0, 2.0, 0.0, 0.0, 0.0),
    )
    path = _dataset(tmp_path, head, _brik([DATA] * 2, ["<f"] * 2))
    image = AfniImage.load(path)
    image.save(tmp_path / "same+orig.HEAD")
    same = AfniImage.load(tmp_path / "same+orig.HEAD")
    assert same.header.labels == ["a", "b"]
    assert same.header.taxis == (2.0, "second")
    image.data = np.asarray(image.data)[..., :1]
    image.save(tmp_path / "one+orig.HEAD")
    one = AfniImage.load(tmp_path / "one+orig.HEAD")
    assert one.header.nvals == 1
    assert one.header.labels is None
    assert one.header.taxis is None


def test_an_oblique_dataset_round_trips(tmp_path) -> None:  # noqa: ANN001
    real = np.array(
        [
            [1.9, 0.6, 0.0, -9.0],
            [-0.6, 2.9, 0.0, -21.0],
            [0.0, 0.0, 4.0, -30.0],
        ]
    )
    path = _source(tmp_path, IJK_TO_DICOM_REAL=tuple(real.ravel()))
    image = AfniImage.load(path)
    image.save(tmp_path / "obl+orig.HEAD")
    back = AfniImage.load(tmp_path / "obl+orig.HEAD")
    assert np.allclose(back.transformation.matrix, real)
    # The cardinal grid of the source is kept as it was.
    assert back.header.origin == image.header.origin
    assert back.header.delta == image.header.delta


def test_a_time_series_from_another_format(tmp_path) -> None:  # noqa: ANN001
    voxel = CoordinateSystem(
        name="voxel",
        axes=[Axis(n, "space") for n in "xyz"] + [Axis("t", "time")],
        order="F",
    )
    physical = CoordinateSystem(
        name="physical",
        axes=[Axis(n, "space", unit="mm") for n in "xyz"]
        + [Axis("t", "time", unit="millisecond")],
    )
    scaling = Scaling(input=voxel, output=physical, scale=(1, 1, 1, 800))
    data = np.stack([DATA] * 4, axis=-1)
    image = SingleScaleImage(data=data, transformations=[scaling])
    io.save(image, tmp_path / "ts+orig.HEAD")
    back = AfniImage.load(tmp_path / "ts+orig.HEAD")
    assert back.header.taxis == (800.0, "millisecond")
    assert back.header["TAXIS_NUMS"][:3] == (4, 0, 77001)
    assert back.system.axes[3].type == "time"
    assert np.array_equal(back.data, data)


def test_an_image_round_trips_through_nifti(tmp_path) -> None:  # noqa: ANN001
    nb = pytest.importorskip("nibabel")
    image = AfniImage.load(
        _source(tmp_path, orient=(1, 5, 2), delta=(-2.0, -3.0, -4.0))
    )
    nifti = tmp_path / "out.nii"
    io.save(SingleScaleImage.from_instance(image), nifti)
    assert np.allclose(
        nb.load(str(nifti)).affine,
        DICOM_TO_RAS @ image.transformation.homogeneous_matrix,
    )
    back = tmp_path / "back+orig.HEAD"
    io.save(SingleScaleImage.from_instance(io.images.load(nifti)), back)
    reloaded = AfniImage.load(back)
    assert np.array_equal(np.asarray(reloaded.data), DATA)
    assert reloaded.header.orient == (1, 5, 2)
    assert np.allclose(
        reloaded.transformation.matrix, image.transformation.matrix
    )


def test_shapes_afni_can_and_cannot_store(tmp_path) -> None:  # noqa: ANN001
    flat = SingleScaleImage(
        data=np.ones((3, 2), np.float32), transformations=[]
    )
    io.save(flat, tmp_path / "flat+orig.HEAD")
    assert AfniImage.load(tmp_path / "flat+orig.HEAD").data.shape == (3, 2, 1)
    five = SingleScaleImage(data=np.ones((2,) * 5), transformations=[])
    with pytest.raises(WriterError):
        io.save(five, tmp_path / "five+orig.HEAD")


def test_a_dataset_cannot_be_written_to_a_stream(tmp_path) -> None:  # noqa: ANN001
    image = AfniImage.load(_source(tmp_path))
    with pytest.raises(WriterError):
        image.to_bytes()
    with pytest.raises(WriterError):
        image.save(_io.BytesIO())
