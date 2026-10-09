"""Tests for reading and writing FreeSurfer LTA files.

LTA is a registered text format, found by extension or by content and
written by `io.save`. Every supported file type is round-tripped, as are
the views that read any file as a voxel-to-voxel, physical-to-physical or
RAS-to-RAS affine.
"""

import io as _io
import warnings
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops.datamodel import systems as _systems
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import (
    Confidence,
    FileWriter,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations import TransformationFormat
from brainhops.io.transformations.freesurfer.lta import (
    LtaPhysicalSystem,
    LtaStruct,
    LtaTransformation,
    LtaTransformationPhysToPhys,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
    LtaType,
    LtaValidity,
    LtaVoxelSystem,
)

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# Entries with many decimals, so that a lossy round trip shows.
MATRIX = (
    (0.9876543210123456, 0.1, -0.05, 1.25),
    (-0.1, 0.99, 0.2, -2.5),
    (0.05, -0.2, 1.01, 0.3333333333333333),
    (0.0, 0.0, 0.0, 1.0),
)

SRC = LtaStruct.SrcVolumeInfo(
    valid=LtaValidity.VOLUME_INFO_VALID,
    filename="src.nii.gz",
    volume=(10, 12, 14),
    voxelsize=(1.0, 2.0, 3.0),
    xras=(-1.0, 0.0, 0.0),
    yras=(0.0, 0.0, 1.0),
    zras=(0.0, 1.0, 0.0),
    cras=(1.0, 2.0, 3.0),
)
DST = LtaStruct.DstVolumeInfo(
    valid=LtaValidity.VOLUME_INFO_VALID,
    filename="dst.nii.gz",
    volume=(8, 9, 10),
    voxelsize=(2.0, 2.0, 2.0),
    cras=(-4.0, 5.0, 6.0),
)

# What FreeSurfer writes, including the lines after the volumes.
FREESURFER = """\
# transform file /subjects/bert/mri/transforms/reg.lta
# created by bert on Mon Jan  1 00:00:00 2024

type      = 1 # LINEAR_RAS_TO_RAS
nxforms   = 1
mean      = 0.0000 0.0000 0.0000
sigma     = 1.0000
1 4 4
+1.000000  +0.000000  +0.000000  +2.000000
+0.000000  +1.000000  +0.000000  -3.000000
+0.000000  +0.000000  +1.000000  +4.000000
+0.000000  +0.000000  +0.000000  +1.000000
src volume info
valid = 1  # volume info valid
filename = /subjects/bert/mri/orig.mgz
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = 5.000000000000000e+00 -1.000000000000000e+01 2.000000000000000e+01
dst volume info
valid = 1  # volume info valid
filename = /subjects/bert/mri/T1.mgz
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = 5.000000000000000e+00 -1.000000000000000e+01 2.000000000000000e+01
subject bert
fscale 0.100000
"""

# Every supported LTA type, with the systems it reads.
TYPES = [
    (LtaType.LINEAR_VOX_TO_VOX, LtaVoxelSystem),
    (LtaType.LINEAR_RAS_TO_RAS, _systems.RASmm),
    (LtaType.LINEAR_PHYSVOX_TO_PHYSVOX, LtaPhysicalSystem),
    (LtaType.LINEAR_RSA_TO_RSA, _systems.RSAmm),
]
TYPE_IDS = ["vox", "ras", "physvox", "rsa"]

# Each view, with the type written once matrix and systems are set.
VIEWS = [
    (LtaTransformationVoxToVox, LtaType.LINEAR_VOX_TO_VOX),
    (LtaTransformationPhysToPhys, LtaType.LINEAR_PHYSVOX_TO_PHYSVOX),
    (LtaTransformationRASToRAS, LtaType.LINEAR_RAS_TO_RAS),
]
VIEW_IDS = ["vox", "phys", "ras"]


def _struct(lta_type: LtaType = LtaType.LINEAR_RAS_TO_RAS) -> LtaStruct:
    return LtaStruct(
        type=lta_type,
        affine=LtaStruct.Affine(matrix=MATRIX),
        src=SRC,
        dst=DST,
    )


def _write(tmp_path: Path, struct: LtaStruct, name: str = "x.lta") -> Path:
    file = tmp_path / name
    struct.save(file)
    return file


# ----------------------------------------------------------------------
#   THE STRUCT
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lta_type", [t for t, _ in TYPES], ids=TYPE_IDS)
def test_a_struct_round_trips_through_text(lta_type: LtaType) -> None:
    # Regression: to_text and from_text raised AttributeError.
    struct = _struct(lta_type)
    text = struct.to_text()
    assert text.startswith("type")
    assert text.endswith("\n")
    assert LtaStruct.from_text(text) == struct


def test_a_struct_without_volumes_round_trips() -> None:
    struct = LtaStruct(
        type=LtaType.LINEAR_RAS_TO_RAS,
        affine=LtaStruct.Affine(matrix=MATRIX),
    )
    back = LtaStruct.from_text(struct.to_text())
    assert back == struct
    assert back.src is None and back.dst is None


def test_the_matrix_is_written_at_full_precision() -> None:
    back = LtaStruct.from_text(_struct().to_text())
    assert back.affine.matrix == MATRIX


def test_a_struct_follows_the_parser_contract(tmp_path) -> None:  # noqa: ANN001
    struct = _struct()
    assert LtaStruct.from_bytes(struct.to_bytes()) == struct
    data = struct.to_bytes(encoding="latin-1")
    assert LtaStruct.from_bytes(data, encoding="latin-1") == struct
    buffer = _io.StringIO()
    struct.to_fileobj(buffer)
    buffer.seek(0)
    assert LtaStruct.load(buffer) == struct
    struct.save(tmp_path / "x.lta")
    assert LtaStruct.load(tmp_path / "x.lta") == struct
    assert LtaStruct.load(str(tmp_path / "x.lta")) == struct
    assert LtaStruct.from_lines(struct.to_lines()) == struct


def test_a_freesurfer_file_is_read() -> None:
    # Comments, a commented type, a commented valid and trailing lines.
    struct = LtaStruct.from_text(FREESURFER)
    assert struct.type is LtaType.LINEAR_RAS_TO_RAS
    assert struct.sigma == 1.0
    assert struct.affine.matrix[0] == (1.0, 0.0, 0.0, 2.0)
    assert struct.src.filename == "/subjects/bert/mri/orig.mgz"
    assert struct.src.volume == (256, 256, 256)
    assert struct.dst.cras == (5.0, -10.0, 20.0)


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LtaStruct, LtaTransformation])
def test_lta_content_is_sniffed(cls: type) -> None:
    text = _struct().to_text()
    assert cls.sniff_text(text) == Confidence.LIKELY
    assert cls.sniff_bytes(text.encode()) == Confidence.LIKELY
    assert cls.sniff_lines(text.splitlines()) == Confidence.LIKELY
    assert cls.sniff_fileobj(_io.StringIO(text)) == Confidence.LIKELY
    assert cls.sniff_text(FREESURFER) == Confidence.LIKELY


@pytest.mark.parametrize("cls", [LtaStruct, LtaTransformation])
def test_other_content_is_not_sniffed(cls: type) -> None:
    assert cls.sniff_text("1 0 0 0\n0 1 0 0\n") == Confidence.NO
    assert cls.sniff_text("") == Confidence.NO
    with pytest.raises(SnifferContentError):
        cls.sniff_text("1 0 0 0\n", error=True)


@pytest.mark.parametrize("cls", [LtaStruct, LtaTransformation])
def test_binary_content_is_not_sniffed(cls: type, tmp_path) -> None:  # noqa: ANN001
    # A binary file named .lta is not sniffed, rather than failing to decode.
    binary = b"\x00\x01type = 1\x9a\xff"
    file = tmp_path / "binary.lta"
    file.write_bytes(binary)
    assert cls.sniff_bytes(binary) == Confidence.NO
    assert cls.sniff(file) == Confidence.NO
    with open(file, "rb") as f:
        assert cls.sniff(f) == Confidence.NO
    with pytest.raises(SnifferContentError):
        cls.sniff(file, error=True)


def test_a_file_is_sniffed_by_its_content_alone(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path, _struct(), "transform.txt")
    assert LtaTransformation.sniff(file) == Confidence.LIKELY
    assert io.sniff(file) is LtaTransformation
    assert type(io.load(file)) is LtaTransformation


def test_content_given_as_a_str_is_a_path(tmp_path) -> None:  # noqa: ANN001
    # Regression: text in place of a path raised "File name too long".
    text = _struct().to_text()
    assert LtaStruct.sniff(text) == Confidence.NO
    assert LtaTransformation.sniff(text) == Confidence.NO
    with pytest.raises(FileNotFoundError):
        LtaStruct.sniff(text, error=True)
    with pytest.raises(FileNotFoundError):
        LtaStruct.from_filename(text)
    assert LtaStruct.sniff(tmp_path / "missing.lta") == Confidence.NO


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


def test_lta_is_a_registered_writable_format() -> None:
    from brainhops.io import Format

    assert LtaTransformation in Format._REGISTRY
    assert LtaTransformation in TransformationFormat._REGISTRY
    assert issubclass(LtaTransformation, FileWriter)
    assert LtaTransformation.EXTENSIONS == (".lta",)
    # The views read the same files, so registering them would be ambiguous.
    for view, _ in VIEWS:
        assert view not in Format._REGISTRY


@pytest.mark.parametrize(
    "load",
    [
        io.load,
        io.transformations.load,
        TransformationFormat.from_any,
        LtaTransformation.from_any,
        LtaTransformation.load,
    ],
    ids=["io.load", "transformations.load", "from_any", "own", "own.load"],
)
@pytest.mark.parametrize("lta_type, system", TYPES, ids=TYPE_IDS)
def test_every_type_is_read_through_dispatch(
    tmp_path,  # noqa: ANN001
    load: tx.Callable,
    lta_type: LtaType,
    system: type,
) -> None:
    file = _write(tmp_path, _struct(lta_type))
    xform = load(file)
    assert type(xform) is LtaTransformation
    assert xform.struct == _struct(lta_type)
    assert isinstance(xform.input, system)
    assert isinstance(xform.output, system)
    np.testing.assert_array_equal(xform.matrix, np.asarray(MATRIX)[:-1])
    # A path given as a str is read the same way.
    assert load(str(file)).struct == xform.struct


HINTS = [
    "lta",
    "freesurfer",
    "affine",
    "freesurfer.lta",
    "affine.lta",
    "xform.freesurfer.lta",
]


def test_lta_declares_its_hints() -> None:
    from brainhops.io.base.specs import format_hints

    # The family, format and affine hints compose into dotted hints.
    assert format_hints(LtaTransformation) >= set(HINTS)


@pytest.mark.parametrize("hint", HINTS)
def test_a_structured_source_is_read_with_a_hint(tmp_path, hint: str) -> None:  # noqa: ANN001
    from brainhops.io.base.specs import SourceSpec

    file = _write(tmp_path, _struct(), "transform.txt")
    xform = io.load(SourceSpec(path=str(file), hints=(hint,)))
    assert type(xform) is LtaTransformation


@pytest.mark.parametrize("hint", HINTS)
def test_a_hint_selects_the_lta_reader(tmp_path, hint: str) -> None:  # noqa: ANN001
    file = _write(tmp_path, _struct(), "transform.txt")
    assert io.sniff(file, hint=hint) is LtaTransformation
    assert type(io.load(file, hint=hint)) is LtaTransformation
    assert type(io.transformations.load(file, hint=hint)) is (
        LtaTransformation
    )


def test_a_freesurfer_file_is_read_through_dispatch(tmp_path) -> None:  # noqa: ANN001
    file = tmp_path / "reg.lta"
    file.write_text(FREESURFER)
    xform = io.load(file)
    assert type(xform) is LtaTransformation
    assert isinstance(xform.input, _systems.RASmm)
    np.testing.assert_array_equal(xform.matrix[:, -1], [2.0, -3.0, 4.0])


# ----------------------------------------------------------------------
#   ROUND TRIPS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lta_type, system", TYPES, ids=TYPE_IDS)
def test_every_type_round_trips_through_save(
    tmp_path,  # noqa: ANN001
    lta_type: LtaType,
    system: type,
) -> None:
    first = _write(tmp_path, _struct(lta_type), "first.lta")
    xform = io.load(first)
    io.save(xform, tmp_path / "second.lta")
    # An untouched transformation is written as the struct it was read from.
    assert (tmp_path / "second.lta").read_text() == first.read_text()
    back = io.load(tmp_path / "second.lta")
    assert back.struct == xform.struct
    assert isinstance(back.input, system)
    np.testing.assert_array_equal(back.matrix, xform.matrix)


@pytest.mark.parametrize("lta_type, system", TYPES, ids=TYPE_IDS)
def test_every_type_round_trips_once_rebuilt(
    tmp_path,  # noqa: ANN001
    lta_type: LtaType,
    system: type,
) -> None:
    # Setting matrix and systems makes the writer build a new struct.
    xform = io.load(_write(tmp_path, _struct(lta_type)))
    rebuilt = LtaTransformation(
        matrix=xform.matrix, input=xform.input, output=xform.output
    )
    assert rebuilt.to_struct() is not rebuilt.struct
    io.save(rebuilt, tmp_path / "rebuilt.lta")
    back = io.load(tmp_path / "rebuilt.lta")
    assert back.struct.type is lta_type
    np.testing.assert_array_equal(back.matrix, xform.matrix)
    assert isinstance(back.input, system)
    if system in (LtaVoxelSystem, LtaPhysicalSystem):
        # The geometry is that of the systems.
        assert back.struct.src == SRC
        assert back.struct.dst == DST
        assert back.input == xform.input
        assert back.output == xform.output


@pytest.mark.parametrize("view, written", VIEWS, ids=VIEW_IDS)
@pytest.mark.parametrize("lta_type", [t for t, _ in TYPES], ids=TYPE_IDS)
def test_a_view_round_trips(
    tmp_path,  # noqa: ANN001
    view: type,
    written: LtaType,
    lta_type: LtaType,
) -> None:
    file = _write(tmp_path, _struct(lta_type))
    xform = view.load(file)
    assert type(xform) is view
    # An untouched view is written as read.
    io.save(xform, tmp_path / "same.lta")
    assert (tmp_path / "same.lta").read_text() == file.read_text()
    np.testing.assert_allclose(
        view.load(tmp_path / "same.lta").matrix, xform.matrix
    )
    # A rebuilt view is written in its own type with the same meaning.
    rebuilt = view(
        struct=xform.struct,
        matrix=xform.matrix,
        input=xform.input,
        output=xform.output,
    )
    io.save(rebuilt, tmp_path / "rebuilt.lta")
    back = io.load(tmp_path / "rebuilt.lta")
    assert back.struct.type is written
    np.testing.assert_allclose(back.matrix, xform.matrix, atol=1e-12)
    np.testing.assert_allclose(
        LtaTransformationRASToRAS.load(tmp_path / "rebuilt.lta").matrix,
        LtaTransformationRASToRAS.load(file).matrix,
        atol=1e-12,
    )


def test_the_ras_view_of_a_ras_file_is_its_matrix(tmp_path) -> None:  # noqa: ANN001
    # Regression: the RAS-to-RAS view computed a physical-to-physical matrix.
    file = _write(tmp_path, _struct(LtaType.LINEAR_RAS_TO_RAS))
    xform = LtaTransformationRASToRAS.load(file)
    np.testing.assert_allclose(xform.matrix, np.asarray(MATRIX)[:-1])


def test_the_ras_view_of_an_rsa_file_reorders_its_axes(tmp_path) -> None:  # noqa: ANN001
    # Regression: the axes are R, S, A, so RSA (1, 2, 3) is RAS (1, 3, 2).
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    struct = LtaStruct(
        type=LtaType.LINEAR_RSA_TO_RSA,
        affine=LtaStruct.Affine(matrix=tuple(map(tuple, matrix))),
        src=SRC,
        dst=DST,
    )
    xform = LtaTransformationRASToRAS.load(_write(tmp_path, struct))
    np.testing.assert_allclose(xform.matrix[:, 3], [1.0, 3.0, 2.0])
    np.testing.assert_allclose(xform.matrix[:, :3], np.eye(3))


def test_a_transformation_follows_the_parser_contract(tmp_path) -> None:  # noqa: ANN001
    xform = LtaTransformation.from_struct(_struct())
    text = _struct().to_text()
    assert xform.to_text() == text
    assert xform.to_bytes() == text.encode()
    assert list(xform.to_lines()) == list(_struct().to_lines())
    buffer = _io.StringIO()
    xform.to_fileobj(buffer)
    assert buffer.getvalue() == text
    xform.save(tmp_path / "x.lta")
    assert (tmp_path / "x.lta").read_text() == text
    assert LtaTransformation.from_bytes(text.encode()).struct == _struct()
    assert LtaTransformation.from_text(text).struct == _struct()


def test_a_transformation_is_saved_to_an_unnamed_file() -> None:
    buffer = _io.StringIO()
    io.save(LtaTransformation.from_struct(_struct()), buffer)
    assert buffer.getvalue() == _struct().to_text()


def test_keyword_options_override_the_file(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path, _struct())
    matrix = np.eye(4)[:-1]
    xform = io.load(file, matrix=matrix)
    np.testing.assert_array_equal(xform.matrix, matrix)


# ----------------------------------------------------------------------
#   SAVING OTHER AFFINES
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "system, lta_type",
    [
        (_systems.RASmm, LtaType.LINEAR_RAS_TO_RAS),
        (_systems.RSAmm, LtaType.LINEAR_RSA_TO_RSA),
    ],
    ids=["ras", "rsa"],
)
def test_an_anatomical_affine_is_saved_as_lta(
    tmp_path,  # noqa: ANN001
    system: type,
    lta_type: LtaType,
) -> None:
    matrix = np.asarray(MATRIX)[:-1]
    affine = Affine(matrix, input=system(), output=system())
    io.save(affine, tmp_path / "x.lta")
    back = io.load(tmp_path / "x.lta")
    assert back.struct.type is lta_type
    np.testing.assert_array_equal(back.matrix, matrix)
    assert isinstance(back.input, system)
    # FreeSurfer writes the geometry blocks even when they are unknown.
    assert back.struct.src.valid == 0
    assert back.struct.dst.valid == 0


def test_an_affine_between_lta_voxel_systems_is_saved_as_lta(tmp_path) -> None:  # noqa: ANN001
    src = LtaVoxelSystem.from_struct(SRC)
    dst = LtaVoxelSystem.from_struct(DST)
    affine = Affine(np.asarray(MATRIX)[:-1], input=src, output=dst)
    io.save(affine, tmp_path / "x.lta")
    back = io.load(tmp_path / "x.lta")
    assert back.struct.type is LtaType.LINEAR_VOX_TO_VOX
    assert (back.struct.src, back.struct.dst) == (SRC, DST)


@pytest.mark.parametrize(
    "affine",
    [
        Affine(np.asarray(MATRIX)[:-1]),
        Affine(
            np.asarray(MATRIX)[:-1],
            input=_systems.RASmm(),
            output=_systems.LPSmm(),
        ),
        Affine(
            np.asarray(MATRIX)[:-1],
            input=_systems.VoxelCoordinateSystem(),
            output=_systems.VoxelCoordinateSystem(),
        ),
    ],
    ids=["unspecified", "ras-to-lps", "voxel"],
)
def test_an_affine_lta_cannot_encode_is_refused(
    tmp_path,  # noqa: ANN001
    affine: Affine,
) -> None:
    # An affine without LTA systems is refused, and no file is created.
    with pytest.raises(UnrepresentableTransformationError):
        io.save(affine, tmp_path / "x.lta")
    assert not (tmp_path / "x.lta").exists()
    assert issubclass(UnrepresentableTransformationError, WriterError)


def test_a_non_3d_affine_is_refused(tmp_path) -> None:  # noqa: ANN001
    affine = Affine(
        np.eye(3)[:-1], input=_systems.RASmm(), output=_systems.RASmm()
    )
    with pytest.raises(UnrepresentableTransformationError, match="shape"):
        io.save(affine, tmp_path / "x.lta")


# ----------------------------------------------------------------------
#   DEPRECATED
# ----------------------------------------------------------------------


def test_from_is_deprecated_but_still_reads(tmp_path) -> None:  # noqa: ANN001
    struct = _struct()
    text = struct.to_text()
    file = _write(tmp_path, struct)
    with pytest.deprecated_call():
        # Regression: text in place of a path raised "File name too long".
        assert LtaStruct.from_(text) == struct
    with pytest.deprecated_call():
        assert LtaStruct.from_(str(file)) == struct
    with pytest.deprecated_call():
        assert LtaStruct.from_(file) == struct
    with pytest.deprecated_call():
        assert LtaStruct.from_(text.encode()) == struct
    with pytest.deprecated_call():
        assert LtaStruct.from_(text.splitlines()) == struct
    with pytest.deprecated_call():
        assert LtaTransformation.from_(struct).struct is struct
    with pytest.deprecated_call():
        assert LtaTransformation.from_(text).struct == struct


def test_from_warns_once_in_terms_of_its_own_class() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LtaTransformation.from_(_struct().to_text())
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert messages[0].startswith("LtaTransformation.from_()")
