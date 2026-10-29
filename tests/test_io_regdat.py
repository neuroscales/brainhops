"""Tests for reading and writing FreeSurfer `register.dat` files.

A `register.dat` file is a registered text format, found by its content
and written by `io.save`. Its matrix maps the tkregister RAS space of the
reference volume to that of the movable volume. The file holds no
geometry, so the geometries of the volumes are given to the
transformation, and the scanner RAS affine is checked against the LTA
files that FreeSurfer builds from the same matrix and the same
geometries. The exact conversions between `register.dat` and LTA, and
the LTA files of type `REGISTER_DAT`, are checked against the formulas of
FreeSurfer (`utils/transform.cpp` and `utils/mri.cpp`, commit 4f8728e).
"""

import io as _io
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import replace

import brainhops.io as io
from brainhops.datamodel import systems as _systems
from brainhops.datamodel.geometry import Geometry
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, DisplacementField
from brainhops.errors import ConversionError
from brainhops.io import FileFormat
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.freesurfer import FreesurferTkrSystem
from brainhops.io.common.freesurfer._geometry import fs_vox2tkr
from brainhops.io.metadata import MetadataFormat
from brainhops.io.transformations import TransformationFormat
from brainhops.io.transformations.freesurfer.lta import (
    LtaMetadata,
    LtaRaw,
    LtaTransformation,
    LtaTransformationPhysToPhys,
    LtaTransformationRASToRAS,
    LtaTransformationVoxToVox,
    LtaType,
    LtaValidity,
)
from brainhops.io.transformations.freesurfer.lta._matrix_utils import (
    get_vox2ras,
    volume_geometry,
)
from brainhops.io.transformations.freesurfer.regdat import (
    RegisterDatMetadata,
    RegisterDatRaw,
    RegisterDatTransformation,
    RegisterDatTransformationRASToRAS,
)
from brainhops.io.transformations.matrix import TxtMatrixAffine

# ----------------------------------------------------------------------
#   FIXTURES
# ----------------------------------------------------------------------

# A file in the layout of `regio_print_register`, with 16 significant
# digits that do not all survive a conversion to float and back.
FREESURFER = (
    "bert\n"
    "2.000000\n"
    "3.000000\n"
    "0.150000\n"
    "9.998765432109876e-01 -1.400000000000000e-02 "
    "1.000000000000000e-02 1.250000000000000e+00 \n"
    "1.400000000000000e-02 9.999000000000000e-01 "
    "1.234567890123457e-03 -2.500000000000000e+00 \n"
    "-1.000000000000000e-02 0.000000000000000e+00 "
    "9.999000000000000e-01 3.333333333333333e-01 \n"
    "0 0 0 1\n"
    "round\n"
)

MATRIX = (
    (0.9998765432109876, -0.014, 0.01, 1.25),
    (0.014, 0.9999, 0.001234567890123457, -2.5),
    (-0.01, 0.0, 0.9999, 0.3333333333333333),
    (0.0, 0.0, 0.0, 1.0),
)

# The reference is a conformed anatomical volume, and the movable volume
# an oblique functional volume with anisotropic voxels. Both have a
# centre away from the origin, so that tkregister RAS and scanner RAS
# differ.
_C, _S = np.cos(0.3), np.sin(0.3)
REFERENCE = LtaRaw.VolumeInfo(
    valid=LtaValidity.VOLUME_INFO_VALID,
    filename="orig.mgz",
    volume=(256, 256, 256),
    voxelsize=(1.0, 1.0, 1.0),
    xras=(-1.0, 0.0, 0.0),
    yras=(0.0, 0.0, -1.0),
    zras=(0.0, 1.0, 0.0),
    cras=(5.0, -10.0, 20.0),
)
MOVING = LtaRaw.VolumeInfo(
    valid=LtaValidity.VOLUME_INFO_VALID,
    filename="bold.nii.gz",
    volume=(64, 64, 35),
    voxelsize=(3.0, 3.0, 4.0),
    xras=(_C, _S, 0.0),
    yras=(-_S, _C, 0.0),
    zras=(0.0, 0.0, 1.0),
    cras=(-2.0, 7.5, 13.0),
)

_RAS = _systems.RASmm()


def _write(
    tmp_path: Path, text: str = FREESURFER, name: str = "r.dat"
) -> Path:
    file = tmp_path / name
    file.write_text(text)
    return file


def _tkr2ras(info: LtaRaw.VolumeInfo) -> np.ndarray:
    return get_vox2ras(info) @ np.linalg.inv(
        fs_vox2tkr(info.volume, info.voxelsize)
    )


# ----------------------------------------------------------------------
#   THE RECORD
# ----------------------------------------------------------------------


def test_a_freesurfer_file_is_read() -> None:
    raw = RegisterDatRaw.from_text(FREESURFER)
    assert raw.subject == "bert"
    assert raw.inplane_resolution == 2.0
    assert raw.slice_thickness == 3.0
    assert raw.intensity == 0.15
    assert raw.matrix == MATRIX
    assert raw.float2int == "round"


def test_a_file_is_written_back_byte_for_byte() -> None:
    raw = RegisterDatRaw.from_text(FREESURFER)
    assert raw.to_text() == FREESURFER
    assert replace(raw, lines=None).to_text() == FREESURFER
    # A file that another tool wrote keeps its own layout, which the
    # fields alone do not give.
    text = "bert\n2 3 0.15\n1 0 0 1.25\n0 1 0 -2.5\n0 0 1 0\n0 0 0 1\n"
    raw = RegisterDatRaw.from_text(text)
    assert raw.to_text() == text
    fresh = replace(raw, lines=None)
    assert fresh.to_text() != text
    assert RegisterDatRaw.from_text(fresh.to_text()) == replace(
        raw, lines=tuple(fresh.to_lines())
    )


def test_a_new_record_is_written_as_freesurfer_writes_it() -> None:
    raw = RegisterDatRaw(
        subject="bert",
        inplane_resolution=2.0,
        slice_thickness=3.0,
        intensity=0.15,
        matrix=((1.0, 0.0, 0.0, 1.5),) * 3 + ((0.0, 0.0, 0.0, 1.0),),
    )
    row = (
        "1.000000000000000e+00 0.000000000000000e+00 "
        "0.000000000000000e+00 1.500000000000000e+00 "
    )
    expected = ["bert", "2.000000", "3.000000", "0.150000"]
    expected += [row] * 3 + ["0 0 0 1", "round"]
    assert list(raw.to_lines()) == expected


def test_an_edited_record_is_encoded_again() -> None:
    raw = RegisterDatRaw.from_text(FREESURFER)
    edited = replace(raw, subject="ernie")
    lines = list(edited.to_lines())
    assert lines[0] == "ernie"
    assert RegisterDatRaw.from_lines(lines) == replace(
        edited, lines=tuple(lines)
    )


def test_any_white_space_separates_the_values() -> None:
    # FreeSurfer reads the values with `fscanf`, so their lines do not
    # matter, and words after the rounding method are ignored.
    text = " ".join(FREESURFER.split()) + " ignored"
    raw = RegisterDatRaw.from_text(text)
    assert raw.matrix == MATRIX
    assert raw.float2int == "round"
    assert raw.to_text() == text + "\n"


@pytest.mark.parametrize("name", ["round", "rint", "floor", "tkregister"])
def test_every_rounding_method_is_read(name: str) -> None:
    text = FREESURFER.replace("round\n", name + "\n")
    raw = RegisterDatRaw.from_text(text)
    assert raw.float2int == name
    assert replace(raw, lines=None).to_text().endswith(name + "\n")


def test_a_file_without_a_rounding_method_is_read() -> None:
    text = FREESURFER.replace("round\n", "")
    raw = RegisterDatRaw.from_text(text)
    assert raw.float2int is None
    assert raw.to_text() == text
    assert replace(raw, lines=None).to_text().endswith("0 0 0 1\n")


@pytest.mark.parametrize(
    "text",
    [
        "",
        FREESURFER.replace("round\n", "nearest\n"),
        FREESURFER.replace("0.150000", "bright"),
        "\n".join(FREESURFER.split()[:19]),
    ],
    ids=["empty", "method", "number", "short"],
)
def test_other_content_is_refused(text: str) -> None:
    with pytest.raises(ParserContentError):
        RegisterDatRaw.from_text(text)


def test_options_are_refused() -> None:
    with pytest.raises(TypeError):
        RegisterDatRaw.from_text(FREESURFER, subject="ernie")
    with pytest.raises(TypeError):
        RegisterDatRaw.from_text(FREESURFER).to_text(precision=6)


@pytest.mark.parametrize(
    "fields",
    [{"subject": "two words"}, {"float2int": "nearest"}],
    ids=["subject", "method"],
)
def test_a_record_that_cannot_be_read_back_is_not_written(
    fields: dict,
) -> None:
    with pytest.raises(WriterError):
        RegisterDatRaw(**fields).to_text()


def test_the_metadata_holds_the_record(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    metadata = RegisterDatMetadata.load(file)
    assert isinstance(metadata.raw, RegisterDatRaw)
    assert metadata.to_text() == FREESURFER
    assert type(MetadataFormat.load(file)) is RegisterDatMetadata
    assert RegisterDatMetadata().to_text() == RegisterDatRaw().to_text()


# ----------------------------------------------------------------------
#   SNIFFING AND DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls", [RegisterDatRaw, RegisterDatMetadata, RegisterDatTransformation]
)
def test_register_dat_content_is_sniffed(cls: type) -> None:
    assert cls.sniff_text(FREESURFER) == Confidence.LIKELY
    assert cls.sniff_bytes(FREESURFER.encode()) == Confidence.LIKELY
    assert cls.sniff_fileobj(_io.StringIO(FREESURFER)) == Confidence.LIKELY
    # Without its rounding method, the file has a weaker signature.
    text = FREESURFER.replace("round\n", "")
    assert cls.sniff_text(text) == Confidence.MAYBE


@pytest.mark.parametrize(
    "cls", [RegisterDatRaw, RegisterDatMetadata, RegisterDatTransformation]
)
def test_other_content_is_not_sniffed(cls: type) -> None:
    plain = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    lta = LtaMetadata.from_raw(LtaRaw()).to_text()
    for text in (
        "",
        plain,
        lta,
        FREESURFER + "more words\n",
        FREESURFER.replace("0 0 0 1", "0 0 1 1"),
        FREESURFER.replace("2.000000", "0"),
    ):
        assert cls.sniff_text(text) == Confidence.NO
    with pytest.raises(SnifferContentError):
        cls.sniff_text(plain, error=True)


def test_other_formats_do_not_claim_a_register_dat() -> None:
    assert LtaTransformation.sniff_text(FREESURFER) == Confidence.NO
    assert TxtMatrixAffine.sniff_text(FREESURFER) == Confidence.NO
    # A subject that is a number does not make the file a matrix either.
    text = FREESURFER.replace("bert", "42").replace("round\n", "")
    assert TxtMatrixAffine.sniff_text(text) == Confidence.NO


def test_a_register_dat_is_found_by_its_content(tmp_path) -> None:  # noqa: ANN001
    for name in ("register.dat", "bbr.reg", "registration.txt"):
        file = _write(tmp_path, name=name)
        assert io.sniff(file) is RegisterDatTransformation
        assert type(io.load(file)) is RegisterDatTransformation
        assert type(io.transformations.load(file)) is (
            RegisterDatTransformation
        )


def test_a_plain_matrix_named_dat_is_not_a_register_dat(tmp_path) -> None:  # noqa: ANN001
    file = tmp_path / "matrix.dat"
    np.savetxt(file, np.eye(4))
    assert RegisterDatTransformation.sniff(file) == Confidence.NO
    assert type(io.load(file)) is TxtMatrixAffine


def test_register_dat_is_a_registered_writable_format() -> None:
    assert RegisterDatTransformation in FileFormat._REGISTRY
    assert RegisterDatTransformation in TransformationFormat._REGISTRY
    assert RegisterDatTransformation.EXTENSIONS == (".dat", ".reg")
    # The view reads the same files, so registering it would be ambiguous.
    assert RegisterDatTransformationRASToRAS not in FileFormat._REGISTRY


@pytest.mark.parametrize("hint", ["regdat", "freesurfer.regdat"])
def test_a_hint_selects_the_reader(tmp_path, hint: str) -> None:  # noqa: ANN001
    file = _write(tmp_path, name="registration")
    assert type(io.load(file, hint=hint)) is RegisterDatTransformation


# ----------------------------------------------------------------------
#   THE TRANSFORMATION
# ----------------------------------------------------------------------


def test_the_file_maps_two_unknown_tkregister_spaces(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path))
    assert np.array_equal(xform.matrix, np.asarray(MATRIX)[:3])
    assert not xform.matrix.flags.writeable
    for system, name in ((xform.input, "reference"), (xform.output, "moving")):
        assert type(system) is FreesurferTkrSystem
        assert system.name == name
        assert system.geometry is None
        # tkregister RAS is not scanner RAS.
        assert not isinstance(system, _systems.RASmm)
    assert xform.input != xform.output


def test_the_geometries_are_options_of_the_reader(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    xform = io.load(file, reference=REFERENCE, moving=MOVING)
    # An LTA geometry block is held as the geometry of the data model.
    assert isinstance(xform.reference, Geometry)
    assert isinstance(xform.moving, Geometry)
    assert _same_place(xform.reference, REFERENCE)
    assert _same_place(xform.moving, MOVING)
    assert xform.input.geometry is xform.reference
    assert xform.input.name == "reference"
    assert xform.output.geometry is xform.moving
    with pytest.raises(TypeError):
        RegisterDatTransformation.load(file, subject="ernie")


def test_an_untouched_file_is_written_byte_for_byte(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    xform = io.load(file, reference=REFERENCE, moving=MOVING)
    xform.save(tmp_path / "copy.dat")
    assert (tmp_path / "copy.dat").read_text() == FREESURFER
    io.save(xform, tmp_path / "io.dat")
    assert (tmp_path / "io.dat").read_text() == FREESURFER
    # The geometries are not written, so setting them changes no record.
    xform.moving = None
    assert xform.consolidate().metadata is xform.metadata


def test_a_set_matrix_keeps_the_other_fields(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path))
    xform.data = np.diag([2.0, 3.0, 4.0, 1.0])[:3]
    xform.save(tmp_path / "set.dat")
    raw = RegisterDatRaw.load(tmp_path / "set.dat")
    assert (raw.subject, raw.intensity, raw.float2int) == (
        "bert",
        0.15,
        "round",
    )
    assert raw.inplane_resolution == 2.0
    assert raw.slice_thickness == 3.0
    assert np.array_equal(raw.matrix, np.diag([2.0, 3.0, 4.0, 1.0]))


def test_a_new_record_takes_the_voxel_size_of_the_movable_volume() -> None:
    xform = RegisterDatTransformation(np.eye(4)[:3], moving=MOVING)
    raw = xform.consolidate().metadata.raw
    assert (raw.inplane_resolution, raw.slice_thickness) == (3.0, 4.0)
    assert raw.subject == "subject-unknown"
    assert raw.float2int == "round"


def test_a_copy_shares_the_record_and_the_geometries(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    copy = RegisterDatTransformation.from_instance(xform)
    assert copy.metadata is xform.metadata
    assert copy.reference is xform.reference
    assert "_cache_data" not in vars(xform)


@pytest.mark.parametrize(
    "systems",
    [
        (_systems.RASmm(), FreesurferTkrSystem()),
        (_systems.LPSmm(), _systems.LPSmm()),
        (None, None),
    ],
    ids=["mixed", "lps", "unspecified"],
)
def test_other_systems_are_refused(systems: tuple) -> None:
    xform = RegisterDatTransformation(np.eye(4)[:3])
    xform.input, xform.output = systems
    if systems == (None, None):
        # Unset systems are the tkregister spaces of the format.
        assert xform.consolidate().metadata is not None
        return
    with pytest.raises(UnrepresentableTransformationError):
        xform.consolidate()


def test_a_tkregister_affine_is_not_saved_as_scanner_ras(tmp_path) -> None:  # noqa: ANN001
    # Without the geometries, the affine between scanner spaces is not
    # known, so the file is not written as an LTA file.
    xform = io.load(_write(tmp_path))
    with pytest.raises(WriterError, match="geometries"):
        io.save(xform, tmp_path / "x.lta")
    assert not (tmp_path / "x.lta").exists()
    # A relabelled affine between tkregister spaces is written with the
    # type of FreeSurfer for them, never as LINEAR_RAS_TO_RAS.
    lta = LtaTransformation(
        np.asarray(MATRIX)[:3],
        input=FreesurferTkrSystem(),
        output=FreesurferTkrSystem(),
    )
    assert lta.consolidate().metadata.raw.type == LtaType.REGISTER_DAT


# ----------------------------------------------------------------------
#   SCANNER RAS
# ----------------------------------------------------------------------


def _expected_scanner(tkr: np.ndarray) -> np.ndarray:
    # The inverse of `MRItkReg2Native(ref, mov, R)` in `utils/mri.cpp`:
    # Tmov * inv(Kmov) * R * Kref * inv(Tref), written as FreeSurfer
    # writes it, with the voxel-to-RAS matrices of each volume.
    t_ref, t_mov = get_vox2ras(REFERENCE), get_vox2ras(MOVING)
    k_ref = fs_vox2tkr(REFERENCE.volume, REFERENCE.voxelsize)
    k_mov = fs_vox2tkr(MOVING.volume, MOVING.voxelsize)
    return t_mov @ np.linalg.inv(k_mov) @ tkr @ k_ref @ np.linalg.inv(t_ref)


def test_the_ras_view_needs_both_geometries(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    view = RegisterDatTransformationRASToRAS.load(file, reference=REFERENCE)
    assert view.input == _systems.RASmm()
    assert view.output == _systems.RASmm()
    with pytest.raises(ValueError):
        _ = view.data
    view.moving = MOVING
    assert np.allclose(view.matrix, _expected_scanner(np.asarray(MATRIX))[:3])


def test_the_ras_view_is_a_conversion_that_shares_the_record(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    view = xform.to(RegisterDatTransformationRASToRAS)
    assert type(view) is RegisterDatTransformationRASToRAS
    assert view.metadata is xform.metadata
    assert view.moving is xform.moving
    assert np.allclose(view.matrix, _expected_scanner(np.asarray(MATRIX))[:3])


def test_a_scanner_matrix_is_encoded_in_tkregister_ras(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    view = xform.to(RegisterDatTransformationRASToRAS)
    scanner = np.array(view.matrix)
    view.data = scanner
    view.save(tmp_path / "scanner.dat")
    raw = RegisterDatRaw.load(tmp_path / "scanner.dat")
    assert np.allclose(raw.matrix, MATRIX, atol=1e-12)
    assert raw.subject == "bert"
    # A transformation between scanner spaces needs both geometries.
    view.moving = None
    with pytest.raises(UnrepresentableTransformationError):
        view.consolidate()


def test_a_scanner_affine_is_saved_with_its_geometries(tmp_path) -> None:  # noqa: ANN001
    scanner = _expected_scanner(np.asarray(MATRIX))
    affine = Affine(
        scanner[:3], input=_systems.RASmm(), output=_systems.RASmm()
    )
    # No other writable format claims `.dat`, so the reason of
    # register.dat is given, and nothing is written.
    with pytest.raises(WriterError) as raised:
        io.save(affine, tmp_path / "bare.dat")
    assert "RegisterDatTransformation: ConversionError" in str(raised.value)
    assert "reference= and moving=" in str(raised.value)
    assert list(tmp_path.iterdir()) == []
    xform = RegisterDatTransformation.from_instance(
        affine, reference=REFERENCE, moving=MOVING
    )
    xform.save(tmp_path / "placed.dat")
    raw = RegisterDatRaw.load(tmp_path / "placed.dat")
    assert np.allclose(raw.matrix, MATRIX, atol=1e-12)


def test_a_tkregister_affine_is_saved_as_register_dat(tmp_path) -> None:  # noqa: ANN001
    tkr = FreesurferTkrSystem()
    affine = Affine(np.asarray(MATRIX)[:3], input=tkr, output=tkr)
    io.save(affine, tmp_path / "tkr.dat")
    raw = RegisterDatRaw.load(tmp_path / "tkr.dat")
    assert np.allclose(raw.matrix, MATRIX, atol=1e-12)
    # With the geometries, an affine between scanner spaces is saved too.
    scanner = _expected_scanner(np.asarray(MATRIX))
    xform = RegisterDatTransformation.from_instance(
        Affine(scanner[:3], input=_RAS, output=_RAS),
        reference=REFERENCE,
        moving=MOVING,
    )
    io.save(xform, tmp_path / "scanner.dat")
    raw = RegisterDatRaw.load(tmp_path / "scanner.dat")
    assert np.allclose(raw.matrix, MATRIX, atol=1e-12)


def test_the_ras_view_matches_a_vox_to_vox_lta(tmp_path) -> None:  # noqa: ANN001
    # `TransformRegDat2LTA(targ, mov, R)` in `utils/transform.cpp` writes
    # a LINEAR_VOX_TO_VOX LTA from the target to the movable volume, whose
    # matrix is inv(Kmov) * R * Kref. Its scanner affine, which the LTA
    # reader computes from its own geometries, is that of the register.dat.
    tkr = np.asarray(MATRIX)
    k_ref = fs_vox2tkr(REFERENCE.volume, REFERENCE.voxelsize)
    k_mov = fs_vox2tkr(MOVING.volume, MOVING.voxelsize)
    vox2vox = np.linalg.inv(k_mov) @ tkr @ k_ref
    lta = LtaRaw(
        type=LtaType.LINEAR_VOX_TO_VOX,
        affine=LtaRaw.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in vox2vox)
        ),
        src=LtaRaw.SrcVolumeInfo(**_fields(REFERENCE)),
        dst=LtaRaw.DstVolumeInfo(**_fields(MOVING)),
    )
    from_lta = LtaTransformationRASToRAS.from_raw(lta)
    view = RegisterDatTransformationRASToRAS.load(
        _write(tmp_path), reference=REFERENCE, moving=MOVING
    )
    assert np.allclose(view.matrix, from_lta.matrix, atol=1e-10)
    # The same affine, written as a LINEAR_RAS_TO_RAS LTA, holds the same
    # scanner matrix. Assigning the matrix makes the view write its type.
    from_lta.data = np.array(from_lta.matrix)
    from_lta.save(tmp_path / "ras.lta")
    ras = LtaTransformation.load(tmp_path / "ras.lta")
    assert ras.metadata.raw.type == LtaType.LINEAR_RAS_TO_RAS
    assert np.allclose(view.matrix, ras.matrix, atol=1e-10)


def test_the_ras_view_inverts_the_lta_of_lta_convert(tmp_path) -> None:  # noqa: ANN001
    # `lta_convert --inreg` reads the file with `ltaReadRegisterDat`, puts
    # the movable volume in `src` and the reference in `dst`, and changes
    # the type to LINEAR_RAS_TO_RAS with `MRItkReg2Native`, which maps the
    # movable volume to the reference (`utils/transform.cpp`).
    native = np.linalg.inv(_expected_scanner(np.asarray(MATRIX)))
    lta = LtaRaw(
        type=LtaType.LINEAR_RAS_TO_RAS,
        affine=LtaRaw.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in native)
        ),
        src=LtaRaw.SrcVolumeInfo(**_fields(MOVING)),
        dst=LtaRaw.DstVolumeInfo(**_fields(REFERENCE)),
    )
    from_lta = LtaTransformationRASToRAS.from_raw(lta)
    view = RegisterDatTransformationRASToRAS.load(
        _write(tmp_path), reference=REFERENCE, moving=MOVING
    )
    product = view.homogeneous_matrix @ from_lta.homogeneous_matrix
    assert np.allclose(product, np.eye(4), atol=1e-10)


def test_the_tkregister_space_is_that_of_nibabel() -> None:
    nb = pytest.importorskip("nibabel")
    for info in (REFERENCE, MOVING):
        header = nb.freesurfer.mghformat.MGHHeader()
        header.set_data_shape(info.volume)
        header.set_zooms(info.voxelsize)
        header["Mdc"] = np.array([info.xras, info.yras, info.zras])
        header["Pxyz_c"] = info.cras
        assert np.allclose(header.get_affine(), get_vox2ras(info))
        tkr2ras = header.get_affine() @ np.linalg.inv(header.get_vox2ras_tkr())
        assert np.allclose(_tkr2ras(info), tkr2ras)


# ----------------------------------------------------------------------
#   GEOMETRIES
# ----------------------------------------------------------------------


def test_a_geometry_an_image_or_a_block_is_accepted() -> None:
    vox2ras = get_vox2ras(MOVING)
    voxel = _systems.FVoxelCoordinateSystem()
    geometry = Geometry(
        transformation=Affine(vox2ras[:3], input=voxel, output=_RAS),
        shape=MOVING.volume,
    )
    image = SingleScaleImage(
        data=np.zeros(MOVING.volume, dtype=np.uint8),
        transformations=[
            Affine(vox2ras[:3], input=voxel, output=_systems.RASmm())
        ],
    )
    # A world in LPS is converted to RAS.
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps = Geometry(
        transformation=Affine(
            (flip @ vox2ras)[:3], input=voxel, output=_systems.LPSmm()
        ),
        shape=MOVING.volume,
    )
    xform = RegisterDatTransformation(np.eye(4)[:3], moving=geometry)
    assert xform.moving is geometry
    for value in (image, lps, MOVING):
        xform.moving = value
        assert isinstance(xform.moving, Geometry)
        assert _same_place(xform.moving, MOVING)
    xform.moving = None
    assert xform.moving is None


def test_the_geometry_of_an_mgh_image_gives_its_tkregister_space(
    tmp_path,  # noqa: ANN001
) -> None:
    nb = pytest.importorskip("nibabel")
    header = nb.freesurfer.mghformat.MGHHeader()
    header.set_data_shape(MOVING.volume)
    header.set_zooms(MOVING.voxelsize)
    header["Mdc"] = np.array([MOVING.xras, MOVING.yras, MOVING.zras])
    header["Pxyz_c"] = MOVING.cras
    data = np.zeros(MOVING.volume, dtype=np.float32)
    nb.save(nb.MGHImage(data, None, header), str(tmp_path / "bold.mgz"))
    image = io.images.load(tmp_path / "bold.mgz")
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=image)
    loaded = nb.load(str(tmp_path / "bold.mgz")).header
    tkr2ras = loaded.get_affine() @ np.linalg.inv(loaded.get_vox2ras_tkr())
    view = xform.to(RegisterDatTransformationRASToRAS)
    expected = (
        tkr2ras @ np.asarray(MATRIX) @ np.linalg.inv(_tkr2ras(REFERENCE))
    )
    assert np.allclose(view.homogeneous_matrix, expected, atol=1e-4)


def _bad_geometries() -> dict:
    vox2ras = get_vox2ras(MOVING)
    voxel = _systems.FVoxelCoordinateSystem()

    def place(transformation: object, shape: tuple = (4, 4, 4)) -> Geometry:
        return Geometry(transformation=transformation, shape=shape)

    return {
        "4-D": SingleScaleImage(
            data=np.zeros((4, 4, 4, 2)),
            transformations=[Affine(np.eye(5)[:4])],
        ),
        "unknown world": place(Affine(vox2ras[:3])),
        "tkregister world": place(
            Affine(vox2ras[:3], input=voxel, output=FreesurferTkrSystem())
        ),
        "field": place(
            DisplacementField(np.zeros((4, 4, 4, 3)), input=voxel, output=_RAS)
        ),
        "singular": place(Affine(np.zeros((3, 4)), input=voxel, output=_RAS)),
        "invalid block": LtaRaw.VolumeInfo(volume=(4, 4, 4)),
    }


@pytest.mark.parametrize("name", sorted(_bad_geometries()))
def test_a_geometry_that_does_not_place_a_volume_is_refused(
    name: str,
) -> None:
    value = _bad_geometries()[name]
    with pytest.raises(ValueError, match="the reference volume|invalid"):
        RegisterDatTransformation(reference=value)
    xform = RegisterDatTransformation()
    with pytest.raises(ValueError):
        xform.moving = value
    assert xform.moving is None


def test_a_value_that_is_not_a_geometry_is_refused() -> None:
    with pytest.raises(TypeError, match="Geometry, an image or an"):
        RegisterDatTransformation(reference=np.eye(4))


def test_the_geometries_are_never_written(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    bare = io.load(file)
    placed = io.load(file, reference=REFERENCE, moving=MOVING)
    assert placed.to_text() == bare.to_text() == FREESURFER
    assert placed.consolidate().metadata is placed.metadata


# ----------------------------------------------------------------------
#   CONVERSIONS WITH LTA
# ----------------------------------------------------------------------


def test_a_register_dat_converts_exactly_to_an_lta(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    lta = xform.to(LtaTransformation)
    record = lta.metadata.raw
    # The type is the one that `lta_convert --inreg` writes, and the map
    # is that of the register.dat: from the reference to the movable
    # volume, so the reference is the source.
    assert record.type == LtaType.LINEAR_RAS_TO_RAS
    assert _same_place(volume_geometry(record.src), REFERENCE)
    assert _same_place(volume_geometry(record.dst), MOVING)
    scanner = _expected_scanner(np.asarray(MATRIX))
    assert np.allclose(lta.homogeneous_matrix, scanner, atol=1e-12)
    # `lta_convert --inreg` writes the inverse, from the movable volume to
    # the reference (`MRItkReg2Native`), which `--invert` turns into this
    # file.
    native = np.linalg.inv(scanner)
    assert np.allclose(np.linalg.inv(lta.homogeneous_matrix), native)
    # The view in voxels is the LTA of `TransformRegDat2LTA(targ, mov, R)`
    # in `utils/transform.cpp`: inv(Kmov) * R * Kref, from the target.
    vox2vox = xform.to(LtaTransformationVoxToVox)
    expected = np.linalg.inv(_vox2tkr(MOVING)) @ np.asarray(MATRIX)
    expected = expected @ _vox2tkr(REFERENCE)
    assert np.allclose(vox2vox.homogeneous_matrix, expected, atol=1e-10)
    assert type(xform.to(LtaTransformationPhysToPhys)) is (
        LtaTransformationPhysToPhys
    )
    # The scanner view converts to the same file.
    view = xform.to(RegisterDatTransformationRASToRAS)
    again = view.to(LtaTransformation)
    assert np.allclose(again.homogeneous_matrix, scanner, atol=1e-12)


def test_a_register_dat_is_saved_as_an_lta(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    io.save(xform, tmp_path / "r.lta")
    lta = io.load(tmp_path / "r.lta")
    assert type(lta) is LtaTransformation
    assert lta.metadata.raw.type == LtaType.LINEAR_RAS_TO_RAS
    scanner = _expected_scanner(np.asarray(MATRIX))
    assert np.allclose(lta.homogeneous_matrix, scanner, atol=1e-12)


def test_a_conversion_to_lta_needs_both_geometries(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    for options in ({}, {"reference": REFERENCE}, {"moving": MOVING}):
        xform = io.load(file, **options)
        with pytest.raises(ConversionError, match="geometries"):
            xform.to(LtaTransformation)
        with pytest.raises(ConversionError, match="geometries"):
            LtaTransformation.from_instance(xform)


@pytest.mark.parametrize(
    "view",
    [LtaTransformation, LtaTransformationVoxToVox, LtaTransformationRASToRAS],
)
def test_an_lta_converts_exactly_to_a_register_dat(view: type) -> None:
    lta = view.from_raw(_vox2vox_lta())
    xform = lta.to(RegisterDatTransformation)
    assert type(xform) is RegisterDatTransformation
    # The source of the LTA is the reference of the register.dat.
    assert _same_place(xform.reference, REFERENCE)
    assert _same_place(xform.moving, MOVING)
    record = xform.metadata.raw
    assert np.allclose(record.matrix, MATRIX, atol=1e-12)
    assert (record.inplane_resolution, record.slice_thickness) == (3.0, 4.0)
    ras = lta.to(RegisterDatTransformationRASToRAS)
    scanner = _expected_scanner(np.asarray(MATRIX))
    assert np.allclose(ras.homogeneous_matrix, scanner, atol=1e-10)


def test_an_lta_is_saved_as_a_register_dat(tmp_path) -> None:  # noqa: ANN001
    # A scanner RAS affine is converted to tkregister RAS, and is not
    # written as it is (it was relabelled before the exact converter).
    ras = LtaTransformationRASToRAS.from_raw(_vox2vox_lta())
    ras.data = np.array(ras.matrix)
    assert ras.consolidate().metadata.raw.type == LtaType.LINEAR_RAS_TO_RAS
    io.save(ras, tmp_path / "r.dat")
    raw = RegisterDatRaw.load(tmp_path / "r.dat")
    assert np.allclose(raw.matrix, MATRIX, atol=1e-10)


def test_a_conversion_from_lta_needs_both_geometries() -> None:
    record = replace(
        _vox2vox_lta(),
        dst=LtaRaw.DstVolumeInfo(valid=LtaValidity.VOLUME_INFO_INVALID),
    )
    lta = LtaTransformation.from_raw(record)
    with pytest.raises(ConversionError, match="geometry"):
        lta.to(RegisterDatTransformation)
    with pytest.raises(TypeError, match="reference="):
        LtaTransformation.from_raw(_vox2vox_lta()).to(
            RegisterDatTransformation, reference=MOVING
        )


def test_a_round_trip_through_lta_keeps_the_matrix(tmp_path) -> None:  # noqa: ANN001
    xform = io.load(_write(tmp_path), reference=REFERENCE, moving=MOVING)
    back = xform.to(LtaTransformation).to(RegisterDatTransformation)
    assert np.allclose(back.metadata.raw.matrix, MATRIX, atol=1e-12)
    back = xform.to(LtaTransformationVoxToVox).to(RegisterDatTransformation)
    assert np.allclose(back.metadata.raw.matrix, MATRIX, atol=1e-12)


# ----------------------------------------------------------------------
#   LTA FILES OF TYPE REGISTER_DAT
# ----------------------------------------------------------------------

# An LTA file of type 14, as `LTAprint` in `utils/transform.cpp` writes
# it after `lta_convert --inreg ... --ltatkreg`: the matrix of the
# register.dat, with the movable volume as `src` and the reference as
# `dst` (`ltaReadRegisterDat`).


def _register_dat_lta() -> LtaRaw:
    return LtaRaw(
        type=LtaType.REGISTER_DAT,
        sigma=1.0,
        affine=LtaRaw.Affine(matrix=MATRIX),
        src=LtaRaw.SrcVolumeInfo(**_fields(MOVING)),
        dst=LtaRaw.DstVolumeInfo(**_fields(REFERENCE)),
    )


def test_an_lta_of_type_register_dat_is_read(tmp_path) -> None:  # noqa: ANN001
    text = LtaMetadata.from_raw(_register_dat_lta()).to_text()
    assert text.startswith("type      = 14")
    file = tmp_path / "tkreg.lta"
    file.write_text(text)
    lta = io.load(file)
    assert type(lta) is LtaTransformation
    assert lta.metadata.raw.type == LtaType.REGISTER_DAT
    # The transformation maps the source to the destination, between
    # their tkregister spaces, so it is the inverse of the stored matrix.
    assert type(lta.input) is FreesurferTkrSystem
    assert lta.input.name == "bold.nii.gz"
    assert lta.output.name == "orig.mgz"
    assert np.allclose(
        lta.homogeneous_matrix, np.linalg.inv(np.asarray(MATRIX))
    )
    # An untouched file is written as it was read.
    lta.save(tmp_path / "copy.lta")
    assert (tmp_path / "copy.lta").read_text() == text


def test_the_views_of_a_register_dat_lta_follow_freesurfer() -> None:
    record = _register_dat_lta()
    # `LTAchangeType(lta, LINEAR_RAS_TO_RAS)` computes
    # `MRItkReg2Native(ref=dst, mov=src, R)`, from `src` to `dst`.
    ras = LtaTransformationRASToRAS.from_raw(record)
    native = np.linalg.inv(_expected_scanner(np.asarray(MATRIX)))
    assert np.allclose(ras.homogeneous_matrix, native, atol=1e-10)
    vox = LtaTransformationVoxToVox.from_raw(record)
    vox2ras_src, vox2ras_dst = get_vox2ras(MOVING), get_vox2ras(REFERENCE)
    expected = np.linalg.inv(vox2ras_dst) @ native @ vox2ras_src
    assert np.allclose(vox.homogeneous_matrix, expected, atol=1e-10)
    phys = LtaTransformationPhysToPhys.from_raw(record)
    assert phys.matrix.shape == (3, 4)


def test_an_assigned_tkregister_matrix_is_written_as_register_dat() -> None:
    lta = LtaTransformation.from_raw(_register_dat_lta())
    lta.data = np.array(lta.matrix)
    record = lta.consolidate().metadata.raw
    assert record.type == LtaType.REGISTER_DAT
    assert np.allclose(record.affine.matrix, MATRIX, atol=1e-12)
    assert record.src == _register_dat_lta().src
    # A geometry held by a tkregister system goes to the record.
    xform = LtaTransformation(
        np.eye(4)[:3],
        input=FreesurferTkrSystem(geometry=volume_geometry(REFERENCE)),
        output=FreesurferTkrSystem(geometry=volume_geometry(MOVING)),
    )
    record = xform.consolidate().metadata.raw
    assert _same_place(volume_geometry(record.src), REFERENCE)
    assert _same_place(volume_geometry(record.dst), MOVING)


def test_a_register_dat_lta_converts_without_its_geometries() -> None:
    record = replace(_register_dat_lta(), src=None, dst=None)
    xform = LtaTransformation.from_raw(record).to(RegisterDatTransformation)
    assert xform.reference is None and xform.moving is None
    assert np.allclose(
        xform.metadata.raw.matrix, np.linalg.inv(np.asarray(MATRIX))
    )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _fields(info: LtaRaw.VolumeInfo) -> dict:
    names = ("valid", "filename", "volume", "voxelsize")
    names += ("xras", "yras", "zras", "cras")
    return {name: getattr(info, name) for name in names}


def _vox2tkr(info: LtaRaw.VolumeInfo) -> np.ndarray:
    return fs_vox2tkr(info.volume, info.voxelsize)


def _same_place(geometry: Geometry, info: LtaRaw.VolumeInfo) -> bool:
    # Whether a geometry places a volume as an LTA geometry block does.
    matrix = geometry.transformation.homogeneous_matrix
    if geometry.transformation.output == _systems.LPSmm():
        matrix = np.diag([-1.0, -1.0, 1.0, 1.0]) @ matrix
    return tuple(geometry.shape) == tuple(info.volume) and np.allclose(
        matrix, get_vox2ras(info), atol=1e-12
    )


def _vox2vox_lta() -> LtaRaw:
    # The LTA of `TransformRegDat2LTA(targ, mov, R)` in
    # `utils/transform.cpp`: LINEAR_VOX_TO_VOX from the target to the
    # movable volume, whose matrix is inv(Kmov) * R * Kref.
    vox2vox = np.linalg.inv(_vox2tkr(MOVING)) @ np.asarray(MATRIX)
    vox2vox = vox2vox @ _vox2tkr(REFERENCE)
    return LtaRaw(
        type=LtaType.LINEAR_VOX_TO_VOX,
        affine=LtaRaw.Affine(
            matrix=tuple(tuple(float(v) for v in row) for row in vox2vox)
        ),
        src=LtaRaw.SrcVolumeInfo(**_fields(REFERENCE)),
        dst=LtaRaw.DstVolumeInfo(**_fields(MOVING)),
    )
