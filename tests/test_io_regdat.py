"""Tests for reading and writing FreeSurfer `register.dat` files.

A `register.dat` file is a registered text format, found by its content
and written by `io.save`. Its matrix maps the tkregister RAS space of the
reference volume to that of the movable volume. The file holds no
geometry, so the scanner RAS affine is checked against the LTA files that
FreeSurfer builds from the same matrix and the same geometries.
"""

import io as _io
from pathlib import Path

import numpy as np
import pytest
from bagof.magic import replace

import brainhops.io as io
from brainhops.datamodel import systems as _systems
from brainhops.datamodel.transformations import Affine
from brainhops.io import Format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common.freesurfer._geometry import fs_vox2tkr
from brainhops.io.metadata import MetadataFormat
from brainhops.io.transformations import TransformationFormat
from brainhops.io.transformations.freesurfer.lta import (
    LtaMetadata,
    LtaRaw,
    LtaTransformation,
    LtaTransformationRASToRAS,
    LtaType,
    LtaValidity,
)
from brainhops.io.transformations.freesurfer.lta._matrix_utils import (
    _get_vox2ras,
)
from brainhops.io.transformations.freesurfer.regdat import (
    RegisterDatMetadata,
    RegisterDatRaw,
    RegisterDatTkrSystem,
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


def _write(
    tmp_path: Path, text: str = FREESURFER, name: str = "r.dat"
) -> Path:
    file = tmp_path / name
    file.write_text(text)
    return file


def _tkr2ras(info: LtaRaw.VolumeInfo) -> np.ndarray:
    return _get_vox2ras(info) @ np.linalg.inv(
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
    assert RegisterDatTransformation in Format._REGISTRY
    assert RegisterDatTransformation in TransformationFormat._REGISTRY
    assert RegisterDatTransformation.EXTENSIONS == (".dat", ".reg")
    # The view reads the same files, so registering it would be ambiguous.
    assert RegisterDatTransformationRASToRAS not in Format._REGISTRY


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
        assert type(system) is RegisterDatTkrSystem
        assert system.name == name
        assert system.raw is None
        # tkregister RAS is not scanner RAS.
        assert not isinstance(system, _systems.RASmm)
    assert xform.input != xform.output


def test_the_geometries_are_options_of_the_reader(tmp_path) -> None:  # noqa: ANN001
    file = _write(tmp_path)
    xform = io.load(file, reference=REFERENCE, moving=MOVING)
    assert xform.reference is REFERENCE
    assert xform.moving is MOVING
    assert xform.input.raw is REFERENCE
    assert xform.input.name == "orig.mgz"
    assert xform.output.raw is MOVING
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
    assert copy.reference is REFERENCE
    assert "_cache_data" not in vars(xform)


@pytest.mark.parametrize(
    "systems",
    [
        (_systems.RASmm(), RegisterDatTkrSystem()),
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
    xform = io.load(_write(tmp_path))
    with pytest.raises(UnrepresentableTransformationError):
        io.save(xform, tmp_path / "x.lta")


# ----------------------------------------------------------------------
#   SCANNER RAS
# ----------------------------------------------------------------------


def _expected_scanner(tkr: np.ndarray) -> np.ndarray:
    # The inverse of `MRItkReg2Native(ref, mov, R)` in `utils/mri.cpp`:
    # Tmov * inv(Kmov) * R * Kref * inv(Tref), written as FreeSurfer
    # writes it, with the voxel-to-RAS matrices of each volume.
    t_ref, t_mov = _get_vox2ras(REFERENCE), _get_vox2ras(MOVING)
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
    assert view.moving is MOVING
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
    with pytest.raises(UnrepresentableTransformationError):
        io.save(affine, tmp_path / "bare.dat")
    xform = RegisterDatTransformation.from_instance(
        affine, reference=REFERENCE, moving=MOVING
    )
    xform.save(tmp_path / "placed.dat")
    raw = RegisterDatRaw.load(tmp_path / "placed.dat")
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
        assert np.allclose(header.get_affine(), _get_vox2ras(info))
        tkr2ras = header.get_affine() @ np.linalg.inv(header.get_vox2ras_tkr())
        assert np.allclose(_tkr2ras(info), tkr2ras)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _fields(info: LtaRaw.VolumeInfo) -> dict:
    names = ("valid", "filename", "volume", "voxelsize")
    names += ("xras", "yras", "zras", "cras")
    return {name: getattr(info, name) for name in names}
