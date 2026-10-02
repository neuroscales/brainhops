"""
Tests for AFNI transformations: affine matrices (`.aff12.1D`, `.1D`) and
nonlinear warps (`_WARP` datasets, as AFNI datasets or NIfTI files).

The fixtures are written from the formats' specifications, as AFNI's
sources describe them (see `brainhops.io.transformations.afni`), with
encoders independent of the readers: the `.HEAD` attribute by attribute,
the `.BRIK` packed sub-brick by sub-brick with `x` fastest, and NIfTI
files through `nibabel`. The obliquity correction is checked against
NiTransforms' formula (`nitransforms/io/afni.py`), re-implemented here.
"""

import gzip
import io as _io

import numpy as np
import pytest

import brainhops.io as io
from brainhops.datamodel import systems as S
from brainhops.datamodel import transformations as T
from brainhops.io.base.afni import afni_cardinal_matrix
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.base.specs import format_hints
from brainhops.io.images.afni import AfniImage
from brainhops.io.transformations import load as load_xform
from brainhops.io.transformations import sniff as sniff_xform
from brainhops.io.transformations.afni import (
    AfniAffine,
    AfniBrikWarp,
    cardinal_to_real,
)
from brainhops.io.transformations.matrix import TxtMatrixAffine

nb = pytest.importorskip("nibabel")
from brainhops.io.transformations.afni import AfniNiftiWarp  # noqa: E402

LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
FLIP = np.array([-1.0, -1.0, 1.0])

# A rigid-ish base-to-source matrix, as 3dAllineate would save it.
M1 = np.array(
    [
        [0.99, -0.05, 0.02, 3.5],
        [0.06, 1.01, -0.1, -2.25],
        [-0.01, 0.09, 0.98, 7.0],
    ]
)
M2 = M1 + np.array([[0, 0, 0, 1.0], [0, 0, 0, -1.0], [0, 0, 0, 0.5]])
ALLINEATE = "# 3dAllineate matrices (DICOM-to-DICOM, row-by-row):\n"


def _row(matrix: np.ndarray) -> str:
    return " " + " ".join(f"{v:13.6g}" for v in np.ravel(matrix)) + "\n"


def _homog(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3] = np.asarray(matrix)[:3]
    return out


def _apply(xform, points: np.ndarray, space) -> np.ndarray:  # noqa: ANN001
    """Map points (in `space`) through a transformation."""
    field = T.CoordinatesField(field=np.asarray(points, float), output=space)
    out = T.Sequence(transformations=[field, xform]).compute()
    return np.asarray(out.to(T.CoordinatesField).field)


# ----------------------------------------------------------------------
#   AFFINE MATRICES
# ----------------------------------------------------------------------


def test_reads_a_3dallineate_matrix(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "anat_al.aff12.1D"
    path.write_text(ALLINEATE + _row(M1))
    assert sniff_xform(path) is AfniAffine
    xform = io.load(path)
    assert type(xform) is AfniAffine
    assert np.allclose(xform.matrix, M1, atol=1e-6)
    assert xform.nvolumes == 1
    assert xform.input == S.LPSmm() and xform.output == S.LPSmm()


def test_sniff_levels() -> None:
    text = ALLINEATE + _row(M1)
    assert AfniAffine.sniff_text(text) == Confidence.CERTAIN
    assert AfniAffine.sniff_text(_row(M1)) == Confidence.WEAK
    # 3 lines of 4 without a .1D name are left to the matrix reader
    three = "".join(" ".join(map(str, r)) + "\n" for r in M1)
    assert AfniAffine.sniff_text(three) == Confidence.NO
    assert AfniAffine.sniff_text("1 2 3\n4 5 6\n") == Confidence.NO
    assert AfniAffine.sniff_text("not numbers\n") == Confidence.NO


def test_rows_of_12_without_a_name_are_read(tmp_path) -> None:  # noqa: ANN001
    xform = load_xform(_row(M1).encode())
    assert type(xform) is AfniAffine
    assert np.allclose(xform.matrix, M1, atol=1e-6)


def test_several_matrices_one_per_volume(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "epi_vr.aff12.1D"
    path.write_text(
        "# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):\n"
        + _row(M1)
        + _row(M2)
    )
    xform = io.load(path)
    assert xform.nvolumes == 2
    assert np.allclose(xform.matrix, M1, atol=1e-6)
    assert np.allclose(xform.matrices, [M1, M2], atol=1e-6)
    assert np.allclose(io.load(path, volume=1).matrix, M2, atol=1e-6)
    assert np.allclose(io.load(path, volume=-1).matrix, M2, atol=1e-6)
    with pytest.raises(ParserContentError):
        AfniAffine.from_file(path, volume=2)


@pytest.mark.parametrize(
    "text",
    [
        "".join(" ".join(f"{v:.17g}" for v in r) + "\n" for r in M1),
        "".join(" ".join(f"{v:.17g}" for v in r) + "\n" for r in _homog(M1)),
    ],
    ids=["cat_matvec", "cat_matvec-4x4"],
)
def test_cat_matvec_layouts(tmp_path, text) -> None:  # noqa: ANN001
    path = tmp_path / "mat.1D"
    path.write_text(text)
    xform = AfniAffine.from_file(path)
    assert np.allclose(xform.matrix, M1)


def test_three_lines_of_four_in_a_1d_file_is_afni(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "mat.1D"
    path.write_text("".join(" ".join(map(str, r)) + "\n" for r in M1))
    assert sniff_xform(path) is AfniAffine


def test_a_square_matrix_in_a_1d_file_stays_a_plain_matrix(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "mat.1D"
    np.savetxt(path, _homog(M1))
    assert sniff_xform(path) is TxtMatrixAffine
    # ... but AFNI reads it, when asked
    assert type(load_xform(path, hint="afni")) is AfniAffine


def test_nine_numbers_are_a_matrix_without_shift() -> None:
    xform = AfniAffine.from_text("1 2 3 4 5 6 7 8 9\n")
    assert np.allclose(
        xform.matrix, [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]
    )


def test_hints() -> None:
    hints = format_hints(AfniAffine)
    assert {"afni", "afni.aff12", "xform", "affine"} <= hints


def test_ras_points_are_mapped_with_x_and_y_negated() -> None:
    xform = AfniAffine.from_text(ALLINEATE + _row(M1))
    points = np.array([[10.0, -20.0, 30.0], [0.0, 0.0, 0.0], [-5, 7, 1]])
    expected = points @ M1[:, :3].T + M1[:, 3]
    # RAS points in, LPS points out: the RAS input is flipped first
    assert np.allclose(_apply(xform, points * FLIP, S.RASmm()), expected)
    # and the RAS matrix is LPS @ M @ LPS (NiTransforms' to_ras)
    affine = T.Sequence(
        [T.Affine(input=S.RASmm(), output=S.RASmm()), xform]
    ).compute()
    ras = _homog(affine.to(T.Affine).matrix)
    if affine.output == S.LPSmm():
        ras = LPS @ ras
    assert np.allclose(ras, LPS @ _homog(M1) @ LPS)


# --- obliquity --------------------------------------------------------


def _oblique(angle: float = 0.3, shift=(1.0, -2.0, 3.0)) -> np.ndarray:  # noqa: ANN001
    """A voxel-to-RAS matrix, rotated about z and x."""
    c, s = np.cos(angle), np.sin(angle)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    rx = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    out = np.eye(4)
    out[:3, :3] = rz @ rx @ np.diag([2.0, 2.5, 3.0])
    out[:3, 3] = shift
    return out


def _nitransforms_card(oblique: np.ndarray) -> np.ndarray:
    """NiTransforms' `_dicom_real_to_card`, on a voxel-to-RAS affine."""
    out = np.eye(4)
    out[:3, 3] = oblique[:3, 3]
    cosines = oblique[:3, :3] / np.abs(oblique[:3, :3]).max(0)
    cosines[np.abs(cosines) < 1.0] = 0
    sizes = np.sqrt((oblique[:3, :3] ** 2).sum(0))
    out[:3, :3] = np.round(sizes, 4) * cosines
    return out


def _nifti(vox2ras: np.ndarray, shape=(6, 5, 4)) -> "nb.Nifti1Image":  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros(shape, np.float32), vox2ras)
    img.header.set_qform(vox2ras, code=1)
    img.header.set_sform(vox2ras, code=1)
    return img


def test_cardinal_to_real_is_the_identity_without_obliquity() -> None:
    vox2ras = np.diag([-2.0, 2.0, 3.0, 1.0])
    vox2ras[:3, 3] = [10, -20, 5]
    assert np.allclose(cardinal_to_real(_nifti(vox2ras)).matrix, np.eye(4)[:3])


def test_obliquity_matches_nitransforms() -> None:
    ref = _nifti(np.diag([2.0, 2.0, 2.0, 1.0]))
    mov = _nifti(_oblique())
    xform = AfniAffine.from_text(ALLINEATE + _row(M1), base=ref, source=mov)
    # NiTransforms' AFNILinearTransform.to_ras, in RAS
    ras = LPS @ _homog(M1) @ LPS
    card = _nitransforms_card(mov.affine)
    ras = mov.affine @ np.linalg.inv(card) @ ras
    assert np.allclose(LPS @ _homog(xform.matrix) @ LPS, ras)


def test_obliquity_of_both_images() -> None:
    ref = _nifti(_oblique(-0.2, (4.0, 5.0, 6.0)))
    mov = _nifti(_oblique(0.35))
    xform = AfniAffine.from_text(
        ALLINEATE + _row(M1), reference=ref, moving=mov
    )
    ras = LPS @ _homog(M1) @ LPS
    ras = ras @ _nitransforms_card(ref.affine) @ np.linalg.inv(ref.affine)
    ras = mov.affine @ np.linalg.inv(_nitransforms_card(mov.affine)) @ ras
    assert np.allclose(LPS @ _homog(xform.matrix) @ LPS, ras)
    # writing it with the same images gives back the stored matrix
    plain = AfniAffine(matrix=xform.matrix, base=ref, source=mov)
    again = AfniAffine.from_text(plain.to_text())
    assert np.allclose(again.matrix, M1)


def test_obliquity_of_an_afni_image(tmp_path) -> None:  # noqa: ANN001
    real = LPS @ _oblique()
    image = AfniImage(
        data=np.zeros((6, 5, 4), np.float32),
        transformations=[
            T.Affine(
                input=S.VoxelCoordinateSystem(),
                output=S.LPSmm(name="orig"),
                matrix=real[:3],
            )
        ],
    )
    image.save(tmp_path / "mov+orig.HEAD")
    mov = io.load(tmp_path / "mov+orig.HEAD")
    expected = mov.header.voxel_to_dicom @ np.linalg.inv(
        mov.header.cardinal_matrix
    )
    assert np.allclose(cardinal_to_real(mov).matrix, expected[:3])
    xform = AfniAffine.from_text(_row(M1), source=mov)
    assert np.allclose(_homog(xform.matrix), expected @ _homog(M1))


# --- writing ----------------------------------------------------------


def test_round_trip_keeps_rows_and_comment(tmp_path) -> None:  # noqa: ANN001
    text = "# 3dvolreg matrices (DICOM-to-DICOM, row-by-row):\n"
    text += _row(M1) + _row(M2)
    path = tmp_path / "in.aff12.1D"
    path.write_text(text)
    xform = io.load(path, volume=1)
    xform.save(tmp_path / "out.aff12.1D")
    out = (tmp_path / "out.aff12.1D").read_text()
    assert out.splitlines()[0] == text.splitlines()[0]
    again = io.load(tmp_path / "out.aff12.1D")
    assert np.allclose(again.matrices, [M1, M2], atol=1e-6)


def test_save_an_ras_affine(tmp_path) -> None:  # noqa: ANN001
    ras = LPS @ _homog(M1) @ LPS
    affine = T.Affine(matrix=ras[:3], input=S.RASmm(), output=S.RASmm())
    io.save(affine, tmp_path / "out.aff12.1D")
    text = (tmp_path / "out.aff12.1D").read_text()
    assert "DICOM-to-DICOM" in text.splitlines()[0]
    assert np.allclose(io.load(tmp_path / "out.aff12.1D").matrix, M1)


def test_save_full_precision() -> None:
    xform = AfniAffine(matrix=M1 / 3.0)
    again = AfniAffine.from_text(xform.to_text())
    assert np.array_equal(again.matrix, M1 / 3.0)


def test_save_as_three_lines_of_four() -> None:
    xform = AfniAffine(matrix=M1)
    text = xform.to_text(oneline=False)
    assert len(text.split("\n")[0].split()) == 4
    assert np.allclose(AfniAffine.from_text(text).matrix, M1)
    many = AfniAffine.from_text(_row(M1) + _row(M2))
    with pytest.raises(WriterError):
        many.to_text(oneline=False)


def test_a_failed_write_leaves_no_file(tmp_path) -> None:  # noqa: ANN001
    many = AfniAffine.from_text(_row(M1) + _row(M2))
    with pytest.raises(WriterError):
        many.save(tmp_path / "out.1D", oneline=False)
    assert not (tmp_path / "out.1D").exists()


def test_a_changed_matrix_is_written_alone() -> None:
    many = AfniAffine.from_text(_row(M1) + _row(M2))
    many.matrix = M2
    again = AfniAffine.from_text(many.to_text())
    assert again.nvolumes == 1 and np.allclose(again.matrix, M2)


def test_inverse() -> None:
    xform = AfniAffine.from_text(_row(M1))
    inv = xform.inverse().compute().to(T.Affine).matrix
    assert np.allclose(_homog(inv), np.linalg.inv(_homog(M1)))


# ----------------------------------------------------------------------
#   WARPS: REFERENCE ENCODERS
# ----------------------------------------------------------------------

SHAPE = (5, 4, 3)
# A grid that is not in DICOM order: L2R, P2A, I2S.
ORIENT, ORIGIN, DELTA = (1, 2, 4), (10.0, -20.0, -30.0), (-2.0, 2.0, 3.0)
CARD = afni_cardinal_matrix(ORIENT, ORIGIN, DELTA)
LABELS = "x_delta~y_delta~z_delta~"


def _vectors(nvals: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(scale=2.0, size=(*SHAPE, nvals)).astype(np.float32)


def _attr(kind: str, name: str, values) -> str:  # noqa: ANN001
    if kind == "string":
        return (
            f"\ntype = string-attribute\nname = {name}\n"
            f"count = {len(values)}\n'{values}\n"
        )
    body = " ".join(str(v) for v in values)
    return (
        f"\ntype = {kind}-attribute\nname = {name}\n"
        f"count = {len(values)}\n {body}\n"
    )


def _write_brik(
    tmp_path,  # noqa: ANN001
    vectors: np.ndarray,
    name: str = "anat_WARP+tlrc",
    labels: str = LABELS,
    real: np.ndarray = None,  # noqa: RUF013
    facs=None,  # noqa: ANN001
    dtype: str = "<f4",
    gz: bool = False,
) -> str:
    nvals = vectors.shape[-1]
    code = {"<f4": 3, "<i2": 1}[dtype]
    head = "".join(
        [
            _attr("string", "TYPESTRING", "3DIM_HEAD_ANAT~"),
            _attr(
                "integer",
                "SCENE_DATA",
                [2, 11, 0, -999, -999, -999, -999, -999],
            ),
            _attr("integer", "ORIENT_SPECIFIC", list(ORIENT)),
            _attr("float", "ORIGIN", list(ORIGIN)),
            _attr("float", "DELTA", list(DELTA)),
            _attr("integer", "DATASET_RANK", [3, nvals, 0, 0, 0, 0, 0, 0]),
            _attr("integer", "DATASET_DIMENSIONS", [*SHAPE, 0, 0]),
            _attr("integer", "BRICK_TYPES", [code] * nvals),
            _attr("string", "BYTEORDER_STRING", "LSB_FIRST~"),
        ]
    )
    if labels:
        head += _attr("string", "BRICK_LABS", labels)
    if real is not None:
        head += _attr("float", "IJK_TO_DICOM_REAL", list(real[:3].ravel()))
    stored = vectors
    if facs is not None:
        head += _attr("float", "BRICK_FLOAT_FACS", list(facs))
        stored = np.round(vectors / np.asarray(facs))
    data = b"".join(
        np.asarray(stored[..., i], dtype=dtype).tobytes(order="F")
        for i in range(nvals)
    )
    (tmp_path / f"{name}.HEAD").write_text(head)
    brik = tmp_path / (f"{name}.BRIK" + (".gz" if gz else ""))
    brik.write_bytes(gzip.compress(data) if gz else data)
    return str(tmp_path / f"{name}.HEAD")


def _grid_lps(card: np.ndarray = CARD) -> np.ndarray:
    ijk = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), -1)
    return ijk @ card[:3, :3].T + card[:3, 3]


def _check_warp(warp, vectors: np.ndarray, card: np.ndarray = CARD) -> None:  # noqa: ANN001
    """Every grid point `x` (DICOM) goes to `x + u(x)`."""
    points = _grid_lps(card).reshape(-1, 3)
    moved = _apply(warp, points * FLIP, S.RASmm()) * FLIP
    assert np.allclose(
        moved, points + vectors[..., :3].reshape(-1, 3), atol=1e-4
    )


# ----------------------------------------------------------------------
#   WARPS: BRIK/HEAD
# ----------------------------------------------------------------------


def test_reads_a_brik_warp(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    path = _write_brik(tmp_path, vectors)
    assert sniff_xform(path) is AfniBrikWarp
    assert io.sniff(path) is AfniBrikWarp
    warp = io.load(path)
    assert type(warp) is AfniBrikWarp
    assert warp.input == S.RASmm() and warp.output == S.RASmm()
    assert [type(t).__name__ for t in warp] == [
        "RASToVoxel",
        "DisplacementField",
        "VoxelToRAS",
    ]
    _check_warp(warp, vectors)
    # an image reader still reads it as an image
    assert type(io.images.load(path)) is AfniImage


def test_brik_warp_hints(tmp_path) -> None:  # noqa: ANN001
    hints = format_hints(AfniBrikWarp)
    assert {"afni", "afni.warp", "afni.warp.brik", "brik"} <= hints
    path = _write_brik(tmp_path, _vectors())
    assert type(io.load(path, hint="afni.warp")) is AfniBrikWarp


def test_a_brik_without_warp_labels_is_an_image(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    path = _write_brik(tmp_path, vectors, labels="a~b~c~")
    assert AfniBrikWarp.sniff_filename(path) == Confidence.NO
    assert type(io.load(path)) is AfniImage
    warp = load_xform(path, hint="afni.warp")
    assert type(warp) is AfniBrikWarp
    _check_warp(warp, vectors)


def test_brik_warp_scaled_and_compressed(tmp_path) -> None:  # noqa: ANN001
    vectors = np.round(_vectors() * 100) / 100
    facs = (0.01, 0.01, 0.01)
    path = _write_brik(tmp_path, vectors, facs=facs, dtype="<i2", gz=True)
    _check_warp(io.load(path), vectors)


def test_brik_warp_with_auxiliary_volumes(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors(6)
    labels = "x_delta~y_delta~z_delta~hexvol~BulkEn~ShearEn~"
    warp = io.load(_write_brik(tmp_path, vectors, labels=labels))
    assert type(warp) is AfniBrikWarp
    _check_warp(warp, vectors)


def test_brik_warp_ignores_obliquity(tmp_path) -> None:  # noqa: ANN001
    """The warp lives on the cardinal grid, as `IW3D_from_dataset` reads
    it, even when the header records an oblique one."""
    vectors = _vectors()
    real = CARD.copy()
    real[:3, :3] = _oblique()[:3, :3]
    path = _write_brik(tmp_path, vectors, real=real)
    _check_warp(io.load(path), vectors)


def test_brik_warp_round_trip(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    warp = io.load(_write_brik(tmp_path, vectors))
    warp.save(tmp_path / "copy_WARP+tlrc.BRIK.gz")
    again = io.load(tmp_path / "copy_WARP+tlrc.HEAD")
    assert type(again) is AfniBrikWarp
    assert again.header.labels[:3] == ["x_delta", "y_delta", "z_delta"]
    assert again.header.view == "tlrc"
    assert np.allclose(again.header.cardinal_matrix, CARD)
    assert np.allclose(np.asarray(again.dataobj), vectors)


def test_brik_warp_from_a_chain(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    warp = io.load(_write_brik(tmp_path, vectors))
    fresh = AfniBrikWarp(transformations=tuple(warp))
    fresh.save(tmp_path / "fresh_WARP+orig.HEAD")
    again = io.load(tmp_path / "fresh_WARP+orig.HEAD")
    assert again.header.view == "orig"
    _check_warp(again, vectors)


def test_to_image(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors(6)
    labels = "x_delta~y_delta~z_delta~hexvol~BulkEn~ShearEn~"
    warp = io.load(_write_brik(tmp_path, vectors, labels=labels))
    image = warp.to_image()
    assert type(image) is AfniImage
    assert image.header.labels == ["x_delta", "y_delta", "z_delta"]
    assert image.header.view == "tlrc"
    assert np.allclose(image.header.cardinal_matrix, CARD)
    assert np.asarray(image.data).dtype == np.float32
    assert np.allclose(np.asarray(image.data), vectors[..., :3])
    # saved as an image, it is read back as a warp
    image.save(tmp_path / "image+tlrc.HEAD")
    again = io.load(tmp_path / "image+tlrc.HEAD")
    assert type(again) is AfniBrikWarp
    _check_warp(again, vectors)


def test_to_image_of_a_warp_not_read_from_a_brik(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    for warp in (
        AfniBrikWarp(
            transformations=tuple(io.load(_write_brik(tmp_path, vectors)))
        ),
        io.load(_write_nifti(tmp_path, vectors)),
    ):
        image = warp.to_image()
        assert image.header.labels == ["x_delta", "y_delta", "z_delta"]
        assert np.allclose(image.header.cardinal_matrix, CARD)
        assert np.allclose(np.asarray(image.data), vectors)
        image.save(tmp_path / "image.HEAD")
        _check_warp(io.load(tmp_path / "image.HEAD"), vectors)


def test_an_oblique_grid_cannot_be_written(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    warp = io.load(_write_brik(tmp_path, vectors))
    vox2ras = LPS @ CARD
    vox2ras[:3, :3] = _oblique()[:3, :3]
    from brainhops.io.transformations.base.fields import (
        ras_displacement_chain,
    )

    chain = ras_displacement_chain(np.asarray(vectors), vox2ras)
    with pytest.raises(WriterError):
        AfniBrikWarp(transformations=chain).save(tmp_path / "x+orig.HEAD")
    assert warp is not None


# ----------------------------------------------------------------------
#   WARPS: NIFTI
# ----------------------------------------------------------------------

EXTENSION = (
    "<?xml version='1.0' ?>\n"
    '<AFNI_attributes\n  self_idcode="AFN_abc"\n'
    '  NIfTI_nums="5,4,3,1,3,16"\n  ni_form="ni_group" >\n'
    '<AFNI_atr\n  ni_type="String"\n  ni_dimen="1"\n'
    '  atr_name="TYPESTRING" >\n "3DIM_HEAD_ANAT"\n</AFNI_atr>\n'
    '<AFNI_atr\n  ni_type="String"\n  ni_dimen="1"\n'
    '  atr_name="BRICK_LABS" >\n "x_delta~y_delta~z_delta"\n</AFNI_atr>\n'
    "</AFNI_attributes>\n"
)


def _write_nifti(
    tmp_path,  # noqa: ANN001
    vectors: np.ndarray,
    extension: bool = True,
    vox2ras: np.ndarray = None,  # noqa: RUF013
    name: str = "anat_WARP.nii.gz",
    layout: str = "5d",
) -> str:
    vox2ras = LPS @ CARD if vox2ras is None else vox2ras
    data = vectors.reshape((*SHAPE, 1, -1)) if layout == "5d" else vectors
    img = nb.Nifti1Image(data, vox2ras)
    img.header.set_qform(vox2ras, code=3)
    img.header.set_sform(vox2ras, code=3)
    if extension:
        img.header.extensions.append(
            nb.nifti1.Nifti1Extension(4, EXTENSION.encode() + b"\0")
        )
    nb.save(img, str(tmp_path / name))
    return str(tmp_path / name)


def test_reads_a_nifti_warp(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    path = _write_nifti(tmp_path, vectors)
    assert sniff_xform(path) is AfniNiftiWarp
    assert io.sniff(path) is AfniNiftiWarp
    warp = io.load(path)
    assert type(warp) is AfniNiftiWarp
    _check_warp(warp, vectors)
    assert {"afni.warp", "afni.warp.nifti"} <= format_hints(AfniNiftiWarp)


def test_a_nifti_warp_without_extension_needs_a_hint(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    path = _write_nifti(tmp_path, vectors, extension=False)
    assert AfniNiftiWarp.sniff_filename(path) == Confidence.NO
    assert type(load_xform(path)) is not AfniNiftiWarp
    warp = load_xform(path, hint="afni.warp")
    assert type(warp) is AfniNiftiWarp
    _check_warp(warp, vectors)


def test_a_nifti_warp_with_bricks_on_the_time_axis(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    path = _write_nifti(tmp_path, vectors, extension=False, layout="4d")
    _check_warp(AfniNiftiWarp.from_file(path), vectors)


def test_a_nifti_warp_prefers_the_qform(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    header = nb.Nifti1Header()
    header.set_qform(LPS @ CARD, code=1)
    header.set_sform(np.diag([9.0, 9.0, 9.0, 1.0]), code=1)
    img = nb.Nifti1Image(vectors.reshape((*SHAPE, 1, 3)), None, header)
    assert img.header.get_qform(coded=True)[1] == 1
    assert not np.allclose(img.header.get_sform(), LPS @ CARD)
    img.header.extensions.append(
        nb.nifti1.Nifti1Extension(4, EXTENSION.encode() + b"\0")
    )
    nb.save(img, str(tmp_path / "q.nii"))
    _check_warp(io.load(tmp_path / "q.nii"), vectors)


def test_an_oblique_nifti_warp(tmp_path) -> None:  # noqa: ANN001
    """AFNI reads an oblique grid as a cardinal one: it keeps the
    translation, the closest orientation and the
    column lengths (`thd_niftiread.c`) -- NiTransforms'
    `_dicom_real_to_card` too."""
    vectors = _vectors()
    oblique = _oblique()
    path = _write_nifti(tmp_path, vectors, vox2ras=oblique)
    card = LPS @ _nitransforms_card(oblique)
    _check_warp(io.load(path), vectors, card=card)


def test_nifti_warp_written_as_afni_writes_it(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    warp = io.load(_write_brik(tmp_path, vectors))
    io.save(warp, tmp_path / "out_WARP.nii.gz")
    img = nb.load(str(tmp_path / "out_WARP.nii.gz"))
    assert img.shape == (*SHAPE, 1, 3)
    assert img.get_data_dtype() == np.float32
    assert int(img.header["intent_code"]) == 0
    assert np.allclose(img.affine, LPS @ CARD)
    assert np.allclose(np.asarray(img.dataobj)[..., 0, :], vectors)
    ext = [e for e in img.header.extensions if e.get_code() == 4]
    assert len(ext) == 1 and b"x_delta~y_delta~z_delta" in ext[0].get_content()
    again = io.load(tmp_path / "out_WARP.nii.gz")
    assert type(again) is AfniNiftiWarp
    _check_warp(again, vectors)
    # and back to a BRIK
    io.save(again, tmp_path / "back_WARP+tlrc.HEAD")
    _check_warp(io.load(tmp_path / "back_WARP+tlrc.HEAD"), vectors)


def test_nifti_warp_round_trip_keeps_the_extension(tmp_path) -> None:  # noqa: ANN001
    vectors = _vectors()
    warp = io.load(_write_nifti(tmp_path, vectors))
    warp.save(tmp_path / "copy.nii")
    img = nb.load(str(tmp_path / "copy.nii"))
    (ext,) = [e for e in img.header.extensions if e.get_code() == 4]
    assert b"self_idcode" in ext.get_content()
    assert img.header.get_qform(coded=True)[1] == 3


def test_a_nifti_that_is_not_a_warp_is_refused(tmp_path) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros(SHAPE, np.float32), np.eye(4))
    nb.save(img, str(tmp_path / "plain.nii"))
    with pytest.raises(ParserContentError):
        AfniNiftiWarp.from_file(tmp_path / "plain.nii")


def test_bytes_of_a_dataset_are_refused() -> None:
    with pytest.raises(ParserContentError):
        AfniBrikWarp.from_bytes(b"type = integer-attribute\n")
    assert _io is not None
