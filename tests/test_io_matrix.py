"""
Tests for the plain-matrix affine readers (text, `.npy`, `.npz`, `.mat`).

A plain matrix file stores no convention, so most tests check that each
convention keyword (vector, direction, index base, spaces, images) is
applied as documented, and the rest that the reader stays out of the way
of FLIRT and ITK files, which it could otherwise also read.
"""

# stdlib
import io
import shutil
from pathlib import Path

# dependencies
import numpy as np
import pytest
import scipy.io

from brainhops.datamodel import systems
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    TextFileParser,
)
from brainhops.io.common import _arrays as arrays
from brainhops.io.transformations import FileBasedTransformation, load, sniff
from brainhops.io.transformations.matrix import (
    CsvMatrixAffine,
    Mat73MatrixAffine,
    MatLegacyMatrixAffine,
    MatMatrixAffine,
    MatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    TsvMatrixAffine,
    TxtMatrixAffine,
)

data_dir = Path(__file__).parent / "data"

# The registered readers: one per container (and MATLAB version).
CONCRETE = (
    TxtMatrixAffine,
    CsvMatrixAffine,
    TsvMatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    MatLegacyMatrixAffine,
    Mat73MatrixAffine,
)

# A 3-D affine with a non-trivial linear part and translation.
A = np.array(
    [
        [0.9, -0.1, 0.0, 10.0],
        [0.1, 0.9, 0.2, -5.0],
        [0.0, -0.2, 1.1, 2.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _apply(homog, points):  # noqa: ANN001, ANN202
    points = np.asarray(points, float)
    ones = np.ones(points.shape[:-1] + (1,))
    return (np.concatenate([points, ones], -1) @ homog.T)[..., :-1]


def _homog(xform):  # noqa: ANN001, ANN202
    return np.asarray(xform.homogeneous_matrix)


# ----------------------------------------------------------------------
#   CONTAINERS
# ----------------------------------------------------------------------


def _rows(sep, matrix=A):  # noqa: ANN001, ANN202
    return "\n".join(sep.join(f"{v:.17g}" for v in row) for row in matrix)


@pytest.mark.parametrize(
    "cls, name, text",
    [
        (TxtMatrixAffine, "affine.txt", _rows(" ")),
        (TxtMatrixAffine, "affine.txt", _rows("\t")),
        (TxtMatrixAffine, "affine.txt", _rows(" \t  ")),
        (TxtMatrixAffine, "affine.dat", _rows(" ")),
        (TxtMatrixAffine, "affine.1D", "# AFNI-style\n" + _rows("  ")),
        (CsvMatrixAffine, "affine.csv", _rows(",")),
        (CsvMatrixAffine, "affine.csv", _rows(", ")),
        (TsvMatrixAffine, "affine.tsv", _rows("\t")),
        (
            TxtMatrixAffine,
            "affine.txt",
            "# a comment\n\n"
            + "\n".join(
                " ".join(f"{v:.17g}" for v in row) + "  # trailing"
                for row in A
            )
            + "\n",
        ),
    ],
    ids=[
        "txt-spaces",
        "txt-tabs",
        "txt-mixed",
        "dat",
        "1D",
        "csv",
        "csv-padded",
        "tsv",
        "txt-comments",
    ],
)
def test_text(tmp_path, cls, name, text) -> None:  # noqa: ANN001
    path = tmp_path / name
    path.write_text(text)
    xform = cls.from_file(path)
    assert np.allclose(_homog(xform), A)
    assert xform.CONTAINER == "text"
    # dispatch picks the same reader, from the name
    assert sniff(path) is cls
    assert type(load(path)) is cls


@pytest.mark.parametrize(
    "cls, text",
    [
        (TxtMatrixAffine, _rows(",")),
        (CsvMatrixAffine, _rows(" ")),
        (CsvMatrixAffine, _rows("\t")),
        (TsvMatrixAffine, _rows(" ")),
        (TsvMatrixAffine, _rows(",")),
    ],
)
def test_text_separator_is_strict(cls, text) -> None:  # noqa: ANN001
    """Each text reader reads its own separator only."""
    assert cls.sniff_text(text) == Confidence.NO
    with pytest.raises(ParserContentError):
        cls.from_text(text)


@pytest.mark.parametrize(
    "sep, cls",
    [(" ", TxtMatrixAffine), ("\t", TsvMatrixAffine), (",", CsvMatrixAffine)],
)
def test_unnamed_text_goes_to_one_reader(sep, cls) -> None:  # noqa: ANN001
    """Without a file name, each content has one reader: CSV needs a
    comma, TSV a tab, and whitespace text yields tab-separated content to
    TSV, so the three never tie."""
    content = _rows(sep).encode()
    scores = {c: c.sniff_bytes(content) for c in CONCRETE}
    assert {c for c, score in scores.items() if score} == {cls}
    assert type(load(content, hint="matrix")) is cls


def test_name_beats_signature(tmp_path) -> None:  # noqa: ANN001
    """A `.txt` holding tabs is still a `.txt`, and a single-column
    `.csv` (no comma) still a `.csv`: the signature is only asked for
    when the name says nothing."""
    path = tmp_path / "affine.txt"
    path.write_text(_rows("\t"))
    assert TxtMatrixAffine.sniff(path) == Confidence.LIKELY
    assert sniff(path) is TxtMatrixAffine
    assert TxtMatrixAffine.sniff_bytes(path.read_bytes()) == Confidence.NO

    path = tmp_path / "column.csv"
    path.write_text("1\n2\n3\n")
    assert arrays.CsvArrayParser.sniff(path) == Confidence.LIKELY
    assert arrays.CsvArrayParser.sniff_bytes(b"1\n2\n3\n") == Confidence.NO
    assert arrays.TxtArrayParser.sniff_bytes(b"1\n2\n3\n") > Confidence.NO


def test_text_from_string_content() -> None:
    text = "\n".join(" ".join(str(v) for v in row) for row in A)
    assert np.allclose(_homog(TxtMatrixAffine.from_text(text)), A)
    assert np.allclose(_homog(TxtMatrixAffine.from_bytes(text.encode())), A)


def test_npy(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.npy"
    np.save(path, A)
    xform = load(path)
    assert isinstance(xform, NpyMatrixAffine)
    assert np.allclose(_homog(xform), A)


def test_npy_refuses_pickles(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.npy"
    np.save(path, np.array([A], dtype=object), allow_pickle=True)
    with pytest.raises(ParserContentError):
        NpyMatrixAffine.from_file(path)
    assert NpyMatrixAffine.sniff(path) == Confidence.NO


def test_npz_default_and_key(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "one.npz"
    np.savez(path, affine=A)
    xform = load(path)
    assert isinstance(xform, NpzMatrixAffine)
    assert np.allclose(_homog(xform), A)
    assert xform.variable == "affine"

    path = tmp_path / "two.npz"
    np.savez(path, fwd=A, inv=np.linalg.inv(A))
    with pytest.raises(ParserContentError):
        NpzMatrixAffine.from_file(path)
    xform = NpzMatrixAffine.from_file(path, variable="inv")
    assert np.allclose(_homog(xform), np.linalg.inv(A))
    xform = NpzMatrixAffine.from_file(path, key="fwd")
    assert np.allclose(_homog(xform), A)


@pytest.mark.parametrize("fmt", ["4", "5"])
def test_mat_v4_v5(tmp_path, fmt) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    scipy.io.savemat(path, {"M": A}, format=fmt)
    xform = load(path)
    assert isinstance(xform, MatMatrixAffine)
    assert np.allclose(_homog(xform), A)
    assert xform.variable == "M"


def test_mat_variable_selection(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    scipy.io.savemat(path, {"M": A, "N": np.eye(4), "name": "not numeric"})
    with pytest.raises(ParserContentError):
        MatMatrixAffine.from_file(path)
    xform = MatMatrixAffine.from_file(path, variable="N")
    assert np.allclose(_homog(xform), np.eye(4))
    with pytest.raises(ParserContentError):
        MatMatrixAffine.from_file(path, variable="missing")


def _savemat73(path, **variables):  # noqa: ANN001, ANN003, ANN202
    """Write a MATLAB v7.3 file: an HDF5 file with a 512-byte header."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w", userblock_size=512) as f:
        for name, value in variables.items():
            # MATLAB is column-major: HDF5 sees the transposed array.
            dset = f.create_dataset(name, data=np.asarray(value).T)
            dset.attrs["MATLAB_class"] = np.bytes_(b"double")
    header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: test"
    header = header.ljust(116) + b"\x00" * 8 + b"\x00\x02IM"
    with open(path, "r+b") as f:
        f.write(header.ljust(512, b"\x00"))


def test_mat_v73(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    _savemat73(path, M=A)
    assert arrays.detect_container(path.read_bytes()) == "mat73"
    xform = load(path)
    assert isinstance(xform, Mat73MatrixAffine)
    # v7.3 stores arrays transposed; the reader must undo it.
    assert np.allclose(_homog(xform), A)
    assert xform.variable == "M"


def test_fileobj(tmp_path) -> None:  # noqa: ANN001
    buffer = io.BytesIO()
    np.save(buffer, A)
    buffer.seek(0)
    xform = load(buffer)
    assert isinstance(xform, NpyMatrixAffine)
    assert np.allclose(_homog(xform), A)


# ----------------------------------------------------------------------
#   SHAPES
# ----------------------------------------------------------------------


def test_shapes_3d() -> None:
    linear = A[:3, :3]
    assert np.allclose(_homog(TxtMatrixAffine.from_text(_text(A[:3]))), A)
    xform = TxtMatrixAffine.from_text(_text(linear))
    expected = np.eye(4)
    expected[:3, :3] = linear
    assert np.allclose(_homog(xform), expected)


def test_shapes_2d() -> None:
    a2 = np.array([[0.0, -1.0, 3.0], [1.0, 0.0, 4.0], [0.0, 0.0, 1.0]])
    xform = TxtMatrixAffine.from_text(_text(a2[:2]))
    assert np.allclose(_homog(xform), a2)
    assert isinstance(xform.input, systems.CoordinateSystem2D)
    # (3, 3) is 3-D linear by default, 2-D homogeneous with ndim=2.
    assert _homog(TxtMatrixAffine.from_text(_text(a2))).shape == (4, 4)
    xform = TxtMatrixAffine.from_text(_text(a2), ndim=2)
    assert np.allclose(_homog(xform), a2)


@pytest.mark.parametrize("shape", [(2, 2), (4, 3), (5, 5), (1, 4), (3, 5)])
def test_bad_shapes(shape) -> None:  # noqa: ANN001
    text = _text(np.ones(shape))
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(text)
    assert TxtMatrixAffine.sniff_text(text) == Confidence.NO


def test_projective_matrix_is_refused() -> None:
    bad = A.copy()
    bad[3, 0] = 0.5
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(_text(bad))


def _text(matrix):  # noqa: ANN001, ANN202
    return "\n".join(" ".join(repr(float(v)) for v in row) for row in matrix)


# ----------------------------------------------------------------------
#   CONVENTIONS
# ----------------------------------------------------------------------


def test_defaults() -> None:
    xform = TxtMatrixAffine.from_text(_text(A))
    assert (xform.vector, xform.direction, xform.index_base) == (
        "column",
        "forward",
        (0, 0),
    )
    assert type(xform.input) is systems.CoordinateSystem3D
    assert type(xform.output) is systems.CoordinateSystem3D
    assert np.allclose(np.asarray(xform.raw_matrix), A)


def test_row_vector_convention() -> None:
    xform = TxtMatrixAffine.from_text(_text(A.T), vector="row")
    assert np.allclose(_homog(xform), A)
    # a (4, 3) row-vector matrix is a (3, 4) column-vector one
    xform = TxtMatrixAffine.from_text(_text(A[:3].T), vector="row")
    assert np.allclose(_homog(xform), A)
    assert np.allclose(np.asarray(xform.raw_matrix), A[:3].T)


def test_inverse_direction() -> None:
    xform = TxtMatrixAffine.from_text(_text(A), direction="inverse")
    assert np.allclose(_homog(xform), np.linalg.inv(A))
    assert xform.direction == "inverse"


def test_spaces() -> None:
    xform = TxtMatrixAffine.from_text(_text(A), input="voxel", output="ras")
    assert isinstance(xform.input, systems.VoxelCoordinateSystem)
    assert isinstance(xform.output, systems.RASCoordinateSystem)
    xform = TxtMatrixAffine.from_text(_text(A), input="lps", output="lps")
    assert isinstance(xform.output, systems.LPSCoordinateSystem)
    custom = systems.FVoxelCoordinateSystem()
    xform = TxtMatrixAffine.from_text(_text(A), input=custom)
    assert xform.input is custom
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(_text(A), input="nowhere")


def test_one_based_voxel_to_world() -> None:
    """A MATLAB/SPM voxel-to-world matrix maps index 1 to where a 0-based
    one maps index 0."""
    xform = TxtMatrixAffine.from_text(
        _text(A), input="voxel", output="ras", index_base=1
    )
    assert xform.index_base == (1, 0)
    zero = _apply(_homog(xform), [0, 0, 0])
    assert np.allclose(zero, _apply(A, [1, 1, 1]))


def test_one_based_voxel_to_voxel() -> None:
    xform = TxtMatrixAffine.from_text(
        _text(A), input="voxel", output="voxel", index_base=1
    )
    point0 = np.array([2.0, 3.0, 4.0])
    assert np.allclose(
        _apply(_homog(xform), point0), _apply(A, point0 + 1) - 1
    )
    # per-endpoint bases
    xform = TxtMatrixAffine.from_text(
        _text(A), input="voxel", output="voxel", index_base=(0, 1)
    )
    assert np.allclose(_apply(_homog(xform), point0), _apply(A, point0) - 1)


def test_one_based_needs_a_voxel_space() -> None:
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(_text(A), index_base=1)
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(
            _text(A), input="voxel", output="ras", index_base=(1, 1)
        )


def test_conventions_compose_in_documented_order() -> None:
    """Transpose, then invert, then shift indices."""
    raw = np.linalg.inv(A).T  # row-vector, inverse
    xform = TxtMatrixAffine.from_text(
        _text(raw),
        vector="row",
        direction="inverse",
        input="voxel",
        output="ras",
        index_base=1,
    )
    assert np.allclose(_apply(_homog(xform), [0, 0, 0]), _apply(A, [1, 1, 1]))


def test_images_place_voxel_spaces_in_world() -> None:
    nb = pytest.importorskip("nibabel")
    src_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    src_aff[:3, 3] = [-10, -20, -30]
    tgt_aff = np.diag([-1.0, 1.0, 3.0, 1.0])
    tgt_aff[:3, 3] = [50, -5, 0]
    src = nb.Nifti1Image(np.zeros((4, 4, 4), np.float32), src_aff)
    tgt = nb.Nifti1Image(np.zeros((4, 4, 4), np.float32), tgt_aff)
    vox2vox = A
    xform = TxtMatrixAffine.from_text(_text(vox2vox), source=src, target=tgt)
    assert isinstance(xform.input, systems.RASCoordinateSystem)
    assert isinstance(xform.output, systems.RASCoordinateSystem)
    expected = tgt_aff @ vox2vox @ np.linalg.inv(src_aff)
    assert np.allclose(_homog(xform), expected)

    # an image cannot place a non-voxel endpoint
    with pytest.raises(ParserContentError):
        TxtMatrixAffine.from_text(_text(A), input="ras", source=src)


def test_brainhops_image_places_voxel_space() -> None:
    from brainhops.datamodel import transformations as xforms
    from brainhops.datamodel.images import SingleScaleImage

    vox2world = xforms.Affine(
        matrix=np.diag([2.0, 3.0, 4.0, 1.0])[:3],
        input=systems.VoxelCoordinateSystem(),
        output=systems.LPSCoordinateSystem(),
    )
    image = SingleScaleImage(
        data=np.zeros((2, 2, 2)), transformations=[vox2world]
    )
    xform = TxtMatrixAffine.from_text(_text(A), output="lps", source=image)
    assert isinstance(xform.input, systems.LPSCoordinateSystem)
    assert np.allclose(
        _homog(xform), A @ np.linalg.inv(np.diag([2.0, 3.0, 4.0, 1.0]))
    )


def test_load_passes_conventions(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.npy"
    np.save(path, A.T)
    xform = load(path, vector="row", direction="inverse")
    assert np.allclose(_homog(xform), np.linalg.inv(A))


def test_inverse_of_loaded_matrix() -> None:
    xform = TxtMatrixAffine.from_text(_text(A), input="voxel", output="ras")
    inverse = xform.inverse().compute()
    assert np.allclose(_homog(inverse), np.linalg.inv(A))
    assert isinstance(inverse.input, systems.RASCoordinateSystem)


# ----------------------------------------------------------------------
#   DISPATCH AND COLLISIONS
# ----------------------------------------------------------------------


def test_sniff_is_weak() -> None:
    assert TxtMatrixAffine.sniff_text(_text(A)) == Confidence.WEAK
    assert TxtMatrixAffine.sniff_text("not a matrix") == Confidence.NO
    assert TxtMatrixAffine.sniff_text("1 2\n3") == Confidence.NO


@pytest.mark.parametrize(
    "ext, sep, cls",
    [
        (".txt", " ", TxtMatrixAffine),
        (".dat", " ", TxtMatrixAffine),
        (".1D", " ", TxtMatrixAffine),
        (".csv", ",", CsvMatrixAffine),
        (".tsv", "\t", TsvMatrixAffine),
    ],
)
def test_text_extension_beats_flirt(tmp_path, ext, sep, cls) -> None:  # noqa: ANN001
    path = tmp_path / f"affine{ext}"
    np.savetxt(path, A, delimiter=sep)
    assert sniff(path) is cls
    assert isinstance(load(path), cls)


def test_flirt_mat_stays_flirt(tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("nibabel")
    from brainhops.io.transformations.fsl import FlirtTransform

    path = tmp_path / "src2ref.mat"
    np.savetxt(path, A, fmt="%.8g")
    assert sniff(path) is FlirtTransform
    assert isinstance(load(path), FlirtTransform)
    # ... but the matrix reader can be asked for explicitly
    xform = load(path, hint="matrix")
    assert isinstance(xform, TxtMatrixAffine)
    assert np.allclose(_homog(xform), A, atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
def test_itk_mat_stays_itk(ndim) -> None:  # noqa: ANN001
    from brainhops.io.transformations.itk.mat import MatTransform

    path = data_dir / f"itk_affine{ndim}d_0GenericAffine.mat"
    for cls in CONCRETE:
        assert cls.sniff(path) == Confidence.NO
    assert sniff(path) is MatTransform
    assert isinstance(load(path), MatTransform)


def test_itk_tfm_as_txt_stays_itk(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.itk.tfm import TfmTransform

    path = tmp_path / "affine.txt"
    shutil.copy(data_dir / "itk_affine3d.tfm", path)
    assert TxtMatrixAffine.sniff(path) == Confidence.NO
    assert sniff(path) is TfmTransform


def test_itk_h5_is_not_a_matrix() -> None:
    pytest.importorskip("h5py")
    for cls in CONCRETE:
        assert cls.sniff(data_dir / "itk_affine3d.h5") == Confidence.NO


def test_large_files_are_not_sniffed(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "big.npy"
    np.save(path, np.zeros((1024, 1024)))
    assert NpyMatrixAffine.sniff(path) == Confidence.NO


# ----------------------------------------------------------------------
#   ONE CLASS PER CONTAINER
# ----------------------------------------------------------------------


def _write_all(tmp_path):  # noqa: ANN001, ANN202
    """One file per container, each holding `A`, keyed by its reader."""
    files = {}
    path = tmp_path / "affine.txt"
    np.savetxt(path, A)
    files[TxtMatrixAffine] = path
    path = tmp_path / "affine.csv"
    np.savetxt(path, A, delimiter=",")
    files[CsvMatrixAffine] = path
    path = tmp_path / "affine.tsv"
    np.savetxt(path, A, delimiter="\t")
    files[TsvMatrixAffine] = path
    path = tmp_path / "affine.npy"
    np.save(path, A)
    files[NpyMatrixAffine] = path
    path = tmp_path / "affine.npz"
    np.savez(path, M=A)
    files[NpzMatrixAffine] = path
    path = tmp_path / "affine_v5.mat"
    scipy.io.savemat(path, {"M": A})
    files[MatLegacyMatrixAffine] = path
    path = tmp_path / "affine_v73.mat"
    _savemat73(path, M=A)
    files[Mat73MatrixAffine] = path
    return files


def test_each_class_reads_only_its_container(tmp_path) -> None:  # noqa: ANN001
    files = _write_all(tmp_path)
    for reader, path in files.items():
        for cls in CONCRETE:
            if cls is reader:
                assert cls.sniff(path) > Confidence.NO
                assert np.allclose(_homog(cls.from_file(path)), A)
            else:
                assert cls.sniff(path) == Confidence.NO, (cls, path)
                if (cls, reader) == (TxtMatrixAffine, TsvMatrixAffine):
                    # tabs are whitespace: asked to, `.txt` reads a TSV
                    # (but leaves it to TSV in dispatch)
                    cls.from_file(path)
                    continue
                with pytest.raises(ParserContentError):
                    cls.from_file(path)
        # dispatch agrees, with or without the "matrix" hint
        assert type(load(path)) is reader
        assert type(load(path, hint="matrix")) is reader


def test_mat_v4_is_not_text() -> None:
    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"M": A}, format="4")
    content = buffer.getvalue()
    assert MatLegacyMatrixAffine.sniff_bytes(content) == Confidence.WEAK
    assert MatMatrixAffine.sniff_bytes(content) == Confidence.WEAK
    for cls in (TxtMatrixAffine, CsvMatrixAffine, TsvMatrixAffine):
        assert cls.sniff_bytes(content) == Confidence.NO
        with pytest.raises(ParserContentError):
            cls.from_bytes(content)
    assert MatMatrixAffine.sniff_text(_text(A)) == Confidence.NO
    assert MatLegacyMatrixAffine.sniff_text(_text(A)) == Confidence.NO


@pytest.mark.parametrize(
    "hint, cls",
    [
        ("matrix.txt", TxtMatrixAffine),
        ("matrix.csv", CsvMatrixAffine),
        ("matrix.tsv", TsvMatrixAffine),
        ("matrix.npy", NpyMatrixAffine),
        ("matrix.npz", NpzMatrixAffine),
        ("matrix.mat", MatLegacyMatrixAffine),
        ("matrix.mat", Mat73MatrixAffine),
        ("matrix.mat.73", Mat73MatrixAffine),
        ("mat.73", Mat73MatrixAffine),
        ("affine.matrix.npy", NpyMatrixAffine),
    ],
)
def test_container_hints(tmp_path, hint, cls) -> None:  # noqa: ANN001
    files = _write_all(tmp_path)
    assert type(load(files[cls], hint=hint)) is cls
    # a container hint does not read another container
    other = files[
        TxtMatrixAffine if cls is not TxtMatrixAffine else NpyMatrixAffine
    ]
    with pytest.raises(ParserContentError):
        load(other, hint=hint)


def test_container_hint_overrides_extension(tmp_path) -> None:  # noqa: ANN001
    """A text matrix in a `.mat` file is FLIRT's unless told otherwise."""
    path = tmp_path / "affine.mat"
    np.savetxt(path, A)
    xform = load(path, hint="matrix.txt")
    assert type(xform) is TxtMatrixAffine


def test_base_is_abstract(tmp_path) -> None:  # noqa: ANN001
    assert MatrixAffine not in FileBasedTransformation._REGISTRY
    for cls in CONCRETE:
        assert cls in FileBasedTransformation._REGISTRY
        assert issubclass(cls, MatrixAffine)
    # the base reads no container
    with pytest.raises(NotImplementedError):
        MatrixAffine.from_text(_text(A))
    with pytest.raises(NotImplementedError):
        MatrixAffine.sniff_text(_text(A))
    # ... and neither does the generic array parser
    with pytest.raises(NotImplementedError):
        arrays.ArrayParser.from_bytes(b"")


def test_array_parsers_are_generic() -> None:
    """The container parsers read any numeric array, with no affine
    semantics, and declare their own extensions and hints."""

    class Raw:
        """A format that keeps the array as is."""

        @classmethod
        def from_array(cls, array, key=None, **kwargs) -> tuple:  # noqa: ANN001
            return key, array

    def raw(parser: type) -> type:
        return type("Raw" + parser.__name__, (Raw, parser), {})

    buffer = io.BytesIO()
    np.save(buffer, np.arange(24.0).reshape(2, 3, 4))
    content = buffer.getvalue()
    name, array = raw(arrays.NpyArrayParser).from_bytes(content)
    assert name is None and array.shape == (2, 3, 4)
    assert arrays.NpyArrayParser.sniff_bytes(content) == Confidence.WEAK
    # the text base is lenient about separators; its readers are not
    lines = ["1 2;3", "4,5\t6"]
    name, array = raw(arrays.TextArrayParser).from_lines(lines)
    assert array.shape == (2, 3)
    with pytest.raises(ParserContentError):
        raw(arrays.CsvArrayParser).from_lines(lines)
    name, array = raw(arrays.CsvArrayParser).from_lines(["1, 2,3", "4,5 , 6"])
    assert array.shape == (2, 3)
    expected = {
        arrays.TxtArrayParser: (("txt",), (".txt", ".dat", ".1D")),
        arrays.CsvArrayParser: (("csv",), (".csv",)),
        arrays.TsvArrayParser: (("tsv",), (".tsv",)),
        arrays.NpyArrayParser: (("npy",), (".npy",)),
        arrays.NpzArrayParser: (("npz",), (".npz",)),
        arrays.MatArrayParser: (("mat",), (".mat",)),
        arrays.Mat73ArrayParser: (("mat73", "73"), (".mat",)),
    }
    for parser, (hints, extensions) in expected.items():
        assert parser.__dict__["HINTS"] == hints
        assert parser.EXTENSIONS == extensions
    # the legacy reader adds no hint of its own
    assert "HINTS" not in arrays.MatLegacyArrayParser.__dict__
    assert arrays.MatLegacyArrayParser.HINTS == ("mat",)
    assert not arrays.TextArrayParser.EXTENSIONS
    assert "HINTS" not in arrays.TextArrayParser.__dict__


def test_text_parsers_reuse_text_plumbing() -> None:
    """The text readers are `TextFileParser`s, the binary ones
    `BinaryFileParser`s: undecodable bytes are not text."""
    for parser in (arrays.TxtArrayParser, arrays.CsvArrayParser):
        assert issubclass(parser, TextFileParser)
        assert not issubclass(parser, BinaryFileParser)
        assert parser._READ_MODE == "rt"
        assert parser.sniff_bytes(b"\xff\xfe1 2\n") == Confidence.NO
        assert parser.sniff_bytes(b"1 2\x00\n3 4") == Confidence.NO
    for parser in (arrays.NpyArrayParser, arrays.MatArrayParser):
        assert issubclass(parser, BinaryFileParser)
        assert parser._READ_MODE == "rb"


@pytest.mark.parametrize(
    "fmt, variant",
    [
        ("4", MatLegacyMatrixAffine),
        ("5", MatLegacyMatrixAffine),
        ("7.3", Mat73MatrixAffine),
    ],
)
def test_mat_dispatcher(tmp_path, fmt, variant) -> None:  # noqa: ANN001
    """`MatMatrixAffine` reads every MATLAB version: its score is the
    best of its variants', and it returns an object of the variant whose
    container the file is."""
    path = tmp_path / "affine.mat"
    if fmt == "7.3":
        _savemat73(path, M=A)
    else:
        scipy.io.savemat(path, {"M": A}, format=fmt)
    other = (set(MatMatrixAffine.VARIANTS) - {variant}).pop()
    assert MatMatrixAffine.sniff(path) == variant.sniff(path) > 0
    assert other.sniff(path) == Confidence.NO
    xform = MatMatrixAffine.from_file(path)
    assert type(xform) is variant
    assert np.allclose(_homog(xform), A)
    with pytest.raises(ParserContentError):
        other.from_file(path)
    # the generic parser dispatches the same way
    content = path.read_bytes()
    legacy, v73 = arrays.MatArrayParser.VARIANTS
    parser, other = (v73, legacy) if fmt == "7.3" else (legacy, v73)
    score = arrays.MatArrayParser.sniff_bytes(content)
    assert score == parser.sniff_bytes(content) > 0
    assert other.sniff_bytes(content) == Confidence.NO
    # dispatch never sees a tie between the dispatcher and a variant
    assert sniff(path) is variant
    assert type(load(path, hint="mat")) is variant


def test_mat_dispatcher_is_not_registered() -> None:
    """Only the variants are registered: the dispatcher would compete
    with its own subclasses for the same files."""
    assert MatMatrixAffine not in FileBasedTransformation._REGISTRY
    with pytest.raises(ParserContentError):
        MatMatrixAffine.from_text(_text(A))
    assert MatMatrixAffine.sniff_text(_text(A)) == Confidence.NO
