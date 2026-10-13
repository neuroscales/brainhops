"""Tests of the affine matrices of Greedy, Convert3D and ITK-SNAP.

The files below are written in the layouts of the programs. Greedy prints
a `vnl_matrix`, which puts a space after every number of a row, while
`c3d_affine_tool -o` and ITK-SNAP end each row without one. All three
print six significant digits.
"""

import numpy as np
import pytest

from brainhops.datamodel.systems import LPSmm, RASmm
from brainhops.datamodel.transformations import Affine
from brainhops.io.base.parsers import (
    ParserContentError,
    SnifferContentError,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.transformations import load
from brainhops.io.transformations.c3d import (
    C3dAffine,
    C3dAffineMetadata,
    C3dAffineRaw,
)
from brainhops.io.transformations.fsl.flirt import FlirtTransform
from brainhops.io.transformations.matrix import TxtMatrixAffine

GREEDY = (
    "0.998512 0.0332118 -0.0429862 1.52817 \n"
    "-0.0291645 0.995731 0.0874301 -3.17266 \n"
    "0.0457061 -0.0860369 0.995243 7.04213 \n"
    "0 0 0 1 \n"
)
"""A matrix in the layout of Greedy, with a space after every number."""

C3D = (
    "0.998512 0.0332118 -0.0429862 1.52817\n"
    "-0.0291645 0.995731 0.0874301 -3.17266\n"
    "0.0457061 -0.0860369 0.995243 7.04213\n"
    "0 0 0 1\n"
)
"""The same matrix in the layout of `c3d_affine_tool` and ITK-SNAP."""

MATRIX = np.array(
    [
        [0.998512, 0.0332118, -0.0429862, 1.52817],
        [-0.0291645, 0.995731, 0.0874301, -3.17266],
        [0.0457061, -0.0860369, 0.995243, 7.04213],
    ]
)


def _write(tmp_path, text: str, name: str = "affine.mat"):  # noqa: ANN001, ANN202
    path = tmp_path / name
    path.write_bytes(text.encode())
    return path


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("text", [GREEDY, C3D])
@pytest.mark.parametrize("hint", ["c3d", "greedy", "itksnap"])
def test_a_hint_selects_the_format(tmp_path, text: str, hint: str) -> None:  # noqa: ANN001
    xform = load(_write(tmp_path, text), hint=hint)
    assert type(xform) is C3dAffine
    assert xform.input == RASmm()
    assert xform.output == RASmm()
    assert np.array_equal(xform.matrix, MATRIX)


def test_the_layout_of_the_rows_is_read(tmp_path) -> None:  # noqa: ANN001
    assert C3dAffine.load(_write(tmp_path, GREEDY)).metadata.raw.row_end == " "
    assert C3dAffine.load(_write(tmp_path, C3D)).metadata.raw.row_end == ""


def test_the_content_never_selects_the_format(tmp_path) -> None:  # noqa: ANN001
    # A plain matrix holds no marker, so the readers that claimed it before
    # still do: FLIRT for a `.mat` file and the plain matrix for a `.txt`.
    mat = _write(tmp_path, GREEDY, "affine.mat")
    txt = _write(tmp_path, C3D, "affine.txt")
    assert C3dAffine.sniff(mat) == 0
    assert C3dAffineMetadata.sniff(mat) == 0
    with pytest.raises(SnifferContentError):
        C3dAffine.sniff(mat, error=True)
    assert type(load(mat)) is FlirtTransform
    assert type(load(txt)) is TxtMatrixAffine
    # A hint selects the format whatever the name of the file.
    assert type(load(txt, hint="greedy")) is C3dAffine


def test_any_whitespace_separates_the_numbers(tmp_path) -> None:  # noqa: ANN001
    # `c3d_affine_tool`, Greedy and ITK-SNAP read sixteen numbers with
    # `operator>>`, whatever separates them.
    text = " ".join(C3D.split()) + "\n"
    path = _write(tmp_path, text)
    xform = C3dAffine.load(path)
    assert np.array_equal(xform.matrix, MATRIX)
    target = tmp_path / "copy.mat"
    xform.save(target)
    assert target.read_bytes() == path.read_bytes()


@pytest.mark.parametrize(
    "text",
    [
        "1 0 0\n0 1 0\n0 0 1\n",
        C3D + "0\n",
        C3D.replace("0 0 0 1", "0 0 0.5 1"),
        C3D.replace("0.995731", "nope"),
        "",
    ],
    ids=["2-D", "17 numbers", "projective", "word", "empty"],
)
def test_a_file_that_is_not_a_3d_affine_is_refused(
    tmp_path,  # noqa: ANN001
    text: str,
) -> None:
    with pytest.raises(ParserContentError):
        C3dAffine.load(_write(tmp_path, text))


def test_the_reader_and_the_writer_take_no_option(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path, C3D)
    with pytest.raises(TypeError):
        C3dAffine.load(path, digits=6)
    with pytest.raises(TypeError):
        C3dAffineMetadata.load(path, digits=6)
    with pytest.raises(TypeError):
        C3dAffine.load(path).to_text(digits=6)


# ----------------------------------------------------------------------
#   CONVENTIONS
# ----------------------------------------------------------------------


def test_an_itk_affine_is_flipped_as_c3d_affine_tool_flips_it() -> None:
    # `itk_read` of `c3d_affine_tool` and `MapITKTransformToRASMatrix` of
    # Greedy negate these six entries of the ITK matrix and offset, which
    # is the conjugation of the matrix by diag(-1, -1, 1, 1).
    itk = np.arange(1.0, 13.0).reshape(3, 4)
    ras = itk.copy()
    for row, col in ((2, 0), (2, 1), (0, 2), (1, 2), (0, 3), (1, 3)):
        ras[row, col] *= -1
    xform = Affine(itk, input=LPSmm(), output=LPSmm()).to(C3dAffine)
    assert type(xform) is C3dAffine
    assert np.allclose(xform.matrix, ras)


def test_an_affine_without_systems_is_taken_as_ras() -> None:
    xform = Affine(MATRIX).to(C3dAffine)
    assert np.array_equal(xform.matrix, MATRIX)
    assert xform.metadata is None


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("text", [GREEDY, C3D])
def test_an_untouched_file_is_written_byte_for_byte(
    tmp_path,  # noqa: ANN001
    text: str,
) -> None:
    path = _write(tmp_path, text)
    xform = C3dAffine.load(path)
    assert xform.to_text() == text
    target = tmp_path / "copy.mat"
    xform.save(target)
    assert target.read_bytes() == path.read_bytes()


def test_a_new_matrix_keeps_the_layout_of_the_file(tmp_path) -> None:  # noqa: ANN001
    xform = C3dAffine.load(_write(tmp_path, GREEDY))
    record = xform.metadata.raw
    xform.data = np.diag([2.0, 3.0, 4.0, 1.0])[:3]
    # The assigned matrix is pending, and the record is unchanged.
    assert xform.metadata.raw is record
    assert xform.to_text() == ("2 0 0 0 \n0 3 0 0 \n0 0 4 0 \n0 0 0 1 \n")
    assert xform.metadata.raw is record


def test_numbers_are_written_with_full_precision() -> None:
    value = np.array(MATRIX)
    value[0, 3] = 0.1 + 0.2
    text = C3dAffine(matrix=value).to_text()
    assert text.splitlines()[0].split()[3] == repr(0.1 + 0.2)
    assert text.endswith("0 0 0 1\n")
    assert np.array_equal(C3dAffine.from_text(text).matrix, value)


def test_a_decoded_matrix_is_read_only(tmp_path) -> None:  # noqa: ANN001
    xform = C3dAffine.load(_write(tmp_path, C3D))
    with pytest.raises(ValueError):
        xform.data[0, 0] = 2.0


def test_a_record_with_a_stale_text_is_formatted_again() -> None:
    record = C3dAffineRaw.from_text(C3D)
    edited = C3dAffineRaw(
        matrix=((2.0, 0.0, 0.0, 0.0),) + record.matrix[1:],
        row_end=record.row_end,
        text=record.text,
    )
    assert edited.to_text().startswith("2 0 0 0\n")


def test_systems_that_are_not_ras_are_refused(tmp_path) -> None:  # noqa: ANN001
    xform = C3dAffine.load(_write(tmp_path, C3D))
    xform.input = LPSmm()
    with pytest.raises(UnrepresentableTransformationError):
        xform.save(tmp_path / "lps.mat")
    assert not (tmp_path / "lps.mat").exists()


def test_metadata_without_a_record_is_not_written() -> None:
    with pytest.raises(WriterError):
        C3dAffineMetadata().to_text()
