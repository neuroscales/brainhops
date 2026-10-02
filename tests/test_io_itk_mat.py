"""ITK binary MATLAB transforms (`.mat`), the files ANTs writes affines to.

The fixtures are `<prefix>0GenericAffine.mat` files written by SimpleITK
through `itk::MatlabTransformIO`, the writer `antsRegistration` uses, and
`tests/data/generate_itk_fixtures.py` stores the affine SimpleITK reports
for each beside it. Other variants -- big-endian, single precision,
older class names, chains -- are encoded below, byte for byte.
"""

# stdlib
from pathlib import Path

# dependencies
import numpy as np
import pytest
import typing_extensions as tx
from numpy.typing import ArrayLike

# internals
from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.base.parsers import ParserContentError
from brainhops.io.transformations import itk
from brainhops.io.transformations.itk.mat import MATTransform

data_dir = Path(__file__).parent / "data"

FILES_MAT = sorted(data_dir.glob("*GenericAffine.mat"))
NDIMS = [3, 2]


def _fixture(ndim: int) -> Path:
    return data_dir / f"itk_affine{ndim}d_0GenericAffine.mat"


def _expected(ndim: int) -> np.ndarray:
    return np.load(data_dir / f"itk_affine{ndim}d_0GenericAffine_expected.npy")


def _affine(transform: xforms.Transformation) -> np.ndarray:
    return np.asarray(transform.compute().to(xforms.Affine, lossy=True).matrix)


def _variable(
    name: str, values: ArrayLike, order: str = "<", precision: str = "f8"
) -> bytes:
    """Encode one MATLAB v4 column vector, as `vnl_matlab_write` does."""
    values = np.asarray(values, dtype=order + precision)
    mopt = (0 if order == "<" else 1000) + (0 if precision == "f8" else 10)
    raw = name.encode("ascii") + b"\0"
    header = np.array([mopt, values.size, 1, 0, len(raw)], dtype=order + "i4")
    return header.tobytes() + raw + values.tobytes()


def _parameters(ndim: int) -> tx.Tuple[np.ndarray, np.ndarray]:
    """The parameters and fixed parameters of a fixture."""
    block = MATTransform.from_file(_fixture(ndim))[0]
    return np.asarray(block.parameters), np.asarray(block.fixed_parameters)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("filename", FILES_MAT)
def test_read_mat(filename: Path) -> None:
    transform = MATTransform.from_file(filename)
    assert isinstance(transform, itk.ITKTransform)
    for block in transform.transformations:
        assert isinstance(block, itk.ITKStruct)
        assert isinstance(block, xforms.Sequence)
        assert all(isinstance(t, xforms.Transformation) for t in block)


@pytest.mark.parametrize("ndim", NDIMS)
def test_generic_affine_is_the_affine_itk_reads(ndim: int) -> None:
    transform = io.load(_fixture(ndim))
    (block,) = transform.transformations
    assert block.type == itk.ITKTransformClass.AffineTransform
    assert block.ndim_input == block.ndim_output == ndim
    np.testing.assert_allclose(np.asarray(block.center), [4, 5, 6][:ndim])

    matrix = _affine(transform)
    assert matrix.shape == (ndim, ndim + 1)
    np.testing.assert_allclose(matrix, _expected(ndim), atol=1e-12)


def test_generic_affine_fixtures_still_match_simpleitk() -> None:
    """The stored expectations are still what ITK itself reports."""
    sitk = pytest.importorskip("SimpleITK")
    for ndim in NDIMS:
        transform = sitk.ReadTransform(str(_fixture(ndim)))
        expected = _expected(ndim)
        np.testing.assert_allclose(
            expected[:, :ndim],
            np.asarray(transform.GetMatrix()).reshape(ndim, ndim),
        )
        np.testing.assert_allclose(
            expected[:, ndim], transform.TransformPoint([0.0] * ndim)
        )


@pytest.mark.parametrize("ndim", NDIMS)
def test_big_endian_and_single_precision_read_the_same(ndim: int) -> None:
    """`vnl_matlab_write` writes in the native byte order of the machine,
    and a `float` transform in single precision."""
    parameters, fixed = _parameters(ndim)
    name = f"AffineTransform_double_{ndim}_{ndim}"
    for order in "<>":
        content = _variable(name, parameters, order) + _variable(
            "fixed", fixed, order
        )
        transform = MATTransform.from_bytes(content)
        np.testing.assert_allclose(_affine(transform), _expected(ndim))

    name = f"AffineTransform_float_{ndim}_{ndim}"
    content = _variable(name, parameters, ">", "f4") + _variable(
        "fixed", fixed, ">", "f4"
    )
    (block,) = MATTransform.from_bytes(content).transformations
    assert block.precision == itk.ITKPrecision.Float
    np.testing.assert_allclose(_affine(block), _expected(ndim), rtol=1e-6)


def test_matrix_offset_transform_base_is_an_affine() -> None:
    """Older ANTs releases name their affines after the base class."""
    parameters, fixed = _parameters(3)
    content = _variable(
        "MatrixOffsetTransformBase_double_3_3", parameters
    ) + _variable("fixed", fixed)
    (block,) = MATTransform.from_bytes(content).transformations
    assert block.type == itk.ITKTransformClass.MatrixOffsetTransformBase
    np.testing.assert_allclose(_affine(block), _expected(3))


def test_a_chain_keeps_every_block_in_order() -> None:
    """A chain repeats the `fixed` name, so it cannot be read as a
    mapping of names to values. The composite's own pair is skipped."""
    parameters, fixed = _parameters(3)
    translation = [10.0, 20.0, 30.0]
    content = b"".join(
        [
            _variable("CompositeTransform_double_3_3", translation),
            _variable("fixed", []),
            _variable("AffineTransform_double_3_3", parameters),
            _variable("fixed", fixed),
            _variable("TranslationTransform_double_3_3", translation),
            _variable("fixed", []),
        ]
    )
    transform = MATTransform.from_bytes(content)
    affine, shift = transform.transformations
    assert affine.type == itk.ITKTransformClass.AffineTransform
    assert shift.type == itk.ITKTransformClass.TranslationTransform
    np.testing.assert_allclose(_affine(affine), _expected(3))
    np.testing.assert_allclose(np.asarray(shift.parameters), translation)


def test_a_truncated_file_is_refused() -> None:
    content = _fixture(3).read_bytes()
    with pytest.raises(ParserContentError):
        MATTransform.from_bytes(content[:-8])


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_dispatched(filename: Path) -> None:
    assert io.transformations.sniff(filename) is MATTransform
    assert io.sniff(filename) is MATTransform
    assert type(io.transformations.load(filename)) is MATTransform
    assert type(io.load(filename)) is MATTransform


@pytest.mark.parametrize("hint", ["itk", "ants", "mat", "itk.mat", "ants.mat"])
@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_selected_by_its_hints(filename: Path, hint: str) -> None:
    assert io.sniff(filename, hint=hint) is MATTransform
    assert type(io.load(filename, hint=hint)) is MATTransform


@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_dispatched_from_an_open_file(filename: Path) -> None:
    with open(filename, "rb") as f:
        assert type(io.load(f)) is MATTransform


def test_sniffer_claims_only_itk_matlab_files() -> None:
    content = _fixture(3).read_bytes()
    assert MATTransform.sniff_bytes(content) == 1.0
    # A MATLAB v4 variable that is not named after an ITK class.
    assert MATTransform.sniff_bytes(_variable("data", [1.0, 2.0])) == 0.0
    # A MATLAB v5 file starts with a text header.
    v5 = b"MATLAB 5.0 MAT-file, Platform: GLNXA64".ljust(128, b" ")
    assert MATTransform.sniff_bytes(v5) == 0.0
    # A FLIRT matrix is text.
    flirt = b"1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    assert MATTransform.sniff_bytes(flirt) == 0.0
    assert MATTransform.sniff_bytes(b"") == 0.0
    assert MATTransform.sniff_text("1 0 0 0") == 0.0


def test_flirt_and_itk_mat_files_go_to_their_own_readers(tmp_path) -> None:  # noqa: ANN001
    """Both formats use `.mat`, so only their content tells them apart."""
    pytest.importorskip("nibabel")
    from brainhops.io.transformations.fsl.flirt import FLIRTTransform

    flirt = tmp_path / "src2ref.mat"
    np.savetxt(str(flirt), np.eye(4), fmt="%.8g")
    ants = tmp_path / "out0GenericAffine.mat"
    ants.write_bytes(_fixture(3).read_bytes())

    # Each sniffer scores the other format's file as a firm "no".
    assert FLIRTTransform.sniff(ants) == 0.0
    assert FLIRTTransform.sniff_bytes(ants.read_bytes()) == 0.0
    assert MATTransform.sniff(flirt) == 0.0
    assert FLIRTTransform.sniff(flirt) > 0.0
    assert MATTransform.sniff(ants) > 0.0

    assert io.transformations.sniff(flirt) is FLIRTTransform
    assert io.transformations.sniff(ants) is MATTransform
    assert type(io.transformations.load(flirt)) is FLIRTTransform
    assert type(io.transformations.load(ants)) is MATTransform
