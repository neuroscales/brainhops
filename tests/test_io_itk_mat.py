"""Tests for ITK binary MATLAB transforms (.mat), as written by ANTs.

The <prefix>0GenericAffine.mat fixtures were written by SimpleITK, and
tests/data/generate_itk_fixtures.py stores the affine that SimpleITK
reports beside each one. Other variants are encoded byte for byte here.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx
from numpy.typing import ArrayLike

from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations import itk
from brainhops.io.transformations.itk.mat import MatTransform

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
    name: str,
    values: ArrayLike,
    order: str = "<",
    precision: str = "f8",
    rowwise: bool = False,
    cols: int = 1,
) -> bytes:
    """Encode one MATLAB v4 variable as vnl_matlab_write does."""
    values = np.asarray(values, dtype=order + precision)
    mopt = (
        (0 if order == "<" else 1000)
        + (100 if rowwise else 0)
        + (0 if precision == "f8" else 10)
    )
    raw = name.encode("ascii") + b"\0"
    rows = values.size // cols
    header = np.array([mopt, rows, cols, 0, len(raw)], dtype=order + "i4")
    return header.tobytes() + raw + values.tobytes()


def _parameters(ndim: int) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Return the parameters and fixed parameters of a fixture."""
    block = MatTransform.from_file(_fixture(ndim))[0]
    return np.asarray(block.parameters), np.asarray(block.fixed_parameters)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("filename", FILES_MAT)
def test_read_mat(filename: Path) -> None:
    transform = MatTransform.from_file(filename)
    assert isinstance(transform, itk.ItkTransform)
    for block in transform.transformations:
        assert isinstance(block, itk.ItkStruct)
        assert isinstance(block, xforms.Sequence)
        assert all(isinstance(t, xforms.Transformation) for t in block)


@pytest.mark.parametrize("ndim", NDIMS)
def test_generic_affine_is_the_affine_itk_reads(ndim: int) -> None:
    transform = io.load(_fixture(ndim))
    (block,) = transform.transformations
    assert block.type == itk.ItkTransformClass.AffineTransform
    assert block.ndim_input == block.ndim_output == ndim
    np.testing.assert_allclose(np.asarray(block.center), [4, 5, 6][:ndim])

    matrix = _affine(transform)
    assert matrix.shape == (ndim, ndim + 1)
    np.testing.assert_allclose(matrix, _expected(ndim), atol=1e-12)


def test_generic_affine_fixtures_still_match_simpleitk() -> None:
    """The stored expectations are still what ITK reports."""
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
    """Big-endian and float files, as other machines and types write them."""
    parameters, fixed = _parameters(ndim)
    name = f"AffineTransform_double_{ndim}_{ndim}"
    for order in "<>":
        content = _variable(name, parameters, order) + _variable(
            "fixed", fixed, order
        )
        transform = MatTransform.from_bytes(content)
        np.testing.assert_allclose(_affine(transform), _expected(ndim))

    name = f"AffineTransform_float_{ndim}_{ndim}"
    content = _variable(name, parameters, ">", "f4") + _variable(
        "fixed", fixed, ">", "f8"
    )
    (block,) = MatTransform.from_bytes(content).transformations
    assert block.precision == itk.ItkPrecision.Float
    np.testing.assert_allclose(_affine(block), _expected(ndim), rtol=1e-6)


def test_matrix_offset_transform_base_is_an_affine() -> None:
    """Older ANTs releases name affines after the base class."""
    parameters, fixed = _parameters(3)
    content = _variable(
        "MatrixOffsetTransformBase_double_3_3", parameters
    ) + _variable("fixed", fixed)
    (block,) = MatTransform.from_bytes(content).transformations
    assert block.type == itk.ItkTransformClass.MatrixOffsetTransformBase
    np.testing.assert_allclose(_affine(block), _expected(3))


def test_a_composite_keeps_every_block_in_application_order() -> None:
    """The composite pair is skipped and the last block applies first."""
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
    transform = MatTransform.from_bytes(content)
    shift, affine = transform.transformations
    assert affine.type == itk.ItkTransformClass.AffineTransform
    assert shift.type == itk.ItkTransformClass.TranslationTransform
    np.testing.assert_allclose(_affine(affine), _expected(3))
    np.testing.assert_allclose(np.asarray(shift.parameters), translation)


def test_a_plain_list_reads_the_transform_at_position() -> None:
    """Without a composite pair, the transform at `position` is read."""
    from brainhops.io.base.parsers import ParserContentError

    parameters, fixed = _parameters(3)
    translation = [10.0, 20.0, 30.0]
    content = b"".join(
        [
            _variable("AffineTransform_double_3_3", parameters),
            _variable("fixed", fixed),
            _variable("TranslationTransform_double_3_3", translation),
            _variable("fixed", []),
        ]
    )
    with pytest.warns(UserWarning, match="holds 2 transforms"):
        (affine,) = MatTransform.from_bytes(content).transformations
    np.testing.assert_allclose(_affine(affine), _expected(3))
    (shift,) = MatTransform.from_bytes(content, position=1).transformations
    np.testing.assert_allclose(np.asarray(shift.parameters), translation)
    with pytest.raises(ParserContentError, match="no transform 2"):
        MatTransform.from_bytes(content, position=2)


def test_variables_are_read_in_pairs_as_itk_reads_them() -> None:
    """The variable after the parameters is fixed whatever its name."""
    parameters, fixed = _parameters(3)
    content = _variable(
        "AffineTransform_double_3_3", parameters, rowwise=True
    ) + _variable("center", fixed, rowwise=True)
    (block,) = MatTransform.from_bytes(content).transformations
    np.testing.assert_allclose(_affine(block), _expected(3))


def test_encoded_variants_read_as_itk_reads_them(tmp_path) -> None:  # noqa: ANN001
    """ITK reads every encoded variant to the same mapping."""
    sitk = pytest.importorskip("SimpleITK")
    parameters, fixed = _parameters(3)
    name = "AffineTransform_double_3_3"
    variants = [
        _variable(name, parameters, ">") + _variable("fixed", fixed, ">"),
        _variable("AffineTransform_float_3_3", parameters, ">", "f4")
        + _variable("fixed", fixed, ">"),
        _variable(name, parameters, rowwise=True)
        + _variable("center", fixed, rowwise=True),
        _variable("MatrixOffsetTransformBase_double_3_3", parameters)
        + _variable("fixed", fixed),
    ]
    points = np.random.default_rng(0).normal(size=(5, 3)) * 10
    for index, content in enumerate(variants):
        path = tmp_path / f"variant{index}.mat"
        path.write_bytes(content)
        transform = sitk.ReadTransform(str(path))
        matrix = _affine(MatTransform.from_file(path))
        for point in points:
            np.testing.assert_allclose(
                transform.TransformPoint(tuple(point)),
                matrix[:, :3] @ point + matrix[:, 3],
                rtol=1e-6,
            )


def test_what_itk_refuses_is_refused() -> None:
    content = _fixture(3).read_bytes()
    parameters, fixed = _parameters(3)
    name = "AffineTransform_double_3_3"
    refused = [
        # The values are truncated.
        content[:-8],
        # The fixed parameters are missing, but ITK reads variables in pairs.
        _variable(name, parameters),
        # The variables are not column vectors, the only shape that ITK reads.
        _variable(name, parameters, cols=12) + _variable("fixed", fixed),
        _variable(name, parameters, cols=2) + _variable("fixed", fixed),
    ]
    for content in refused:
        with pytest.raises(ParserContentError):
            MatTransform.from_bytes(content)


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_dispatched(filename: Path) -> None:
    assert io.transformations.sniff(filename) is MatTransform
    assert io.sniff(filename) is MatTransform
    assert type(io.transformations.load(filename)) is MatTransform
    assert type(io.load(filename)) is MatTransform


@pytest.mark.parametrize("hint", ["itk", "ants", "mat", "itk.mat", "ants.mat"])
@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_selected_by_its_hints(filename: Path, hint: str) -> None:
    assert io.sniff(filename, hint=hint) is MatTransform
    assert type(io.load(filename, hint=hint)) is MatTransform


@pytest.mark.parametrize("filename", FILES_MAT)
def test_mat_is_dispatched_from_an_open_file(filename: Path) -> None:
    with open(filename, "rb") as f:
        assert type(io.load(f)) is MatTransform


def test_sniffer_claims_only_itk_matlab_files() -> None:
    content = _fixture(3).read_bytes()
    assert MatTransform.sniff_bytes(content) == 1.0
    # A MATLAB v4 variable that is not named after an ITK class is refused.
    assert MatTransform.sniff_bytes(_variable("data", [1.0, 2.0])) == 0.0
    # MATLAB v5 starts with a text header.
    v5 = b"MATLAB 5.0 MAT-file, Platform: GLNXA64".ljust(128, b" ")
    assert MatTransform.sniff_bytes(v5) == 0.0
    # A FLIRT matrix is text.
    flirt = b"1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    assert MatTransform.sniff_bytes(flirt) == 0.0
    assert MatTransform.sniff_bytes(b"") == 0.0
    assert MatTransform.sniff_text("1 0 0 0") == 0.0


def test_flirt_and_itk_mat_files_go_to_their_own_readers(tmp_path) -> None:  # noqa: ANN001
    """FLIRT and ITK files share .mat and differ only by content."""
    pytest.importorskip("nibabel")
    from brainhops.io.transformations.fsl.flirt import FlirtTransform

    flirt = tmp_path / "src2ref.mat"
    np.savetxt(str(flirt), np.eye(4), fmt="%.8g")
    ants = tmp_path / "out0GenericAffine.mat"
    ants.write_bytes(_fixture(3).read_bytes())

    # Each sniffer gives the file of the other a firm no.
    assert FlirtTransform.sniff(ants) == 0.0
    assert FlirtTransform.sniff_bytes(ants.read_bytes()) == 0.0
    assert MatTransform.sniff(flirt) == 0.0
    assert FlirtTransform.sniff(flirt) > 0.0
    assert MatTransform.sniff(ants) > 0.0

    assert io.transformations.sniff(flirt) is FlirtTransform
    assert io.transformations.sniff(ants) is MatTransform
    assert type(io.transformations.load(flirt)) is FlirtTransform
    assert type(io.transformations.load(ants)) is MatTransform


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _random_affine(ndim: int) -> np.ndarray:
    rng = np.random.default_rng(ndim)
    linear = np.eye(ndim) + rng.normal(size=(ndim, ndim)) * 0.1
    shift = rng.normal(size=(ndim, 1)) * 10
    return np.concatenate([linear, shift], axis=1)


def _from_affine(matrix: np.ndarray) -> MatTransform:
    return MatTransform([xforms.Affine(matrix)])


@pytest.mark.parametrize("ndim", NDIMS)
def test_a_fixture_is_written_back_byte_for_byte(ndim: int) -> None:
    """A block read from a file keeps its class and center."""
    content = _fixture(ndim).read_bytes()
    assert MatTransform.from_bytes(content).to_bytes() == content


@pytest.mark.parametrize("ndim", NDIMS)
def test_an_affine_round_trips(ndim: int, tmp_path) -> None:  # noqa: ANN001
    matrix = _random_affine(ndim)
    path = tmp_path / "out0GenericAffine.mat"
    _from_affine(matrix).save(path)
    transform = io.load(path)
    assert type(transform) is MatTransform
    (block,) = transform.transformations
    assert block.type == itk.ItkTransformClass.AffineTransform
    assert block.precision == itk.ItkPrecision.Double
    # A brainhops affine has no center, so the ITK center is the origin.
    np.testing.assert_array_equal(np.asarray(block.fixed_parameters), 0)
    np.testing.assert_allclose(_affine(transform), matrix, atol=1e-12)


@pytest.mark.parametrize("ndim", NDIMS)
@pytest.mark.parametrize("precision", ["double", "float"])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_scipy_reads_what_is_written(
    ndim: int, precision: str, byteorder: str
) -> None:
    """scipy.io.loadmat reads both variables at their precision."""
    from io import BytesIO

    scipy_io = pytest.importorskip("scipy.io")
    matrix = _random_affine(ndim)
    content = _from_affine(matrix).to_bytes(
        byteorder=byteorder, precision=precision
    )
    variables = scipy_io.loadmat(BytesIO(content))
    name = f"AffineTransform_{precision}_{ndim}_{ndim}"
    assert {name, "fixed"} <= set(variables)
    parameters, fixed = variables[name], variables["fixed"]
    # scipy keeps the byte order of the file.
    itemsize = 8 if precision == "double" else 4
    assert parameters.shape == (ndim * (ndim + 1), 1)
    assert (parameters.dtype.kind, parameters.dtype.itemsize) == (
        "f",
        itemsize,
    )
    assert fixed.shape == (ndim, 1)
    assert (fixed.dtype.kind, fixed.dtype.itemsize) == ("f", 8)
    np.testing.assert_array_equal(fixed, 0)
    np.testing.assert_allclose(
        parameters[: ndim * ndim, 0].reshape(ndim, ndim),
        matrix[:, :-1],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        parameters[ndim * ndim :, 0], matrix[:, -1], rtol=1e-6
    )

    # The reader reads the file back the same.
    (block,) = MatTransform.from_bytes(content).transformations
    assert block.precision == precision
    np.testing.assert_allclose(_affine(block), matrix, rtol=1e-6, atol=1e-6)


def test_the_default_encoding_is_little_endian_double() -> None:
    content = _from_affine(_random_affine(3)).to_bytes()
    mopt, rows, cols, imagf, namlen = np.frombuffer(content, "<i4", count=5)
    assert (mopt, rows, cols, imagf) == (0, 12, 1, 0)
    assert content[20 : 20 + namlen] == b"AffineTransform_double_3_3\0"


@pytest.mark.parametrize("precision", ["double", "float"])
def test_a_centered_block_keeps_its_center(precision: str) -> None:
    """A centered block is written with its center as given."""
    parameters, fixed = _parameters(3)
    block = itk.ItkStruct(
        type="AffineTransform",
        precision=precision,
        ndim_input=3,
        ndim_output=3,
        parameters=parameters,
        fixed_parameters=fixed,
    )
    content = MatTransform(transformations=[block]).to_bytes()
    (back,) = MatTransform.from_bytes(content).transformations
    assert back.precision == precision
    np.testing.assert_allclose(np.asarray(back.fixed_parameters), fixed)
    np.testing.assert_allclose(_affine(back), _expected(3), rtol=1e-6)


def test_io_save_writes_a_mat_transform(tmp_path) -> None:  # noqa: ANN001
    transform = io.load(_fixture(3))
    path = tmp_path / "copy0GenericAffine.mat"
    io.save(transform, path)
    assert path.read_bytes() == _fixture(3).read_bytes()


def test_a_concrete_transformation_is_written_as_an_affine() -> None:
    shift = xforms.Translation([1.0, 2.0, 3.0])
    content = MatTransform([shift]).to_bytes()
    (block,) = MatTransform.from_bytes(content).transformations
    assert block.type == itk.ItkTransformClass.AffineTransform
    expected = np.concatenate([np.eye(3), [[1.0], [2.0], [3.0]]], axis=1)
    np.testing.assert_allclose(_affine(block), expected)


def test_what_itk_cannot_hold_is_refused(tmp_path) -> None:  # noqa: ANN001
    from brainhops.datamodel import systems

    matrix = _random_affine(3)
    refused = [
        # The affine is not in the space that ITK works in.
        MatTransform(
            [
                xforms.Affine(
                    matrix, input=systems.RASmm(), output=systems.RASmm()
                )
            ]
        ),
        # The matrix is not square.
        MatTransform([xforms.Affine(np.ones((2, 4)))]),
        # The file holds a chain of blocks, or no block at all.
        MatTransform(
            transformations=[xforms.Affine(matrix), xforms.Affine(matrix)]
        ),
        MatTransform(transformations=[]),
    ]
    path = tmp_path / "refused.mat"
    for transform in refused:
        with pytest.raises(UnrepresentableTransformationError):
            transform.save(path)
    # The content is built before the file is opened.
    assert not path.exists()


def test_written_files_are_read_by_itk(tmp_path) -> None:  # noqa: ANN001
    sitk = pytest.importorskip("SimpleITK")
    for ndim in NDIMS:
        for precision in ("double", "float"):
            matrix = _random_affine(ndim)
            path = tmp_path / f"out{ndim}{precision}.mat"
            _from_affine(matrix).save(path, precision=precision)
            transform = sitk.ReadTransform(str(path))
            point = np.arange(ndim, dtype=float) + 1
            np.testing.assert_allclose(
                transform.TransformPoint(tuple(point)),
                matrix[:, :-1] @ point + matrix[:, -1],
                rtol=1e-5,
            )


@pytest.mark.parametrize("ndim", NDIMS)
def test_ants_use_inverse_is_the_inverse(ndim: int) -> None:
    """An entry [file.mat,1] in an ANTs list is the inverse of the file."""
    inverse = ~io.load(_fixture(ndim))
    homogeneous = np.eye(ndim + 1)
    homogeneous[:ndim] = _expected(ndim)
    np.testing.assert_allclose(
        _affine(inverse), np.linalg.inv(homogeneous)[:ndim], atol=1e-12
    )
