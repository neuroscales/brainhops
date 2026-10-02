"""
Tests for the plain-matrix affine reader (text, `.npy`, `.npz`, `.mat`).

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
from brainhops.io.base import arrays
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.transformations import load, sniff
from brainhops.io.transformations.matrix import MatrixAffine

data_dir = Path(__file__).parent / "data"

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


@pytest.mark.parametrize(
    "text",
    [
        "\n".join(" ".join(f"{v:.17g}" for v in row) for row in A),
        "\n".join(",".join(f"{v:.17g}" for v in row) for row in A),
        "\n".join("\t".join(f"{v:.17g}" for v in row) for row in A),
        "# a comment\n\n"
        + "\n".join(
            " ".join(f"{v:.17g}" for v in row) + "  # trailing" for row in A
        )
        + "\n",
    ],
    ids=["spaces", "commas", "tabs", "comments"],
)
def test_text(tmp_path, text) -> None:  # noqa: ANN001
    path = tmp_path / "affine.txt"
    path.write_text(text)
    xform = MatrixAffine.from_file(path)
    assert np.allclose(_homog(xform), A)
    assert xform.container == "text"


def test_text_from_string_content() -> None:
    text = "\n".join(" ".join(str(v) for v in row) for row in A)
    assert np.allclose(_homog(MatrixAffine.from_text(text)), A)
    assert np.allclose(_homog(MatrixAffine.from_bytes(text.encode())), A)


def test_npy(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.npy"
    np.save(path, A)
    xform = load(path)
    assert isinstance(xform, MatrixAffine)
    assert np.allclose(_homog(xform), A)
    assert xform.container == "npy"


def test_npy_refuses_pickles(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.npy"
    np.save(path, np.array([A], dtype=object), allow_pickle=True)
    with pytest.raises(ParserContentError):
        MatrixAffine.from_file(path)
    assert MatrixAffine.sniff(path) == Confidence.NO


def test_npz_default_and_key(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "one.npz"
    np.savez(path, affine=A)
    xform = load(path)
    assert np.allclose(_homog(xform), A)
    assert xform.variable == "affine"

    path = tmp_path / "two.npz"
    np.savez(path, fwd=A, inv=np.linalg.inv(A))
    with pytest.raises(ParserContentError):
        MatrixAffine.from_file(path)
    xform = MatrixAffine.from_file(path, variable="inv")
    assert np.allclose(_homog(xform), np.linalg.inv(A))
    xform = MatrixAffine.from_file(path, key="fwd")
    assert np.allclose(_homog(xform), A)


@pytest.mark.parametrize("fmt", ["4", "5"])
def test_mat_v4_v5(tmp_path, fmt) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    scipy.io.savemat(path, {"M": A}, format=fmt)
    xform = load(path)
    assert isinstance(xform, MatrixAffine)
    assert np.allclose(_homog(xform), A)
    assert (xform.container, xform.variable) == ("mat", "M")


def test_mat_variable_selection(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "affine.mat"
    scipy.io.savemat(path, {"M": A, "N": np.eye(4), "name": "not numeric"})
    with pytest.raises(ParserContentError):
        MatrixAffine.from_file(path)
    xform = MatrixAffine.from_file(path, variable="N")
    assert np.allclose(_homog(xform), np.eye(4))
    with pytest.raises(ParserContentError):
        MatrixAffine.from_file(path, variable="missing")


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
    assert isinstance(xform, MatrixAffine)
    # v7.3 stores arrays transposed; the reader must undo it.
    assert np.allclose(_homog(xform), A)
    assert (xform.container, xform.variable) == ("mat", "M")


def test_fileobj(tmp_path) -> None:  # noqa: ANN001
    buffer = io.BytesIO()
    np.save(buffer, A)
    buffer.seek(0)
    assert np.allclose(_homog(load(buffer)), A)


# ----------------------------------------------------------------------
#   SHAPES
# ----------------------------------------------------------------------


def test_shapes_3d() -> None:
    linear = A[:3, :3]
    assert np.allclose(_homog(MatrixAffine.from_text(_text(A[:3]))), A)
    xform = MatrixAffine.from_text(_text(linear))
    expected = np.eye(4)
    expected[:3, :3] = linear
    assert np.allclose(_homog(xform), expected)


def test_shapes_2d() -> None:
    a2 = np.array([[0.0, -1.0, 3.0], [1.0, 0.0, 4.0], [0.0, 0.0, 1.0]])
    xform = MatrixAffine.from_text(_text(a2[:2]))
    assert np.allclose(_homog(xform), a2)
    assert isinstance(xform.input, systems.CoordinateSystem2D)
    # (3, 3) is 3-D linear by default, 2-D homogeneous with ndim=2.
    assert _homog(MatrixAffine.from_text(_text(a2))).shape == (4, 4)
    xform = MatrixAffine.from_text(_text(a2), ndim=2)
    assert np.allclose(_homog(xform), a2)


@pytest.mark.parametrize("shape", [(2, 2), (4, 3), (5, 5), (1, 4), (3, 5)])
def test_bad_shapes(shape) -> None:  # noqa: ANN001
    text = _text(np.ones(shape))
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(text)
    assert MatrixAffine.sniff_text(text) == Confidence.NO


def test_projective_matrix_is_refused() -> None:
    bad = A.copy()
    bad[3, 0] = 0.5
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(_text(bad))


def _text(matrix):  # noqa: ANN001, ANN202
    return "\n".join(" ".join(repr(float(v)) for v in row) for row in matrix)


# ----------------------------------------------------------------------
#   CONVENTIONS
# ----------------------------------------------------------------------


def test_defaults() -> None:
    xform = MatrixAffine.from_text(_text(A))
    assert (xform.vector, xform.direction, xform.index_base) == (
        "column",
        "forward",
        (0, 0),
    )
    assert type(xform.input) is systems.CoordinateSystem3D
    assert type(xform.output) is systems.CoordinateSystem3D
    assert np.allclose(np.asarray(xform.raw_matrix), A)


def test_row_vector_convention() -> None:
    xform = MatrixAffine.from_text(_text(A.T), vector="row")
    assert np.allclose(_homog(xform), A)
    # a (4, 3) row-vector matrix is a (3, 4) column-vector one
    xform = MatrixAffine.from_text(_text(A[:3].T), vector="row")
    assert np.allclose(_homog(xform), A)
    assert np.allclose(np.asarray(xform.raw_matrix), A[:3].T)


def test_inverse_direction() -> None:
    xform = MatrixAffine.from_text(_text(A), direction="inverse")
    assert np.allclose(_homog(xform), np.linalg.inv(A))
    assert xform.direction == "inverse"


def test_spaces() -> None:
    xform = MatrixAffine.from_text(_text(A), input="voxel", output="ras")
    assert isinstance(xform.input, systems.VoxelCoordinateSystem)
    assert isinstance(xform.output, systems.RASCoordinateSystem)
    xform = MatrixAffine.from_text(_text(A), input="lps", output="lps")
    assert isinstance(xform.output, systems.LPSCoordinateSystem)
    custom = systems.FVoxelCoordinateSystem()
    xform = MatrixAffine.from_text(_text(A), input=custom)
    assert xform.input is custom
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(_text(A), input="nowhere")


def test_one_based_voxel_to_world() -> None:
    """A MATLAB/SPM voxel-to-world matrix maps index 1 to where a 0-based
    one maps index 0."""
    xform = MatrixAffine.from_text(
        _text(A), input="voxel", output="ras", index_base=1
    )
    assert xform.index_base == (1, 0)
    zero = _apply(_homog(xform), [0, 0, 0])
    assert np.allclose(zero, _apply(A, [1, 1, 1]))


def test_one_based_voxel_to_voxel() -> None:
    xform = MatrixAffine.from_text(
        _text(A), input="voxel", output="voxel", index_base=1
    )
    point0 = np.array([2.0, 3.0, 4.0])
    assert np.allclose(
        _apply(_homog(xform), point0), _apply(A, point0 + 1) - 1
    )
    # per-endpoint bases
    xform = MatrixAffine.from_text(
        _text(A), input="voxel", output="voxel", index_base=(0, 1)
    )
    assert np.allclose(_apply(_homog(xform), point0), _apply(A, point0) - 1)


def test_one_based_needs_a_voxel_space() -> None:
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(_text(A), index_base=1)
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(
            _text(A), input="voxel", output="ras", index_base=(1, 1)
        )


def test_conventions_compose_in_documented_order() -> None:
    """Transpose, then invert, then shift indices."""
    raw = np.linalg.inv(A).T  # row-vector, inverse
    xform = MatrixAffine.from_text(
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
    xform = MatrixAffine.from_text(_text(vox2vox), source=src, target=tgt)
    assert isinstance(xform.input, systems.RASCoordinateSystem)
    assert isinstance(xform.output, systems.RASCoordinateSystem)
    expected = tgt_aff @ vox2vox @ np.linalg.inv(src_aff)
    assert np.allclose(_homog(xform), expected)

    # an image cannot place a non-voxel endpoint
    with pytest.raises(ParserContentError):
        MatrixAffine.from_text(_text(A), input="ras", source=src)


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
    xform = MatrixAffine.from_text(_text(A), output="lps", source=image)
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
    xform = MatrixAffine.from_text(_text(A), input="voxel", output="ras")
    inverse = xform.inverse().compute()
    assert np.allclose(_homog(inverse), np.linalg.inv(A))
    assert isinstance(inverse.input, systems.RASCoordinateSystem)


# ----------------------------------------------------------------------
#   DISPATCH AND COLLISIONS
# ----------------------------------------------------------------------


def test_sniff_is_weak() -> None:
    assert MatrixAffine.sniff_text(_text(A)) == Confidence.WEAK
    assert MatrixAffine.sniff_text("not a matrix") == Confidence.NO
    assert MatrixAffine.sniff_text("1 2\n3") == Confidence.NO


@pytest.mark.parametrize("ext", [".txt", ".csv", ".tsv", ".dat"])
def test_text_extension_beats_flirt(tmp_path, ext) -> None:  # noqa: ANN001
    path = tmp_path / f"affine{ext}"
    np.savetxt(path, A, delimiter="," if ext == ".csv" else " ")
    assert sniff(path) is MatrixAffine
    assert isinstance(load(path), MatrixAffine)


def test_flirt_mat_stays_flirt(tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("nibabel")
    from brainhops.io.transformations.fsl import FLIRTTransform

    path = tmp_path / "src2ref.mat"
    np.savetxt(path, A, fmt="%.8g")
    assert sniff(path) is FLIRTTransform
    assert isinstance(load(path), FLIRTTransform)
    # ... but the matrix reader can be asked for explicitly
    xform = load(path, hint="matrix")
    assert isinstance(xform, MatrixAffine)
    assert np.allclose(_homog(xform), A, atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
def test_itk_mat_stays_itk(ndim) -> None:  # noqa: ANN001
    from brainhops.io.transformations.itk.mat import MatTransform

    path = data_dir / f"itk_affine{ndim}d_0GenericAffine.mat"
    assert MatrixAffine.sniff(path) == Confidence.NO
    assert sniff(path) is MatTransform
    assert isinstance(load(path), MatTransform)


def test_itk_tfm_as_txt_stays_itk(tmp_path) -> None:  # noqa: ANN001
    from brainhops.io.transformations.itk.tfm import TFMTransform

    path = tmp_path / "affine.txt"
    shutil.copy(data_dir / "itk_affine3d.tfm", path)
    assert MatrixAffine.sniff(path) == Confidence.NO
    assert sniff(path) is TFMTransform


def test_itk_h5_is_not_a_matrix() -> None:
    pytest.importorskip("h5py")
    assert MatrixAffine.sniff(data_dir / "itk_affine3d.h5") == Confidence.NO


def test_large_files_are_not_sniffed(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "big.npy"
    np.save(path, np.zeros((1024, 1024)))
    assert MatrixAffine.sniff(path) == Confidence.NO
