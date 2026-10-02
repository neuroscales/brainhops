"""
NiftyReg transformations: `reg_aladin` affines, and the NIfTI fields and
control-point grids of `reg_f3d` and `reg_transform`.

NiftyReg is not a Python dependency, so the fixtures are written with
`nibabel` in the layout NiftyReg writes (`reg_createControlPointGrid`,
`reg_createDeformationField`), and the expected maps are computed by
NumPy ports of NiftyReg's own kernels: `reg_cubic_spline_getDeformationField3D`
(composition branch, with `get_GridValues` sliding beyond the grid),
`reg_linear_spline_getDeformationField3D`, and `reg_defField_compose3D`.
"""

import io as _io
from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as bio  # noqa: E402
from brainhops.datamodel import systems  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.base._dispatch import format_hints  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.transformations.base.affines import RASToRAS  # noqa: E402
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ItkNiftiDisplacementField,
)
from brainhops.io.transformations.matrix import TxtMatrixAffine  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
)
from brainhops.io.transformations.niftyreg import (  # noqa: E402
    NiftyRegAffine,
    NiftyRegControlPointGrid,
    NiftyRegDeformationField,
    NiftyRegDisplacementField,
    NiftyRegVelocityField,
    NiftyRegVelocityGrid,
)

load = bio.transformations.load

# A reference voxel-to-world affine with a permutation, a flip,
# anisotropic spacing and an offset, so that a grid read the wrong way
# cannot pass.
REF_VOX2RAS = np.array(
    [
        [0.0, -1.5, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 1.25, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
REF_SHAPE = (9, 8, 7)
SPACING = 3.0
"""Control-point spacing, in reference voxels along every axis."""

AFFINE = np.array(
    [
        [0.98, -0.17, 0.02, 4.5],
        [0.17, 0.97, -0.05, -2.25],
        [-0.01, 0.06, 1.03, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _apply(xform, points: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through a RAS-to-RAS transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points, float))
    chain = list(xform) if isinstance(xform, xforms.Sequence) else [xform]
    out = xforms.Sequence(transformations=[points, *chain]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _world(ijk: np.ndarray, vox2world: np.ndarray) -> np.ndarray:
    return np.asarray(ijk, float) @ vox2world[:3, :3].T + vox2world[:3, 3]


def _grid(shape, vox2world) -> np.ndarray:  # noqa: ANN001
    ijk = np.stack(np.meshgrid(*map(np.arange, shape), indexing="ij"), -1)
    return _world(ijk, vox2world)


def _points(rng: np.random.RandomState, n: int, lo, hi) -> np.ndarray:  # noqa: ANN001
    """Random world points whose reference voxel coordinates are in
    `[lo, hi]` along every axis."""
    ijk = rng.uniform(lo, hi, size=(n, 3)) * (np.asarray(REF_SHAPE) - 1)
    return _world(ijk, REF_VOX2RAS)


# ----------------------------------------------------------------------
#   NUMPY PORTS OF NIFTYREG'S KERNELS
# ----------------------------------------------------------------------


def _slide(values: np.ndarray, vox2world: np.ndarray, idx) -> np.ndarray:  # noqa: ANN001
    """`get_GridValues` / `get_SlidedValues` (positions): the value at an
    integer index, or beyond the grid the nearest value shifted by the
    world offset, so that its displacement is kept."""
    idx = np.asarray(idx)
    shape = np.asarray(values.shape[:3])
    clamped = np.clip(idx, 0, shape - 1)
    shift = vox2world[:3, :3] @ (idx - clamped)
    return values[tuple(clamped)] + shift


def _bspline_basis(t: float) -> np.ndarray:
    """`get_BSplineBasisValues`."""
    return np.array(
        [
            (1 - t) ** 3 / 6,
            (3 * t**3 - 6 * t**2 + 4) / 6,
            (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6,
            t**3 / 6,
        ]
    )


def _niftyreg_cubic(
    positions: np.ndarray, vox2world: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """`reg_cubic_spline_getDeformationField3D`, composition branch."""
    world2vox = np.linalg.inv(vox2world)
    out = np.zeros_like(points)
    for n, point in enumerate(points):
        voxel = world2vox[:3, :3] @ point + world2vox[:3, 3]
        pre = np.floor(voxel).astype(int)
        basis = [_bspline_basis(t) for t in voxel - pre]
        pre -= 1
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    w = basis[0][a] * basis[1][b] * basis[2][c]
                    idx = pre + (a, b, c)
                    out[n] += w * _slide(positions, vox2world, idx)
    return out


def _niftyreg_linear(
    positions: np.ndarray, vox2world: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """`reg_defField_compose3D` (and the linear spline grid): trilinear,
    sliding beyond the grid."""
    world2vox = np.linalg.inv(vox2world)
    out = np.zeros_like(points)
    for n, point in enumerate(points):
        voxel = world2vox[:3, :3] @ point + world2vox[:3, 3]
        pre = np.floor(voxel).astype(int)
        rel = voxel - pre
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    w = np.prod(
                        [r if k else 1 - r for r, k in zip(rel, (a, b, c))]
                    )
                    idx = pre + (a, b, c)
                    out[n] += w * _slide(positions, vox2world, idx)
    return out


# ----------------------------------------------------------------------
#   FIXTURES, IN NIFTYREG'S LAYOUT
# ----------------------------------------------------------------------


def _nreg_image(
    vectors: np.ndarray,
    vox2world: np.ndarray,
    kind: int,
    sform: bool = True,
    qform: bool = False,
    p2: float = 0.0,
) -> "nb.Nifti1Image":
    """A NiftyReg transformation: `(X, Y, Z, 1, 3)` `VECTOR` image named
    `NREG_TRANS`, with its type in `intent_p1`."""
    image = nb.Nifti1Image(
        np.asarray(vectors, "f4")[:, :, :, None, :], vox2world
    )
    header = image.header
    header.set_sform(vox2world, code=1 if sform else 0)
    header.set_qform(vox2world, code=1 if qform else 0)
    header.set_intent(1007, name="NREG_TRANS")
    header["intent_p1"] = kind
    header["intent_p2"] = p2
    return image


def _cpp_vox2world(ref_vox2world: np.ndarray) -> np.ndarray:
    """`reg_createControlPointGrid`: the reference orientation, scaled to
    the spacing, with the origin one control point before."""
    grid = ref_vox2world.copy()
    grid[:3, :3] = ref_vox2world[:3, :3] * SPACING
    grid[:3, 3] = _world([-1, -1, -1], grid)
    return grid


def _cpp_shape() -> tuple:
    return tuple(int(np.ceil(n / SPACING)) + 3 for n in REF_SHAPE)


def _random_cpp(rng: np.random.RandomState, scale: float = 2.0):  # noqa: ANN202
    """Control-point positions: the affine positions plus a random
    perturbation, as `reg_f3d -aff` would start from and optimise."""
    vox2world = _cpp_vox2world(REF_VOX2RAS)
    shape = _cpp_shape()
    identity = _grid(shape, vox2world)
    positions = _world(identity, AFFINE)
    positions += scale * rng.standard_normal(positions.shape)
    return positions.astype("f4").astype(float), vox2world


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(1234)


# ----------------------------------------------------------------------
#   AFFINE
# ----------------------------------------------------------------------


def _write_aladin(path: Path, matrix: np.ndarray) -> Path:
    """`reg_tool_WriteAffineFile`: `%.7g`, space separated."""
    with open(path, "w") as f:
        for row in matrix:
            f.write(" ".join(f"{value:.7g}" for value in row) + "\n")
    return path


def test_aladin_affine_maps_reference_ras_to_floating_ras(
    tmp_path: Path,
) -> None:
    """`Affine * Reference = Floating`: the matrix is read as it is, as a
    RAS-to-RAS affine."""
    path = _write_aladin(tmp_path / "aff.txt", AFFINE)
    xform = load(path, hint="niftyreg")
    assert isinstance(xform, NiftyRegAffine)
    assert isinstance(xform, RASToRAS)
    assert xform.input == systems.RASmm()
    assert xform.output == systems.RASmm()
    np.testing.assert_allclose(xform.homogeneous_matrix, AFFINE, atol=1e-6)
    points = np.array([[1.0, 2.0, 3.0], [-10.0, 5.0, 0.5]])
    np.testing.assert_allclose(
        _apply(xform, points), _world(points, AFFINE), atol=1e-5
    )


@pytest.mark.parametrize("hint", ["niftyreg", "niftyreg.aladin", "aladin"])
def test_aladin_affine_hints(tmp_path: Path, hint: str) -> None:
    path = _write_aladin(tmp_path / "aff.txt", AFFINE)
    assert isinstance(load(path, hint=hint), NiftyRegAffine)


def test_aladin_affine_is_not_claimed_without_a_hint(tmp_path: Path) -> None:
    """A bare `(4, 4)` matrix says nothing of NiftyReg: the generic
    matrix reader keeps it."""
    path = _write_aladin(tmp_path / "aff.txt", AFFINE)
    assert isinstance(load(path), TxtMatrixAffine)
    assert NiftyRegAffine.sniff_file(path) == Confidence.WEAK


@pytest.mark.parametrize(
    "rows",
    [
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n",
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 2 1\n",
    ],
)
def test_aladin_affine_rejects_other_matrices(rows: str) -> None:
    with pytest.raises(ParserContentError):
        NiftyRegAffine.from_text(rows)


def test_aladin_affine_round_trips(tmp_path: Path) -> None:
    xform = NiftyRegAffine.from_file(
        _write_aladin(tmp_path / "aff.txt", AFFINE)
    )
    out = tmp_path / "out.txt"
    xform.save(out)
    rows = [line.split() for line in out.read_text().splitlines()]
    assert len(rows) == 4 and all(len(row) == 4 for row in rows)
    back = NiftyRegAffine.from_file(out)
    np.testing.assert_array_equal(back.matrix, xform.matrix)


def test_aladin_affine_is_written_from_a_matrix() -> None:
    xform = NiftyRegAffine(matrix=AFFINE[:3])
    text = xform.to_text()
    np.testing.assert_allclose(np.loadtxt(_io.StringIO(text)), AFFINE, atol=0)


def test_plain_affine_is_not_saved_as_niftyreg(tmp_path: Path) -> None:
    """A plain `Affine` does not say it maps RAS to RAS, so `save` does
    not write it as a NiftyReg affine."""
    with pytest.raises(WriterError):
        bio.save(xforms.Affine(matrix=AFFINE[:3]), tmp_path / "a.txt")


# ----------------------------------------------------------------------
#   DISPATCH OF THE NIFTI FILES
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, cls",
    [
        (0, NiftyRegDeformationField),
        (1, NiftyRegDisplacementField),
        (2, NiftyRegControlPointGrid),
        (3, NiftyRegVelocityField),
        (4, NiftyRegVelocityField),
        (5, NiftyRegVelocityGrid),
        (6, NiftyRegControlPointGrid),
    ],
)
def test_intent_p1_picks_the_reader(
    tmp_path: Path, rng: np.random.RandomState, kind: int, cls: type
) -> None:
    path = tmp_path / "xform.nii.gz"
    vectors = rng.standard_normal((*_cpp_shape(), 3))
    nb.save(_nreg_image(vectors, REF_VOX2RAS, kind), path)
    xform = load(path)
    assert type(xform) is cls
    assert xform.niftyreg_type == kind


def test_generic_vector_readers_decline_niftyreg_files() -> None:
    image = _nreg_image(np.zeros((3, 4, 5, 3)), REF_VOX2RAS, 1)
    assert NiftiRASCoordinatesField.sniff_nibabel(image) == Confidence.NO
    assert ItkNiftiDisplacementField.sniff_nibabel(image) == Confidence.NO


def test_other_vector_images_are_not_niftyreg() -> None:
    image = _nreg_image(np.zeros((3, 4, 5, 3)), REF_VOX2RAS, 1)
    image.header.set_intent(1007, name="")
    image.header["intent_p1"] = 1
    for cls in (NiftyRegDisplacementField, NiftyRegDeformationField):
        assert cls.sniff_nibabel(image) == Confidence.NO


@pytest.mark.parametrize(
    "cls, hints",
    [
        (NiftyRegAffine, {"niftyreg.aladin", "affine"}),
        (NiftyRegControlPointGrid, {"niftyreg.cpp", "niftyreg.f3d"}),
        (NiftyRegDeformationField, {"niftyreg.deformation", "niftyreg.def"}),
        (NiftyRegDisplacementField, {"niftyreg.displacement"}),
        (NiftyRegVelocityGrid, {"niftyreg.velocity"}),
        (NiftyRegVelocityField, {"niftyreg.velocity"}),
    ],
)
def test_hints(cls: type, hints: set) -> None:
    assert hints <= format_hints(cls)


def test_two_dimensional_fields_are_refused(tmp_path: Path) -> None:
    image = nb.Nifti1Image(np.zeros((5, 6, 1, 1, 2), "f4"), REF_VOX2RAS)
    image.header.set_intent(1007, name="NREG_TRANS")
    image.header["intent_p1"] = 1
    path = tmp_path / "disp2d.nii.gz"
    nb.save(image, path)
    with pytest.raises(ParserContentError, match="Two-dimensional"):
        NiftyRegDisplacementField.from_file(path)


# ----------------------------------------------------------------------
#   DENSE FIELDS
# ----------------------------------------------------------------------


def _deformation(rng: np.random.RandomState) -> np.ndarray:
    identity = _grid(REF_SHAPE, REF_VOX2RAS)
    positions = _world(identity, AFFINE) + rng.standard_normal(identity.shape)
    return positions.astype("f4").astype(float)


def test_deformation_field_matches_niftyreg(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    """Positions at the voxels, trilinear in between, and the nearest
    displacement slid beyond the grid (`reg_defField_compose3D`)."""
    positions = _deformation(rng)
    path = tmp_path / "def.nii.gz"
    nb.save(_nreg_image(positions, REF_VOX2RAS, 0), path)
    xform = load(path)
    assert isinstance(xform, NiftyRegDeformationField)
    assert len(xform) == 3
    assert xform.input == systems.RASmm() and xform.output == systems.RASmm()

    ijk = np.array([[1, 2, 3], [0, 7, 6], [8, 0, 0]])
    np.testing.assert_allclose(
        _apply(xform, _world(ijk, REF_VOX2RAS)),
        positions[tuple(ijk.T)],
        atol=1e-4,
    )
    points = np.concatenate(
        [_points(rng, 20, 0, 1), _points(rng, 10, -0.3, 1.3)]
    )
    np.testing.assert_allclose(
        _apply(xform, points),
        _niftyreg_linear(positions, REF_VOX2RAS, points),
        atol=1e-4,
    )


def test_displacement_field_matches_niftyreg(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    """`displacement = deformation - position`, so the field maps
    `x -> x + u(x)`."""
    positions = _deformation(rng)
    disp = positions - _grid(REF_SHAPE, REF_VOX2RAS)
    path = tmp_path / "disp.nii.gz"
    nb.save(_nreg_image(disp, REF_VOX2RAS, 1), path)
    xform = load(path)
    assert isinstance(xform, NiftyRegDisplacementField)
    points = np.concatenate(
        [_points(rng, 20, 0, 1), _points(rng, 10, -0.3, 1.3)]
    )
    np.testing.assert_allclose(
        _apply(xform, points),
        _niftyreg_linear(positions, REF_VOX2RAS, points),
        atol=1e-4,
    )


def test_qform_is_used_when_sform_code_is_zero(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    positions = _deformation(rng)
    image = _nreg_image(positions, REF_VOX2RAS, 0, sform=False, qform=True)
    # A misleading sform with code 0: NiftyReg ignores it.
    image.header.set_sform(np.diag([5.0, 5.0, 5.0, 1.0]), code=0)
    path = tmp_path / "def.nii.gz"
    nb.save(image, path)
    xform = load(path)
    points = _points(rng, 10, 0, 1)
    np.testing.assert_allclose(
        _apply(xform, points),
        _niftyreg_linear(positions, REF_VOX2RAS, points),
        atol=1e-4,
    )


def test_no_xform_code_falls_back_to_pixel_sizes(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    """With both codes zero, `nifti1_io` makes the qform a diagonal of
    pixel sizes with no offset, and NiftyReg uses it."""
    vox2world = np.diag([2.0, 3.0, 1.5, 1.0])
    identity = _grid(REF_SHAPE, vox2world)
    positions = identity + rng.standard_normal(identity.shape)
    positions = positions.astype("f4").astype(float)
    image = _nreg_image(positions, vox2world, 0, sform=False, qform=False)
    path = tmp_path / "def.nii.gz"
    nb.save(image, path)
    xform = load(path)
    ijk = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(
        _apply(xform, _world(ijk, vox2world)),
        positions[tuple(ijk.T)],
        atol=1e-4,
    )


@pytest.mark.parametrize("kind", [0, 1])
def test_dense_fields_round_trip(
    tmp_path: Path, rng: np.random.RandomState, kind: int
) -> None:
    vectors = _deformation(rng)
    if kind == 1:
        vectors = vectors - _grid(REF_SHAPE, REF_VOX2RAS)
    path = tmp_path / "in.nii.gz"
    nb.save(_nreg_image(vectors, REF_VOX2RAS, kind), path)
    out = tmp_path / "out.nii.gz"
    load(path).save(out)
    back = nb.load(out)
    header = back.header
    assert header.get_intent() == ("vector", (), "NREG_TRANS")
    assert float(header["intent_p1"]) == kind
    assert back.shape == (*REF_SHAPE, 1, 3)
    assert int(header["sform_code"]) > 0
    np.testing.assert_allclose(back.get_sform(), REF_VOX2RAS, atol=1e-6)
    np.testing.assert_allclose(
        back.get_fdata()[:, :, :, 0, :], vectors, atol=1e-4
    )


def test_displacement_field_is_written_from_a_chain(
    rng: np.random.RandomState,
) -> None:
    """A field built from the shared RAS displacement chain writes as a
    NiftyReg displacement field."""
    from brainhops.io.transformations.base.fields import (
        ras_displacement_chain,
    )

    disp = rng.standard_normal((*REF_SHAPE, 3))
    field = NiftyRegDisplacementField()
    field.transformations = ras_displacement_chain(disp, REF_VOX2RAS)
    image = field.to_nibabel()
    assert float(image.header["intent_p1"]) == 1
    np.testing.assert_allclose(
        np.asarray(image.dataobj)[:, :, :, 0, :], disp, atol=1e-10
    )


# ----------------------------------------------------------------------
#   CONTROL-POINT GRIDS
# ----------------------------------------------------------------------


def test_control_point_grid_matches_niftyreg(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    """The cubic B-spline of the control-point positions, placed by the
    grid's header, and slid beyond the grid."""
    positions, vox2world = _random_cpp(rng)
    path = tmp_path / "cpp.nii.gz"
    nb.save(_nreg_image(positions, vox2world, 2), path)
    xform = load(path)
    assert isinstance(xform, NiftyRegControlPointGrid)
    assert xform.order == 3 and xform.coeff
    assert xform.affine is None
    assert xform.displacement.coeff and xform.displacement.order == 3
    points = np.concatenate(
        [_points(rng, 25, 0, 1), _points(rng, 10, -0.8, 1.8)]
    )
    np.testing.assert_allclose(
        _apply(xform, points),
        _niftyreg_cubic(positions, vox2world, points),
        atol=1e-4,
    )


def test_identity_grid_is_the_identity(tmp_path: Path) -> None:
    """`reg_createControlPointGrid` fills the grid with the positions of
    its control points: the identity."""
    vox2world = _cpp_vox2world(REF_VOX2RAS)
    positions = _grid(_cpp_shape(), vox2world)
    path = tmp_path / "cpp.nii.gz"
    nb.save(_nreg_image(positions, vox2world, 2), path)
    points = _points(np.random.RandomState(0), 20, -0.5, 1.5)
    np.testing.assert_allclose(_apply(load(path), points), points, atol=1e-4)


def test_linear_grid_matches_niftyreg(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    positions, vox2world = _random_cpp(rng)
    path = tmp_path / "lin.nii.gz"
    nb.save(_nreg_image(positions, vox2world, 6), path)
    xform = load(path, hint="niftyreg.cpp")
    assert xform.order == 1 and not xform.coeff
    points = np.concatenate(
        [_points(rng, 20, 0, 1), _points(rng, 10, -0.5, 1.5)]
    )
    np.testing.assert_allclose(
        _apply(xform, points),
        _niftyreg_linear(positions, vox2world, points),
        atol=1e-4,
    )


def _with_extension(image: "nb.Nifti1Image", matrix: np.ndarray) -> None:
    """The affine extension of `reg_createSymmetricControlPointGrids`:
    a raw `mat44` in a `NIFTI_ECODE_IGNORE` extension, twice."""
    content = np.asarray(matrix, "<f4").tobytes() + bytes(8)
    for _ in range(2):
        image.header.extensions.append(nb.nifti1.Nifti1Extension(0, content))


def test_grid_extension_affine_is_applied_first(
    tmp_path: Path, rng: np.random.RandomState
) -> None:
    """`reg_spline_getDeformationField` maps the reference position
    through the extension affine, then evaluates the spline there."""
    positions, vox2world = _random_cpp(rng, scale=1.0)
    half = AFFINE.copy()
    half[:3, 3] *= 0.5
    image = _nreg_image(positions, vox2world, 2)
    _with_extension(image, half)
    path = tmp_path / "cpp.nii.gz"
    nb.save(image, path)
    xform = load(path)
    assert len(xform) == 4
    assert isinstance(xform.affine, RASToRAS)
    np.testing.assert_allclose(
        xform.affine.homogeneous_matrix, half.astype("f4"), atol=0
    )
    points = _points(rng, 20, 0.1, 0.9)
    expected = _niftyreg_cubic(positions, vox2world, _world(points, half))
    np.testing.assert_allclose(_apply(xform, points), expected, atol=1e-4)

    out = tmp_path / "out.nii.gz"
    xform.save(out)
    back = load(out)
    assert len(back) == 4
    np.testing.assert_allclose(_apply(back, points), expected, atol=1e-4)


@pytest.mark.parametrize("kind", [2, 6])
def test_grid_round_trips(
    tmp_path: Path, rng: np.random.RandomState, kind: int
) -> None:
    positions, vox2world = _random_cpp(rng)
    path = tmp_path / "cpp.nii.gz"
    nb.save(_nreg_image(positions, vox2world, kind), path)
    out = tmp_path / "out.nii.gz"
    load(path).save(out)
    back = nb.load(out)
    assert float(back.header["intent_p1"]) == kind
    assert back.header.get_intent()[2] == "NREG_TRANS"
    np.testing.assert_allclose(back.get_sform(), vox2world, atol=1e-6)
    np.testing.assert_allclose(back.header.get_zooms()[:3], [6.0, 4.5, 3.75])
    np.testing.assert_allclose(
        back.get_fdata()[:, :, :, 0, :], positions, atol=1e-4
    )


def test_grid_refuses_sampled_values() -> None:
    from brainhops.io.transformations.base.fields import (
        ras_displacement_chain,
    )

    grid = NiftyRegControlPointGrid()
    grid.transformations = ras_displacement_chain(
        np.zeros((4, 4, 4, 3)), np.eye(4), order=2, coeff=True
    )
    with pytest.raises(WriterError, match="order 2"):
        grid.to_nibabel()


# ----------------------------------------------------------------------
#   VELOCITIES
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", [3, 4, 5])
def test_velocities_are_read_but_not_used(
    tmp_path: Path, rng: np.random.RandomState, kind: int
) -> None:
    vectors = rng.standard_normal((*_cpp_shape(), 3))
    image = _nreg_image(vectors, REF_VOX2RAS, kind, p2=-6)
    _with_extension(image, AFFINE)
    path = tmp_path / "vel.nii.gz"
    nb.save(image, path)
    xform = load(path, hint="niftyreg.velocity")
    assert xform.squaring_steps == -6
    assert len(xform.extension_affines) == 2
    np.testing.assert_allclose(
        np.asarray(xform.data)[:, :, :, 0, :], vectors, atol=1e-6
    )
    for call in (
        lambda: xform.compute(),
        lambda: xform.inverse(),
        lambda: xform.to(xforms.Sequence),
    ):
        with pytest.raises(NotImplementedError, match="reg_transform"):
            call()

    out = tmp_path / "out.nii.gz"
    xform.save(out)
    back = nb.load(out)
    assert float(back.header["intent_p1"]) == kind
    assert float(back.header["intent_p2"]) == -6
    assert len(back.header.extensions) == 2
    np.testing.assert_allclose(back.get_fdata(), image.get_fdata(), atol=0)
