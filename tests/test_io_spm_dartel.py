"""Tests for DARTEL flow fields (`u_rc1*.nii`).

No field written by SPM ships with the tests, so files are written with
nibabel in the layout that `spm_dartel_warp` writes: a float32 array of
shape (X, Y, Z, 1, 3), described as `"Flow Field"`, whose qform and sform
are the `mat0` and the `mat` of the imported image, both coded
`Aligned`. The map is checked against `_spm_dartel`, a literal
transcription of `expdef`, `smalldef` and `composition` in SPM12's
`src/shoot_dartel.c` and `src/shoot_diffeo3d.c`, with the circulant
boundary of `src/shoot_boundary.c` and MATLAB's 1-based voxels, and of
`get_dartel` in `spm_deformations.m`, which is independent of the reader.
"""

from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import replace

from brainhops import io
from brainhops.backends import backend
from brainhops.datamodel import transformations as xforms
from brainhops.errors import ConversionError
from brainhops.io.base import TransformationSpec
from brainhops.io.base.parsers import (
    ParserContentError,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.spm import (
    DartelFlowField,
    SpmCoordinatesField,
)

nb = pytest.importorskip("nibabel")

DIM = (9, 10, 8)
"""Shape of the grid of the field."""

# The sform (`mat`) places the grid in the space of the template, and the
# qform (`mat0`) is the same grid rigidly moved back into the space of the
# subject, as `spm_preproc_write8` writes them (`mat0 = R \\ mat`).
SFORM = np.array(
    [
        [-1.5, 0.0, 0.0, 60.0],
        [0.0, 1.5, 0.0, -80.0],
        [0.0, 0.0, 1.5, -50.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _rigid() -> np.ndarray:
    c, s = np.cos(0.1), np.sin(0.1)
    return np.array(
        [
            [c, -s, 0.0, 2.0],
            [s, c, 0.0, -3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


QFORM = np.linalg.inv(_rigid()) @ SFORM


def _velocity(seed: int = 0, dtype: str = "float32") -> np.ndarray:
    # A smooth velocity of a few voxels, periodic over the grid.
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.arange(d) / d for d in DIM], indexing="ij")
    components = []
    for _ in range(3):
        a, b, c = rng.integers(1, 3, 3)
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(2 * np.pi * (a * grid[0] + b * grid[1] + c * grid[2]))
        components.append(2.0 * rng.uniform(0.5, 1.0) * wave + phase / 10)
    return np.stack(components, -1).astype(dtype)


def _save(
    path: Path, velocity: tx.Optional[np.ndarray] = None, **header: tx.Any
) -> Path:
    """Write a flow field as `spm_dartel_warp` writes it."""
    velocity = _velocity() if velocity is None else velocity
    image = nb.Nifti1Image(velocity[:, :, :, None, :], SFORM)
    image.header.set_qform(header.get("qform", QFORM), code=2)
    image.header.set_sform(SFORM, code=2)
    image.header["descrip"] = header.get("descrip", b"Flow Field")
    nb.save(image, str(path))
    return path


# ----------------------------------------------------------------------
#   THE MAP, AS SPM COMPUTES IT
# ----------------------------------------------------------------------


def _composition(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Transcribe `composition(dm, mm, B, A, C)`, which gives `C = B(A)`.

    The positions are numbered from 1. The displacement of `B` is sampled
    trilinearly at `A`, with indices wrapped around the grid, and added to
    `A`.
    """
    dm = B.shape[:3]
    x = A[..., 0] - 1.0
    y = A[..., 1] - 1.0
    z = A[..., 2] - 1.0
    ix, iy, iz = (np.floor(t).astype(int) for t in (x, y, z))
    dx1, dy1, dz1 = x - ix, y - iy, z - iz
    dx2, dy2, dz2 = 1 - dx1, 1 - dy1, 1 - dz1
    ix0, ix1 = np.mod(ix, dm[0]), np.mod(ix + 1, dm[0])
    iy0, iy1 = np.mod(iy, dm[1]), np.mod(iy + 1, dm[1])
    iz0, iz1 = np.mod(iz, dm[2]), np.mod(iz + 1, dm[2])
    corners = {
        (a, b, c): ((ix0, ix1)[a], (iy0, iy1)[b], (iz0, iz1)[c])
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    }
    C = np.empty_like(A)
    for d in range(3):

        def k(a: int, b: int, c: int, d: int = d) -> np.ndarray:
            i, j, l = corners[a, b, c]
            # Subtract the coordinate of the sampled point.
            return B[i, j, l, d] - ((i, j, l)[d] + 1.0)

        C[..., d] = (
            ((k(0, 0, 0) * dx2 + k(1, 0, 0) * dx1) * dy2)
            + ((k(0, 1, 0) * dx2 + k(1, 1, 0) * dx1) * dy1)
        ) * dz2 + (
            ((k(0, 0, 1) * dx2 + k(1, 0, 1) * dx1) * dy2)
            + ((k(0, 1, 1) * dx2 + k(1, 1, 1) * dx1) * dy1)
        ) * dz1
        C[..., d] += (x, y, z)[d] + 1.0
    return C


def _expdef(v: np.ndarray, K: int, sc: float) -> np.ndarray:
    """Transcribe `dartel3('Exp', v, [K sc])`, numbered from 1.

    `smalldef` gives `Id + v * sc / 2^K`, which is composed with itself
    `K` times.
    """
    grid = np.meshgrid(
        *[np.arange(1, d + 1) for d in v.shape[:3]], indexing="ij"
    )
    t = np.stack(grid, -1) + v * (sc / 2**K)
    for _ in range(K):
        t = _composition(t, t)
    return t


def _spm_dartel(
    path: Path, times: tx.Sequence[int], K: int = 6
) -> tx.Tuple[np.ndarray, np.ndarray]:
    """Transcribe `[Def, mat] = get_dartel(job)` without a template.

    `spm_dartel_integrate` gives `dartel3('Exp', u, [K, 1])` for a forward
    flow (`times = [0 1]`) and `[K, -1]` for a backward one. The forward
    deformation is placed on the grid of `mat0` and carried by `mat`, and
    the backward one the other way round. The matrices of SPM number
    voxels from 1. The velocity and the matrices are read from the file,
    as SPM reads them.
    """
    image = nb.load(str(path))
    velocity = np.asarray(image.dataobj)[:, :, :, 0, :]
    one = np.eye(4)
    one[:3, 3] = -1.0
    mat = image.header.get_sform() @ one
    mat0 = image.header.get_qform() @ one
    forward = list(times) == [0, 1]
    y0 = _expdef(velocity.astype(float), K, 1.0 if forward else -1.0)
    carry, grid = (mat, mat0) if forward else (mat0, mat)
    Def = y0 @ carry[:3, :3].T + carry[:3, 3]
    return Def, grid


def _grid_points(mat: np.ndarray) -> np.ndarray:
    """Return the world coordinates of the voxels of a grid placed by an
    SPM matrix."""
    grid = np.stack(
        np.meshgrid(*[np.arange(1, d + 1) for d in DIM], indexing="ij"), -1
    )
    return grid @ mat[:3, :3].T + mat[:3, 3]


def _mapped(xform: tx.Any, points: np.ndarray) -> np.ndarray:
    field = xforms.CoordinatesField(field=points)
    return np.asarray(xform.compute()(field, compute=True).field)


# ----------------------------------------------------------------------
#   THE CONVENTIONS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("array_backend", ["numpy", "dask"])
def test_the_forward_flow_is_what_spm_computes(
    tmp_path: Path, array_backend: str
) -> None:
    if array_backend == "dask":
        pytest.importorskip("dask")
    path = _save(tmp_path / "u_rc1sub.nii")
    Def, mat = _spm_dartel(path, [0, 1])
    with backend(array_backend):
        field = io.load(path)
        assert type(field) is DartelFlowField
        mapped = _mapped(field, _grid_points(mat))
    # The velocity is stored in single precision, in which SPM integrates
    # it too.
    assert np.allclose(mapped, Def, atol=1e-4)


def test_the_backward_flow_is_the_inverse(tmp_path: Path) -> None:
    path = _save(tmp_path / "u_rc1sub.nii")
    field = DartelFlowField.load(path)
    Def, mat = _spm_dartel(path, [1, 0])
    with backend("numpy"):
        mapped = _mapped(field.inverse(), _grid_points(mat))
    assert np.allclose(mapped, Def, atol=1e-4)


def test_in_double_precision_the_integration_is_exact(tmp_path: Path) -> None:
    velocity = _velocity(dtype="float64")
    path = _save(tmp_path / "u_rc1sub.nii", velocity)
    field = DartelFlowField.load(path)
    Def, mat = _spm_dartel(path, [0, 1])
    with backend("numpy"):
        mapped = _mapped(field, _grid_points(mat))
    assert np.allclose(mapped, Def, rtol=0, atol=1e-10)


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_the_steps_are_an_option_of_the_reader(
    tmp_path: Path, steps: int
) -> None:
    velocity = _velocity(dtype="float64")
    path = _save(tmp_path / "u_rc1sub.nii", velocity)
    field = DartelFlowField.load(path, steps=steps)
    assert field.steps == steps
    assert field.velocity.steps == steps
    Def, mat = _spm_dartel(path, [0, 1], K=steps)
    with backend("numpy"):
        mapped = _mapped(field, _grid_points(mat))
    assert np.allclose(mapped, Def, rtol=0, atol=1e-10)


def test_the_default_steps_are_those_of_spm12(tmp_path: Path) -> None:
    path = _save(tmp_path / "u_rc1sub.nii")
    assert DartelFlowField.load(path).steps == 6
    # A specification gives the option as text.
    spec = TransformationSpec.from_arg(f"{path}|dartel|steps:3")
    field = io.transformations.load(spec)
    assert type(field) is DartelFlowField
    assert field.steps == 3
    assert field.velocity.steps == 3


def test_the_chain_has_three_slots(tmp_path: Path) -> None:
    field = DartelFlowField.load(_save(tmp_path / "u_rc1sub.nii"))
    ras2voxel, velocity, voxel2ras = field.transformations
    assert field.ras2voxel is ras2voxel
    assert field.velocity is velocity
    assert field.voxel2ras is voxel2ras
    assert isinstance(ras2voxel, RASToVoxel)
    assert isinstance(voxel2ras, VoxelToRAS)
    assert type(velocity) is xforms.StationaryVelocityField
    assert velocity.degree == 1
    assert velocity.bound == "grid-wrap"
    stored = nb.load(str(tmp_path / "u_rc1sub.nii")).header
    assert np.allclose(
        np.linalg.inv(ras2voxel.homogeneous_matrix), stored.get_qform()
    )
    assert np.allclose(voxel2ras.homogeneous_matrix, stored.get_sform())
    assert np.array_equal(np.asarray(velocity.data), _velocity())


def test_a_header_without_an_sform_uses_the_qform(tmp_path: Path) -> None:
    path = _save(tmp_path / "u_rc1sub.nii")
    image = nb.load(str(path), mmap=False)
    image.header.set_sform(None, code=0)
    # The voxels are read first, since the file is replaced.
    data = np.asarray(image.dataobj)
    nb.save(nb.Nifti1Image(data, None, image.header), str(path))
    field = DartelFlowField.load(path)
    assert np.allclose(
        field.voxel2ras.homogeneous_matrix,
        np.linalg.inv(field.ras2voxel.homogeneous_matrix),
    )


def test_a_header_without_a_qform_is_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "u_rc1sub.nii")
    image = nb.load(str(path), mmap=False)
    image.header.set_qform(None, code=0)
    # The voxels are read first, since the file is replaced.
    data = np.asarray(image.dataobj)
    nb.save(nb.Nifti1Image(data, None, image.header), str(path))
    field = DartelFlowField.load(path)
    with pytest.raises(ParserContentError, match="qform"):
        _ = field.transformations


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, descrip",
    [
        ("u_rc1sub.nii", b"Flow Field"),
        ("u_rc1sub.nii.gz", b"edited"),
        ("flow.nii", b"Flow Field"),
    ],
)
def test_the_file_is_recognised(
    tmp_path: Path, name: str, descrip: bytes
) -> None:
    # The description of DARTEL, or else the `u_` prefix, tells the field
    # from a field of coordinates.
    path = _save(tmp_path / name, descrip=descrip)
    assert type(io.load(path)) is DartelFlowField
    assert type(io.load(path, hint="dartel")) is DartelFlowField


def test_a_deformation_field_is_not_a_flow_field(tmp_path: Path) -> None:
    path = _save(tmp_path / "y_rc1sub.nii", descrip=b"Deformation")
    assert type(io.load(path)) is SpmCoordinatesField


def test_bad_steps_are_refused(tmp_path: Path) -> None:
    path = _save(tmp_path / "u_rc1sub.nii")
    for steps in (-1, 2.5, True):
        with pytest.raises(TypeError):
            DartelFlowField.load(path, steps=steps)


# ----------------------------------------------------------------------
#   WRITING AND CONVERTING
# ----------------------------------------------------------------------


def _chain(
    velocity: np.ndarray,
    first: np.ndarray = QFORM,
    last: np.ndarray = SFORM,
    **options: tx.Any,
) -> xforms.Sequence:
    voxel = RASToVoxel().output
    settings = {"steps": 6, "bound": "grid-wrap", **options}
    field_class = xforms.StationaryVelocityField
    if settings.pop("displacement", False):
        # Only a velocity has squaring steps.
        del settings["steps"]
        field_class = xforms.DisplacementField
    return xforms.Sequence(
        [
            RASToVoxel(matrix=np.linalg.inv(first)[:3]),
            field_class(velocity, input=voxel, output=voxel, **settings),
            VoxelToRAS(matrix=last[:3]),
        ]
    )


def test_a_chain_is_written_with_a_qform_and_an_sform(tmp_path: Path) -> None:
    field = DartelFlowField.from_instance(_chain(_velocity()))
    path = tmp_path / "u_new.nii"
    io.save(field, path)
    header = nb.load(str(path)).header
    assert int(header["qform_code"]) == 2
    assert int(header["sform_code"]) == 2
    assert np.allclose(header.get_qform(), QFORM, atol=1e-5)
    assert np.allclose(header.get_sform(), SFORM, atol=1e-5)
    assert header["descrip"].item() == b"Flow Field"
    back = io.load(path)
    assert type(back) is DartelFlowField
    assert np.array_equal(np.asarray(back.data), _velocity())


def test_the_steps_of_a_velocity_become_the_option(tmp_path: Path) -> None:
    field = DartelFlowField.from_instance(_chain(_velocity(), steps=4))
    assert field.steps == 4
    consolidated = field.consolidate()
    assert consolidated.steps == 4
    assert consolidated.velocity.steps == 4
    # The file does not store them, so it is read with the default unless
    # the reader is told.
    path = tmp_path / "u_new.nii"
    field.save(path)
    assert DartelFlowField.load(path).steps == 6
    assert DartelFlowField.load(path, steps=4).velocity.steps == 4


def test_an_assigned_velocity_is_encoded(tmp_path: Path) -> None:
    field = DartelFlowField.load(_save(tmp_path / "u_rc1sub.nii"))
    chain = list(field.transformations)
    chain[1] = replace(chain[1], data=np.asarray(chain[1].data) * 2)
    field.transformations = tuple(chain)
    path = tmp_path / "u_doubled.nii"
    field.save(path)
    back = DartelFlowField.load(path)
    assert np.allclose(np.asarray(back.data), _velocity() * 2)
    header = nb.load(str(path)).header
    assert np.allclose(header.get_qform(), QFORM, atol=1e-5)


@pytest.mark.parametrize(
    "chain, reason",
    [
        (_chain(_velocity(), bound="nearest"), "wraps it around"),
        (_chain(_velocity(), degree=3), "degree 3"),
        (
            _chain(
                _velocity(),
                first=np.array(
                    [
                        [1.0, 0.3, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ),
            "qform",
        ),
        (_chain(_velocity(), displacement=True), "logarithm"),
    ],
)
def test_other_chains_are_refused(
    tmp_path: Path, chain: tx.Any, reason: str
) -> None:
    with pytest.raises(ConversionError, match=reason):
        DartelFlowField.from_instance(chain)
    field = DartelFlowField(transformations=tuple(chain.transformations))
    with pytest.raises(UnrepresentableTransformationError, match=reason):
        field.consolidate()


def test_a_flow_field_is_not_a_deformation_field(tmp_path: Path) -> None:
    # A velocity wraps around its grid, and a deformation field takes the
    # nearest value outside it, so the two maps differ there.
    field = DartelFlowField.load(_save(tmp_path / "u_rc1sub.nii"))
    with pytest.raises(ConversionError, match="extend differently"):
        field.to(SpmCoordinatesField)
