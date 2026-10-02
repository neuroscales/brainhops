"""
ITK displacement and coordinate fields stored as NIfTI vector images.

ITK writes a warp as a `(X, Y, Z, 1, 3)` NIfTI -- `(X, Y, 1, 1, 2)` in
2-D -- with the `VECTOR` (1007) intent, a voxel-to-RAS sform/qform, and
vector values left in LPS. These tests pin that encoding numerically, in
both dimensions -- a known LPS displacement must move RAS points by the
same vector with x and y negated -- and pin how such a file is told
apart from a RAS NIfTI field.
"""

from pathlib import Path

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.backends import available_backends, backend  # noqa: E402
from brainhops.datamodel import systems  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    AmbiguousFormatError,
    WriterError,
)
from brainhops.io.base.specs import format_hints  # noqa: E402
from brainhops.io.transformations.base import (  # noqa: E402
    FileBasedTransformation,
)
from brainhops.io.transformations.base.affines import (  # noqa: E402
    LPSToVoxel,
    VoxelToLPS,
)
from brainhops.io.transformations.base.fields import (  # noqa: E402
    LPSCoordinatesField,
)
from brainhops.io.transformations.itk._systems import (  # noqa: E402
    _make_system,
)
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ItkNiftiCoordinatesField,
    ItkNiftiDisplacementField,
    ItkNiftiField,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiRASDisplacementField,
    NiftiVoxelToRAS,
)

DATA = Path(__file__).parent / "data"

NONE = 0  # NIFTI_INTENT_NONE
DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR, what ITK writes
FNIRT = 2006  # NIFTI_INTENT_FSL_FNIRT_DISPLACEMENT_FIELD

NDIMS = [2, 3]

SHAPES = {2: (4, 5), 3: (4, 5, 6)}
"""Grid shapes: small, and no two axes of the same length."""

# Voxel-to-RAS affines with a permutation, a flip, anisotropic spacing and
# an offset, so that a wrong frame or a wrong rotation of the vectors
# cannot cancel out. The 2-D one is the identity along z, as ITK writes a
# 2-D image.
VOX2RAS = {
    2: np.array(
        [
            [0.0, -3.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, -20.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    3: np.array(
        [
            [0.0, -3.0, 0.0, 10.0],
            [2.0, 0.0, 0.0, -20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def _flip(ndim: int) -> np.ndarray:
    """RAS <-> LPS, on vectors and on points: x and y are negated."""
    return np.array([-1.0, -1.0, 1.0])[:ndim]


def _vox2lps(ndim: int) -> np.ndarray:
    """The homogeneous `(ndim + 1)`-square voxel-to-LPS affine of a grid."""
    full = np.diag([-1.0, -1.0, 1.0, 1.0]) @ VOX2RAS[ndim]
    keep = [*range(ndim), 3]
    return full[keep][:, keep]


def _vox2ras(ndim: int) -> np.ndarray:
    """The homogeneous `(ndim + 1)`-square voxel-to-RAS affine of a grid."""
    keep = [*range(ndim), 3]
    return VOX2RAS[ndim][keep][:, keep]


def _ramp(ndim: int) -> np.ndarray:
    """An `(*shape, ndim)` LPS displacement whose entries name their voxel."""
    grids = np.meshgrid(*map(np.arange, SHAPES[ndim]), indexing="ij")
    return np.stack(
        [(d + 1.0) + 0.1 * (d + 1) * g for d, g in enumerate(grids)], axis=-1
    ).astype("float32")


def _grid_points(vox2world: np.ndarray) -> np.ndarray:
    """The world coordinates of every voxel, as an `(*shape, ndim)` array."""
    ndim = vox2world.shape[0] - 1
    ijk = np.stack(
        np.meshgrid(*map(np.arange, SHAPES[ndim]), indexing="ij"), axis=-1
    )
    return ijk @ vox2world[:ndim, :ndim].T + vox2world[:ndim, ndim]


def _write(path, vectors: np.ndarray, intent: int = VECTOR):  # noqa: ANN001, ANN202
    """Write `(*shape, ndim)` vectors in ITK's NIfTI layout."""
    ndim = vectors.shape[-1]
    layout = (*vectors.shape[:ndim], *(1,) * (3 - ndim), 1, ndim)
    img = nb.Nifti1Image(vectors.reshape(layout), VOX2RAS[ndim])
    img.header.set_intent(intent)
    img.header.set_sform(VOX2RAS[ndim], code=1)
    img.header.set_qform(VOX2RAS[ndim], code=1)
    nb.save(img, str(path))
    return path


def _apply(xform, points_lps: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map LPS points through an LPS-to-LPS transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points_lps, float))
    out = xforms.Sequence(transformations=[points, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _apply_ras(xform, points_ras: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through an LPS-to-LPS transformation."""
    flip = _flip(points_ras.shape[-1])
    return _apply(xform, points_ras * flip) * flip


def _all_nodes_ras(ndim: int) -> np.ndarray:
    return _grid_points(_vox2ras(ndim)).reshape(-1, ndim)


@pytest.fixture(params=NDIMS, ids=lambda n: f"{n}d")
def ndim(request) -> int:  # noqa: ANN001
    return request.param


@pytest.fixture
def itk_warp(tmp_path, ndim):  # noqa: ANN001, ANN201
    """A small ITK displacement field, carrying the LPS ramp."""
    return _write(tmp_path / "warp.nii.gz", _ramp(ndim))


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_field_maps_lps_to_lps(itk_warp, ndim) -> None:  # noqa: ANN001
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    assert field.input == _make_system(ndim)
    assert field.output == _make_system(ndim)
    assert len(field.input.axes) == ndim
    assert list(field) == [
        field.lps2voxel,
        field.displacement,
        field.voxel2lps,
    ]
    assert isinstance(field.lps2voxel, LPSToVoxel)
    assert isinstance(field.displacement, xforms.DisplacementField)
    assert isinstance(field.voxel2lps, VoxelToLPS)
    np.testing.assert_allclose(field.voxel2lps.matrix, _vox2lps(ndim)[:-1])


def test_the_endpoints_are_itks_spaces(itk_warp, ndim) -> None:  # noqa: ANN001
    """(L, P) in 2-D and `LPSmm` in 3-D, both in millimetres, and the
    pixel or voxel grid in between."""
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    world = field.input
    if ndim == 3:
        assert isinstance(world, systems.LPSmm)
    else:
        assert isinstance(world, systems.SpatialCoordinateSystem2D)
        orientations = [axis.orientation.value for axis in world.axes]
        assert orientations == ["right-to-left", "anterior-to-posterior"]
    assert all(str(axis.unit) == "millimeter" for axis in world.axes)
    grid = {2: systems.PixelCoordinateSystem, 3: systems.VoxelCoordinateSystem}
    assert isinstance(field.lps2voxel.output, grid[ndim])
    assert field.lps2voxel.input == world
    assert field.voxel2lps.output == world
    assert field.displacement.input == field.lps2voxel.output


def test_the_grid_is_read_from_the_ras_header_as_lps(itk_warp, ndim) -> None:  # noqa: ANN001
    """The header is RAS; ITK's grid is the same one, expressed in LPS."""
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    np.testing.assert_allclose(
        field.lps2voxel.matrix, np.linalg.inv(_vox2lps(ndim))[:-1]
    )


def test_the_singleton_axes_are_dropped(itk_warp, ndim) -> None:  # noqa: ANN001
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    shape = np.asarray(field.displacement.field).shape
    assert shape == (*SHAPES[ndim], ndim)


def test_displacements_are_stored_in_voxel_units(itk_warp, ndim) -> None:  # noqa: ANN001
    """A `DisplacementField` adds its values in the units of its grid."""
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    linear = _vox2lps(ndim)[:ndim, :ndim]
    expected = _ramp(ndim) @ np.linalg.inv(linear).T
    np.testing.assert_allclose(
        field.displacement.field, expected, rtol=1e-6, atol=1e-6
    )


def test_itks_interpolation_is_kept(itk_warp) -> None:  # noqa: ANN001
    """ITK interpolates a displacement field linearly, and extends it with
    its nearest value."""
    displacement = ItkNiftiDisplacementField.from_file(itk_warp).displacement
    assert displacement.order == 1
    assert displacement.bound == "nearest"
    assert not displacement.coeff


def test_a_file_that_is_not_in_itks_layout_is_refused(tmp_path) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros((4, 5, 6), "float32"), VOX2RAS[3])
    path = tmp_path / "plain.nii.gz"
    nb.save(img, str(path))
    with pytest.raises(Exception, match="ITK field"):
        ItkNiftiDisplacementField.from_file(path)


# ----------------------------------------------------------------------
#   SEMANTICS
# ----------------------------------------------------------------------


def test_a_constant_lps_displacement_moves_ras_points_with_x_y_negated(
    tmp_path,  # noqa: ANN001
    ndim,  # noqa: ANN001
) -> None:
    """`u_lps = (1, 2[, 3])` is `(-1, -2[, 3])` in RAS, everywhere."""
    u = np.array([1.0, 2.0, 3.0])[:ndim]
    vectors = np.broadcast_to(u.astype("float32"), (*SHAPES[ndim], ndim))
    field = ItkNiftiDisplacementField.from_file(
        _write(tmp_path / "warp.nii.gz", vectors.copy())
    )
    # On and between the grid nodes alike.
    points_ras = np.array(
        [[10.0, -20.0, 30.0], [5.5, -17.0, 37.0], [1.0, -14.5, 45.0]]
    )[:, :ndim]
    moved = _apply_ras(field, points_ras)
    np.testing.assert_allclose(
        moved - points_ras, [u * _flip(ndim)] * 3, atol=1e-5
    )
    np.testing.assert_allclose(u * _flip(ndim), [-1.0, -2.0, 3.0][:ndim])


def test_a_varying_lps_displacement_maps_every_node(itk_warp, ndim) -> None:  # noqa: ANN001
    """At each voxel: `x_lps -> x_lps + u_lps`, i.e.
    `x_ras -> x_ras + (-u_x, -u_y[, u_z])`."""
    field = ItkNiftiDisplacementField.from_file(itk_warp)
    points_ras = _all_nodes_ras(ndim)
    ramp = _ramp(ndim).reshape(-1, ndim)
    moved = _apply_ras(field, points_ras)
    np.testing.assert_allclose(
        moved, points_ras + ramp * _flip(ndim), rtol=1e-5, atol=1e-4
    )
    # ... which in LPS is the stored vector, unchanged.
    points_lps = points_ras * _flip(ndim)
    np.testing.assert_allclose(
        _apply(field, points_lps) - points_lps, ramp, rtol=1e-5, atol=1e-4
    )


def test_a_3d_dispvect_file_holds_ras_vectors(tmp_path) -> None:  # noqa: ANN001
    """
    ITK 5.4+ reads a three-component `DISPVECT` file as RAS and negates
    its first two components, so the same RAS displacement comes out
    whichever of the two intents stored it.
    """
    lps = _write(tmp_path / "lps.nii.gz", _ramp(3), VECTOR)
    ras = _write(tmp_path / "ras.nii.gz", _ramp(3) * _flip(3), DISPVECT)
    points = _all_nodes_ras(3)
    np.testing.assert_allclose(
        _apply_ras(ItkNiftiDisplacementField.from_file(ras), points),
        _apply_ras(ItkNiftiDisplacementField.from_file(lps), points),
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_2d_dispvect_file_is_not_converted(tmp_path) -> None:  # noqa: ANN001
    """ITK converts only a three-component `DISPVECT` image, so a 2-D one
    is read as LPS, like a `VECTOR` one."""
    lps = _write(tmp_path / "lps.nii.gz", _ramp(2), VECTOR)
    disp = _write(tmp_path / "disp.nii.gz", _ramp(2), DISPVECT)
    np.testing.assert_allclose(
        ItkNiftiDisplacementField.from_file(disp).displacement.field,
        ItkNiftiDisplacementField.from_file(lps).displacement.field,
    )


def test_coordinates_and_displacements_agree(tmp_path, ndim) -> None:  # noqa: ANN001
    """A field of LPS positions maps points where the matching field of
    LPS displacements does."""
    disp = _write(tmp_path / "disp.nii.gz", _ramp(ndim))
    positions = _grid_points(_vox2lps(ndim)) + _ramp(ndim)
    coords = _write(tmp_path / "coords.nii.gz", positions.astype("float32"))
    disp = ItkNiftiDisplacementField.from_file(disp)
    coords = ItkNiftiCoordinatesField.from_file(coords)
    assert coords.input == _make_system(ndim)
    assert coords.output == _make_system(ndim)
    assert isinstance(coords.lps2voxel, LPSToVoxel)
    assert isinstance(coords.coordinates, LPSCoordinatesField)
    assert coords.coordinates.output == _make_system(ndim)
    points = _all_nodes_ras(ndim)
    np.testing.assert_allclose(
        _apply_ras(coords, points),
        _apply_ras(disp, points),
        rtol=1e-5,
        atol=1e-4,
    )


def _outside_points(ndim: int, region: str) -> np.ndarray:
    """
    LPS points mostly outside of the field's grid, as `(*shape, ndim)`.

    The grid spans `[-10, 2] x [14, 20] x [30, 50]` mm in LPS (its first
    two axes in 2-D). `"before"` puts every point before the start of
    every axis and `"after"` past its end; `"around"` spans `[-60, 60]` mm
    on every axis, and `"centred"` is a 3x3x3 grid over `[-20, 20]` mm. Points
    that all lie before an axis once made the sampler crop the field to
    nothing and read whatever memory followed it.
    """
    corners = _grid_points(_vox2lps(ndim)).reshape(-1, ndim)
    low, high = corners.min(axis=0), corners.max(axis=0)
    if region == "before":
        axes = [np.linspace(lo - 40.0, lo - 5.0, 3) for lo in low]
    elif region == "after":
        axes = [np.linspace(hi + 5.0, hi + 40.0, 3) for hi in high]
    elif region == "around":
        axes = [np.linspace(-60.0, 60.0, 5)] * ndim
    else:
        axes = [np.linspace(-20.0, 20.0, 3)] * ndim
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _clamped_voxels(points_lps: np.ndarray) -> np.ndarray:
    """The voxel coordinates of LPS points, clamped to the grid."""
    ndim = points_lps.shape[-1]
    lps2vox = np.linalg.inv(_vox2lps(ndim))
    voxels = points_lps @ lps2vox[:ndim, :ndim].T + lps2vox[:ndim, ndim]
    return np.clip(voxels, 0, np.asarray(SHAPES[ndim]) - 1)


@pytest.mark.parametrize(
    "array_backend",
    [
        "numpy",
        pytest.param(
            "dask",
            marks=pytest.mark.skipif(
                "dask" not in available_backends(),
                reason="dask is not installed",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "cls", [ItkNiftiDisplacementField, ItkNiftiCoordinatesField]
)
@pytest.mark.parametrize("region", ["before", "after", "around", "centred"])
def test_points_outside_the_grid_take_the_nearest_vector(
    tmp_path,  # noqa: ANN001
    ndim: int,
    cls,  # noqa: ANN001
    array_backend: str,
    region: str,
) -> None:
    """
    Outside of its grid, a field is extended with its nearest vector, as
    ITK's `DisplacementFieldTransform` does -- and the same on every call.

    The stored vectors are linear in the voxel index, so linear
    interpolation with a `nearest` boundary samples them at the voxel
    clamped to the grid.
    """
    points = _outside_points(ndim, region)
    voxels = _clamped_voxels(points)
    d = np.arange(ndim)
    ramp = (d + 1.0) + 0.1 * (d + 1) * voxels
    vox2lps = _vox2lps(ndim)
    if cls is ItkNiftiCoordinatesField:
        vectors = _grid_points(vox2lps) + _ramp(ndim)
        world = voxels @ vox2lps[:ndim, :ndim].T + vox2lps[:ndim, ndim]
        expected = world + ramp
    else:
        vectors = _ramp(ndim)
        expected = points + ramp
    path = _write(tmp_path / "field.nii.gz", vectors.astype("float32"))
    with backend(array_backend):
        field = cls.from_file(path)
        source = xforms.CoordinatesField(field=points, output=field.input)
        for _ in range(3):
            sequence = xforms.Sequence([source, field])
            out = sequence.compute().to(xforms.CoordinatesField).field
            np.testing.assert_allclose(
                np.asarray(out), expected, rtol=1e-5, atol=1e-4
            )


def test_a_dispvect_file_read_as_itk_coordinates_is_converted_to_lps(
    tmp_path,  # noqa: ANN001
) -> None:
    """
    A three-component `DISPVECT` file holds RAS vectors. Read as ITK LPS
    coordinates, through the class or a hint, they are converted the way
    ITK converts a `DISPVECT` file, so the positions are the same points.
    """
    coords_ras = (_grid_points(VOX2RAS[3]) + 1.5).astype("float32")
    path = tmp_path / "ras.nii.gz"
    _write(path, coords_ras, DISPVECT)
    itk = ItkNiftiCoordinatesField.from_file(path)
    np.testing.assert_allclose(
        np.asarray(itk.coordinates.field) * _flip(3), coords_ras, rtol=1e-6
    )


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


def test_an_itk_vector_file_is_ambiguous_without_a_hint(itk_warp) -> None:  # noqa: ANN001
    """
    `VECTOR` is what ITK writes, but the header does not say which frame
    the vectors are in, so neither reader may guess.
    """
    assert io.transformations.sniff(itk_warp) is None
    with pytest.raises(AmbiguousFormatError):
        io.transformations.load(itk_warp)
    with pytest.raises(AmbiguousFormatError):
        io.load(itk_warp)
    assert ItkNiftiDisplacementField.sniff(itk_warp) == pytest.approx(
        NiftiRASCoordinatesField.sniff(itk_warp)
    )


def test_a_vector_file_named_mapping_is_not_claimed(tmp_path, ndim) -> None:  # noqa: ANN001
    """
    SPM12 and brainhops name their RAS maps `"Mapping"`; ITK writes no
    intent name. So such a file, even in ITK's layout, is left to the RAS
    readers -- and no longer ties -- while a hint still reaches it.
    """
    path = _write(tmp_path / "map.nii.gz", _ramp(ndim), VECTOR)
    img = nb.load(str(path))
    img.header.set_intent(VECTOR, name="Mapping")
    nb.save(img, str(path))
    assert ItkNiftiDisplacementField.sniff(path) == 0
    assert ItkNiftiCoordinatesField.sniff(path) == 0
    sniffed = io.transformations.sniff(path)
    assert sniffed is not None
    assert not issubclass(sniffed, ItkNiftiField)
    assert not isinstance(io.transformations.load(path), ItkNiftiField)
    loaded = io.transformations.load(path, hint="itk")
    assert type(loaded) is ItkNiftiDisplacementField


def test_a_brainhops_coordinates_field_loads_without_a_hint(tmp_path) -> None:  # noqa: ANN001
    """
    brainhops writes a field of RAS coordinates as `VECTOR`, named
    `"Mapping"`, in the same `(X, Y, Z, 1, 3)` layout ITK uses. The name
    is what keeps it from tying with an ITK field, while a bare `VECTOR`
    file in that layout stays ambiguous.
    """
    coords = (_grid_points(VOX2RAS[3]) + 1.5).astype("float32")
    path = tmp_path / "coords.nii.gz"
    NiftiRASCoordinatesField(field=coords).save(path)
    header = nb.load(str(path)).header
    assert header.get_intent() == ("vector", (), "Mapping")
    assert header.get_data_shape() == (*SHAPES[3], 1, 3)
    assert io.transformations.sniff(path) is NiftiRASCoordinatesField
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField
    assert type(io.load(path)) is NiftiRASCoordinatesField

    bare = _write(tmp_path / "bare.nii.gz", coords, VECTOR)
    assert nb.load(str(bare)).header.get_intent()[2] == ""
    with pytest.raises(AmbiguousFormatError):
        io.transformations.load(bare)


@pytest.mark.parametrize(
    "hint",
    [
        "itk",
        "itk.displacements",
        "displacements",
        "ants",
        "ants.displacements",
    ],
)
def test_a_hint_selects_the_itk_displacement_reader(itk_warp, hint) -> None:  # noqa: ANN001
    loaded = io.transformations.load(itk_warp, hint=hint)
    assert type(loaded) is ItkNiftiDisplacementField
    assert type(io.load(itk_warp, hint=hint)) is ItkNiftiDisplacementField


@pytest.mark.parametrize("hint", ["itk.coordinates", "ants.coordinates"])
def test_a_hint_selects_the_itk_coordinates_reader(itk_warp, hint) -> None:  # noqa: ANN001
    loaded = io.transformations.load(itk_warp, hint=hint)
    assert type(loaded) is ItkNiftiCoordinatesField


def test_a_hint_selects_the_ras_reader(tmp_path) -> None:  # noqa: ANN001
    """The RAS reader is still reachable: `coordinates` alone is RAS,
    since the ITK coordinates reader never claims a file on content."""
    path = _write(tmp_path / "warp.nii.gz", _ramp(3))
    loaded = io.transformations.load(path, hint="coordinates")
    assert type(loaded) is NiftiRASCoordinatesField


@pytest.mark.parametrize(
    "intent, ras",
    [
        (DISPVECT, NiftiRASDisplacementField),
        (NONE, NiftiRASCoordinatesField),
    ],
)
def test_non_itk_intents_stay_with_the_ras_reader(
    tmp_path,  # noqa: ANN001
    intent,  # noqa: ANN001
    ras,  # noqa: ANN001
) -> None:
    """ITK never writes these for a vector image by default: `DISPVECT`
    holds RAS displacements, and a field without an intent code is read
    as RAS coordinates."""
    path = _write(tmp_path / "field.nii.gz", _ramp(3), intent)
    assert io.transformations.sniff(path) is ras
    assert type(io.transformations.load(path)) is ras
    # ... but an explicit hint still reads them as ITK.
    for hint in ("itk", "ants"):
        loaded = io.transformations.load(path, hint=hint)
        assert type(loaded) is ItkNiftiDisplacementField


def test_a_2d_field_without_itks_intent_is_not_claimed(tmp_path) -> None:  # noqa: ANN001
    """No RAS reader claims a two-component field either, so it stays the
    plain NIfTI affine it was before, and a hint still reaches it."""
    path = _write(tmp_path / "field.nii.gz", _ramp(2), NONE)
    assert ItkNiftiDisplacementField.sniff(path) == 0
    assert type(io.transformations.load(path)) is NiftiVoxelToRAS
    loaded = io.transformations.load(path, hint="itk")
    assert type(loaded) is ItkNiftiDisplacementField


def test_fsl_intents_are_not_claimed(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "field.nii.gz", _ramp(3), FNIRT)
    assert ItkNiftiDisplacementField.sniff(path) == 0


@pytest.mark.parametrize(
    "shape",
    [
        (4, 5, 6, 3),  # no singleton time axis
        (4, 5, 6, 2, 3),  # a time series of vectors
        (4, 5, 6, 1, 2),  # two components on a 3-D grid
        (4, 5, 1, 1, 4),  # four components
        (4, 5, 6),  # a plain volume
    ],
)
def test_other_layouts_are_not_claimed(tmp_path, shape) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros(shape, "float32"), VOX2RAS[3])
    img.header.set_intent(VECTOR)
    path = tmp_path / "field.nii.gz"
    nb.save(img, str(path))
    assert ItkNiftiDisplacementField.sniff(path) == 0


def test_coordinates_are_never_claimed_on_content(itk_warp) -> None:  # noqa: ANN001
    assert ItkNiftiCoordinatesField.sniff(itk_warp) == 0


# ----------------------------------------------------------------------
#   THE ANTS HINT
# ----------------------------------------------------------------------
#
# ANTs writes its transformations through ITK's writers, so `ants` is an
# alias of `itk`: it must select exactly the readers `itk` selects.


def test_ants_and_itk_name_the_same_readers() -> None:
    readers = FileBasedTransformation._REGISTRY
    itk = {cls for cls in readers if "itk" in format_hints(cls)}
    ants = {cls for cls in readers if "ants" in format_hints(cls)}
    assert itk
    assert ants == itk


def _ants_outputs(tmp_path) -> list:  # noqa: ANN001
    """Files in each ITK format that ANTs writes, and brainhops reads."""
    outputs = [
        # `ConvertTransformFile` text output, and B-spline `BSpline.txt`
        tmp_path / "out0Affine.txt",
        DATA / "itk_affine3d.tfm",
        DATA / "itk_bspline3d.tfm",
        # warps, `<prefix><n>Warp.nii.gz`, in 2-D and 3-D
        _write(tmp_path / "out1Warp.nii.gz", _ramp(3)),
        _write(tmp_path / "out1InverseWarp.nii.gz", _ramp(2)),
    ]
    outputs[0].write_text((DATA / "itk_affine3d.tfm").read_text())
    try:
        import h5py  # noqa: F401
    except ImportError:
        pass
    else:
        # composite transforms, `<prefix>Composite.h5`
        outputs += [
            DATA / "itk_composite_displacement3d.h5",
            DATA / "itk_affine3d.h5",
        ]
    return outputs


def test_the_ants_hint_resolves_as_the_itk_hint(tmp_path) -> None:  # noqa: ANN001
    for path in _ants_outputs(tmp_path):
        itk = io.transformations.load(path, hint="itk")
        ants = io.transformations.load(path, hint="ants")
        assert type(ants) is type(itk), path.name
        assert type(io.load(path, hint="ants")) is type(itk), path.name
        assert io.transformations.sniff(path, hint="ants") is (
            io.transformations.sniff(path, hint="itk")
        )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_itks_encoding_is_written(itk_warp, tmp_path, ndim) -> None:  # noqa: ANN001
    out = tmp_path / "out.nii.gz"
    ItkNiftiDisplacementField.from_file(itk_warp).to_file(out)
    img = nb.load(str(out))
    layout = (*SHAPES[ndim], *(1,) * (3 - ndim), 1, ndim)
    assert img.shape == layout
    assert int(img.header["intent_code"]) == VECTOR
    assert int(img.header["sform_code"]) == 1
    assert int(img.header["qform_code"]) == 1
    np.testing.assert_allclose(img.affine, VOX2RAS[ndim], atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(img.dataobj).reshape(*SHAPES[ndim], ndim),
        _ramp(ndim),
        rtol=1e-6,
    )


def test_a_dispvect_file_is_written_back_as_lps_vectors(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "ras.nii.gz", _ramp(3) * _flip(3), DISPVECT)
    out = tmp_path / "out.nii.gz"
    ItkNiftiDisplacementField.from_file(path).to_file(out)
    img = nb.load(str(out))
    assert int(img.header["intent_code"]) == VECTOR
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[:, :, :, 0, :], _ramp(3), rtol=1e-6
    )


@pytest.mark.parametrize(
    "cls", [ItkNiftiDisplacementField, ItkNiftiCoordinatesField]
)
def test_the_fields_round_trip(tmp_path, cls, ndim) -> None:  # noqa: ANN001
    vectors = _ramp(ndim)
    if cls is ItkNiftiCoordinatesField:
        vectors = (_grid_points(_vox2lps(ndim)) + vectors).astype("float32")
    path = _write(tmp_path / "in.nii.gz", vectors)
    out = tmp_path / "out.nii.gz"
    first = cls.from_file(path)
    first.to_file(out)
    second = cls.from_file(out)
    assert type(second) is cls
    assert second.input == first.input
    assert len(second) == len(first)
    for a, b in zip(first, second):
        assert type(a) is type(b)
    np.testing.assert_allclose(
        np.asarray(second[1].field), np.asarray(first[1].field), rtol=1e-6
    )
    np.testing.assert_allclose(second[0].matrix, first[0].matrix, atol=1e-5)
    points = _all_nodes_ras(ndim)
    np.testing.assert_allclose(
        _apply_ras(second, points),
        _apply_ras(first, points),
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_field_built_in_memory_is_written_in_itks_encoding(ndim) -> None:  # noqa: ANN001
    """With no header to copy, the grid comes from the chain itself."""
    vox2lps = _vox2lps(ndim)
    world = _make_system(ndim)
    vectors = _ramp(ndim)
    field = ItkNiftiDisplacementField(
        transformations=[
            LPSToVoxel(matrix=np.linalg.inv(vox2lps)[:-1], input=world),
            xforms.DisplacementField(
                field=vectors @ np.linalg.inv(vox2lps[:ndim, :ndim]).T
            ),
            VoxelToLPS(matrix=vox2lps[:-1], output=world),
        ]
    )
    assert field.input == world
    img = field.to_nibabel()
    assert img.shape == (*SHAPES[ndim], *(1,) * (3 - ndim), 1, ndim)
    assert int(img.header["intent_code"]) == VECTOR
    assert int(img.header["sform_code"]) == 1
    np.testing.assert_allclose(img.affine, VOX2RAS[ndim], atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(img.dataobj).reshape(*SHAPES[ndim], ndim),
        vectors,
        rtol=1e-5,
        atol=1e-5,
    )


def test_spline_coefficients_are_not_written() -> None:
    voxel = systems.VoxelCoordinateSystem()
    field = ItkNiftiDisplacementField(
        transformations=[
            LPSToVoxel(matrix=np.eye(4)[:3]),
            xforms.DisplacementField(
                field=np.zeros((*SHAPES[3], 3)),
                input=voxel,
                output=voxel,
                order=3,
                coeff=True,
            ),
            VoxelToLPS(matrix=np.eye(4)[:3]),
        ]
    )
    with pytest.raises(WriterError):
        field.to_nibabel()


# ----------------------------------------------------------------------
#   AGAINST ITK ITSELF
# ----------------------------------------------------------------------

_SITK_GEOMETRY = {
    2: dict(
        spacing=(2.0, 3.0),
        origin=(10.0, -20.0),
        direction=(0.0, -1.0, 1.0, 0.0),
        indices=[(0.5, 1.25), (2.0, 3.5), (1.7, 0.2)],
    ),
    3: dict(
        spacing=(2.0, 3.0, 4.0),
        origin=(10.0, -20.0, 30.0),
        direction=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        indices=[(0.5, 1.25, 2.0), (2.0, 3.0, 4.5), (1.7, 0.2, 3.3)],
    ),
}


def test_points_move_where_itk_moves_them(tmp_path, ndim) -> None:  # noqa: ANN001
    """
    A field written by ITK maps points where ITK maps them, and a field
    written by brainhops is mapped by ITK the same way. Needs SimpleITK,
    which the suite does not require.
    """
    sitk = pytest.importorskip("SimpleITK")
    geometry = _SITK_GEOMETRY[ndim]

    # ITK's own image of LPS vectors, with a non-trivial geometry. The
    # array is handed over in ITK's C order: the last spatial axis first.
    array = _ramp(ndim).transpose(*reversed(range(ndim)), ndim)
    image = sitk.GetImageFromArray(array.astype("float64"), isVector=True)
    image.SetSpacing(geometry["spacing"])
    image.SetOrigin(geometry["origin"])
    image.SetDirection(geometry["direction"])
    path = tmp_path / "sitk.nii.gz"
    sitk.WriteImage(image, str(path))

    # What ITK writes is what this module expects: ITK's layout, the
    # `VECTOR` intent, and x and y negated in the RAS geometry.
    header = nb.load(str(path)).header
    assert int(header["intent_code"]) == VECTOR
    assert header.get_data_shape() == (
        *SHAPES[ndim],
        *(1,) * (3 - ndim),
        1,
        ndim,
    )
    origin = header.get_best_affine()[:ndim, 3]
    np.testing.assert_allclose(origin, geometry["origin"] * _flip(ndim))

    def itk_map(file, points_lps):  # noqa: ANN001, ANN202
        field = sitk.Cast(sitk.ReadImage(str(file)), sitk.sitkVectorFloat64)
        transform = sitk.DisplacementFieldTransform(field)
        return np.array([transform.TransformPoint(p) for p in points_lps])

    # Off-grid points inside the field, so interpolation is exercised.
    points = np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint(index)
            for index in geometry["indices"]
        ]
    )

    ours = ItkNiftiDisplacementField.from_file(path)
    np.testing.assert_allclose(
        _apply(ours, points), itk_map(path, points), atol=1e-4
    )

    out = tmp_path / "ours.nii.gz"
    ours.to_file(out)
    np.testing.assert_allclose(
        itk_map(out, points), itk_map(path, points), atol=1e-4
    )
