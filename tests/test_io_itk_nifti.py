"""
ITK displacement and coordinate fields stored as NIfTI vector images.

ITK writes a warp as a `(X, Y, Z, 1, 3)` NIfTI with the `VECTOR` (1007)
intent, a voxel-to-RAS sform/qform, and vector values left in LPS. These
tests pin that encoding numerically -- a known LPS displacement must
move RAS points by the same vector with x and y negated -- and pin how
such a file is told apart from a RAS NIfTI field.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel import systems  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    AmbiguousFormatError,
    WriterError,
)
from brainhops.io.transformations.base.affines import (  # noqa: E402
    LPSToVoxel,
    VoxelToLPS,
)
from brainhops.io.transformations.base.fields import (  # noqa: E402
    LPSCoordinatesField,
)
from brainhops.io.transformations.itk.nifti import (  # noqa: E402
    ITKNiftiCoordinatesField,
    ITKNiftiDisplacementField,
)
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
)

NONE = 0  # NIFTI_INTENT_NONE
DISPVECT = 1006  # NIFTI_INTENT_DISPVECT
VECTOR = 1007  # NIFTI_INTENT_VECTOR, what ITK writes
FNIRT = 2006  # NIFTI_INTENT_FSL_FNIRT_DISPLACEMENT_FIELD

SHAPE = (4, 5, 6)
"""Grid shape: small, and no two axes of the same length."""

FLIP = np.array([-1.0, -1.0, 1.0])
"""RAS <-> LPS, on vectors and on points."""

# A voxel-to-RAS affine with a permutation, a flip, anisotropic spacing
# and an offset, so that a wrong frame or a wrong rotation of the
# vectors cannot cancel out.
VOX2RAS = np.array(
    [
        [0.0, -3.0, 0.0, 10.0],
        [2.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 4.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
VOX2LPS = np.diag([-1.0, -1.0, 1.0, 1.0]) @ VOX2RAS


def _ramp() -> np.ndarray:
    """An `(X, Y, Z, 3)` LPS displacement whose entries name their voxel."""
    i, j, k = np.meshgrid(*map(np.arange, SHAPE), indexing="ij")
    return np.stack(
        [1.0 + 0.1 * i, 2.0 + 0.2 * j, 3.0 + 0.3 * k], axis=-1
    ).astype("float32")


def _grid_points(vox2world: np.ndarray) -> np.ndarray:
    """The world coordinates of every voxel, as an `(X, Y, Z, 3)` array."""
    ijk = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), axis=-1)
    return ijk @ vox2world[:3, :3].T + vox2world[:3, 3]


def _write(path, vectors: np.ndarray, intent: int = VECTOR):  # noqa: ANN001, ANN202
    """Write `(X, Y, Z, 3)` vectors in ITK's NIfTI layout."""
    img = nb.Nifti1Image(vectors[:, :, :, None, :], VOX2RAS)
    img.header.set_intent(intent)
    img.header.set_sform(VOX2RAS, code=1)
    img.header.set_qform(VOX2RAS, code=1)
    nb.save(img, str(path))
    return path


def _apply(xform, points_lps: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map LPS points through an LPS-to-LPS transformation."""
    points = xforms.CoordinatesField(field=np.asarray(points_lps, float))
    out = xforms.Sequence(transformations=[points, *xform]).compute()
    return np.asarray(out.to(xforms.CoordinatesField).field)


def _apply_ras(xform, points_ras: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Map RAS points through an LPS-to-LPS transformation."""
    return _apply(xform, points_ras * FLIP) * FLIP


@pytest.fixture
def itk_warp(tmp_path):  # noqa: ANN001, ANN201
    """A small ITK displacement field, carrying the LPS ramp."""
    return _write(tmp_path / "warp.nii.gz", _ramp())


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_the_field_maps_lps_to_lps(itk_warp) -> None:  # noqa: ANN001
    field = ITKNiftiDisplacementField.from_file(itk_warp)
    assert isinstance(field.input, systems.LPSmm)
    assert isinstance(field.output, systems.LPSmm)
    assert list(field) == [
        field.lps2voxel,
        field.displacement,
        field.voxel2lps,
    ]
    assert isinstance(field.lps2voxel, LPSToVoxel)
    assert isinstance(field.displacement, xforms.DisplacementField)
    assert isinstance(field.voxel2lps, VoxelToLPS)
    np.testing.assert_allclose(field.voxel2lps.matrix, VOX2LPS[:3])


def test_the_grid_is_read_from_the_ras_header_as_lps(itk_warp) -> None:  # noqa: ANN001
    """The header is RAS; ITK's grid is the same one, expressed in LPS."""
    field = ITKNiftiDisplacementField.from_file(itk_warp)
    np.testing.assert_allclose(
        field.lps2voxel.matrix, np.linalg.inv(VOX2LPS)[:3]
    )


def test_the_singleton_time_axis_is_dropped(itk_warp) -> None:  # noqa: ANN001
    field = ITKNiftiDisplacementField.from_file(itk_warp)
    assert np.asarray(field.displacement.field).shape == (*SHAPE, 3)


def test_displacements_are_stored_in_voxel_units(itk_warp) -> None:  # noqa: ANN001
    """A `DisplacementField` adds its values in the units of its grid."""
    field = ITKNiftiDisplacementField.from_file(itk_warp)
    expected = _ramp() @ np.linalg.inv(VOX2LPS[:3, :3]).T
    np.testing.assert_allclose(
        field.displacement.field, expected, rtol=1e-6, atol=1e-6
    )


def test_itks_interpolation_is_kept(itk_warp) -> None:  # noqa: ANN001
    """ITK interpolates a displacement field linearly, and extends it with
    its nearest value."""
    displacement = ITKNiftiDisplacementField.from_file(itk_warp).displacement
    assert displacement.order == 1
    assert displacement.bound == "nearest"
    assert not displacement.coeff


# ----------------------------------------------------------------------
#   SEMANTICS
# ----------------------------------------------------------------------


def test_a_constant_lps_displacement_moves_ras_points_with_x_y_negated(
    tmp_path,  # noqa: ANN001
) -> None:
    """`u_lps = (1, 2, 3)` is `(-1, -2, 3)` in RAS, everywhere."""
    vectors = np.broadcast_to(
        np.array([1.0, 2.0, 3.0], "float32"), (*SHAPE, 3)
    ).copy()
    field = ITKNiftiDisplacementField.from_file(
        _write(tmp_path / "warp.nii.gz", vectors)
    )
    # On and between the grid nodes alike.
    points_ras = np.array(
        [[10.0, -20.0, 30.0], [5.5, -17.0, 37.0], [1.0, -14.5, 45.0]]
    )
    moved = _apply_ras(field, points_ras)
    np.testing.assert_allclose(
        moved - points_ras, [[-1.0, -2.0, 3.0]] * 3, atol=1e-5
    )


def test_a_varying_lps_displacement_maps_every_node(itk_warp) -> None:  # noqa: ANN001
    """At each voxel: `x_lps -> x_lps + u_lps`, i.e.
    `x_ras -> x_ras + (-u_x, -u_y, u_z)`."""
    field = ITKNiftiDisplacementField.from_file(itk_warp)
    points_ras = _grid_points(VOX2RAS).reshape(-1, 3)
    moved = _apply_ras(field, points_ras)
    expected = points_ras + (_ramp() * FLIP).reshape(-1, 3)
    np.testing.assert_allclose(moved, expected, rtol=1e-5, atol=1e-4)
    # ... which in LPS is the stored vector, unchanged.
    moved_lps = _apply(field, points_ras * FLIP)
    np.testing.assert_allclose(
        moved_lps - points_ras * FLIP,
        _ramp().reshape(-1, 3),
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_dispvect_file_holds_ras_vectors(tmp_path) -> None:  # noqa: ANN001
    """
    ITK 5.4+ reads a three-component `DISPVECT` file as RAS and negates
    its first two components, so the same RAS displacement comes out
    whichever of the two intents stored it.
    """
    lps = _write(tmp_path / "lps.nii.gz", _ramp(), VECTOR)
    ras = _write(tmp_path / "ras.nii.gz", _ramp() * FLIP, DISPVECT)
    points = _grid_points(VOX2RAS).reshape(-1, 3)
    np.testing.assert_allclose(
        _apply_ras(ITKNiftiDisplacementField.from_file(ras), points),
        _apply_ras(ITKNiftiDisplacementField.from_file(lps), points),
        rtol=1e-5,
        atol=1e-4,
    )


def test_coordinates_and_displacements_agree(tmp_path) -> None:  # noqa: ANN001
    """A field of LPS positions maps points where the matching field of
    LPS displacements does."""
    disp = _write(tmp_path / "disp.nii.gz", _ramp())
    coords = _write(
        tmp_path / "coords.nii.gz",
        (_grid_points(VOX2LPS) + _ramp()).astype("float32"),
    )
    disp = ITKNiftiDisplacementField.from_file(disp)
    coords = ITKNiftiCoordinatesField.from_file(coords)
    assert isinstance(coords.input, systems.LPSmm)
    assert isinstance(coords.output, systems.LPSmm)
    assert isinstance(coords.lps2voxel, LPSToVoxel)
    assert isinstance(coords.coordinates, LPSCoordinatesField)
    points = _grid_points(VOX2RAS).reshape(-1, 3)
    np.testing.assert_allclose(
        _apply_ras(coords, points),
        _apply_ras(disp, points),
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_ras_field_read_as_itk_coordinates_maps_the_same_points(
    tmp_path,  # noqa: ANN001
) -> None:
    """
    A RAS coordinates field written by brainhops (`DISPVECT`) means the
    same map when read as ITK LPS coordinates: its vectors are converted
    the way ITK converts a `DISPVECT` file.
    """
    coords_ras = (_grid_points(VOX2RAS) + 1.5).astype("float32")
    path = tmp_path / "ras.nii.gz"
    _write(path, coords_ras, DISPVECT)
    itk = ITKNiftiCoordinatesField.from_file(path)
    np.testing.assert_allclose(
        np.asarray(itk.coordinates.field) * FLIP, coords_ras, rtol=1e-6
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
    assert ITKNiftiDisplacementField.sniff(itk_warp) == pytest.approx(
        NiftiRASCoordinatesField.sniff(itk_warp)
    )


@pytest.mark.parametrize("hint", ["itk", "itk.displacements", "displacements"])
def test_a_hint_selects_the_itk_displacement_reader(itk_warp, hint) -> None:  # noqa: ANN001
    loaded = io.transformations.load(itk_warp, hint=hint)
    assert type(loaded) is ITKNiftiDisplacementField
    assert type(io.load(itk_warp, hint=hint)) is ITKNiftiDisplacementField


def test_a_hint_selects_the_itk_coordinates_reader(itk_warp) -> None:  # noqa: ANN001
    loaded = io.transformations.load(itk_warp, hint="itk.coordinates")
    assert type(loaded) is ITKNiftiCoordinatesField


def test_a_hint_selects_the_ras_reader(itk_warp) -> None:  # noqa: ANN001
    """The RAS reader is still reachable: `coordinates` alone is RAS,
    since the ITK coordinates reader never claims a file on content."""
    loaded = io.transformations.load(itk_warp, hint="coordinates")
    assert type(loaded) is NiftiRASCoordinatesField


@pytest.mark.parametrize("intent", [DISPVECT, NONE])
def test_non_itk_intents_stay_with_the_ras_reader(tmp_path, intent) -> None:  # noqa: ANN001
    """ITK never writes these for a vector image by default."""
    path = _write(tmp_path / "field.nii.gz", _ramp(), intent)
    assert io.transformations.sniff(path) is NiftiRASCoordinatesField
    assert type(io.transformations.load(path)) is NiftiRASCoordinatesField
    # ... but an explicit hint still reads them as ITK.
    loaded = io.transformations.load(path, hint="itk")
    assert type(loaded) is ITKNiftiDisplacementField


def test_fsl_intents_are_not_claimed(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "field.nii.gz", _ramp(), FNIRT)
    assert ITKNiftiDisplacementField.sniff(path) == 0


@pytest.mark.parametrize(
    "shape",
    [
        (*SHAPE, 3),  # no singleton time axis
        (*SHAPE, 2, 3),  # a time series of vectors
        (4, 5, 1, 1, 2),  # a 2-D field, which is not supported
        SHAPE,  # a plain volume
    ],
)
def test_other_layouts_are_not_claimed(tmp_path, shape) -> None:  # noqa: ANN001
    img = nb.Nifti1Image(np.zeros(shape, "float32"), VOX2RAS)
    img.header.set_intent(VECTOR)
    path = tmp_path / "field.nii.gz"
    nb.save(img, str(path))
    assert ITKNiftiDisplacementField.sniff(path) == 0


def test_coordinates_are_never_claimed_on_content(itk_warp) -> None:  # noqa: ANN001
    assert ITKNiftiCoordinatesField.sniff(itk_warp) == 0


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def test_itks_encoding_is_written(itk_warp, tmp_path) -> None:  # noqa: ANN001
    out = tmp_path / "out.nii.gz"
    ITKNiftiDisplacementField.from_file(itk_warp).to_file(out)
    img = nb.load(str(out))
    assert img.shape == (*SHAPE, 1, 3)
    assert int(img.header["intent_code"]) == VECTOR
    assert int(img.header["sform_code"]) == 1
    assert int(img.header["qform_code"]) == 1
    np.testing.assert_allclose(img.affine, VOX2RAS, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[:, :, :, 0, :], _ramp(), rtol=1e-6
    )


def test_a_dispvect_file_is_written_back_as_lps_vectors(tmp_path) -> None:  # noqa: ANN001
    path = _write(tmp_path / "ras.nii.gz", _ramp() * FLIP, DISPVECT)
    out = tmp_path / "out.nii.gz"
    ITKNiftiDisplacementField.from_file(path).to_file(out)
    img = nb.load(str(out))
    assert int(img.header["intent_code"]) == VECTOR
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[:, :, :, 0, :], _ramp(), rtol=1e-6
    )


@pytest.mark.parametrize(
    "cls, vectors",
    [
        (ITKNiftiDisplacementField, _ramp()),
        (
            ITKNiftiCoordinatesField,
            (_grid_points(VOX2LPS) + _ramp()).astype("float32"),
        ),
    ],
)
def test_the_fields_round_trip(tmp_path, cls, vectors) -> None:  # noqa: ANN001
    path = _write(tmp_path / "in.nii.gz", vectors)
    out = tmp_path / "out.nii.gz"
    first = cls.from_file(path)
    first.to_file(out)
    second = cls.from_file(out)
    assert type(second) is cls
    assert len(second) == len(first)
    for a, b in zip(first, second):
        assert type(a) is type(b)
    np.testing.assert_allclose(
        np.asarray(second[1].field), np.asarray(first[1].field), rtol=1e-6
    )
    np.testing.assert_allclose(second[0].matrix, first[0].matrix, atol=1e-5)
    points = _grid_points(VOX2RAS).reshape(-1, 3)
    np.testing.assert_allclose(
        _apply_ras(second, points),
        _apply_ras(first, points),
        rtol=1e-5,
        atol=1e-4,
    )


def test_a_field_built_in_memory_is_written_in_itks_encoding() -> None:
    """With no header to copy, the grid comes from the chain itself."""
    vox2lps = VOX2LPS[:3]
    lps2vox = np.linalg.inv(VOX2LPS)[:3]
    voxel = systems.VoxelCoordinateSystem()
    vectors = _ramp()
    field = ITKNiftiDisplacementField(
        transformations=[
            LPSToVoxel(matrix=lps2vox),
            xforms.DisplacementField(
                field=vectors @ np.linalg.inv(VOX2LPS[:3, :3]).T,
                input=voxel,
                output=voxel,
            ),
            VoxelToLPS(matrix=vox2lps),
        ]
    )
    img = field.to_nibabel()
    assert img.shape == (*SHAPE, 1, 3)
    assert int(img.header["intent_code"]) == VECTOR
    assert int(img.header["sform_code"]) == 1
    np.testing.assert_allclose(img.affine, VOX2RAS, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(img.dataobj)[:, :, :, 0, :], vectors, rtol=1e-5, atol=1e-5
    )


def test_spline_coefficients_are_not_written() -> None:
    voxel = systems.VoxelCoordinateSystem()
    field = ITKNiftiDisplacementField(
        transformations=[
            LPSToVoxel(matrix=np.eye(4)[:3]),
            xforms.DisplacementField(
                field=np.zeros((*SHAPE, 3)),
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


def test_points_move_where_itk_moves_them(tmp_path) -> None:  # noqa: ANN001
    """
    A field written by ITK maps points where ITK maps them, and a field
    written by brainhops is mapped by ITK the same way. Needs SimpleITK,
    which the suite does not require.
    """
    sitk = pytest.importorskip("SimpleITK")

    # ITK's own image of LPS vectors, with a non-trivial geometry.
    image = sitk.GetImageFromArray(
        _ramp().transpose(2, 1, 0, 3).astype("float64"), isVector=True
    )
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, -20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "sitk.nii.gz"
    sitk.WriteImage(image, str(path))

    # What ITK writes is what this module expects.
    header = nb.load(str(path)).header
    assert int(header["intent_code"]) == VECTOR
    assert header.get_data_shape() == (*SHAPE, 1, 3)

    def itk_map(file, points_lps):  # noqa: ANN001, ANN202
        field = sitk.Cast(sitk.ReadImage(str(file)), sitk.sitkVectorFloat64)
        transform = sitk.DisplacementFieldTransform(field)
        return np.array([transform.TransformPoint(p) for p in points_lps])

    # Off-grid points inside the field, so interpolation is exercised.
    points = np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint(index)
            for index in [(0.5, 1.25, 2.0), (2.0, 3.0, 4.5), (1.7, 0.2, 3.3)]
        ]
    )

    ours = ITKNiftiDisplacementField.from_file(path)
    np.testing.assert_allclose(
        _apply(ours, points), itk_map(path, points), atol=1e-4
    )

    out = tmp_path / "ours.nii.gz"
    ours.to_file(out)
    np.testing.assert_allclose(
        itk_map(out, points), itk_map(path, points), atol=1e-4
    )
