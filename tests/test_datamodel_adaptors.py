"""Tests for the coordinate-system adaptor.

The adaptor bridges two coordinate systems that meet at a composition
boundary. These tests exercise the matching kernel and the bridge it
builds against the named systems in ``systems.py``, the wiring that
inserts a bridge inside sequence composition, and two end-to-end
demonstrations that apply an FSL and an ITK transform across an image's
coordinate system.
"""

from pathlib import Path

import numpy as np
import pytest

import brainhops.datamodel  # noqa: F401  (registers the adaptor)
from brainhops.datamodel._xform_adaptors import adapt, bridge
from brainhops.datamodel.axes import SpatialAxis
from brainhops.datamodel.orientation import (
    LeftToRight,
    Orientation,
    PosteriorToAnterior,
    RightToLeft,
)
from brainhops.datamodel.systems import (
    CoordinateSystem,
    CRASCoordinateSystem,
    CVoxelCoordinateSystem,
    FRASCoordinateSystem,
    FVoxelCoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    AdaptationError,
    Affine,
    Identity,
    Permutation,
    Scaling,
    Sequence,
    Transformation,
    is_identity,
)

data_dir = Path(__file__).parent / "data"


def _homogeneous(transformation: Transformation) -> np.ndarray:
    """The homogeneous affine a transformation reduces to."""
    return np.asarray(transformation.compute().to(Affine).homogeneous_matrix)


# ----------------------------------------------------------------------
#   MATCHING KERNEL AGAINST THE NAMED SYSTEMS
# ----------------------------------------------------------------------


def test_equal_systems_need_no_bridge() -> None:
    result = bridge(RASCoordinateSystem(), RASCoordinateSystem())
    assert isinstance(result, Identity)


def test_ras_to_lps_is_a_sign_flip() -> None:
    result = bridge(RASCoordinateSystem(), LPSCoordinateSystem())
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


def test_lps_to_ras_is_the_same_flip() -> None:
    result = bridge(LPSCoordinateSystem(), RASCoordinateSystem())
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


def test_fvoxel_to_cvoxel_is_a_reversal_permutation() -> None:
    result = bridge(FVoxelCoordinateSystem(), CVoxelCoordinateSystem())
    assert isinstance(result, Permutation)
    np.testing.assert_array_equal(result.permutation, [2, 1, 0])


def test_fras_to_cras_is_a_reversal_permutation() -> None:
    result = bridge(FRASCoordinateSystem(), CRASCoordinateSystem())
    assert isinstance(result, Permutation)
    np.testing.assert_array_equal(result.permutation, [2, 1, 0])


@pytest.mark.parametrize(
    "source, target",
    [
        (RASCoordinateSystem(), LPSCoordinateSystem()),
        (FVoxelCoordinateSystem(), CVoxelCoordinateSystem()),
        (FRASCoordinateSystem(), CRASCoordinateSystem()),
    ],
)
def test_bridge_round_trips_to_identity(
    source: CoordinateSystem, target: CoordinateSystem
) -> None:
    # A bridge and its opposite compose to the identity and simplify away,
    # because the adaptor emits only exactly-invertible pieces.
    forward = bridge(source, target)
    backward = bridge(target, source)
    result = Sequence([forward, backward]).compute()
    assert is_identity(result, compute=True)


def test_matches_by_orientation_when_names_differ() -> None:
    # The source names its axes x, y, z; the target names them by their
    # anatomical orientation. They still match on (type, orientation).
    source = CoordinateSystem(
        name="named",
        axes=[
            SpatialAxis(name="x", orientation=LeftToRight()),
            SpatialAxis(name="y", orientation=PosteriorToAnterior()),
        ],
    )
    target = CoordinateSystem(
        name="oriented",
        axes=[
            SpatialAxis(name="ap", orientation=PosteriorToAnterior()),
            SpatialAxis(name="lr", orientation=LeftToRight()),
        ],
    )
    result = bridge(source, target)
    # x (left-to-right) feeds target axis 1, y (posterior-to-anterior)
    # feeds target axis 0.
    assert isinstance(result, Permutation)
    np.testing.assert_array_equal(result.permutation, [1, 0])


# ----------------------------------------------------------------------
#   UNITS
# ----------------------------------------------------------------------


def test_unit_difference_is_a_scaling() -> None:
    source = CoordinateSystem(
        name="mm", axes=[SpatialAxis(name="x", unit="millimeter")]
    )
    target = CoordinateSystem(
        name="um", axes=[SpatialAxis(name="x", unit="micrometer")]
    )
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    # One millimetre is a thousand micrometres.
    np.testing.assert_allclose(result.scale, [1000.0])


def test_unit_present_on_one_side_only_raises() -> None:
    source = CoordinateSystem(
        name="mm",
        axes=[
            SpatialAxis(
                name="x", unit="millimeter", orientation=LeftToRight()
            )
        ],
    )
    target = CoordinateSystem(
        name="index",
        axes=[SpatialAxis(name="x", unit=None, orientation=LeftToRight())],
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


# ----------------------------------------------------------------------
#   ORIENTATION-FLIP ORIGIN OFFSET (array-index versus world)
# ----------------------------------------------------------------------


def _oriented_index_system(
    name: str, orientation: Orientation
) -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=[SpatialAxis(name="i", unit=None, orientation=orientation)],
    )


def test_world_flip_carries_no_offset() -> None:
    # Between world systems a reversed axis is a pure sign flip.
    source = CoordinateSystem(
        name="R",
        axes=[
            SpatialAxis(
                name="x", unit="millimeter", orientation=LeftToRight()
            )
        ],
    )
    target = CoordinateSystem(
        name="L",
        axes=[
            SpatialAxis(
                name="x", unit="millimeter", orientation=RightToLeft()
            )
        ],
    )
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0])


def test_array_index_flip_carries_the_extent_offset() -> None:
    # Between array-index systems a reversed axis of extent n maps index x
    # to index (n - 1) - x, a sign flip followed by a shift of (n - 1).
    n = 10
    source = _oriented_index_system("LR", LeftToRight())
    target = _oriented_index_system("RL", RightToLeft())
    result = bridge(source, target, extents=[n])
    matrix = np.asarray(result.compute().to(Affine).homogeneous_matrix)
    np.testing.assert_array_equal(matrix, [[-1.0, n - 1], [0.0, 1.0]])
    for x in (0, 9, 4):
        y = (matrix @ np.array([float(x), 1.0]))[0]
        assert y == pytest.approx((n - 1) - x)


def test_array_index_flip_without_extent_raises() -> None:
    source = _oriented_index_system("LR", LeftToRight())
    target = _oriented_index_system("RL", RightToLeft())
    with pytest.raises(AdaptationError):
        bridge(source, target)


def test_array_index_flip_round_trips() -> None:
    n = 7
    source = _oriented_index_system("LR", LeftToRight())
    target = _oriented_index_system("RL", RightToLeft())
    result = Sequence(
        [
            bridge(source, target, extents=[n]),
            bridge(target, source, extents=[n]),
        ]
    ).compute()
    assert is_identity(result, compute=True)


# ----------------------------------------------------------------------
#   FAILURE POLICY
# ----------------------------------------------------------------------


def test_unmatched_axis_raises_by_default() -> None:
    source = CoordinateSystem(
        name="ab", axes=[SpatialAxis(name="a"), SpatialAxis(name="b")]
    )
    target = CoordinateSystem(
        name="cd", axes=[SpatialAxis(name="c"), SpatialAxis(name="d")]
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


def test_positional_fallback_is_opt_in_and_warns() -> None:
    source = CoordinateSystem(
        name="ab", axes=[SpatialAxis(name="a"), SpatialAxis(name="b")]
    )
    target = CoordinateSystem(
        name="cd", axes=[SpatialAxis(name="c"), SpatialAxis(name="d")]
    )
    with pytest.warns(UserWarning):
        result = bridge(source, target, allow_positional=True)
    # Same names paired by position, same units: nothing to change.
    assert is_identity(result, compute=True)


# ----------------------------------------------------------------------
#   THE adapt() CONTRACT: A CONTAINER OF BOTH TRANSFORMS
# ----------------------------------------------------------------------


def test_adapt_returns_a_sequence_containing_both_transforms() -> None:
    first = Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=LPSCoordinateSystem(),
        output=LPSCoordinateSystem(),
    )
    result = adapt(first, second)
    assert isinstance(result, Sequence)
    assert result.transformations[0] is first
    assert result.transformations[-1] is second
    # A flip bridge sits between the two ingested transforms.
    assert any(isinstance(t, Scaling) for t in result.transformations)


def test_adapt_without_mismatch_inserts_no_bridge() -> None:
    first = Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    result = adapt(first, second)
    assert list(result.transformations) == [first, second]


# ----------------------------------------------------------------------
#   WIRING INTO SEQUENCE COMPOSITION
# ----------------------------------------------------------------------


def test_sequence_composition_inserts_the_flip_bridge() -> None:
    rng = np.random.default_rng(0)
    first_matrix = rng.standard_normal((3, 4))
    second_matrix = rng.standard_normal((3, 4))
    first = Affine(
        matrix=first_matrix,
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    second = Affine(
        matrix=second_matrix,
        input=LPSCoordinateSystem(),
        output=LPSCoordinateSystem(),
    )
    result = Sequence([first, second]).compute()

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])

    def homogeneous(matrix: np.ndarray) -> np.ndarray:
        stacked = np.eye(4)
        stacked[:3, :] = matrix
        return stacked

    expected = homogeneous(second_matrix) @ flip @ homogeneous(first_matrix)
    np.testing.assert_allclose(_homogeneous(result), expected)


def test_matching_systems_compose_without_a_bridge() -> None:
    rng = np.random.default_rng(1)
    first_matrix = rng.standard_normal((3, 4))
    second_matrix = rng.standard_normal((3, 4))
    first = Affine(
        matrix=first_matrix,
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    second = Affine(
        matrix=second_matrix,
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    result = Sequence([first, second]).compute()

    def homogeneous(matrix: np.ndarray) -> np.ndarray:
        stacked = np.eye(4)
        stacked[:3, :] = matrix
        return stacked

    expected = homogeneous(second_matrix) @ homogeneous(first_matrix)
    np.testing.assert_allclose(_homogeneous(result), expected)


def test_opposite_bridges_cancel_in_a_sequence() -> None:
    # A round trip RAS -> LPS -> RAS composes to the identity, so a
    # sequence that crosses into LPS and back leaves no residue.
    matrix = np.eye(3, 4)
    ras_to_lps = Affine(
        matrix=matrix,
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    lps = Affine(
        matrix=matrix,
        input=LPSCoordinateSystem(),
        output=LPSCoordinateSystem(),
    )
    lps_to_ras = Affine(
        matrix=matrix,
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    result = Sequence([ras_to_lps, lps, lps_to_ras]).compute()
    assert is_identity(result, compute=True)


# ----------------------------------------------------------------------
#   END-TO-END: A TRANSFORM ACROSS AN IMAGE'S COORDINATE SYSTEM
# ----------------------------------------------------------------------


def test_itk_transform_applied_to_a_nifti_image_bridges_ras_and_lps() -> None:
    """Apply an ITK (LPS) transform after a NIfTI image's voxel-to-RAS map.

    The NIfTI image is placed in RAS world coordinates by its header. The
    ITK transform acts in LPS. Composing the two crosses the RAS/LPS
    boundary, and the adaptor inserts the sign flip that reconciles them.
    """
    images = pytest.importorskip("brainhops.io.images")
    transformations = pytest.importorskip("brainhops.io.transformations")

    image = images.load(str(data_dir / "fsl" / "ref.nii.gz"))
    voxel_to_world = image.transformation  # voxel -> RAS
    # An ITK affine acts in LPS.
    itk = transformations.load(str(data_dir / "itk_affine3d.tfm"))

    result = Sequence([voxel_to_world, itk]).compute()

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected = _homogeneous(itk) @ flip @ np.asarray(
        voxel_to_world.homogeneous_matrix
    )
    got = _homogeneous(result)
    np.testing.assert_allclose(got, expected)
    # The flip is load-bearing: without it the composition differs.
    assert not np.allclose(got, _homogeneous(itk) @ np.asarray(
        voxel_to_world.homogeneous_matrix
    ))


def test_fsl_transform_applied_to_a_zarr_image_bridges_lps_and_ras(
    tmp_path: Path,
) -> None:
    """Apply an FSL (RAS) transform after a Zarr image's voxel-to-LPS map.

    The Zarr image is read with a supplied voxel-to-world geometry that
    places it in LPS. The FSL transform acts in RAS. The adaptor bridges
    the LPS/RAS boundary with the sign flip when the two are composed.
    """
    pytest.importorskip("abczarr")
    nb = pytest.importorskip("nibabel")
    from brainhops.io.images.zarr import ZarrImage
    from brainhops.io.transformations.fsl.flirt import FLIRTTransform

    path = str(tmp_path / "image.zarr")
    data = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    ZarrImage(data=data).save(path)

    voxel_to_lps = Affine(
        matrix=np.array(
            [
                [-1.5, 0.0, 0.0, 2.0],
                [0.0, -1.5, 0.0, 3.0],
                [0.0, 0.0, 1.5, 4.0],
            ]
        ),
        output=LPSCoordinateSystem(),
    )
    image = ZarrImage.load(path, transformation=voxel_to_lps)
    voxel_to_world = image.transformation  # voxel -> LPS

    reference = nb.load(str(data_dir / "fsl" / "ref.nii.gz"))
    moving = nb.load(str(data_dir / "fsl" / "src.nii.gz"))
    flirt_matrix = np.eye(4)
    flirt_matrix[0, 3] = 5.0
    fsl = FLIRTTransform(
        flirt_matrix=flirt_matrix, reference=reference, moving=moving
    )  # RAS -> RAS

    result = Sequence([voxel_to_world, fsl]).compute()

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS -> RAS
    expected = _homogeneous(fsl) @ flip @ _homogeneous(voxel_to_world)
    got = _homogeneous(result)
    np.testing.assert_allclose(got, expected)
    without_flip = _homogeneous(fsl) @ _homogeneous(voxel_to_world)
    assert not np.allclose(got, without_flip)
