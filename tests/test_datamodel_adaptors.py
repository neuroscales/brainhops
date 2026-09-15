"""Tests for the coordinate-system adaptor.

The adaptor bridges two coordinate systems that meet at a composition
boundary. These tests exercise the matching kernel and the bridge it
builds against the named systems in ``systems.py``, the wiring that
inserts a bridge inside sequence composition, and two end-to-end
demonstrations that apply an FSL and an ITK transform across an image's
coordinate system.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import brainhops.datamodel  # noqa: F401  (registers the adaptor)
from brainhops.datamodel._xform_adaptors import adapt, bridge
from brainhops.datamodel.axes import (
    A,
    R,
    S,
    SpatialAxis,
    TimeAxis,
)
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
    FLPSCoordinateSystem,
    FRASCoordinateSystem,
    FVoxelCoordinateSystem,
    LPSCoordinateSystem,
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    AdaptationError,
    Affine,
    CartesianField,
    Identity,
    Inverse,
    Permutation,
    Scaling,
    Sequence,
    SubspaceTransformation,
    Transformation,
    Translation,
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
            SpatialAxis(name="x", unit="millimeter", orientation=LeftToRight())
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
            SpatialAxis(name="x", unit="millimeter", orientation=LeftToRight())
        ],
    )
    target = CoordinateSystem(
        name="L",
        axes=[
            SpatialAxis(name="x", unit="millimeter", orientation=RightToLeft())
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
    expected = (
        _homogeneous(itk)
        @ flip
        @ np.asarray(voxel_to_world.homogeneous_matrix)
    )
    got = _homogeneous(result)
    np.testing.assert_allclose(got, expected)
    # The flip is load-bearing: without it the composition differs.
    assert not np.allclose(
        got, _homogeneous(itk) @ np.asarray(voxel_to_world.homogeneous_matrix)
    )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="zarr I/O requires zarr-python 3 (Python 3.11+)",
)
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


# ----------------------------------------------------------------------
#   REGRESSION: BRIDGING PAST NESTING AND BARE INVERSES (finding 1)
# ----------------------------------------------------------------------


def _ras_affine() -> Affine:
    return Affine(
        matrix=np.eye(3, 4),
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )


def _lps_affine() -> Affine:
    return Affine(
        matrix=np.eye(3, 4),
        input=LPSCoordinateSystem(),
        output=LPSCoordinateSystem(),
    )


def test_singly_nested_sequence_inserts_the_flip_bridge() -> None:
    # A mismatch that sits inside a nested sequence is bridged just as it
    # would be at the top level. Without recursing into the nesting, the
    # RAS/LPS boundary would compose with no flip.
    a, b = _ras_affine(), _lps_affine()
    flat = Sequence([a, b]).compute()
    nested = Sequence([Sequence([a, b])]).compute()
    np.testing.assert_allclose(_homogeneous(nested), _homogeneous(flat))
    np.testing.assert_array_equal(
        np.diag(_homogeneous(nested)), [-1, -1, 1, 1]
    )


def test_nested_sequence_beside_a_sibling_inserts_the_flip_bridge() -> None:
    a, b = _ras_affine(), _lps_affine()
    c = _lps_affine()
    flat = Sequence([a, b]).compute()
    nested = Sequence([Sequence([a, b]), c]).compute()
    # The A -> B boundary inside the nested sequence still gets its flip.
    inner_flip = _homogeneous(c) @ _homogeneous(flat)
    np.testing.assert_allclose(_homogeneous(nested), inner_flip)


def test_bare_inverse_boundary_inserts_the_flip_bridge() -> None:
    # A boundary hidden behind a generic Inverse(forward=B) is bridged
    # against the system the inverse actually presents. Inverting an
    # LPS -> LPS affine presents LPS at the boundary, so an RAS output
    # meeting it needs the flip.
    a = _ras_affine()
    b = _lps_affine()
    result = Sequence([a, Inverse(forward=b)]).compute()
    np.testing.assert_array_equal(
        np.diag(_homogeneous(result)), [-1, -1, 1, 1]
    )


# ----------------------------------------------------------------------
#   REGRESSION: IMPLICIT POSITIONAL FALLBACK FOR ARRAY AXES (finding 2)
# ----------------------------------------------------------------------


def _xyz_index_system() -> CoordinateSystem:
    return CoordinateSystem(
        name="xyz",
        axes=[
            SpatialAxis(name="x", unit=None),
            SpatialAxis(name="y", unit=None),
            SpatialAxis(name="z", unit=None),
        ],
    )


def test_compute_pairs_underspecified_voxel_axes_by_position_and_warns() -> (
    None
):
    # dim0/dim1/dim2 and x/y/z are unoriented and unitless, so compute()
    # pairs them by position and warns, rather than raising.
    first = Affine(
        matrix=np.eye(3, 4),
        input=VoxelCoordinateSystem(),
        output=VoxelCoordinateSystem(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=_xyz_index_system(),
        output=_xyz_index_system(),
    )
    with pytest.warns(UserWarning):
        result = Sequence([first, second]).compute()
    np.testing.assert_allclose(_homogeneous(result), np.eye(4))


def test_explicit_bridge_does_not_fall_back_to_position() -> None:
    # The explicit entry point keeps error-by-default. The implicit
    # fallback is reached only through compute().
    with pytest.raises(AdaptationError):
        bridge(VoxelCoordinateSystem(), _xyz_index_system())


def test_oriented_against_unoriented_is_not_paired_by_position() -> None:
    # When two axes of the same type remain on each side, so the pairing is
    # no longer unambiguous, an oriented axis and an unoriented one are a
    # genuine mismatch. The implicit fallback declines and it is reported,
    # rather than pairing them by position.
    oriented = CoordinateSystem(
        name="oriented",
        axes=[
            SpatialAxis(name="a", unit=None, orientation=LeftToRight()),
            SpatialAxis(name="b", unit=None),
        ],
    )
    plain = CoordinateSystem(
        name="plain",
        axes=[
            SpatialAxis(name="c", unit=None),
            SpatialAxis(name="d", unit=None),
        ],
    )
    first = Affine(matrix=np.eye(2, 3), input=oriented, output=oriented)
    second = Affine(matrix=np.eye(2, 3), input=plain, output=plain)
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


# ----------------------------------------------------------------------
#   REGRESSION: ARRAY FLIP INSIDE compute() VIA GRID EXTENTS (finding 3)
# ----------------------------------------------------------------------


def _oriented_named_index_system(
    name: str, orientation: Orientation
) -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=[SpatialAxis(name="i", unit=None, orientation=orientation)],
    )


def test_array_flip_in_a_sequence_uses_an_adjacent_grid_extent() -> None:
    # A reversed array-index axis needs its extent. An interior grid
    # carries it as its shape, and compute() threads it into the bridge.
    lr = _oriented_named_index_system("LR", LeftToRight())
    rl = _oriented_named_index_system("RL", RightToLeft())
    n = 10
    before = Affine(matrix=np.eye(1, 2), input=lr, output=lr)
    grid = CartesianField(shape=(n,), input=lr, output=lr)
    after = Affine(matrix=np.eye(1, 2), input=rl, output=rl)
    result = Sequence([before, grid, after]).compute()
    matrix = np.asarray(result.to(Affine).homogeneous_matrix)
    np.testing.assert_array_equal(matrix, [[-1.0, n - 1], [0.0, 1.0]])


def test_array_flip_without_an_extent_source_raises_actionably() -> None:
    # With no grid to supply the extent, the error tells the caller to
    # bridge explicitly with extents, from where the caller actually is.
    lr = _oriented_named_index_system("LR", LeftToRight())
    rl = _oriented_named_index_system("RL", RightToLeft())
    before = Affine(matrix=np.eye(1, 2), input=lr, output=lr)
    after = Affine(matrix=np.eye(1, 2), input=rl, output=rl)
    with pytest.raises(AdaptationError) as excinfo:
        Sequence([before, after]).compute()
    message = str(excinfo.value)
    assert "extents=" in message
    assert "adapt(" in message or "bridge(" in message


# ----------------------------------------------------------------------
#   REGRESSION: NON-COLLINEAR ORIENTED AXES BY NAME (finding 4)
# ----------------------------------------------------------------------


def test_non_collinear_axes_matched_by_name_are_rejected() -> None:
    # x pointing left-to-right and x pointing anterior-to-posterior lie on
    # different anatomical lines, so aligning them is a rotation, not a
    # flip. The shared name must not force a spurious sign flip.
    source = CoordinateSystem(
        name="lr",
        axes=[SpatialAxis(name="x", orientation=LeftToRight())],
    )
    target = CoordinateSystem(
        name="ap",
        axes=[SpatialAxis(name="x", orientation=PosteriorToAnterior())],
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


# ----------------------------------------------------------------------
#   REGRESSION: EXACT UNIT RATIOS (finding 5)
# ----------------------------------------------------------------------


def test_unit_ratio_round_trip_is_exact() -> None:
    # A ratio and its reciprocal multiply back to exactly one, which a
    # division of the two scales does not guarantee (mm / um is
    # 1000.0000000000001).
    def _system(unit: str) -> CoordinateSystem:
        return CoordinateSystem(
            name=unit, axes=[SpatialAxis(name="x", unit=unit)]
        )

    pairs = [
        ("millimeter", "micrometer"),
        ("millimeter", "nanometer"),
        ("micrometer", "nanometer"),
        ("centimeter", "millimeter"),
        ("meter", "millimeter"),
    ]
    for a, b in pairs:
        forward = bridge(_system(a), _system(b))
        backward = bridge(_system(b), _system(a))
        product = float(forward.scale[0]) * float(backward.scale[0])
        assert product == 1.0
    # mm -> um is exactly a thousand.
    mm_to_um = bridge(_system("millimeter"), _system("micrometer"))
    assert float(mm_to_um.scale[0]) == 1000.0


# ----------------------------------------------------------------------
#   REGRESSION: NAMED VOXEL SYSTEMS CLASSIFY AS ARRAY-INDEX (finding 6)
# ----------------------------------------------------------------------


def test_named_ras_lps_voxel_systems_are_array_index() -> None:
    # fRAS and fLPS are voxel grids, even though their axes carry a
    # millimetre unit and an orientation. A flip between them is an
    # array-index flip, so it carries the origin offset and needs the
    # extent.
    with pytest.raises(AdaptationError):
        bridge(FRASCoordinateSystem(), FLPSCoordinateSystem())
    result = bridge(
        FRASCoordinateSystem(),
        FLPSCoordinateSystem(),
        extents={"x": 4, "y": 5, "z": 6},
    )
    matrix = np.asarray(result.compute().to(Affine).homogeneous_matrix)
    # x and y are reversed (offsets 3 and 4); z is unchanged.
    np.testing.assert_array_equal(np.diag(matrix), [-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_array_equal(matrix[:3, 3], [3.0, 4.0, 0.0])


def test_world_ras_lps_flip_stays_a_pure_sign_flip() -> None:
    # Plain RAS and LPS are world systems, so a flip between them carries
    # no offset even with an extent available.
    result = bridge(RASCoordinateSystem(), LPSCoordinateSystem())
    assert isinstance(result, Scaling)
    assert not isinstance(result, Translation)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


# ----------------------------------------------------------------------
#   SUBSET-AXIS WRAPPING: A 3D TRANSFORM ACROSS A 4D (x, y, z, t) SPACE
# ----------------------------------------------------------------------


def _ras_time_system(name: str) -> CoordinateSystem:
    """A 4D system with RAS spatial axes and a trailing time axis."""
    return CoordinateSystem(name=name, axes=[R, A, S, TimeAxis(name="t")])


def _voxel_to_ras_time(matrix: np.ndarray) -> Affine:
    """A voxel-to-world affine over the 4D (x, y, z, t) space."""
    return Affine(
        matrix=matrix,
        input=_ras_time_system("voxel"),
        output=_ras_time_system("world"),
    )


def _lps_affine_3d(matrix: np.ndarray) -> Affine:
    """A 3D affine that acts in LPS, like an ITK or ANTs transform."""
    lps = LPSCoordinateSystem()
    return Affine(matrix=matrix, input=lps, output=lps)


def _embed_spatial(matrix_3x4: np.ndarray) -> np.ndarray:
    """A 3D affine embedded into a 4D homogeneous matrix, identity on t."""
    embedded = np.eye(5)
    embedded[:3, :3] = matrix_3x4[:, :3]
    embedded[:3, 4] = matrix_3x4[:, 3]
    return embedded


def _homogeneous_of_each(sequence: Sequence) -> np.ndarray:
    """The full matrix a sequence applies, built element by element.

    A subspace-wrapped result stays a `SubspaceTransformation` rather than
    folding into a single affine, so the composed matrix is read by
    reducing each element to an affine and multiplying in application
    order.
    """
    matrix = None
    for transformation in sequence:
        element = np.asarray(
            transformation.compute().to(Affine).homogeneous_matrix
        )
        matrix = element if matrix is None else element @ matrix
    return matrix


def test_subset_transform_is_wrapped_in_a_subspace() -> None:
    # A 3D LPS transform meeting a 4D (x, y, z, t) boundary is lifted into
    # the 4D space by a SubspaceTransformation over the three spatial axes.
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(np.eye(3, 4))
    result = adapt(first, second)
    assert isinstance(result, Sequence)
    assert result.transformations[0] is first
    wrapped = result.transformations[-1]
    assert isinstance(wrapped, SubspaceTransformation)
    np.testing.assert_array_equal(wrapped.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(wrapped.output_axes, [0, 1, 2])


def test_subspace_wrap_is_identity_on_the_extra_axis() -> None:
    # The wrapped transform acts on the spatial axes and leaves the time
    # axis untouched: its full affine is the identity on the t row and
    # column.
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(itk_matrix)
    wrapped = adapt(first, second).transformations[-1]
    full = np.asarray(wrapped.to(Affine).homogeneous_matrix)
    # The spatial block is the ITK affine conjugated by the RAS/LPS flip on
    # its input side: itk @ diag(-1, -1, 1).
    flip = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(full[:3, :3], itk_matrix[:, :3] @ flip)
    np.testing.assert_allclose(full[:3, 4], itk_matrix[:, 3])
    # The time axis passes through unchanged.
    np.testing.assert_array_equal(full[3], [0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(full[:, 3], [0.0, 0.0, 0.0, 1.0, 0.0])


def test_subspace_wrap_round_trips_through_inverse() -> None:
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(itk_matrix)
    wrapped = adapt(first, second).transformations[-1]
    inverse = wrapped.inverse()
    assert isinstance(inverse, SubspaceTransformation)
    np.testing.assert_array_equal(inverse.input_axes, [0, 1, 2])
    forward = np.asarray(wrapped.to(Affine).homogeneous_matrix)
    backward = np.asarray(inverse.to(Affine).homogeneous_matrix)
    np.testing.assert_allclose(backward @ forward, np.eye(5), atol=1e-12)


def test_spatial_transform_across_a_4d_image_via_compute() -> None:
    # End to end: a 3D LPS transform applied after a 4D voxel-to-RAS map.
    # Composing the two crosses from a 4D (x, y, z, t) space into a 3D LPS
    # transform. The adaptor lifts the transform onto the spatial axes and
    # leaves time alone, and the composition maps the spatial coordinates
    # by the transform while carrying the time coordinate through.
    voxel_matrix = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(voxel_matrix)
    second = _lps_affine_3d(itk_matrix)

    result = Sequence([first, second]).compute()
    assert isinstance(result, Sequence)
    wrapped = [t for t in result if isinstance(t, SubspaceTransformation)]
    assert len(wrapped) == 1

    got = _homogeneous_of_each(result)
    flip4 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0])
    voxel_homogeneous = np.eye(5)
    voxel_homogeneous[:4, :] = voxel_matrix
    expected = _embed_spatial(itk_matrix) @ flip4 @ voxel_homogeneous
    np.testing.assert_allclose(got, expected)

    # Numerically: the spatial coordinates are mapped by the spatial
    # transform, and the time coordinate is left unchanged.
    point = np.array([1.0, 1.0, 1.0, 7.0, 1.0])
    mapped = got @ point
    world_time = voxel_matrix[3, :4] @ point[:4] + voxel_matrix[3, 4]
    assert mapped[3] == pytest.approx(world_time)
    # The flip is load-bearing: without it the spatial block differs.
    without_flip = _embed_spatial(itk_matrix) @ voxel_homogeneous
    assert not np.allclose(got, without_flip)


def test_wrapped_result_round_trips_through_compute() -> None:
    # The lifted transform composes and inverts through compute(): a
    # sequence and its reverse cancel to the identity on the full 4D space.
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(itk_matrix)
    forward = Sequence([first, second]).compute()
    forward_matrix = _homogeneous_of_each(forward)
    backward_matrix = _homogeneous_of_each(forward.inverse())
    np.testing.assert_allclose(
        backward_matrix @ forward_matrix, np.eye(5), atol=1e-12
    )


def test_extra_spatial_axis_is_not_absorbed_and_raises() -> None:
    # The extra axis on the fuller side must be a genuine pass-through. A
    # fourth spatial axis with no counterpart is a real dimensionality
    # mismatch, not a pass-through, so the wrapping declines and the
    # adaptor raises rather than silently dropping the axis.
    four_spatial = CoordinateSystem(
        name="four-spatial",
        axes=[R, A, S, SpatialAxis(name="extra")],
    )
    first = Affine(
        matrix=np.eye(4, 5), input=four_spatial, output=four_spatial
    )
    second = _lps_affine_3d(np.eye(3, 4))
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


def test_bridge_refuses_a_dimensionality_mismatch() -> None:
    # A bridge reorders, rescales, and flips axes, and never adds or drops
    # one, so it refuses two systems of different sizes outright.
    with pytest.raises(AdaptationError):
        bridge(_ras_time_system("4d"), LPSCoordinateSystem())


def test_embedding_a_3d_transform_into_4d_is_out_of_scope() -> None:
    # A 3D transform followed by a 4D neighbour would have to invent the
    # extra axis, which is an embedding rather than a subspace lift. It is
    # out of scope, and the boundary is refused.
    first = _lps_affine_3d(np.eye(3, 4))
    second = _voxel_to_ras_time(np.eye(4, 5))
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


def test_itk_3d_transform_applied_to_a_4d_image_wraps_the_spatial_axes(
    tmp_path: Path,
) -> None:
    """Apply a real ITK (LPS) 3D transform after a 4D voxel-to-RAS map.

    The ITK transform is read from an ITK text file, so it needs no ``itk``
    package. It acts in LPS over three spatial axes. Composing it after a
    4D voxel-to-RAS map lifts it onto the spatial axes of the 4D space and
    leaves the time axis unchanged.
    """
    transformations = pytest.importorskip("brainhops.io.transformations")
    itk = transformations.load(str(data_dir / "itk_affine3d.tfm"))

    voxel_matrix = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    voxel_to_world = _voxel_to_ras_time(voxel_matrix)

    result = Sequence([voxel_to_world, itk]).compute()
    wrapped = [t for t in result if isinstance(t, SubspaceTransformation)]
    assert len(wrapped) == 1
    np.testing.assert_array_equal(wrapped[0].input_axes, [0, 1, 2])

    got = _homogeneous_of_each(result)
    itk_3d = np.asarray(itk.compute().to(Affine).homogeneous_matrix)[:3, :]
    flip4 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0])
    voxel_homogeneous = np.eye(5)
    voxel_homogeneous[:4, :] = voxel_matrix
    expected = _embed_spatial(itk_3d) @ flip4 @ voxel_homogeneous
    np.testing.assert_allclose(got, expected)
    # Time passes through untouched: the composed t row is the voxel map's
    # own identity on time, and the wrap adds nothing to it.
    np.testing.assert_array_equal(got[3], [0.0, 0.0, 0.0, 1.0, 0.0])
    # The flip is load-bearing.
    without_flip = _embed_spatial(itk_3d) @ voxel_homogeneous
    assert not np.allclose(got, without_flip)
