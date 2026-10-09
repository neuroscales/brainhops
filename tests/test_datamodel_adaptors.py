"""Tests of the adaptor that bridges two coordinate systems.

The adaptor reconciles the systems that meet at a composition boundary.
The tests exercise the matching of axes and the bridges it builds against
the named systems, the insertion of bridges during composition, the
embedding of a transform into a space with more axes, and end-to-end
examples with FSL and ITK transforms applied to images.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel._transformations.compute.adaptors import (
    _match_axes,
    adapt,
    bridge,
    embed,
)
from brainhops.datamodel._transformations.compute.compose import compose
from brainhops.datamodel.axes import (
    A,
    Axis,
    LeftToRightAxis,
    PosteriorToAnteriorAxis,
    R,
    RightToLeftAxis,
    S,
    SpaceAxis,
    TimeAxis,
)
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import (
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
    LPSmm,
    RASCoordinateSystem,
    RASmm,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import (
    Affine,
    CartesianField,
    CoordinatesField,
    DisplacementField,
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
from brainhops.errors import (
    AdaptationError,
    CompositionError,
)

# Physical anatomical axes in mm. R, A and S fix a direction but not a
# metric, so a world system that meets a transform in mm must state one.
Rmm, Amm, Smm = RASmm().axes

data_dir = Path(__file__).parent / "data"


def _homogeneous(transformation: Transformation) -> np.ndarray:
    """Return the homogeneous affine that a transformation reduces to."""
    return np.asarray(transformation.compute().to(Affine).homogeneous_matrix)


# ----------------------------------------------------------------------
#   MATCHING KERNEL AGAINST THE NAMED SYSTEMS
# ----------------------------------------------------------------------


def test_equal_systems_need_no_bridge() -> None:
    result = bridge(RASmm(), RASmm())
    assert isinstance(result, Identity)


def test_ras_to_lps_is_a_sign_flip() -> None:
    result = bridge(RASmm(), LPSmm())
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


def test_lps_to_ras_is_the_same_flip() -> None:
    result = bridge(LPSmm(), RASmm())
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
        (RASmm(), LPSmm()),
        (FVoxelCoordinateSystem(), CVoxelCoordinateSystem()),
        (FRASCoordinateSystem(), CRASCoordinateSystem()),
    ],
)
def test_bridge_round_trips_to_identity(
    source: CoordinateSystem, target: CoordinateSystem
) -> None:
    # Opposite bridges cancel, because the adaptor emits only exactly
    # invertible pieces.
    forward = bridge(source, target)
    backward = bridge(target, source)
    result = Sequence([forward, backward]).compute()
    assert is_identity(result, compute=True)


def test_matches_by_orientation_when_names_differ() -> None:
    # The source axes are named and the target axes are oriented; they still
    # match on type and orientation.
    source = CoordinateSystem(
        name="named",
        axes=[
            SpaceAxis(name="x", orientation=LeftToRight()),
            SpaceAxis(name="y", orientation=PosteriorToAnterior()),
        ],
    )
    target = CoordinateSystem(
        name="oriented",
        axes=[
            SpaceAxis(name="ap", orientation=PosteriorToAnterior()),
            SpaceAxis(name="lr", orientation=LeftToRight()),
        ],
    )
    result = bridge(source, target)
    # x (left to right) feeds target axis 1, and y (posterior to anterior)
    # feeds target axis 0.
    assert isinstance(result, Permutation)
    np.testing.assert_array_equal(result.permutation, [1, 0])


# ----------------------------------------------------------------------
#   UNITS
# ----------------------------------------------------------------------


def test_unit_difference_is_a_scaling() -> None:
    source = CoordinateSystem(
        name="mm", axes=[SpaceAxis(name="x", unit="millimeter")]
    )
    target = CoordinateSystem(
        name="um", axes=[SpaceAxis(name="x", unit="micrometer")]
    )
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    # One millimeter is 1000 micrometers.
    np.testing.assert_allclose(result.scale, [1000.0])


def test_an_unspecified_unit_is_compatible_with_any_unit() -> None:
    # An unspecified unit is never a reason to refuse, and matches at a ratio
    # of one.
    source = CoordinateSystem(name="mm", axes=[LeftToRightAxis(unit="mm")])
    target = CoordinateSystem(
        name="unspecified",
        axes=[SpaceAxis(name="x", unit=None, orientation=LeftToRight())],
    )
    assert is_identity(bridge(source, target), compute=True)
    assert is_identity(bridge(target, source), compute=True)


def test_an_unspecified_anatomical_system_bridges_to_millimetres() -> None:
    assert is_identity(bridge(RASCoordinateSystem(), RASmm()), compute=True)
    result = bridge(LPSmm(), RASCoordinateSystem())
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


def test_a_sample_matched_to_a_physical_unit_raises() -> None:
    sampled = CoordinateSystem(
        axes=[SpaceAxis(name="x", unit="index", orientation=LeftToRight())]
    )
    world = CoordinateSystem(axes=[LeftToRightAxis(name="x", unit="mm")])
    for source, target in ((sampled, world), (world, sampled)):
        with pytest.raises(AdaptationError, match="sampled.*millimeter"):
            bridge(source, target)


# ----------------------------------------------------------------------
#   ORIENTATION-FLIP ORIGIN OFFSET (array-index versus world)
# ----------------------------------------------------------------------


def _oriented_index_system(
    name: str, orientation: Orientation
) -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=[SpaceAxis(name="i", unit="index", orientation=orientation)],
    )


def test_world_flip_carries_no_offset() -> None:
    # Between world systems, a reversed axis is a pure sign flip.
    source = CoordinateSystem(name="R", axes=[LeftToRightAxis()])
    target = CoordinateSystem(name="L", axes=[RightToLeftAxis()])
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    np.testing.assert_array_equal(result.scale, [-1.0])


def test_array_index_flip_carries_the_extent_offset() -> None:
    # Between array-index systems, a reversed axis of extent n maps x to
    # (n - 1) - x.
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
        name="ab", axes=[SpaceAxis(name="a"), SpaceAxis(name="b")]
    )
    target = CoordinateSystem(
        name="cd", axes=[SpaceAxis(name="c"), SpaceAxis(name="d")]
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


def test_positional_fallback_is_opt_in_and_warns() -> None:
    source = CoordinateSystem(
        name="ab", axes=[SpaceAxis(name="a"), SpaceAxis(name="b")]
    )
    target = CoordinateSystem(
        name="cd", axes=[SpaceAxis(name="c"), SpaceAxis(name="d")]
    )
    with pytest.warns(UserWarning):
        result = bridge(source, target, allow_positional=True)
    # The names pair by position and the units agree, so nothing changes.
    assert is_identity(result, compute=True)


# ----------------------------------------------------------------------
#   THE adapt() CONTRACT: A CONTAINER OF BOTH TRANSFORMS
# ----------------------------------------------------------------------


def test_adapt_returns_a_sequence_containing_both_transforms() -> None:
    first = Affine(
        matrix=np.eye(3, 4),
        input=RASmm(),
        output=RASmm(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=LPSmm(),
        output=LPSmm(),
    )
    result = adapt(first, second)
    assert isinstance(result, Sequence)
    assert result.transformations[0] is first
    assert result.transformations[-1] is second
    # The flip sits between the two transforms.
    assert any(isinstance(t, Scaling) for t in result.transformations)


def test_adapt_without_mismatch_inserts_no_bridge() -> None:
    first = Affine(
        matrix=np.eye(3, 4),
        input=RASmm(),
        output=RASmm(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=RASmm(),
        output=RASmm(),
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
        input=RASmm(),
        output=RASmm(),
    )
    second = Affine(
        matrix=second_matrix,
        input=LPSmm(),
        output=LPSmm(),
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
        input=RASmm(),
        output=RASmm(),
    )
    second = Affine(
        matrix=second_matrix,
        input=RASmm(),
        output=RASmm(),
    )
    result = Sequence([first, second]).compute()

    def homogeneous(matrix: np.ndarray) -> np.ndarray:
        stacked = np.eye(4)
        stacked[:3, :] = matrix
        return stacked

    expected = homogeneous(second_matrix) @ homogeneous(first_matrix)
    np.testing.assert_allclose(_homogeneous(result), expected)


def test_opposite_bridges_cancel_in_a_sequence() -> None:
    # A round trip RAS to LPS to RAS composes to the identity.
    matrix = np.eye(3, 4)
    ras_to_lps = Affine(
        matrix=matrix,
        input=RASmm(),
        output=RASmm(),
    )
    lps = Affine(
        matrix=matrix,
        input=LPSmm(),
        output=LPSmm(),
    )
    lps_to_ras = Affine(
        matrix=matrix,
        input=RASmm(),
        output=RASmm(),
    )
    result = Sequence([ras_to_lps, lps, lps_to_ras]).compute()
    assert is_identity(result, compute=True)


# ----------------------------------------------------------------------
#   END-TO-END: A TRANSFORM ACROSS AN IMAGE'S COORDINATE SYSTEM
# ----------------------------------------------------------------------


def test_itk_transform_applied_to_a_nifti_image_bridges_ras_and_lps() -> None:
    """An ITK transform, which acts in LPS, applied after the voxel-to-RAS map
    of a NIfTI image gets a sign flip from the adaptor.
    """
    # Without nibabel, `brainhops.io.images` imports but has no NIfTI reader,
    # so the load would fail instead of skipping.
    pytest.importorskip("nibabel")
    images = pytest.importorskip("brainhops.io.images")
    transformations = pytest.importorskip("brainhops.io.transformations")

    image = images.load(str(data_dir / "fsl" / "ref.nii.gz"))
    voxel_to_world = image.transformation
    # The ITK affine acts in LPS.
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
    # The flip matters: the composition differs without it.
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
    """An FSL transform, which acts in RAS, applied after the voxel-to-LPS map
    of a Zarr image gets a sign flip from the adaptor.
    """
    pytest.importorskip("abczarr")
    nb = pytest.importorskip("nibabel")
    from brainhops.io.images.zarr import ZarrImage
    from brainhops.io.transformations.fsl.flirt import FlirtTransform

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
        output=LPSmm(),
    )
    image = ZarrImage.load(path, transformation=voxel_to_lps)
    voxel_to_world = image.transformation

    reference = nb.load(str(data_dir / "fsl" / "ref.nii.gz"))
    moving = nb.load(str(data_dir / "fsl" / "src.nii.gz"))
    flirt_matrix = np.eye(4)
    flirt_matrix[0, 3] = 5.0
    fsl = FlirtTransform(
        flirt_matrix=flirt_matrix, reference=reference, moving=moving
    )

    result = Sequence([voxel_to_world, fsl]).compute()

    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
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
        input=RASmm(),
        output=RASmm(),
    )


def _lps_affine() -> Affine:
    return Affine(
        matrix=np.eye(3, 4),
        input=LPSmm(),
        output=LPSmm(),
    )


def test_singly_nested_sequence_inserts_the_flip_bridge() -> None:
    # A mismatch inside a nested sequence is bridged as at the top level.
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
    # The boundary inside the nested sequence still gets the flip.
    inner_flip = _homogeneous(c) @ _homogeneous(flat)
    np.testing.assert_allclose(_homogeneous(nested), inner_flip)


def test_bare_inverse_boundary_inserts_the_flip_bridge() -> None:
    # A boundary behind `Inverse(forward=b)` is bridged against the LPS system
    # that the inverse presents.
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
            SpaceAxis(name="x", unit=None),
            SpaceAxis(name="y", unit=None),
            SpaceAxis(name="z", unit=None),
        ],
    )


def test_compute_pairs_underspecified_voxel_axes_by_position_and_warns() -> (
    None
):
    # Unoriented, unitless axes are paired by position with a warning.
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
    # The explicit entry point raises by default; only `compute()` falls back
    # to positions.
    with pytest.raises(AdaptationError):
        bridge(VoxelCoordinateSystem(), _xyz_index_system())


def test_same_type_group_pairs_by_position_even_when_oriented() -> None:
    # Two spatial axes on each side form groups of equal size, so they pair by
    # order, with a warning, even when only one is oriented.
    oriented = CoordinateSystem(
        name="oriented",
        axes=[
            SpaceAxis(name="a", unit=None, orientation=LeftToRight()),
            SpaceAxis(name="b", unit=None),
        ],
    )
    plain = CoordinateSystem(
        name="plain",
        axes=[
            SpaceAxis(name="c", unit=None),
            SpaceAxis(name="d", unit=None),
        ],
    )
    first = Affine(matrix=np.eye(2, 3), input=oriented, output=oriented)
    second = Affine(matrix=np.eye(2, 3), input=plain, output=plain)
    with pytest.warns(UserWarning):
        result = Sequence([first, second]).compute()
    # An oriented axis paired with an unoriented one passes through.
    np.testing.assert_allclose(_homogeneous(result), np.eye(3))


def test_same_type_group_pairs_by_position_when_united() -> None:
    # All axes have length units, so neither names nor units give a unique
    # pairing; the axes pair by order, and each unit ratio becomes a scaling.
    source = CoordinateSystem(
        name="mm",
        axes=[
            SpaceAxis(name="a", unit="millimeter"),
            SpaceAxis(name="b", unit="millimeter"),
        ],
    )
    target = CoordinateSystem(
        name="um",
        axes=[
            SpaceAxis(name="c", unit="micrometer"),
            SpaceAxis(name="d", unit="micrometer"),
        ],
    )
    first = Affine(matrix=np.eye(2, 3), input=source, output=source)
    second = Affine(matrix=np.eye(2, 3), input=target, output=target)
    with pytest.warns(UserWarning):
        result = Sequence([first, second]).compute()
    # Axis a maps to c and b maps to d, each scaled by 1000 from mm to um.
    np.testing.assert_allclose(
        np.diag(_homogeneous(result)), [1000.0, 1000.0, 1.0]
    )


def test_type_grouped_positional_keeps_each_type_to_its_own_group() -> None:
    # Pairing by absolute position would cross time with space; the fallback
    # pairs within each type and reorders the axes instead.
    source = CoordinateSystem(
        name="s",
        axes=[
            TimeAxis(name="t", unit=None),
            SpaceAxis(name="a", unit=None),
        ],
    )
    target = CoordinateSystem(
        name="t",
        axes=[
            SpaceAxis(name="c", unit=None),
            TimeAxis(name="u", unit=None),
        ],
    )
    first = Affine(matrix=np.eye(2, 3), input=source, output=source)
    second = Affine(matrix=np.eye(2, 3), input=target, output=target)
    with pytest.warns(UserWarning):
        result = Sequence([first, second]).compute()
    # Space feeds space and time feeds time.
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(_homogeneous(result), expected)


def test_typeless_group_pairs_with_a_typed_group_of_equal_count() -> None:
    # A typeless system pairs by position with a spatial system of the same
    # size, with a warning.
    source = CoordinateSystem(
        name="typeless",
        axes=[Axis(name="a"), Axis(name="b"), Axis(name="c")],
    )
    target = CoordinateSystem(
        name="voxel",
        axes=[
            SpaceAxis(name="dim0", unit=None),
            SpaceAxis(name="dim1", unit=None),
            SpaceAxis(name="dim2", unit=None),
        ],
    )
    first = Affine(matrix=np.eye(3, 4), input=source, output=source)
    second = Affine(matrix=np.eye(3, 4), input=target, output=target)
    with pytest.warns(UserWarning):
        result = Sequence([first, second]).compute()
    np.testing.assert_allclose(_homogeneous(result), np.eye(4))


def test_typeless_group_with_no_equal_count_typed_group_raises() -> None:
    # Two typeless axes against a spatial and a time axis are ambiguous, and
    # the mismatch is reported rather than guessed.
    source = CoordinateSystem(
        name="typeless",
        axes=[Axis(name="a"), Axis(name="b")],
    )
    target = CoordinateSystem(
        name="mixed",
        axes=[SpaceAxis(name="c", unit=None), TimeAxis(name="u", unit=None)],
    )
    first = Affine(matrix=np.eye(2, 3), input=source, output=source)
    second = Affine(matrix=np.eye(2, 3), input=target, output=target)
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


def test_type_group_count_mismatch_raises() -> None:
    # The type groups differ in size, so no group can pair by position.
    source = CoordinateSystem(
        name="s",
        axes=[
            SpaceAxis(name="a", unit=None),
            SpaceAxis(name="b", unit=None),
            TimeAxis(name="t", unit=None),
        ],
    )
    target = CoordinateSystem(
        name="t",
        axes=[
            SpaceAxis(name="c", unit=None),
            TimeAxis(name="u", unit=None),
            TimeAxis(name="v", unit=None),
        ],
    )
    first = Affine(matrix=np.eye(3, 4), input=source, output=source)
    second = Affine(matrix=np.eye(3, 4), input=target, output=target)
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


# ----------------------------------------------------------------------
#   TYPE CONFLICTS OVERRIDE A NAME OR UNIT COINCIDENCE (comment #1)
# ----------------------------------------------------------------------


def test_shared_name_across_different_types_is_not_matched() -> None:
    # A type conflict overrides a shared name.
    source = CoordinateSystem(name="s", axes=[SpaceAxis(name="t", unit=None)])
    target = CoordinateSystem(name="t", axes=[TimeAxis(name="t", unit=None)])
    with pytest.raises(AdaptationError):
        bridge(source, target)


def test_shared_unit_across_different_types_is_not_matched() -> None:
    # Axes of different types do not match through a shared unit.
    source = CoordinateSystem(
        name="s", axes=[SpaceAxis(name="a", unit="millimeter")]
    )
    target = CoordinateSystem(
        name="t",
        axes=[Axis(name="b", type="channel", unit="millimeter")],
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


def test_axes_match_by_unit_when_names_differ() -> None:
    # Spatial axes with different names match by unit kind, and the unit
    # ratio becomes a scaling.
    source = CoordinateSystem(
        name="mm", axes=[SpaceAxis(name="a", unit="millimeter")]
    )
    target = CoordinateSystem(
        name="um", axes=[SpaceAxis(name="b", unit="micrometer")]
    )
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    np.testing.assert_allclose(result.scale, [1000.0])


# ----------------------------------------------------------------------
#   REGRESSION: PAIRING DOES NOT DEPEND ON THE ORDER OF THE AXES
# ----------------------------------------------------------------------


def _all_matches(
    source_axes: tx.List[Axis], target_axes: tx.List[Axis]
) -> tx.List[tx.Dict[int, tx.Optional[int]]]:
    """Return the matching for every order of the source and target axes.

    Each matching maps the identity of a target axis to the identity of
    the source axis paired with it, or to None, so that matchings computed
    in different orders can be compared.
    """
    matchings = []
    for sources in itertools.permutations(source_axes):
        for targets in itertools.permutations(target_axes):
            match, _ = _match_axes(list(sources), list(targets), False)
            matchings.append(
                {
                    id(target): None if i is None else id(sources[i])
                    for target, i in zip(targets, match)
                }
            )
    return matchings


def test_a_unit_shared_by_two_target_axes_is_ambiguous() -> None:
    # Two target axes have the same single source candidate by unit. The
    # match is ambiguous, so neither target axis is paired, whatever the
    # order of the axes.
    source = [Axis(name="a", unit="mm"), Axis(name="b")]
    target = [Axis(name="c", unit="mm"), Axis(name="d", unit="mm")]
    for matching in _all_matches(source, target):
        assert matching == {id(target[0]): None, id(target[1]): None}


def test_a_line_shared_by_two_target_axes_is_ambiguous() -> None:
    # Two collinear target axes have the same single source candidate by
    # orientation, and neither one is paired.
    source = [Axis(name="a", orientation=LeftToRight()), Axis(name="b")]
    target = [
        Axis(name="c", orientation=LeftToRight()),
        Axis(name="d", orientation=RightToLeft()),
    ]
    for matching in _all_matches(source, target):
        assert matching == {id(target[0]): None, id(target[1]): None}


def test_a_name_shared_by_two_target_axes_is_ambiguous() -> None:
    # Two target axes with the same name have the same single source
    # candidate by name, and neither one is paired.
    source = [Axis(name="x"), Axis(name="b")]
    target = [Axis(name="x"), Axis(name="x", type="space")]
    for matching in _all_matches(source, target):
        assert matching == {id(target[0]): None, id(target[1]): None}


def test_a_pair_by_unit_frees_the_remaining_candidate() -> None:
    # The spatial target axis "c" can only be paired with "a", because "b"
    # is a channel axis. Once "c" is paired, "a" is no longer a candidate
    # of the untyped target axis "d", which is then paired with "b". The
    # result is the same in every order.
    source = [
        SpaceAxis(name="a", unit="mm"),
        Axis(name="b", type="channel", unit="mm"),
    ]
    target = [SpaceAxis(name="c", unit="mm"), Axis(name="d", unit="mm")]
    for matching in _all_matches(source, target):
        assert matching == {
            id(target[0]): id(source[0]),
            id(target[1]): id(source[1]),
        }


def test_a_bridge_reports_both_ambiguous_axes_in_every_order() -> None:
    # The bridge fails because the unit match is ambiguous, and it reports
    # both target axes as unmatched, whatever their order. The first
    # target axis used to take the source axis, and only the second
    # one was reported.
    source = CoordinateSystem(
        axes=[SpaceAxis(name="a", unit="mm"), SpaceAxis(name="b")]
    )
    c = SpaceAxis(name="c", unit="mm")
    d = SpaceAxis(name="d", unit="mm")
    for target_axes in ([c, d], [d, c]):
        target = CoordinateSystem(axes=target_axes)
        with pytest.raises(AdaptationError) as error:
            bridge(source, target)
        message = str(error.value)
        assert "'c'" in message
        assert "'d'" in message
        assert "'a'" in message
        assert "'b'" in message


# ----------------------------------------------------------------------
#   REGRESSION: ARRAY FLIP INSIDE compute() VIA GRID EXTENTS (finding 3)
# ----------------------------------------------------------------------


def _oriented_named_index_system(
    name: str, orientation: Orientation
) -> CoordinateSystem:
    return CoordinateSystem(
        name=name,
        axes=[SpaceAxis(name="i", unit="index", orientation=orientation)],
    )


def test_array_flip_in_a_sequence_uses_an_adjacent_grid_extent() -> None:
    # A reversed array-index axis needs its extent, which an interior grid
    # supplies through its shape.
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
    # Without an extent, the error says how to bridge explicitly.
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
    # Left-to-right and anterior-to-posterior axes lie on different lines, so a
    # shared name must not force a sign flip.
    source = CoordinateSystem(
        name="lr",
        axes=[SpaceAxis(name="x", orientation=LeftToRight())],
    )
    target = CoordinateSystem(
        name="ap",
        axes=[SpaceAxis(name="x", orientation=PosteriorToAnterior())],
    )
    with pytest.raises(AdaptationError):
        bridge(source, target)


# ----------------------------------------------------------------------
#   REGRESSION: EXACT UNIT RATIOS (finding 5)
# ----------------------------------------------------------------------


def test_unit_ratio_round_trip_is_exact() -> None:
    # A ratio times its reciprocal is exactly one, which dividing the two
    # scales does not guarantee.
    def _system(unit: str) -> CoordinateSystem:
        return CoordinateSystem(
            name=unit, axes=[SpaceAxis(name="x", unit=unit)]
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
    mm_to_um = bridge(_system("millimeter"), _system("micrometer"))
    assert float(mm_to_um.scale[0]) == 1000.0


# ----------------------------------------------------------------------
#   REGRESSION: NAMED VOXEL SYSTEMS CLASSIFY AS ARRAY-INDEX (finding 6)
# ----------------------------------------------------------------------


def test_named_ras_lps_voxel_systems_are_array_index() -> None:
    # fRAS and fLPS are voxel grids despite their unit and orientation, so the
    # flip between them carries an offset and needs the extents.
    with pytest.raises(AdaptationError):
        bridge(FRASCoordinateSystem(), FLPSCoordinateSystem())
    result = bridge(
        FRASCoordinateSystem(),
        FLPSCoordinateSystem(),
        extents={"x": 4, "y": 5, "z": 6},
    )
    matrix = np.asarray(result.compute().to(Affine).homogeneous_matrix)
    # x and y are reversed, with offsets 3 and 4; z is unchanged.
    np.testing.assert_array_equal(np.diag(matrix), [-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_array_equal(matrix[:3, 3], [3.0, 4.0, 0.0])


def test_world_ras_lps_flip_stays_a_pure_sign_flip() -> None:
    # RAS and LPS are world systems, so the flip has no offset.
    result = bridge(RASmm(), LPSmm())
    assert isinstance(result, Scaling)
    assert not isinstance(result, Translation)
    np.testing.assert_array_equal(result.scale, [-1.0, -1.0, 1.0])


# ----------------------------------------------------------------------
#   SUBSET-AXIS WRAPPING: A 3D TRANSFORM ACROSS A 4D (x, y, z, t) SPACE
# ----------------------------------------------------------------------


def _ras_time_system(name: str) -> CoordinateSystem:
    """Return a 4-D system with RAS spatial axes and a trailing time axis."""
    return CoordinateSystem(
        name=name, axes=[Rmm, Amm, Smm, TimeAxis(name="t")]
    )


def _voxel_to_ras_time(matrix: np.ndarray) -> Affine:
    """Return a voxel-to-world affine over (x, y, z, t)."""
    return Affine(
        matrix=matrix,
        input=_ras_time_system("voxel"),
        output=_ras_time_system("world"),
    )


def _lps_affine_3d(matrix: np.ndarray) -> Affine:
    """Return a 3-D affine that acts in LPS, as ITK and ANTs transforms do."""
    lps = LPSmm()
    return Affine(matrix=matrix, input=lps, output=lps)


def _embed_spatial(matrix_3x4: np.ndarray) -> np.ndarray:
    """Embed a 3-D affine in a 4-D homogeneous matrix, as the identity on t."""
    embedded = np.eye(5)
    embedded[:3, :3] = matrix_3x4[:, :3]
    embedded[:3, 4] = matrix_3x4[:, 3]
    return embedded


def _homogeneous_of_each(sequence: Sequence) -> np.ndarray:
    """Return the full matrix that a sequence applies, element by element.

    A result wrapped in subspaces stays a sequence of
    `SubspaceTransformation` instead of folding into one affine.
    """
    matrix = None
    for transformation in sequence:
        element = np.asarray(
            transformation.compute().to(Affine).homogeneous_matrix
        )
        matrix = element if matrix is None else element @ matrix
    return matrix


def test_subset_transform_is_wrapped_in_a_subspace() -> None:
    # A 3-D LPS transform meeting a 4-D (x, y, z, t) boundary is embedded over
    # the three spatial axes.
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(np.eye(3, 4))
    result = adapt(first, second)
    assert isinstance(result, Sequence)
    assert result.transformations[0] is first
    wrapped = result.transformations[-1]
    assert isinstance(wrapped, SubspaceTransformation)
    np.testing.assert_array_equal(wrapped.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(wrapped.output_axes, [0, 1, 2])


def test_unnamed_unoriented_subset_transform_is_wrapped() -> None:
    # A transform over unnamed, unoriented spatial axes is still embedded in
    # the spatial axes, by position and with a warning.
    full = CoordinateSystem(
        name="xyzt",
        axes=[
            SpaceAxis(name="x"),
            SpaceAxis(name="y"),
            SpaceAxis(name="z"),
            TimeAxis(name="t"),
        ],
    )
    first = Affine(matrix=np.eye(4, 5), input=full, output=full)
    plain = CoordinateSystem(
        name="plain",
        axes=[SpaceAxis(), SpaceAxis(), SpaceAxis()],
    )
    second = Affine(matrix=np.eye(3, 4), input=plain, output=plain)
    with pytest.warns(UserWarning):
        result = adapt(first, second)
    assert isinstance(result, Sequence)
    assert result.transformations[0] is first
    wrapped = result.transformations[-1]
    assert isinstance(wrapped, SubspaceTransformation)
    np.testing.assert_array_equal(wrapped.input_axes, [0, 1, 2])
    np.testing.assert_array_equal(wrapped.output_axes, [0, 1, 2])


def test_subspace_wrap_is_identity_on_the_extra_axis() -> None:
    # The wrapped transform leaves the time row and column untouched.
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(itk_matrix)
    wrapped = adapt(first, second).transformations[-1]
    full = np.asarray(wrapped.to(Affine).homogeneous_matrix)
    # The spatial block is the ITK affine after the RAS-to-LPS flip.
    flip = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(full[:3, :3], itk_matrix[:, :3] @ flip)
    np.testing.assert_allclose(full[:3, 4], itk_matrix[:, 3])
    # Time passes through.
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
    # A 3-D LPS transform after a 4-D voxel-to-RAS map is embedded in the
    # spatial axes: spatial coordinates are mapped and time is carried through.
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
    # The embedding does not interpolate, so it folds into a single 4-D
    # affine: the embedded transform, the flip and the voxel map.
    got = _composed_homogeneous(result)
    flip4 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0])
    voxel_homogeneous = np.eye(5)
    voxel_homogeneous[:4, :] = voxel_matrix
    expected = _embed_spatial(itk_matrix) @ flip4 @ voxel_homogeneous
    np.testing.assert_allclose(got, expected)

    # The time coordinate is unchanged.
    point = np.array([1.0, 1.0, 1.0, 7.0, 1.0])
    mapped = got @ point
    world_time = voxel_matrix[3, :4] @ point[:4] + voxel_matrix[3, 4]
    assert mapped[3] == pytest.approx(world_time)
    # The flip matters: the spatial block differs without it.
    without_flip = _embed_spatial(itk_matrix) @ voxel_homogeneous
    assert not np.allclose(got, without_flip)


def test_wrapped_result_round_trips_through_compute() -> None:
    # The embedded transform composed with its inverse gives the identity on
    # the full 4-D space.
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(itk_matrix)
    forward = Sequence([first, second]).compute()
    forward_matrix = _composed_homogeneous(forward)
    backward_matrix = _composed_homogeneous(forward.inverse())
    np.testing.assert_allclose(
        backward_matrix @ forward_matrix, np.eye(5), atol=1e-12
    )


def test_extra_spatial_axis_is_not_absorbed_and_raises() -> None:
    # A fourth spatial axis without a counterpart is a real mismatch, so the
    # adaptor raises instead of dropping the axis.
    four_spatial = CoordinateSystem(
        name="four-spatial",
        axes=[Rmm, Amm, Smm, SpaceAxis(name="extra", unit="mm")],
    )
    first = Affine(
        matrix=np.eye(4, 5), input=four_spatial, output=four_spatial
    )
    second = _lps_affine_3d(np.eye(3, 4))
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


def test_bridge_refuses_a_dimensionality_mismatch() -> None:
    # A bridge reorders, rescales and flips axes, but never adds or drops any.
    with pytest.raises(AdaptationError):
        bridge(_ras_time_system("4d"), LPSmm())


def test_backward_embedding_wraps_a_3d_transform_before_a_4d_one() -> None:
    # When the fuller space is on the output side, the 3-D transform is
    # wrapped and placed before the 4-D map.
    itk_matrix = np.array(
        [[0.9, 0.1, 0.0, 1.0], [-0.1, 0.9, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]
    )
    voxel_matrix = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    first = _lps_affine_3d(itk_matrix)
    second = _voxel_to_ras_time(voxel_matrix)

    result = Sequence([first, second]).compute()
    # The backward embedding folds into one 4-D affine as well, in the
    # opposite order.
    got = _composed_homogeneous(result)
    flip4 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0])
    voxel_homogeneous = np.eye(5)
    voxel_homogeneous[:4, :] = voxel_matrix
    expected = voxel_homogeneous @ flip4 @ _embed_spatial(itk_matrix)
    np.testing.assert_allclose(got, expected)
    # The flip matters: the spatial block differs without it.
    without_flip = voxel_homogeneous @ _embed_spatial(itk_matrix)
    assert not np.allclose(got, without_flip)


def test_extra_spatial_axis_is_not_absorbed_backward_and_raises() -> None:
    # A 4-D output with four spatial axes is a real mismatch, so the adaptor
    # raises instead of inventing an axis.
    four_spatial = CoordinateSystem(
        name="four-spatial",
        axes=[Rmm, Amm, Smm, SpaceAxis(name="extra", unit="mm")],
    )
    first = _lps_affine_3d(np.eye(3, 4))
    second = Affine(
        matrix=np.eye(4, 5), input=four_spatial, output=four_spatial
    )
    with pytest.raises(AdaptationError):
        Sequence([first, second]).compute()


def test_itk_3d_transform_applied_to_a_4d_image_wraps_the_spatial_axes(
    tmp_path: Path,
) -> None:
    """A real ITK transform, read from a text file, is embedded in the spatial
    axes of a 4-D image, with time unchanged.
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
    # The ITK transform folds into one 4-D affine that embeds it on the
    # spatial axes.
    got = _composed_homogeneous(result)
    itk_3d = np.asarray(itk.compute().to(Affine).homogeneous_matrix)[:3, :]
    flip4 = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0])
    voxel_homogeneous = np.eye(5)
    voxel_homogeneous[:4, :] = voxel_matrix
    expected = _embed_spatial(itk_3d) @ flip4 @ voxel_homogeneous
    np.testing.assert_allclose(got, expected)
    # The time row is that of the voxel map; the embedding adds nothing.
    np.testing.assert_array_equal(got[3], [0.0, 0.0, 0.0, 1.0, 0.0])
    # The flip matters.
    without_flip = _embed_spatial(itk_3d) @ voxel_homogeneous
    assert not np.allclose(got, without_flip)


# ----------------------------------------------------------------------
#   is_identity RECOGNIZES A SUBSPACE OF THE IDENTITY UNDER compute
# ----------------------------------------------------------------------


def test_is_identity_recognizes_a_subspace_of_the_identity() -> None:
    # A subspace of the identity, over the same axes, is recognized as the
    # identity from its structure alone.
    subspace = SubspaceTransformation(
        transformation=Identity(),
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
    )
    assert is_identity(subspace) is True
    assert is_identity(subspace, compute=True) is True
    # A subspace that reindexes its axes is never the identity.
    reindex = SubspaceTransformation(
        transformation=Identity(),
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([1, 0, 2]),
    )
    assert is_identity(reindex) is False


def test_is_identity_keeps_a_non_identity_subspace_non_identity() -> None:
    # A subspace with a non-identity inner is never the identity.
    subspace = SubspaceTransformation(
        transformation=Scaling(scale=np.asarray([2.0, 2.0, 2.0])),
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
    )
    assert is_identity(subspace, compute=True) is False


# ----------------------------------------------------------------------
#   END TO END: A 3D TRANSFORM APPLIED TO A 4D (x, y, z, t) IMAGE
# ----------------------------------------------------------------------


def _ras_time(name: str) -> CoordinateSystem:
    return CoordinateSystem(
        name=name, axes=[Rmm, Amm, Smm, TimeAxis(name="t")]
    )


def _spatial3(name: str) -> CoordinateSystem:
    return CoordinateSystem(name=name, axes=[Rmm, Amm, Smm])


def _spatial_warp() -> Sequence:
    # A 3-D world-space warp: world to voxel, a displacement in voxel units,
    # and voxel to world.
    rng = np.random.default_rng(1)
    voxel = _spatial3("warp-voxel")
    world = RASmm()
    w2v = Affine(matrix=np.eye(3, 4), input=world, output=voxel)
    w2v.matrix[:, 3] = [-1.0, -2.0, -3.0]
    disp = rng.normal(size=(6, 7, 5, 3)) * 0.3
    field = DisplacementField(field=disp, input=voxel, output=voxel)
    return Sequence([w2v, field, w2v.inverse()], input=world, output=world)


def _image_4d() -> SingleScaleImage:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 7, 5, 3))
    v2w = np.eye(4, 5)
    v2w[:3, 4] = [1.0, 2.0, 3.0]
    return SingleScaleImage(
        data=data,
        transformations=[
            Affine(
                matrix=v2w, input=_ras_time("voxel"), output=_ras_time("world")
            )
        ],
    )


def _reference_4d(
    image: SingleScaleImage, warp: Sequence, degree: int
) -> np.ndarray:
    # Reference: each time point resliced through the 3-D warp on its own.
    from brainhops._core.bsplines import pull

    data = np.asarray(image.data)
    shape = data.shape
    v2w = np.asarray(image.transformation.matrix)
    voxel = _spatial3("ref-voxel")
    world = RASmm()
    v2w3 = Affine(matrix=v2w[:3][:, [0, 1, 2, 4]], input=voxel, output=world)
    seq3 = Sequence(
        [CartesianField(shape=shape[:3]), v2w3, warp, v2w3.inverse()]
    )
    coords3 = np.asarray(seq3.compute().field)
    ref = np.empty(shape)
    for t in range(shape[3]):
        ref[..., t] = pull(
            data[..., t], coords3, degree=degree, bound="reflect", coeff=False
        )
    return ref


def test_4d_reslice_through_a_3d_warp_field_matches_reference() -> None:
    # A 3-D warp resamples every time point of a 4-D image in the same way.
    image = _image_4d()
    warp = _spatial_warp()
    got = np.asarray(image(warp).reslice(image, degree=1).data)
    ref = _reference_4d(image, warp, degree=1)
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_4d_reslice_through_a_3d_warp_field_degree3() -> None:
    # The tolerance is looser at degree 3, because the reslice prefilters the
    # whole 4-D array while the reference prefilters each volume.
    image = _image_4d()
    warp = _spatial_warp()
    got = np.asarray(image(warp).reslice(image, degree=3).data)
    ref = _reference_4d(image, warp, degree=3)
    np.testing.assert_allclose(got, ref, rtol=5e-3, atol=5e-3)


def test_time_component_is_carried_through_a_3d_warp() -> None:
    # The coordinate field of the reslice holds the exact time coordinate.
    image = _image_4d()
    warp = _spatial_warp()
    grid = image.geometry.grid
    coords = (
        image(warp).transformation.inverse()
        @ image.geometry.transformation
        @ grid
    ).compute()
    shape = np.asarray(image.data).shape
    time = np.asarray(coords.field)[..., 3]
    expected = np.broadcast_to(np.arange(shape[3]), shape)
    np.testing.assert_array_equal(time, expected)


def test_3d_affine_applied_to_a_4d_image_via_reslice() -> None:
    # A 3-D affine applied to a 4-D image reslices end to end.
    image = _image_4d()
    warp_aff = Affine(
        matrix=np.eye(3, 4),
        input=RASmm(),
        output=RASmm(),
    )
    warp_aff.matrix[:, 3] = [0.5, 0.0, 0.0]
    out = image(warp_aff).reslice(image, degree=1)
    assert np.asarray(out.data).shape == np.asarray(image.data).shape


def test_own_geometry_reslice_is_exact_and_never_inverts_a_field(
    monkeypatch,  # noqa: ANN001
) -> None:
    # The warp and its inverse cancel, so the field is never inverted.
    import brainhops._ext.invfield as invfield

    def _boom(*args, **kwargs) -> None:
        raise AssertionError("the field was inverted numerically")

    monkeypatch.setattr(invfield, "inverse", _boom)
    image = _image_4d()
    warp = _spatial_warp()
    warped = image(warp)
    got = np.asarray(warped.reslice(warped, degree=1).data)
    np.testing.assert_array_equal(got, np.asarray(image.data))


# ----------------------------------------------------------------------
#   embed AND THE DISCRETE GUARD
# ----------------------------------------------------------------------


def test_embed_refuses_a_cartesian_field() -> None:
    # A grid defines the sampling domain and is never embedded.
    grid = CartesianField(
        shape=(6, 7, 5),
        input=_spatial3("grid"),
        output=_spatial3("grid"),
    )
    full = _ras_time("full")
    assert embed(grid, full=full, side="input", extents=None) is None
    assert embed(grid, full=full, side="output", extents=None) is None


def _discrete_ras_time(name: str) -> CoordinateSystem:
    # A 4-D system whose third spatial axis is discrete.
    discrete_s = S(discrete=True)
    return CoordinateSystem(
        name=name, axes=[R(), A(), discrete_s, TimeAxis(name="t")]
    )


def test_embed_refuses_an_interpolating_transform_on_a_discrete_axis() -> None:
    # An interpolating transform cannot be embedded in a discrete axis.
    voxel = _spatial3("warp-voxel")
    field = DisplacementField(
        field=np.zeros((6, 7, 5, 3)), input=voxel, output=voxel
    )
    full = _discrete_ras_time("full")
    with pytest.raises(AdaptationError):
        embed(field, full=full, side="input", extents=None)


def test_compose_refuses_interpolating_subspace_on_discrete_axis() -> None:
    # A subspace with a field inner on a discrete axis cannot be applied to a
    # coordinate field.
    voxel = _spatial3("warp-voxel")
    field = DisplacementField(
        field=np.zeros((6, 7, 5, 3)), input=voxel, output=voxel
    )
    full = _discrete_ras_time("full")
    subspace = SubspaceTransformation(
        transformation=field,
        input_axes=np.asarray([0, 1, 2]),
        output_axes=np.asarray([0, 1, 2]),
        input=full,
        output=full,
    )
    coords = CoordinatesField(
        field=np.zeros((6, 7, 5, 3, 4)), input=full, output=full
    )
    with pytest.raises(CompositionError):
        compose(subspace, coords)


# ----------------------------------------------------------------------
#   SPLICE INVARIANTS
# ----------------------------------------------------------------------


def test_adapt_keeps_first_by_identity_for_a_bridge() -> None:
    # A bridge never rebuilds either endpoint.
    first = Affine(
        matrix=np.eye(3, 4),
        input=RASmm(),
        output=RASmm(),
    )
    second = Affine(
        matrix=np.eye(3, 4),
        input=LPSmm(),
        output=LPSmm(),
    )
    result = adapt(first, second, allow_type_grouped_positional=True)
    assert result.transformations[0] is first
    assert result.transformations[-1] is second


def test_adapt_keeps_first_by_identity_for_a_forward_embedding() -> None:
    # A forward embedding wraps the second transform and leaves the first.
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(np.eye(3, 4))
    result = adapt(first, second)
    assert result.transformations[0] is first
    wrapped = result.transformations[-1]
    assert isinstance(wrapped, SubspaceTransformation)


def test_backward_embedding_finds_the_original_leaf_by_identity() -> None:
    # A backward embedding wraps the first transform, which remains reachable
    # unchanged inside the wrapper.
    first = _lps_affine_3d(np.eye(3, 4))
    second = _voxel_to_ras_time(np.eye(4, 5))
    result = adapt(first, second)
    wrapped = result.transformations[0]
    assert isinstance(wrapped, SubspaceTransformation)
    inner = wrapped.transformation
    leaves = inner.transformations if isinstance(inner, Sequence) else [inner]
    assert any(leaf is first for leaf in leaves)


# ----------------------------------------------------------------------
#   adapt() REPORTS THE FULLER ENDPOINT AFTER AN EMBEDDING
# ----------------------------------------------------------------------


def _composed_homogeneous(result: Transformation) -> np.ndarray:
    # Return the full matrix of a result, whether or not it stayed a
    # sequence.
    if isinstance(result, Sequence):
        return _homogeneous_of_each(result)
    return np.asarray(result.compute().to(Affine).homogeneous_matrix)


# The affine is nontrivial, so that a dropped or misplaced axis is visible.
_EMBED_AFFINE_3D = np.array(
    [[1.3, 0.2, -0.1, 4.0], [0.0, 0.9, 0.3, -2.0], [0.1, 0.0, 1.1, 1.0]]
)


def test_adapt_forward_embedding_output_matches_its_last_piece() -> None:
    # The sequence leaves its coordinates in the fuller output system.
    first = _voxel_to_ras_time(np.eye(4, 5))
    second = _lps_affine_3d(np.eye(3, 4))
    result = adapt(first, second)
    last = result.transformations[-1]
    assert len(result.output.axes) == len(last.output.axes)
    assert len(result.output.axes) == 4


def test_adapt_backward_embedding_input_matches_its_first_piece() -> None:
    # The sequence reads its coordinates from the fuller input system.
    first = _lps_affine_3d(np.eye(3, 4))
    second = _voxel_to_ras_time(np.eye(4, 5))
    result = adapt(first, second)
    head = result.transformations[0]
    assert len(result.input.axes) == len(head.input.axes)
    assert len(result.input.axes) == 4


def test_adapt_forward_embedding_composes_like_the_flat_sequence() -> None:
    # A nested adapted sequence computes like the flat one.
    v2w = _voxel_to_ras_time(np.eye(4, 5))
    aff = _lps_affine_3d(_EMBED_AFFINE_3D)
    flat = Sequence([v2w, aff, v2w.inverse()]).compute()
    nested = Sequence([adapt(v2w, aff), v2w.inverse()]).compute()
    np.testing.assert_allclose(
        _composed_homogeneous(flat), _composed_homogeneous(nested), atol=1e-12
    )


def test_adapt_backward_embedding_composes_like_the_flat_sequence() -> None:
    v2w = _voxel_to_ras_time(np.eye(4, 5))
    aff = _lps_affine_3d(_EMBED_AFFINE_3D)
    flat = Sequence([v2w.inverse(), aff, v2w]).compute()
    nested = Sequence([v2w.inverse(), adapt(aff, v2w)]).compute()
    np.testing.assert_allclose(
        _composed_homogeneous(flat), _composed_homogeneous(nested), atol=1e-12
    )


# ----------------------------------------------------------------------
#   REGRESSION: A REBUILT VOXEL SUBSPACE IS STILL ARRAY-INDEX
# ----------------------------------------------------------------------


def _embedded_matrix(spatial: list, unit: str) -> np.ndarray:
    # Embed a transform whose own x axis points the other way into the first
    # three axes of a 4-D system, which reverses x. Only the axis units say
    # whether the axes index an array.
    full = CoordinateSystem(axes=[*spatial, TimeAxis(name="t", unit=unit)])
    sub = CoordinateSystem(
        axes=[
            RightToLeftAxis(name="x"),
            PosteriorToAnteriorAxis(name="y"),
            S(name="z"),
        ]
    )
    transform = Scaling(scale=np.ones(3), input=sub, output=sub)
    wrapped = embed(
        transform, full=full, side="input", extents={"x": 4, "y": 5, "z": 6}
    )
    return np.asarray(wrapped.compute().to(Affine).homogeneous_matrix)


def test_embedding_in_voxel_axes_keeps_the_extent_offset() -> None:
    # Voxel axes count samples, so reversing x maps i to 3 - i.
    matrix = _embedded_matrix(list(FRASCoordinateSystem().axes), "index")
    np.testing.assert_array_equal(np.diag(matrix), [-1, 1, 1, 1, 1])
    np.testing.assert_array_equal(matrix[:4, 4], [3, 0, 0, 0])


def test_embedding_in_world_axes_is_a_pure_sign_flip() -> None:
    matrix = _embedded_matrix(list(RASmm().axes), "s")
    np.testing.assert_array_equal(np.diag(matrix), [-1, 1, 1, 1, 1])
    np.testing.assert_array_equal(matrix[:4, 4], [0, 0, 0, 0])


@pytest.mark.parametrize(
    "src, dst, kind, factor",
    [
        # Regression (#257): non-SI units convert by their true scale.
        ("inch", "millimeter", "space", 25.4),
        ("millimeter", "inch", "space", 1 / 25.4),
        ("foot", "meter", "space", 0.3048),
        ("yard", "foot", "space", 3.0),
        ("mile", "kilometer", "space", 1.609344),
        ("angstrom", "nanometer", "space", 0.1),
        ("minute", "second", "time", 60.0),
        ("hour", "second", "time", 3600.0),
        ("hour", "minute", "time", 60.0),
        ("day", "hour", "time", 24.0),
        ("second", "millisecond", "time", 1000.0),
    ],
)
def test_non_si_unit_difference_is_a_scaling(
    src: str, dst: str, kind: str, factor: float
) -> None:
    Ax = SpaceAxis if kind == "space" else TimeAxis
    source = CoordinateSystem(name="a", axes=[Ax(name="x", unit=src)])
    target = CoordinateSystem(name="b", axes=[Ax(name="x", unit=dst)])
    result = bridge(source, target)
    assert isinstance(result, Scaling)
    np.testing.assert_allclose(result.scale, [factor], rtol=1e-12)


# ----------------------------------------------------------------------
#   Sequence.bridge()
# ----------------------------------------------------------------------


def test_bridge_method_inserts_the_flip_and_keeps_both_leaves() -> None:
    a, b = _ras_affine(), _lps_affine()
    bridged = Sequence([a, b]).bridge()
    first, flip, last = bridged.transformations
    # Nothing is composed, and neither leaf is rebuilt.
    assert first is a
    assert last is b
    np.testing.assert_array_equal(np.diag(_homogeneous(flip)), [-1, -1, 1, 1])


def test_bridge_method_leaves_matching_systems_alone() -> None:
    a, b = _ras_affine(), _ras_affine()
    bridged = Sequence([a, b]).bridge()
    assert list(bridged.transformations) == [a, b]


def test_bridge_method_keeps_nesting_and_bridges_inside_it() -> None:
    a, b = _ras_affine(), _lps_affine()
    bridged = Sequence([Sequence([a, b])]).bridge()
    (inner,) = bridged.transformations
    assert isinstance(inner, Sequence)
    assert len(inner) == 3
    # Flattening afterwards gives the chain that composition works on.
    assert len(bridged.flatten(endpoints=False)) == 3


def test_bridge_method_raises_when_a_boundary_cannot_be_bridged() -> None:
    # A sampled axis has no bridge to a physical one.
    sampled = CoordinateSystem(
        axes=[SpaceAxis(name="x", unit="index", orientation=LeftToRight())]
    )
    world = CoordinateSystem(axes=[LeftToRightAxis(name="x", unit="mm")])
    a = Affine(matrix=np.eye(1, 2), input=sampled, output=sampled)
    b = Affine(matrix=np.eye(1, 2), input=world, output=world)
    with pytest.raises(AdaptationError, match="sampled.*millimeter"):
        Sequence([a, b]).bridge()
