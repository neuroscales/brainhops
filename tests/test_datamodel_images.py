"""Unit tests for the image data model (reslice, indexing, geometry)."""

import numpy as np
import pytest

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import (
    RASCoordinateSystem,
    VoxelCoordinateSystem,
)
from brainhops.datamodel.transformations import Affine


def _voxel_to_ras(scale: float = 2.0) -> Affine:
    """A voxel-to-world affine that scales each axis by ``scale``."""
    matrix = np.diag([scale, scale, scale, 1.0])[:-1]
    return Affine(
        matrix=matrix,
        input=VoxelCoordinateSystem(),
        output=RASCoordinateSystem(),
    )


def _image(scale: float = 2.0) -> SingleScaleImage:
    """A small single-scale image with a voxel-to-world transformation."""
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    return SingleScaleImage(data=data, transformations=[_voxel_to_ras(scale)])


def test_reslice_onto_own_grid_returns_working_single_scale_image() -> None:
    img = _image()

    resliced = img.reslice()

    assert isinstance(resliced, SingleScaleImage)
    assert resliced.shape == img.shape
    # Resampling onto the image's own grid returns the same values.
    assert np.allclose(np.asarray(resliced), np.asarray(img))


def test_reslice_no_argument_with_identity_transform_round_trips() -> None:
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    identity = Affine(
        matrix=np.eye(4)[:-1],
        input=VoxelCoordinateSystem(),
        output=VoxelCoordinateSystem(),
    )
    img = SingleScaleImage(data=data, transformations=[identity])

    resliced = img.reslice()

    assert isinstance(resliced, SingleScaleImage)
    assert resliced.shape == img.shape
    # Resampling an identity-transformed image onto its own grid returns
    # the same data.
    assert np.allclose(np.asarray(resliced), data)


def test_call_then_reslice_returns_single_scale_image() -> None:
    img = _image()
    identity = Affine(
        matrix=np.eye(4)[:-1],
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )

    moved = img(identity)

    assert isinstance(moved, SingleScaleImage)
    # The applied transformation is appended as the preferred one.
    assert len(moved.transformations) == len(img.transformations) + 1

    # Reslicing onto its own grid resamples the image without moving it.
    resliced = moved.reslice()

    assert isinstance(resliced, SingleScaleImage)
    assert resliced.shape == img.shape
    assert np.allclose(np.asarray(resliced), np.asarray(img))
    matrix = np.asarray(resliced.transformation.compute().to(Affine).matrix)
    assert np.allclose(matrix, np.asarray(_voxel_to_ras().matrix))


def test_getitem_preserves_geometry() -> None:
    img = _image()

    sub = img[:, 1:3, :]

    assert isinstance(sub, SingleScaleImage)
    assert sub.shape == (2, 2, 4)
    # The sampling grid follows the indexed shape.
    assert sub.grid.shape == (2, 2, 4)

    # The world position of the sub-image origin is the world position of
    # the voxel the slice started from. The voxel spacing is unchanged.
    matrix = np.asarray(sub.transformation.compute().to(Affine).matrix)
    expected = np.asarray(_voxel_to_ras().matrix)
    expected[1, -1] = 2.0
    assert np.allclose(matrix, expected)


def test_setting_transformation_to_a_new_transform_appends_it() -> None:
    img = _image()
    other = _voxel_to_ras(scale=3.0)

    img.transformation = other

    assert len(img.transformations) == 2
    assert img.transformation is other


def test_reselecting_existing_transformation_does_not_duplicate() -> None:
    first = _voxel_to_ras(scale=2.0)
    second = _voxel_to_ras(scale=3.0)
    img = SingleScaleImage(
        data=np.ones((2, 3, 4)), transformations=[first, second]
    )

    # Re-selecting a transformation already present reorders it to the end
    # rather than adding a copy.
    img.transformation = first

    assert len(img.transformations) == 2
    assert img.transformation is first
    assert img.transformations[0] is second


def test_selecting_transformation_by_index_reorders() -> None:
    first = _voxel_to_ras(scale=2.0)
    second = _voxel_to_ras(scale=3.0)
    img = SingleScaleImage(
        data=np.ones((2, 3, 4)), transformations=[first, second]
    )

    img.transformation = 0

    assert len(img.transformations) == 2
    assert img.transformation is first


def test_selecting_transformation_by_unknown_name_raises() -> None:
    img = _image()

    with pytest.raises(KeyError):
        img.transformation = "no-such-space"


def test_multiscale_reslice_onto_own_grid() -> None:
    level0 = _image()
    model_to_world = Affine(
        matrix=np.eye(4)[:-1],
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    pyramid = MultiScaleImage(
        images=[level0], transformations=[model_to_world]
    )

    resliced = pyramid.reslice()

    assert isinstance(resliced, SingleScaleImage)
    assert resliced.shape == level0.shape
    assert np.allclose(np.asarray(resliced), np.asarray(level0))
    # The pyramid-level transform is the identity, so the resliced
    # voxel-to-world matrix is the highest-resolution level's own.
    matrix = np.asarray(resliced.transformation.compute().to(Affine).matrix)
    assert np.allclose(matrix, np.asarray(_voxel_to_ras().matrix))


def test_multiscale_geometry_grid_lives_in_level_zero_voxel_space() -> None:
    level0 = _image()
    model_to_world = Affine(
        matrix=np.eye(4)[:-1],
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    pyramid = MultiScaleImage(
        images=[level0], transformations=[model_to_world]
    )

    geometry = pyramid.geometry

    # The grid is declared in the highest-resolution level's voxel space,
    # not in the model space that the pyramid-level transform maps from.
    grid_space = getattr(geometry.grid.output, "name", None)
    voxel_space = getattr(level0.transformation.input, "name", None)
    model_space = getattr(model_to_world.input, "name", None)
    assert grid_space == voxel_space == "voxel"
    assert grid_space != model_space


def test_reslice_selects_the_multiscale_level_that_matches_the_target() -> (
    None
):
    # An image whose transformation carries a two-level displacement field
    # reslices with the level whose resolution matches the target grid. A
    # target grid twice as coarse selects the coarse level, so the result
    # matches reslicing with that level substituted by hand, and differs
    # from reslicing with the fine level.
    import numpy as np

    from brainhops.datamodel.transformations import Identity, Scaling

    # OmeZarrField is an OME-Zarr file format, so it is available only when
    # abczarr is installed. It is used here only as a convenient builder of a
    # multiscale field for the reslice check.
    pytest.importorskip("abczarr")
    from brainhops.io.transformations.zarr import OmeZarrField

    voxel2world = Affine(matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    rng = np.random.default_rng(3)
    fine = rng.normal(size=(8, 8, 2)) * 0.5
    coarse = rng.normal(size=(4, 4, 2)) * 0.5
    field = OmeZarrField(
        raw_levels=[fine, coarse],
        voxel2world=voxel2world,
        level_transforms=[Identity(), Scaling(scale=[2.0, 2.0])],
        axes=[
            Axis(type="space"),
            Axis(type="space"),
            Axis(type="displacement"),
        ],
    )

    data = np.arange(64, dtype=float).reshape(8, 8)
    coarse_target = Affine(matrix=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))

    auto = np.asarray(
        SingleScaleImage(data=data, transformations=[field]).reslice(
            coarse_target
        )
    )
    by_coarse = np.asarray(
        SingleScaleImage(
            data=data, transformations=[field.to_singlescale(1)]
        ).reslice(coarse_target)
    )
    by_fine = np.asarray(
        SingleScaleImage(
            data=data, transformations=[field.to_singlescale(0)]
        ).reslice(coarse_target)
    )

    assert np.allclose(auto, by_coarse)
    assert not np.allclose(auto, by_fine)


def _pyramid() -> MultiScaleImage:
    model_to_world = Affine(
        matrix=np.eye(4)[:-1],
        input=RASCoordinateSystem(),
        output=RASCoordinateSystem(),
    )
    return MultiScaleImage(images=[_image()], transformations=[model_to_world])


def test_multiscale_reslice_without_copy_may_share_the_level_data() -> None:
    pyramid = _pyramid()

    resliced = pyramid.reslice(copy=False)

    # Onto its own grid the reslice only gathers, and the selected level is
    # handed over without a copy, so the result is a view of its data.
    assert np.shares_memory(resliced.data, pyramid.images[0].data)


def test_multiscale_reslice_with_copy_is_fresh() -> None:
    pyramid = _pyramid()
    level0 = pyramid.images[0].data
    before = level0.copy()

    resliced = pyramid.reslice(copy=True)

    assert np.array_equal(resliced.data, level0)
    assert resliced.data is not level0
    assert not np.shares_memory(resliced.data, level0)
    resliced.data[...] = -1
    assert np.array_equal(level0, before)


# ----------------------------------------------------------------------
#   METADATA
# ----------------------------------------------------------------------


def _dwi() -> SingleScaleImage:
    """A 4-D diffusion series: three spatial axes and a time axis."""
    from brainhops.datamodel.metadata import Metadata
    from brainhops.datamodel.systems import CoordinateSystem

    system = CoordinateSystem(
        axes=[
            Axis("i", "space"),
            Axis("j", "space"),
            Axis("k", "space"),
            Axis("t", "time"),
        ]
    )
    placement = Affine(
        matrix=np.diag([2.0, 2.0, 2.0, 1.0, 1.0])[:-1],
        input=system,
        output=RASCoordinateSystem(),
    )
    return SingleScaleImage(
        data=np.zeros((4, 5, 6, 3)),
        transformations=[placement],
        metadata=Metadata(
            bvalues=(0.0, 1000.0, 2000.0),
            phase_encoding_direction="j-",
            slice_timing=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            history=("acquired",),
        ),
    )


def test_getitem_selects_the_entries_of_the_indexed_axis() -> None:
    dwi = _dwi()
    first = dwi[..., :2]
    assert first.shape == (4, 5, 6, 2)
    assert first.metadata.bvalues == (0.0, 1000.0)
    # The spatial axes are untouched: the spatial fields are kept.
    assert first.metadata.slice_timing == dwi.metadata.slice_timing
    assert first.metadata.phase_encoding_direction == (
        dwi.metadata.phase_encoding_direction
    )
    # An integer drops the time axis, and with it the b-values.
    assert dwi[..., 1].metadata.bvalues is None


def test_getitem_maps_a_direction_through_a_flip() -> None:
    from brainhops.datamodel.metadata import EncodingDirection

    flipped = _dwi()[:, ::-1]
    assert flipped.metadata.phase_encoding_direction == EncodingDirection("j")
    assert flipped.metadata.slice_timing is None
    assert flipped.metadata.bvalues == (0.0, 1000.0, 2000.0)
    # A single component is an index of the first axis.
    assert _dwi()[::-1].metadata.phase_encoding_direction == (
        EncodingDirection("j-")
    )


def test_getitem_appends_to_the_history() -> None:
    dwi = _dwi()
    assert dwi[:, ::-1, :, :2].metadata.history == ("acquired", "getitem")
    assert dwi[...].metadata.history == ("acquired", "getitem")
    assert dwi.metadata.history == ("acquired",)
    assert _image()[0].metadata is None


def test_reslice_derives_the_metadata() -> None:
    from brainhops.datamodel.metadata import EncodingDirection

    dwi = _dwi()
    swap = np.diag([2.0, 2.0, 2.0, 1.0, 1.0])[[1, 0, 2, 3]]
    target = Affine(
        matrix=swap,
        input=dwi.transformation.input,
        output=RASCoordinateSystem(),
    )
    resliced = dwi.reslice(target, degree=0)
    assert resliced.metadata.phase_encoding_direction == (
        EncodingDirection("i-")
    )
    assert resliced.metadata.slice_timing is None
    assert resliced.metadata.bvalues == (0.0, 1000.0, 2000.0)
    assert resliced.metadata.history == ("acquired", "reslice")
    # A call keeps the metadata as it is.
    moved = dwi(Affine(matrix=np.eye(5)[:-1]))
    assert moved.metadata == dwi.metadata
    assert moved.metadata is not dwi.metadata
