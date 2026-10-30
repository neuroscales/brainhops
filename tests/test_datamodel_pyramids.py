"""Tests for the scales that multiscale images and fields build.

`with_scales` reads its argument by type: `None` builds as many scales
as the rules allow, an int builds that number of scales, a sequence of
factors builds one scale per factor, and a sequence of scales is taken
as it is. The conversions take the same values under the keyword
`nscales=`. An axis is selected by its position, which may count from
the end, or by its name.
"""

import numpy as np
import pytest

from brainhops.datamodel.axes import SpaceAxis, TimeAxis
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    CoordinateField,
    DisplacementField,
    MultiscaleField,
    Sequence,
)


def _image(
    shape: tuple = (64, 64, 32, 5), sizes: tuple = (1, 1, 2)
) -> MultiScaleImage:
    # An image whose axes are x, y, z and t, with anisotropic voxels.
    axes = [SpaceAxis(name=name) for name in "xyz"] + [TimeAxis(name="t")]
    axes = axes[: len(shape)]
    matrix = np.eye(len(shape), len(shape) + 1)
    for axis, size in enumerate(sizes):
        matrix[axis, axis] = size
    matrix[:3, -1] = [10.0, 20.0, 30.0][: min(3, len(shape))]
    data = np.random.default_rng(0).random(shape).astype("float32")
    geometry = Affine(matrix=matrix, input=CoordinateSystem(axes=axes))
    return MultiScaleImage(
        images=[SingleScaleImage(data, transformations=[geometry])]
    )


def _matrix(level: SingleScaleImage) -> np.ndarray:
    return np.asarray(level.transformation.to(Affine).matrix)


def _shapes(pyramid: MultiScaleImage) -> list:
    return [tuple(level.data.shape) for level in pyramid.images]


# ---- what the argument `scales` is --------------------------------------


def test_none_builds_as_many_scales_as_the_rules_allow() -> None:
    pyramid = _image((256, 256, 32, 5), (1, 1, 1)).with_scales()

    # z has 32 voxels and freezes, and x and y are halved while they
    # have more than 32 voxels.
    assert _shapes(pyramid) == [
        (256, 256, 32, 5),
        (128, 128, 32, 5),
        (64, 64, 32, 5),
        (32, 32, 32, 5),
    ]


def test_an_int_counts_every_scale() -> None:
    pyramid = _image((128, 128, 128, 1), (1, 1, 1)).with_scales(3)

    assert pyramid.nscales == 3


def test_an_int_beyond_the_rules_builds_fewer_scales() -> None:
    image = _image((128, 128, 128, 1), (1, 1, 1))

    assert image.with_scales(10).nscales == 3
    with pytest.raises(ValueError, match="only"):
        image.with_scales(10, strict=True)


def test_a_sequence_of_factors_builds_one_scale_per_factor() -> None:
    pyramid = _image().with_scales([2, (4, 4, 2), {"x": 8, None: 4}])

    assert _shapes(pyramid) == [
        (64, 64, 32, 5),
        (32, 32, 16, 5),
        (16, 16, 16, 5),
        (8, 16, 8, 5),
    ]


def test_a_sequence_of_scales_is_taken_as_it_is() -> None:
    pyramid = _image().with_scales([2, 4])

    kept = pyramid.with_scales(list(pyramid.images[:2]))

    assert kept.nscales == 2
    assert kept.images[1] is pyramid.images[1]


@pytest.mark.parametrize(
    "scales, error",
    [
        ([], ValueError),
        ("2", TypeError),
        ({"x": 2}, TypeError),
        (True, TypeError),
        (2.5, TypeError),
    ],
)
def test_a_wrong_argument_is_refused(scales: object, error: type) -> None:
    with pytest.raises(error):
        _image().with_scales(scales)


def test_a_sequence_that_mixes_factors_and_scales_is_refused() -> None:
    pyramid = _image()

    with pytest.raises(TypeError, match="mixes"):
        pyramid.with_scales([pyramid.images[0], 2])


def test_factor_is_refused_with_a_sequence() -> None:
    with pytest.raises(TypeError, match="factor"):
        _image().with_scales([2, 4], factor=2)


# ---- the factors of each axis ------------------------------------------


@pytest.mark.parametrize(
    "factor",
    [
        {"x": 4, "y": 2, "z": 2},
        {0: 4, 1: 2, 2: 2},
        {-4: 4, -3: 2, -2: 2},
        {"x": 4, 1: 2, -2: 2},
        (4, 2, 2),
        (4, 2, 2, 1),
    ],
)
def test_an_axis_is_selected_by_name_or_by_position(factor: object) -> None:
    pyramid = _image().with_scales([factor])

    assert _shapes(pyramid)[1] == (16, 32, 16, 5)


@pytest.mark.parametrize(
    "factor, shape",
    [
        ({"z": 1, None: 2}, (32, 32, 32, 5)),
        ({"t": 1, None: 2}, (32, 32, 16, 5)),
        ({"x": 4, None: 2}, (16, 32, 16, 5)),
        ({"t": 5, None: 2}, (32, 32, 16, 1)),
    ],
)
def test_the_key_none_gives_a_factor_to_the_other_spatial_axes(
    factor: dict, shape: tuple
) -> None:
    pyramid = _image().with_scales([factor])

    assert _shapes(pyramid)[1] == shape


def test_an_int_downsamples_the_spatial_axes_only() -> None:
    pyramid = _image().with_scales([2, 4])

    assert _shapes(pyramid) == [
        (64, 64, 32, 5),
        (32, 32, 16, 5),
        (16, 16, 8, 5),
    ]


def test_the_step_of_the_rules_takes_a_factor_per_axis() -> None:
    pyramid = _image((96, 96, 32, 5), (1, 1, 1)).with_scales(
        2, factor={"x": 3, "y": 3}
    )

    assert _shapes(pyramid)[1] == (32, 32, 32, 5)


def test_a_step_of_one_freezes_an_axis() -> None:
    pyramid = _image((128, 128, 128, 1), (1, 1, 1)).with_scales(
        factor={"z": 1, None: 2}
    )

    assert all(shape[2] == 128 for shape in _shapes(pyramid))
    assert _shapes(pyramid)[-1][:2] == (32, 32)


def test_the_step_of_the_rules_reaches_a_time_axis_by_name() -> None:
    pyramid = _image((64, 64, 32, 64), (1, 1, 1)).with_scales(
        2, factor={"t": 2}
    )

    assert _shapes(pyramid)[1] == (64, 64, 32, 32)


def test_anisotropic_voxels_become_roughly_isotropic_first() -> None:
    pyramid = _image((128, 128, 64, 5), (1, 1, 2)).with_scales(3)

    # z waits one step, until x and y have voxels of 2.
    assert _shapes(pyramid) == [
        (128, 128, 64, 5),
        (64, 64, 64, 5),
        (32, 32, 32, 5),
    ]


def test_an_axis_above_the_minimum_size_is_halved_below_it() -> None:
    pyramid = _image((128, 40, 32, 1), (1, 1, 1)).with_scales()

    assert _shapes(pyramid) == [
        (128, 40, 32, 1),
        (64, 20, 32, 1),
        (32, 20, 32, 1),
    ]


@pytest.mark.parametrize(
    "factors, error",
    [
        ([{"w": 2}], KeyError),
        ([{4: 2}], IndexError),
        ([{-5: 2}], IndexError),
        ([{"x": 2, 0: 2}], ValueError),
        ([(2, 2)], ValueError),
        ([0], ValueError),
        ([{"x": 2.5}], TypeError),
        ([4, 2], ValueError),
    ],
)
def test_wrong_factors_are_refused(factors: object, error: type) -> None:
    with pytest.raises(error):
        _image().with_scales(factors)


def test_the_default_names_only_select_axes() -> None:
    # Without known axes, the first three axes are named x, y and z to
    # select them, but these names are never stored.
    image = SingleScaleImage(np.zeros((64, 64, 40)))

    pyramid = image.to(MultiScaleImage, nscales=2, factor={"x": 4, None: 2})

    assert _shapes(pyramid)[1] == (16, 32, 20)
    for level in pyramid.images:
        assert getattr(level.transformation, "input", None) is None


# ---- the placement and the values of a level ---------------------------


def test_a_frozen_axis_keeps_its_size_and_its_placement() -> None:
    built = _image().with_scales([{"x": 2, "y": 2}])

    assert _shapes(built)[1] == (32, 32, 32, 5)
    fine, coarse = _matrix(built.images[0]), _matrix(built.images[1])
    np.testing.assert_allclose(coarse[:, 2:4], fine[:, 2:4])
    np.testing.assert_allclose(coarse[2, -1], fine[2, -1])
    np.testing.assert_allclose(np.diag(coarse[:2, :2]), [2.0, 2.0])
    np.testing.assert_allclose(coarse[:2, -1], fine[:2, -1] + 0.5)


def test_the_exact_placement_follows_the_window_of_each_factor() -> None:
    pyramid = _image().with_scales([(3, 2, 1)])

    coarse = _matrix(pyramid.images[1])

    assert _shapes(pyramid)[1] == (21, 32, 32, 5)
    np.testing.assert_allclose(np.diag(coarse[:3, :3]), [3.0, 2.0, 2.0])
    np.testing.assert_allclose(coarse[:3, -1], [11.0, 20.5, 30.0])


@pytest.mark.parametrize(
    "align, count, scale, shift",
    [
        ("exact", 21, 3.0, 1.0),
        ("edges", 22, 65 / 22, (65 / 22 - 1) / 2),
        ("centers", 23, 64 / 22, 0.0),
    ],
)
def test_each_alignment_places_the_level_by_its_actual_scale(
    align: str, count: int, scale: float, shift: float
) -> None:
    pyramid = _image((65, 64, 32, 1), (1, 1, 1)).with_scales(
        [{"x": 3}], align=align
    )

    coarse = _matrix(pyramid.images[1])

    assert _shapes(pyramid)[1] == (count, 64, 32, 1)
    np.testing.assert_allclose(coarse[0, 0], scale)
    np.testing.assert_allclose(coarse[0, -1], 10.0 + shift)


@pytest.mark.parametrize("align", ["exact", "edges", "centers"])
def test_a_linear_image_stays_exact_where_its_level_lies(align: str) -> None:
    # The values of a linear image are its world coordinate along x,
    # which every level keeps at the centre of its voxels, but at the
    # edges.
    image = _image((49, 8, 8, 1), (1, 1, 1))
    data = np.broadcast_to(
        np.arange(49.0)[:, None, None, None] + 10.0, (49, 8, 8, 1)
    )
    image.images[0].data = data.copy()

    pyramid = image.with_scales([{"x": 4}], align=align, mode="binomial")

    coarse = pyramid.images[1]
    world = _matrix(coarse)[0, 0] * np.arange(coarse.data.shape[0])
    world = world + _matrix(coarse)[0, -1]
    np.testing.assert_allclose(
        np.asarray(coarse.data)[2:-2, 0, 0, 0], world[2:-2], rtol=1e-6
    )


def test_a_level_of_multiples_is_computed_from_the_level_before() -> None:
    image = _image((24, 24, 4, 1))
    pyramid = image.with_scales([(2, 2, 1), (6, 4, 1)])
    direct = image.with_scales([(6, 4, 1)])

    # The data are in single precision, whose rounding depends on the
    # order of the sums.
    np.testing.assert_allclose(
        np.asarray(pyramid.images[2].data),
        np.asarray(direct.images[1].data),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "mode", ["nearest", "mode", "vote_binomial", "vote_gaussian", "max", "min"]
)
def test_a_label_mode_keeps_the_labels_of_an_image(mode: str) -> None:
    labels = np.random.default_rng(4).integers(0, 6, (16, 16, 8)) * 10
    pyramid = MultiScaleImage(
        images=[SingleScaleImage(labels.astype("int32"))]
    )

    built = pyramid.with_scales(2, mode=mode, min_size=4)

    coarse = np.asarray(built.images[1].data)
    assert coarse.dtype == np.int32
    assert set(np.unique(coarse)) <= set(np.unique(labels))
    np.testing.assert_allclose(_matrix(built.images[1])[:, -1], 0.5)


@pytest.mark.parametrize("mode", ["max", "min"])
@pytest.mark.parametrize("align", ["edges", "centers"])
def test_a_block_mode_needs_the_exact_alignment(mode: str, align: str) -> None:
    with pytest.raises(ValueError, match="exact"):
        _image().with_scales(2, mode=mode, align=align)


@pytest.mark.parametrize("mode", ["mode", "vote_binomial", "vote_gaussian"])
@pytest.mark.parametrize("align", ["edges", "centers"])
def test_a_vote_keeps_the_labels_under_every_alignment(
    mode: str, align: str
) -> None:
    labels = np.random.default_rng(5).integers(0, 6, (17, 15, 8)) * 10
    pyramid = MultiScaleImage(
        images=[SingleScaleImage(labels.astype("uint8"))]
    )

    built = pyramid.with_scales(2, mode=mode, align=align, min_size=4)

    coarse = np.asarray(built.images[1].data)
    assert coarse.dtype == np.uint8
    assert set(np.unique(coarse)) <= set(np.unique(labels))


def test_an_unknown_mode_or_alignment_is_refused() -> None:
    with pytest.raises(ValueError, match="modes"):
        _image().with_scales(2, mode="median")
    with pytest.raises(ValueError, match="align"):
        _image().with_scales(2, align="corners")


# ---- the scales that are kept ------------------------------------------


def test_the_scales_that_match_the_request_are_kept() -> None:
    pyramid = _image((256, 256, 64, 1), (1, 1, 1)).with_scales(3)

    extended = pyramid.with_scales(4)

    assert extended.nscales == 4
    assert extended.images[1] is pyramid.images[1]
    assert extended.images[2] is pyramid.images[2]


def test_the_scales_that_differ_from_the_request_are_rebuilt() -> None:
    pyramid = _image((256, 256, 64, 1), (1, 1, 1)).with_scales([2, 4])

    rebuilt = pyramid.with_scales([2, {"x": 8, None: 4}])

    assert rebuilt.images[1] is pyramid.images[1]
    assert rebuilt.images[2] is not pyramid.images[2]
    assert _shapes(rebuilt)[2] == (32, 64, 16, 1)


def test_a_scale_of_another_alignment_is_rebuilt() -> None:
    pyramid = _image((255, 64, 32, 1), (1, 1, 1)).with_scales([{"x": 2}])

    rebuilt = pyramid.with_scales([{"x": 2}], align="edges")

    assert rebuilt.images[1] is not pyramid.images[1]
    assert _shapes(rebuilt)[1] == (128, 64, 32, 1)


def test_scales_out_of_order_are_refused() -> None:
    pyramid = _image().with_scales([2])

    with pytest.raises(ValueError, match="fine to coarse"):
        pyramid.with_scales(pyramid.images[::-1])


def test_scales_of_another_space_are_refused() -> None:
    pyramid = _image().with_scales([2])
    axes = [SpaceAxis(name=n) for n in "xyz"] + [TimeAxis(name="t")]
    other = SingleScaleImage(
        np.zeros((32, 32, 16, 5)),
        transformations=[
            Affine(
                matrix=_matrix(pyramid.images[1]),
                output=CoordinateSystem(name="other", axes=axes),
            )
        ],
    )
    first = pyramid.images[0]
    first.transformations = [
        Affine(
            matrix=_matrix(first),
            output=CoordinateSystem(name="world", axes=axes),
        )
    ]

    with pytest.raises(ValueError, match="same space"):
        pyramid.with_scales([first, other])


# ---- conversions -------------------------------------------------------


def test_an_image_converts_to_a_pyramid_of_several_scales() -> None:
    image = _image((128, 128, 64, 1), (1, 1, 1)).images[0]

    pyramid = image.to(MultiScaleImage, nscales=3, mode="max")
    explicit = image.to(MultiScaleImage, nscales=[2, {"x": 4, "y": 4, "z": 2}])

    assert isinstance(pyramid, MultiScaleImage)
    assert pyramid.nscales == 3
    assert _shapes(explicit)[2] == (32, 32, 32, 1)


def test_a_data_model_pyramid_keeps_one_scale_by_default() -> None:
    image = _image((128, 128, 64, 1), (1, 1, 1)).images[0]

    assert image.to(MultiScaleImage).nscales == 1
    assert image.to(MultiScaleImage, align="edges", nscales=None).nscales > 1


def test_a_pyramid_converts_to_more_scales() -> None:
    pyramid = _image((128, 128, 64, 1), (1, 1, 1))

    assert pyramid.to(MultiScaleImage, nscales=3).nscales == 3


def test_the_options_alone_are_refused_when_nothing_is_built() -> None:
    pyramid = _image((128, 128, 64, 1), (1, 1, 1))

    with pytest.raises(TypeError, match="nscales"):
        pyramid.to(MultiScaleImage, mode="max")


def test_a_single_scale_image_takes_no_scales() -> None:
    image = _image().images[0]

    with pytest.raises(TypeError, match="multiscale"):
        image.to(SingleScaleImage, nscales=2)


# ---- fields ------------------------------------------------------------


def _field_chain(kind: str = "displacement") -> Sequence:
    # A field on a grid of 2 mm voxels, from the world to the world.
    grid = Affine(matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3])
    if kind == "displacement":
        values = np.empty((40, 40, 20, 3))
        values[:] = [1.0, -2.0, 0.5]
        return grid @ DisplacementField(field=values) @ grid.inverse()
    index = np.stack(np.meshgrid(*map(np.arange, (40, 40, 20)), indexing="ij"))
    values = np.moveaxis(index, 0, -1) * 2.0 + 1.0
    return CoordinateField(field=values) @ grid.inverse()


def _grid_field(output: CoordinateSystem) -> Sequence:
    # A field of coordinates in a voxel grid, from the grid `voxels` of
    # 2 mm voxels to `output`, which shifts each voxel by half a voxel.
    voxels = CoordinateSystem(
        name="voxels", axes=[SpaceAxis(name=n) for n in "ijk"]
    )
    if output.name == "voxels":
        output = voxels
    grid = Affine(matrix=np.diag([2.0, 2.0, 2.0, 1.0])[:3], input=voxels)
    index = np.stack(np.meshgrid(*map(np.arange, (40, 40, 20)), indexing="ij"))
    values = np.moveaxis(index, 0, -1) + 0.5
    field = CoordinateField(field=values, input=voxels, output=output)
    return field @ grid.inverse()


def test_a_field_converts_to_a_multiscale_field() -> None:
    field = _field_chain().to(MultiscaleField, nscales=2, min_size=8)

    assert isinstance(field, MultiscaleField)
    assert field.nscales == 2
    coarse = field.to_singlescale(1)
    # A displacement of 1 voxel of the finest grid is half a voxel of a
    # grid downsampled by 2, along each of its three axes.
    values = np.asarray(coarse.transformations[1].field)
    np.testing.assert_allclose(values[3, 3, 3], [0.5, -1.0, 0.25])


def test_a_data_model_field_keeps_one_scale_by_default() -> None:
    field = _field_chain().to(MultiscaleField)

    assert field.nscales == 1
    assert field.with_scales(min_size=8).nscales == 4


def test_a_field_of_world_coordinates_is_only_averaged() -> None:
    multiscale = _field_chain("coordinate").to(
        MultiscaleField, nscales=[(2, 2, 2), (4, 4, 2)]
    )

    for index in range(1, multiscale.nscales):
        scale = multiscale.to_singlescale(index)
        world2voxel = np.asarray(scale.transformations[0].matrix)
        values = np.asarray(scale.transformations[1].field)
        grid = np.stack(
            np.meshgrid(*map(np.arange, values.shape[:-1]), indexing="ij"),
            axis=-1,
        )
        voxel2world = np.linalg.inv(np.vstack([world2voxel, [0, 0, 0, 1]]))
        expected = grid @ voxel2world[:3, :3].T + voxel2world[:3, 3]
        # The coordinates are those of the voxels, plus 1.
        np.testing.assert_allclose(values, expected + 1.0, atol=1e-9)


def test_the_voxel_coordinates_of_the_field_grid_follow_the_grid() -> None:
    # The coarse voxel `j` lies at the fine voxel `2 * j + 0.5`, whose
    # value is `2 * j + 1`, which is the voxel `j + 0.25` of the coarse
    # grid.
    voxels = CoordinateSystem(name="voxels")
    field = _grid_field(voxels).to(MultiscaleField, nscales=[(2, 2, 1)])

    values = np.asarray(field.to_singlescale(1).transformations[1].field)

    np.testing.assert_allclose(values[1, 2, 3], [1.25, 2.25, 3.5])


def test_the_voxel_coordinates_of_another_grid_are_only_averaged() -> None:
    other = CoordinateSystem(
        name="other", axes=[SpaceAxis(name=n) for n in "ijk"]
    )
    field = _grid_field(other).to(MultiscaleField, nscales=[(2, 2, 1)])

    values = np.asarray(field.to_singlescale(1).transformations[1].field)

    np.testing.assert_allclose(values[1, 2, 3], [3.0, 5.0, 3.5])


@pytest.mark.parametrize("align", ["exact", "edges", "centers"])
def test_the_map_of_a_field_of_displacements_is_kept(align: str) -> None:
    # A constant displacement of 1 mm along x is 1 / s voxels of a level
    # of actual scale `s`, whatever the alignment.
    grid = Affine(matrix=np.diag([1.0, 1.0, 1.0, 1.0])[:3])
    values = np.zeros((65, 64, 20, 3))
    values[..., 0] = 1.0
    chain = grid @ DisplacementField(field=values) @ grid.inverse()

    field = chain.to(MultiscaleField, nscales=[{0: 3}], align=align)

    scale = field.to_singlescale(1)
    world2voxel = np.asarray(scale.transformations[0].matrix)
    coarse = np.asarray(scale.transformations[1].field)
    np.testing.assert_allclose(coarse[..., 0], world2voxel[0, 0])
    np.testing.assert_allclose(coarse[..., 1:], 0.0)


def test_a_field_is_sampled_at_its_nodes_with_nearest() -> None:
    # Under the exact alignment, the mode "nearest" keeps the vector of
    # the first node of each window of 2 nodes, while the coarse voxel
    # lies at the centre of the window.
    chain = _field_chain("coordinate")
    field = chain.to(MultiscaleField, nscales=[2], mode="nearest")

    scale = field.to_singlescale(1)
    values = np.asarray(scale.transformations[1].field)
    finest = np.asarray(field.to_singlescale(0).transformations[1].field)
    np.testing.assert_array_equal(values, finest[::2, ::2, ::2])
    world2voxel = np.asarray(scale.transformations[0].matrix)
    np.testing.assert_allclose(world2voxel[:, -1], -0.25)
    np.testing.assert_allclose(np.diag(world2voxel[:, :3]), 0.25)


def test_the_vector_axis_of_a_field_is_never_downsampled() -> None:
    field = _field_chain().to(MultiscaleField)

    with pytest.raises(ValueError, match="vectors"):
        field.with_scales([{-1: 2}])
    with pytest.raises(ValueError, match="vectors"):
        field.with_scales(factor={3: 2})
    for mode in ("mode", "vote_binomial", "vote_gaussian"):
        with pytest.raises(ValueError, match="field"):
            field.with_scales(2, mode=mode)


def test_the_key_none_never_reaches_the_vector_axis_of_a_field() -> None:
    field = _field_chain().to(MultiscaleField)

    built = field.with_scales([{None: 2}])

    values = np.asarray(built.to_singlescale(1).transformations[1].field)
    assert values.shape == (20, 20, 10, 3)
