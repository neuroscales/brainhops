"""Tests for the frames of the values of images, and `reframe`.

The cases follow the worked example of
`spec/design/vector-and-tensor-images.md`: a field of vectors in RAS
(`X`), a transformation applied lazily (`U`) and a target geometry (`G`).
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel.axes import Axis
from brainhops.datamodel.content import Content, Covector, Vector
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, DisplacementField

_LA = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
_THETA = 0.3
_LU = np.array(
    [
        [np.cos(_THETA), -np.sin(_THETA), 0.0],
        [np.sin(_THETA), np.cos(_THETA), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_LB = np.diag([1.5, 1.5, 2.0])
_SHAPE = (4, 5, 6)


def _system(
    names: str, name: str, content: tx.Optional[Content] = None
) -> CoordinateSystem:
    axes = [Axis(n, "space") for n in names]
    axes.append(Axis("c", "channel", content=content))
    return CoordinateSystem(axes, name=name)


def _affine(
    linear: np.ndarray,
    shift: tx.Sequence[float],
    input: CoordinateSystem,
    output: CoordinateSystem,
) -> Affine:
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = linear
    matrix[:3, -1] = shift
    return Affine(matrix=matrix, input=input, output=output)


def _apply(matrix: np.ndarray, values: tx.Any) -> np.ndarray:
    return np.einsum("...ij,...j->...i", matrix, np.asarray(values))


@pytest.fixture
def values() -> np.ndarray:
    return np.random.default_rng(0).random((*_SHAPE, 3))


@pytest.fixture
def a() -> Affine:
    voxel = _system("ijk", "voxel", Vector())
    return _affine(_LA, [1.0, 2.0, 3.0], voxel, _system("xyz", "RAS"))


@pytest.fixture
def x(values: np.ndarray, a: Affine) -> SingleScaleImage:
    # The vectors are stored in RAS, as in a NIfTI displacement field.
    return SingleScaleImage(values, [a], frames={"c": a})


@pytest.fixture
def u() -> Affine:
    ras = _system("xyz", "RAS")
    return _affine(_LU, [0.5, 0.0, 0.0], ras, ras)


@pytest.fixture
def g() -> SingleScaleImage:
    # The array system of the target describes its channels without content.
    b = _affine(
        _LB, [-1.0, -1.0, 0.0], _system("ijk", "voxel"), _system("xyz", "RAS")
    )
    return SingleScaleImage(np.zeros((*_SHAPE, 3)), [b])


# --- the frames field -------------------------------------------------


def test_frames_have_one_slot_per_axis(x: SingleScaleImage, a: Affine) -> None:
    assert x.frames == (None, None, None, a)


def test_frames_default_to_the_array_system(
    values: np.ndarray, a: Affine
) -> None:
    image = SingleScaleImage(values, [a])

    assert image.frames == (None,) * 4


def test_an_axis_without_content_has_no_frame(
    values: np.ndarray, a: Affine
) -> None:
    image = SingleScaleImage(values, [a], frames={0: a})

    with pytest.raises(ValueError):
        _ = image.frames


def test_frames_need_one_slot_per_axis(values: np.ndarray, a: Affine) -> None:
    image = SingleScaleImage(values, [a], frames=(None, a))

    with pytest.raises(ValueError):
        _ = image.frames


def test_frames_refuse_an_unknown_axis_name(
    values: np.ndarray, a: Affine
) -> None:
    image = SingleScaleImage(values, [a], frames={"nope": a})

    with pytest.raises(KeyError):
        _ = image.frames


# --- operations that keep the values ----------------------------------


def test_a_lazy_transformation_keeps_the_frames(
    x: SingleScaleImage, u: Affine, a: Affine
) -> None:
    assert x(u).frames[3] is a


def test_reslicing_onto_the_own_geometry_keeps_the_values(
    x: SingleScaleImage, u: Affine, values: np.ndarray
) -> None:
    resliced = x(u).reslice()

    assert np.allclose(np.asarray(resliced.data), values)
    # The world of the file is no longer the output of a transformation,
    # but the frame still reaches it.
    assert np.allclose(
        np.asarray(resliced.reframe().data), _apply(np.linalg.inv(_LU), values)
    )


def test_reslicing_copies_the_content_onto_the_target_axes(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    resliced = x(u).reslice(g)

    assert resliced.transformation.input.axes[3].content == Vector()
    assert resliced.frames[3] is not None


def test_indexing_keeps_a_view_and_composes_the_frame(
    x: SingleScaleImage, values: np.ndarray
) -> None:
    flipped = x[::-1]

    assert np.shares_memory(np.asarray(flipped.data), values)
    assert np.allclose(
        np.asarray(flipped.reframe().data),
        np.asarray(x.reframe()[::-1].data),
    )


def test_cropping_keeps_the_array_frame(values: np.ndarray, a: Affine) -> None:
    image = SingleScaleImage(values, [a])

    assert image[1:3].frames == (None,) * 4


def test_flipping_composes_the_array_frame(
    values: np.ndarray, a: Affine
) -> None:
    image = SingleScaleImage(values, [a])
    flipped = image[::-1]

    assert flipped.frames[3] is not None
    # In the voxels of the flipped image, the first component changes sign.
    assert np.allclose(
        np.asarray(flipped.reframe(None).data)[..., 0], -values[::-1, ..., 0]
    )


def test_indexing_part_of_a_content_axis_clears_its_content(
    x: SingleScaleImage,
) -> None:
    part = x[..., :2]

    assert part.transformation.input.axes[3].content is None
    assert part.frames[3] is None


# --- reframe ----------------------------------------------------------


def test_reframe_into_the_preferred_world(
    x: SingleScaleImage, u: Affine, values: np.ndarray
) -> None:
    out = x(u).reframe()

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LU), values)
    )
    assert out.frames[3] is out.transformation


def test_reframe_into_the_array_system(
    x: SingleScaleImage, u: Affine, values: np.ndarray
) -> None:
    out = x(u).reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )
    assert out.frames[3] is None


def test_reframe_after_reslicing(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    resliced = x(u).reslice(g)
    sampled = np.asarray(resliced.data)

    world = resliced.reframe()
    voxel = resliced.reframe(None)

    assert np.allclose(
        np.asarray(world.data), _apply(np.linalg.inv(_LU), sampled)
    )
    expected = np.linalg.inv(_LB) @ np.linalg.inv(_LU)
    assert np.allclose(np.asarray(voxel.data), _apply(expected, sampled))


def test_reslice_can_reframe(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    sampled = np.asarray(x(u).reslice(g).data)

    out = x(u).reslice(g, reframe=True)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LU), sampled)
    )


def test_rotation_keeps_the_length_of_vectors(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    resliced = x(u).reslice(g)

    out = resliced.reframe(None, method="rotation")

    assert np.allclose(
        np.linalg.norm(np.asarray(out.data), axis=-1),
        np.linalg.norm(np.asarray(resliced.data), axis=-1),
    )


def test_reframe_into_the_same_frame_returns_the_image(
    x: SingleScaleImage,
) -> None:
    assert x.reframe() is x


def test_reframe_by_name_selects_the_most_preferred(
    x: SingleScaleImage, u: Affine, values: np.ndarray
) -> None:
    # Both transformations end in a system named "RAS".
    out = x(u).reframe("RAS")

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LU), values)
    )


def test_reframe_by_unknown_name_raises(x: SingleScaleImage) -> None:
    with pytest.raises(KeyError):
        x.reframe("MNI")


def test_reframe_skips_a_foreign_frame(values: np.ndarray, a: Affine) -> None:
    foreign = _system("xyz", "moving")
    image = SingleScaleImage(values, [a], frames={"c": foreign})

    assert image.reframe(None) is image
    with pytest.raises(ValueError):
        image.reframe({"c": None})


def test_reframe_a_covector(values: np.ndarray, a: Affine) -> None:
    voxel = _system("ijk", "voxel", Covector())
    image = SingleScaleImage(values, [a.to(input=voxel)])

    out = image.reframe()

    assert np.allclose(np.asarray(out.data), values @ np.linalg.inv(_LA))


def test_reframe_from_a_field_uses_its_jacobian(values: np.ndarray) -> None:
    voxel = _system("ijk", "voxel", Vector())
    grid = np.stack(
        np.meshgrid(
            *[np.arange(s, dtype=float) for s in _SHAPE], indexing="ij"
        ),
        -1,
    )
    linear = np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.1]])
    displacements = np.zeros((*_SHAPE, 1, 4))
    displacements[..., 0, :3] = grid @ (linear - np.eye(3)).T
    field = DisplacementField(data=displacements, input=voxel, output=voxel)
    identity = Affine(matrix=np.eye(5)[:-1], input=voxel, output=voxel)
    image = SingleScaleImage(values, [identity], frames={"c": field})

    out = image.reframe(None)

    # Finite differences are exact for a linear field.
    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(linear), values)
    )


def test_multiscale_reslice_can_reframe(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    pyramid = MultiScaleImage(images=[x(u)])
    sampled = np.asarray(x(u).reslice(g).data)

    out = pyramid.reslice(g, reframe=True)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LU), sampled)
    )


def _sine_field(shape: tx.Tuple[int, ...], voxel: CoordinateSystem) -> tx.Any:
    # A displacement along the first axis that varies along the second.
    grid = np.stack(
        np.meshgrid(
            *[np.arange(s, dtype=float) for s in shape], indexing="ij"
        ),
        -1,
    )
    displacements = np.zeros((*shape, 1, 4))
    displacements[..., 0, 0] = 0.3 * np.sin(grid[..., 1] / 2)
    return DisplacementField(data=displacements, input=voxel, output=voxel)


def test_reframe_uses_one_sided_differences_at_the_edges(
    values: np.ndarray,
) -> None:
    voxel = _system("ijk", "voxel", Vector())
    identity = Affine(matrix=np.eye(5)[:-1], input=voxel, output=voxel)
    field = _sine_field(_SHAPE, voxel)
    image = SingleScaleImage(values, [identity], frames={"c": field})

    out = np.asarray(image.reframe(None).data)

    jacobian = np.eye(3)
    jacobian[0, 1] = 0.3 * (np.sin(1 / 2) - np.sin(0))
    assert np.allclose(
        out[:, 0], _apply(np.linalg.inv(jacobian), values[:, 0])
    )


def test_a_field_is_constant_along_an_axis_of_one_voxel() -> None:
    shape = (4, 5, 1)
    values = np.random.default_rng(0).random((*shape, 3))
    voxel = _system("ijk", "voxel", Vector())
    identity = Affine(matrix=np.eye(5)[:-1], input=voxel, output=voxel)
    scaling = _affine(np.diag([1.0, 1.0, 2.0]), [0.0, 0.0, 0.0], voxel, voxel)
    frame = scaling @ _sine_field(shape, voxel)
    image = SingleScaleImage(values, [identity], frames={"c": frame})

    out = np.asarray(image.reframe(None).data)

    # Inside the grid, the field contributes a shear along the second
    # axis, and along the third axis only the scaling remains.
    jacobian = np.eye(3)
    jacobian[0, 1] = 0.3 * (np.sin(2 / 2) - np.sin(0 / 2)) / 2
    jacobian = np.diag([1.0, 1.0, 2.0]) @ jacobian
    assert np.allclose(
        out[:, 1], _apply(np.linalg.inv(jacobian), values[:, 1])
    )
