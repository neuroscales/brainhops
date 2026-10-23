"""Tests for the frames of the values of images, and `reframe`.

The cases follow the worked example of
`spec/design/vector-and-tensor-images.md`: a field of vectors in RAS
(`X`), a transformation applied lazily (`U`) and a target geometry (`G`).
"""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops.datamodel._frames import explicit_frame, frame_axes
from brainhops.datamodel.axes import (
    Axis,
    ChannelAxis,
    TensorAxis,
    VectorAxis,
)
from brainhops.datamodel.enums import AxisType, ReframeMethod
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata import Channel, Metadata
from brainhops.datamodel.metadata import _vocabulary as vocabulary
from brainhops.datamodel.metadata._derive import Reframed
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Permutation,
    Projection,
    Sequence,
)

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


def _system(names: str, name: str, kind: str = "channel") -> CoordinateSystem:
    axes = [Axis(n, "space") for n in names]
    axes.append(Axis("c", kind))
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
    voxel = _system("ijk", "voxel", "vector")
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
    # The array system of the target describes the components as channels.
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


def test_an_axis_without_components_has_no_frame(
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


def test_reslicing_copies_the_component_axes_onto_the_target_axes(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage
) -> None:
    resliced = x(u).reslice(g)

    assert isinstance(resliced.transformation.input.axes[3], VectorAxis)
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


def test_indexing_part_of_a_component_axis_makes_it_a_channel_axis(
    x: SingleScaleImage,
) -> None:
    part = x[..., :2]

    assert isinstance(part.transformation.input.axes[3], ChannelAxis)
    assert part.frames[3] is None


def test_indexing_a_component_axis_in_full_keeps_its_type(
    x: SingleScaleImage,
) -> None:
    full = x[..., 0:3]

    assert isinstance(full.transformation.input.axes[3], VectorAxis)
    assert full.frames[3] is not None


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


def test_reframe_a_gradient(values: np.ndarray, a: Affine) -> None:
    gradient = VectorAxis("c", valence="covector")
    voxel = CoordinateSystem([*_system("ijk", "voxel").axes[:3], gradient])
    image = SingleScaleImage(values, [a.to(input=voxel)])

    out = image.reframe()

    assert np.allclose(np.asarray(out.data), values @ np.linalg.inv(_LA))


def test_reframe_from_a_field_uses_its_jacobian(values: np.ndarray) -> None:
    voxel = _system("ijk", "voxel", "vector")
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
    voxel = _system("ijk", "voxel", "vector")
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
    voxel = _system("ijk", "voxel", "vector")
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


# --- axes of the components, read from the structure of the frames ----


def _system_of(axes: tx.List[Axis], name: str) -> CoordinateSystem:
    return CoordinateSystem(axes, name=name)


def _matrix(linear: np.ndarray) -> np.ndarray:
    # A homogeneous matrix without its last row.
    size = linear.shape[0]
    matrix = np.zeros((size, size + 1))
    matrix[:, :size] = linear
    return matrix


def test_untyped_axes_are_read_from_the_frame() -> None:
    values = np.random.default_rng(0).random((*_SHAPE, 3))
    voxel = _system_of(
        [Axis("i"), Axis("j"), Axis("k"), Axis("c", "vector")], "voxel"
    )
    world = _system_of([Axis("x"), Axis("y"), Axis("z"), Axis("c")], "world")
    linear = np.eye(4)
    linear[:3, :3] = _LA
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})

    out = image.reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )


def _series(time: tx.Optional[str]) -> tx.Tuple[CoordinateSystem, ...]:
    # The axes of a series of vector fields, with a time axis of type `time`.
    def axes(names: str) -> tx.List[Axis]:
        out = [Axis(n, "space" if time else None) for n in names]
        out.append(Axis("t", time))
        return out

    voxel = _system_of([*axes("ijk"), Axis("c", "vector")], "voxel")
    world = _system_of([*axes("xyz"), Axis("c")], "world")
    return voxel, world


def test_a_time_axis_carried_along_is_left_out() -> None:
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    voxel, world = _series("time")
    linear = np.eye(5)
    linear[:3, :3] = _LA
    linear[3, 3] = 2.5  # the repetition time
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})

    out = image.reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )


def test_a_motion_that_changes_with_time_is_not_differentiated_in_time() -> (
    None
):
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    voxel, world = _series("time")
    linear = np.eye(5)
    linear[:3, :3] = _LA
    linear[:3, 3] = [0.5, 0.0, 0.0]  # the image drifts along x over time
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})

    out = image.reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )


def test_an_ambiguous_frame_is_refused() -> None:
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    # Without types, nothing tells the time axis from the others.
    voxel, world = _series(None)
    linear = np.eye(5)
    linear[:3, :3] = _LA
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})

    with pytest.raises(ValueError):
        image.reframe(None)


def test_vectors_in_a_subspace_follow_the_frame() -> None:
    # Four axes of the same type, and vectors of three components, whose
    # frame maps the first three axes onto its three axes.
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    voxel = _system_of(
        [Axis(n, "space") for n in "ijkl"] + [Axis("c", "vector")],
        "voxel",
    )
    frame_system = _system_of(
        [Axis(n, "space") for n in "xyz"] + [Axis("c")], "frame"
    )
    linear = np.zeros((4, 5))
    linear[:3, :3] = _LA
    linear[3, 4] = 1.0
    matrix = np.zeros((4, 6))
    matrix[:, :5] = linear
    frame = Affine(matrix=matrix, input=voxel, output=frame_system)
    identity = Affine(matrix=np.eye(6)[:-1], input=voxel, output=voxel)
    image = SingleScaleImage(values, [identity], frames={"c": frame})

    out = image.reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )


def test_frames_on_different_axes_cannot_be_compared() -> None:
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    voxel = _system_of(
        [Axis(n, "space") for n in "ijkl"] + [Axis("c", "vector")],
        "voxel",
    )
    frame_system = _system_of(
        [Axis(n, "space") for n in "xyz"] + [Axis("c")], "frame"
    )

    def frame(first: int) -> Affine:
        matrix = np.zeros((4, 6))
        matrix[:3, first : first + 3] = np.eye(3)
        matrix[3, 4] = 1.0
        return Affine(matrix=matrix, input=voxel, output=frame_system)

    identity = Affine(matrix=np.eye(6)[:-1], input=voxel, output=voxel)
    image = SingleScaleImage(values, [identity], frames={"c": frame(0)})

    with pytest.raises(ValueError):
        image.reframe(frame(1))


# --- frames that correspond exactly to the components -----------------


def test_a_given_frame_is_restricted_by_a_projection() -> None:
    values = np.random.default_rng(0).random((*_SHAPE, 2, 3))
    voxel, world = _series("time")
    linear = np.eye(5)
    linear[:3, :3] = _LA
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})

    slot = image.frames[4]

    assert isinstance(slot, Sequence)
    assert isinstance(slot.transformations[-1], Projection)
    assert [axis.name for axis in slot.output.axes] == ["x", "y", "z", "c"]
    # The restricted frame is stored, so that every read returns it.
    assert image.frames[4] is slot


def test_a_stored_frame_must_correspond_exactly() -> None:
    _, world = _series("time")
    voxel = _system_of(
        [*[Axis(n, "space") for n in "ijk"], Axis("t", "time")]
        + [Axis("c", "vector")],
        "voxel",
    )
    linear = np.eye(5)
    linear[:3, :3] = _LA
    a = Affine(matrix=_matrix(linear), input=voxel, output=world)

    # The type of the outputs is not consulted for a stored frame.
    with pytest.raises(ValueError):
        frame_axes(a, voxel.axes, 3)
    assert frame_axes(
        explicit_frame(a, voxel.axes, 3), voxel.axes, 3
    ).cols == [
        0,
        1,
        2,
    ]


def test_a_given_frame_with_too_few_axes_is_refused(
    values: np.ndarray,
) -> None:
    voxel = _system("ijk", "voxel", "vector")
    plane = CoordinateSystem([Axis("x", "space"), Axis("y", "space")])
    matrix = np.zeros((2, 5))
    matrix[0, 0] = matrix[1, 1] = 1.0
    frame = Affine(matrix=matrix, input=voxel, output=plane)
    identity = Affine(matrix=np.eye(5)[:-1], input=voxel, output=voxel)
    image = SingleScaleImage(values, [identity], frames={"c": frame})

    with pytest.raises(ValueError):
        _ = image.frames


def test_a_frame_may_end_with_a_projection(
    values: np.ndarray, a: Affine
) -> None:
    ras = _system("xyz", "RAS")
    projection = Projection(
        dropped=[3], input=ras, output=CoordinateSystem(ras.axes[:3])
    )
    frame = projection @ a
    image = SingleScaleImage(values, [a], frames={"c": frame})

    out = image.reframe(None)

    assert image.frames[3] is frame
    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(_LA), values)
    )


def test_a_frame_may_end_with_a_permutation(
    values: np.ndarray, a: Affine
) -> None:
    ras = _system("xyz", "RAS")
    zxy = CoordinateSystem([ras.axes[i] for i in (2, 0, 1, 3)])
    permutation = Permutation(permutation=[2, 0, 1, 3], input=ras, output=zxy)
    frame = permutation @ a
    image = SingleScaleImage(values, [a], frames={"c": frame})

    out = image.reframe()

    # The components follow z, x and y, and are re-expressed along x, y
    # and z.
    swap = np.eye(3)[[2, 0, 1]]
    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(swap), values)
    )


def test_a_field_followed_by_a_projection_uses_its_jacobian(
    values: np.ndarray,
) -> None:
    voxel = _system("ijk", "voxel", "vector")
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
    projection = Projection(
        dropped=[3], input=voxel, output=CoordinateSystem(voxel.axes[:3])
    )
    identity = Affine(matrix=np.eye(5)[:-1], input=voxel, output=voxel)
    image = SingleScaleImage(
        values, [identity], frames={"c": projection @ field}
    )

    out = image.reframe(None)

    assert np.allclose(
        np.asarray(out.data), _apply(np.linalg.inv(linear), values)
    )


# --- tensors and points -----------------------------------------------


def test_reframe_a_diffusion_tensor(a: Affine) -> None:
    full = np.random.default_rng(0).random((*_SHAPE, 3, 3))
    full = full + np.swapaxes(full, -1, -2)
    axis = TensorAxis("c", layout="symmetric")
    pairs = axis.entries(3)
    packed = np.stack([full[..., i, j] for i, j in pairs], -1)
    voxel = CoordinateSystem([*_system("ijk", "voxel").axes[:3], axis])
    image = SingleScaleImage(packed, [a.to(input=voxel)])

    out = np.asarray(image.reframe().data)

    expected = _LA @ full @ _LA.T
    assert np.allclose(
        out, np.stack([expected[..., i, j] for i, j in pairs], -1)
    )


@pytest.mark.parametrize("kind", ["point", "coordinate"])
def test_points_have_a_frame_but_are_not_reframed(
    values: np.ndarray, a: Affine, kind: str
) -> None:
    voxel = _system("ijk", "voxel", kind)
    a = a.to(input=voxel)
    image = SingleScaleImage(values, [a], frames={"c": a})

    assert image.frames[3] is a
    assert image.reframe() is image
    with pytest.raises(NotImplementedError):
        image.reframe(None)


def test_cropping_moves_the_voxel_frame_of_points(
    values: np.ndarray, a: Affine
) -> None:
    voxel = _system("ijk", "voxel", "point")
    image = SingleScaleImage(values, [a.to(input=voxel)])

    assert image[1:3].frames[3] is not None
    assert image[:, :, :].frames[3] is None


def test_reframe_expands_a_diagonal_tensor(a: Affine) -> None:
    diagonal = np.random.default_rng(0).random((*_SHAPE, 3))
    axis = TensorAxis("c", layout="diagonal")
    voxel = CoordinateSystem([*_system("ijk", "voxel").axes[:3], axis])
    image = SingleScaleImage(diagonal, [a.to(input=voxel)])

    out = image.reframe()

    new_axis = out.transformation.input.axes[3]
    assert new_axis.layout == "symmetric"
    assert out.shape == (*_SHAPE, 6)
    assert out.frames[3] is not None
    assert out.frames[3].input.axes[3] == new_axis
    expected = _LA @ (diagonal[..., None] * np.eye(3)) @ _LA.T
    pairs = new_axis.entries(3)
    assert np.allclose(
        np.asarray(out.data),
        np.stack([expected[..., i, j] for i, j in pairs], -1),
    )
    # The expanded image goes back to the array system in its new layout.
    back = np.asarray(out.reframe(None).data)
    assert np.allclose(
        back,
        np.stack([diagonal[..., i] * (i == j) for i, j in pairs], -1),
    )


def test_reframe_reads_an_ambiguous_size_from_the_frame() -> None:
    # 36 entries are a full matrix in six axes or a symmetric matrix in
    # eight, and the frame tells that there are six.
    names = "abcdef"
    scale = np.arange(1.0, 7.0)
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in names] + [TensorAxis("c")]
    )
    world = CoordinateSystem(
        [Axis(n.upper(), "space") for n in names] + [Axis("c")]
    )
    matrix = np.eye(8)[:-1]
    matrix[:6, :6] = np.diag(scale)
    a = Affine(matrix=matrix, input=voxel, output=world)
    full = np.random.default_rng(0).random((2, 1, 1, 1, 1, 1, 6, 6))
    image = SingleScaleImage(full.reshape(2, 1, 1, 1, 1, 1, 36), [a])

    out = np.asarray(image.reframe().data).reshape(full.shape)

    assert np.allclose(out, scale[:, None] * full * scale[None, :])


def test_reframe_expands_a_layout_that_leaves_the_last_axis_out(
    a: Affine,
) -> None:
    # The layout names the first two axes of a frame of three.
    values = np.random.default_rng(0).random((*_SHAPE, 2))
    axis = TensorAxis("c", layout=((0, 0), (1, 1)))
    voxel = CoordinateSystem([*_system("ijk", "voxel").axes[:3], axis])
    image = SingleScaleImage(values, [a.to(input=voxel)])

    out = image.reframe()

    diagonal = np.concatenate([values, np.zeros((*_SHAPE, 1))], -1)
    expected = _LA @ (diagonal[..., None] * np.eye(3)) @ _LA.T
    pairs = out.transformation.input.axes[3].entries(3)
    assert out.shape == (*_SHAPE, 6)
    assert np.allclose(
        np.asarray(out.data),
        np.stack([expected[..., i, j] for i, j in pairs], -1),
    )


@pytest.mark.parametrize(
    "method",
    ["rotation", ReframeMethod.rotation, "jacobian", ReframeMethod.jacobian],
)
def test_the_method_is_a_member_or_its_value(
    x: SingleScaleImage, u: Affine, g: SingleScaleImage, method: tx.Any
) -> None:
    resliced = x(u).reslice(g)

    out = resliced.reframe(None, method=method)
    expected = resliced.reframe(None, method=str(ReframeMethod(method).value))

    assert np.allclose(np.asarray(out.data), np.asarray(expected.data))


def test_an_unknown_method_is_refused(x: SingleScaleImage) -> None:
    with pytest.raises(ValueError):
        x.reframe(None, method="shear")


def test_reframe_into_an_equal_frame_returns_the_image(
    x: SingleScaleImage, a: Affine
) -> None:
    # Another object with the same matrix, as a frame decoded twice from
    # a header is, gives the same components.
    copy = Affine(matrix=np.asarray(a.matrix), input=a.input, output=a.output)

    assert x.reframe(copy) is x


# --- metadata ---------------------------------------------------------


def _with_metadata(image: SingleScaleImage, **fields: tx.Any) -> tx.Any:
    return SingleScaleImage(
        np.asarray(image.data),
        image.transformations,
        frames=image._frames,
        metadata=Metadata(**fields),
    )


def test_reframe_clears_the_fields_that_describe_the_values(
    x: SingleScaleImage, u: Affine
) -> None:
    image = _with_metadata(
        x,
        description="displacements",
        repetition_time=2.0,
        data_unit="mm",
        display_range=(-1.0, 1.0),
        dtype="int16",
        scale_slope=0.01,
        history=("registered",),
    )

    out = image(u).reframe().metadata

    assert out.data_unit is None
    assert out.display_range is None
    assert out.dtype is None
    assert out.scale_slope is None
    assert out.repetition_time == 2.0
    assert out.description == "displacements"
    assert out.history == ("registered", "reframe")


def test_reframe_keeps_the_entries_along_other_axes(
    x: SingleScaleImage, u: Affine
) -> None:
    # The descriptions of the channels run along channel axes, and the
    # components run along a vector axis, so they are not about them.
    image = _with_metadata(x, channels=(Channel(name="a"),), data_unit="mm")

    out = image(u).reframe().metadata

    assert out.channels == (Channel(name="a"),)
    assert out.data_unit is None


def test_a_reframed_axis_clears_the_entries_along_it() -> None:
    metadata = Metadata(
        channels=(Channel(name="x"), Channel(name="y")), bvalues=(0.0, 1.0)
    )

    out = metadata.derive(Reframed(("channel",)))

    assert out.channels is None
    assert out.bvalues == (0.0, 1.0)


def test_reframe_clears_the_entries_along_an_expanded_tensor_axis(
    a: Affine, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No field of the vocabulary runs along a tensor axis yet, so the
    # descriptions of the channels are made to run along one.
    monkeypatch.setitem(vocabulary.ALONG, "channels", AxisType.tensor)
    diagonal = np.random.default_rng(0).random((*_SHAPE, 3))
    axis = TensorAxis("c", layout="diagonal")
    voxel = CoordinateSystem([*_system("ijk", "voxel").axes[:3], axis])
    channels = tuple(Channel(name=n) for n in ("xx", "yy", "zz"))
    image = SingleScaleImage(
        diagonal, [a.to(input=voxel)], metadata=Metadata(channels=channels)
    )

    # Indexing the tensor axis in full keeps the entries along it.
    assert image[..., :].metadata.channels == channels

    out = image.reframe()

    # The diagonal layout becomes symmetric, so the axis has six entries
    # and the three old descriptions no longer match it.
    assert out.shape == (*_SHAPE, 6)
    assert out.metadata.channels is None


def test_indexing_and_reslicing_keep_the_fields_of_the_values(
    x: SingleScaleImage,
) -> None:
    image = _with_metadata(x, data_unit="mm", display_range=(0.0, 1.0))

    for derived in (image[1:3], image.reslice()):
        assert derived.metadata.data_unit is not None
        assert derived.metadata.display_range == (0.0, 1.0)
