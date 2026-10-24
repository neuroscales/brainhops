"""Tests for the frames of the vectors and tensors of OME-Zarr images.

An OME-Zarr image stores the components of a vector, a tensor or a point
in the system onto which its levels map, called the intrinsic system,
in the order in which the store lists its axes, as RFC-3 orders the
components of a vector. A displacement field stores its vectors in the
same frame, and a coordinate field stores positions in a system that it
does not name. The reader records these frames without converting the
values, and the writer re-expresses the values of each level in the
frame of that level.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx
from bagof.magic import replace

abczarr = pytest.importorskip("abczarr")

from brainhops.datamodel._frames import _structure  # noqa: E402
from brainhops.datamodel.axes import Axis, TensorAxis, VectorAxis  # noqa: E402
from brainhops.datamodel.enums import Valence  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    Scaling,
)
from brainhops.errors import LossyConversionError  # noqa: E402
from brainhops.io import load, save  # noqa: E402
from brainhops.io.images.zarr import OmeZarrImage  # noqa: E402
from brainhops.io.transformations.zarr import OmeZarrField  # noqa: E402
from brainhops.io.transformations.zarr._axes import (  # noqa: E402
    _to_axis,
    axis_to_json,
)

# The OME-Zarr readers need zarr-python 3, which needs Python 3.11.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="OME-Zarr metadata needs zarr-python 3, which needs Python 3.11",
)

_SCALE = np.array([2.0, 3.0, 4.0])
_SHAPE = (4, 5, 6)


def _systems(
    kind: str = "vector", **options: tx.Any
) -> tx.Tuple[CoordinateSystem, CoordinateSystem]:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"] + [Axis("c", kind, **options)],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space", unit="millimeter") for n in "xyz"]
        + [Axis("c", kind, **options)],
        name="world",
    )
    return voxel, world


def _scaling(
    scale: tx.Sequence[float] = tuple(_SCALE),
    kind: str = "vector",
    **options: tx.Any,
) -> Scaling:
    voxel, world = _systems(kind, **options)
    return Scaling(scale=[*scale, 1.0], input=voxel, output=world)


def _values(count: int = 3, shape: tx.Sequence[int] = _SHAPE) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((*shape, count)).astype("f8")


def _write(
    tmp_path: Path, *levels: SingleScaleImage, **options: tx.Any
) -> Path:
    path = tmp_path / "image.ome.zarr"
    OmeZarrImage(images=list(levels)).save(path, **options)
    return path


def _ome(path: Path) -> tx.Dict[str, tx.Any]:
    attributes = json.loads((path / "zarr.json").read_text())["attributes"]
    return attributes["ome"]["multiscales"][0]


def _stored(path: Path, level: int = 0) -> np.ndarray:
    # The stored array has the axes (c, z, y, x), so it is transposed back
    # to (x, y, z, c) without reordering the components.
    return np.asarray(abczarr.open_array(str(path / str(level)))[...]).T


def _linear(frame: tx.Any, ndim: int = 4) -> np.ndarray:
    return _structure(frame, ndim).outputs


# ---- vectors ---------------------------------------------------------


def test_vectors_in_voxels_are_written_in_the_intrinsic_system(
    tmp_path: Path,
) -> None:
    values = _values()
    path = _write(tmp_path, SingleScaleImage(values, [_scaling()]))

    # The components follow the spatial axes in stored order: z, y, x.
    assert np.allclose(_stored(path), (values * _SCALE)[..., ::-1])
    axes = _ome(path)["coordinateSystems"][0]["axes"]
    assert axes[0] == {"name": "c", "type": "vector"}
    assert axes[1]["unit"] == "millimeter"


def test_vectors_in_the_intrinsic_system_are_written_as_they_are(
    tmp_path: Path,
) -> None:
    values = _values()
    scaling = _scaling()
    image = SingleScaleImage(values, [scaling], frames={"c": scaling})

    path = _write(tmp_path, image)

    assert np.allclose(_stored(path), values[..., ::-1])


def test_the_reader_records_the_frame_of_the_store(tmp_path: Path) -> None:
    values = _values()
    path = _write(tmp_path, SingleScaleImage(values, [_scaling()]))

    level = load(path).images[0]

    frame = level.frames[3]
    assert level.frames[3] is frame
    assert frame.output.name == "physical"
    # The frame is the scaling, followed by the permutation into the
    # stored order, which reverses the spatial axes.
    linear = _linear(frame)
    assert np.allclose(linear[1:, :3], np.diag(_SCALE)[::-1])
    # The values are not converted when they are read.
    assert np.allclose(np.asarray(level.data), (values * _SCALE)[..., ::-1])
    assert np.allclose(np.asarray(level.reframe(None).data), values)
    assert level.reframe(frame) is level


def test_an_unchanged_image_is_written_as_it_is(tmp_path: Path) -> None:
    values = _values()
    path = _write(tmp_path, SingleScaleImage(values, [_scaling()]))
    stored = _stored(path)

    again = tmp_path / "again.ome.zarr"
    load(path).save(again)

    assert np.allclose(_stored(again), stored)
    assert np.allclose(
        np.asarray(load(again).images[0].reframe(None).data), values
    )


def test_vectors_follow_a_new_geometry(tmp_path: Path) -> None:
    values = _values()
    path = _write(tmp_path, SingleScaleImage(values, [_scaling()]))
    level = load(path).images[0]
    # The level keeps its values and its frame, but is placed with
    # another scale, so the vectors are re-expressed in its new frame.
    moved = replace(level, transformations=[_scaling([1.0, 1.0, 1.0])])

    again = _write(tmp_path / "moved", moved)

    assert np.allclose(_stored(again), values[..., ::-1])


def test_covectors_keep_their_valence(tmp_path: Path) -> None:
    values = _values()
    image = SingleScaleImage(values, [_scaling(valence="covector")])

    path = _write(tmp_path, image)

    axis = _ome(path)["coordinateSystems"][0]["axes"][0]
    assert axis == {"name": "c", "type": "vector", "valence": "covector"}
    assert np.allclose(_stored(path), (values / _SCALE)[..., ::-1])
    level = load(path).images[0]
    assert level._array_axes()[3].valence == Valence.covector
    assert np.allclose(np.asarray(level.reframe(None).data), values)


def test_a_vector_in_a_foreign_frame_is_refused(tmp_path: Path) -> None:
    foreign = CoordinateSystem([Axis(n, "space") for n in "abc"], name="mni")
    image = SingleScaleImage(_values(), [_scaling()], frames={"c": foreign})

    with pytest.raises(LossyConversionError, match="frame"):
        _write(tmp_path, image)
    assert not (tmp_path / "image.ome.zarr" / "0").exists()

    path = _write(tmp_path, image, lossy=True)
    assert np.allclose(_stored(path), _values())


def test_save_passes_lossy_to_the_writer(tmp_path: Path) -> None:
    foreign = CoordinateSystem([Axis(n, "space") for n in "abc"], name="mni")
    image = SingleScaleImage(_values(), [_scaling()], frames={"c": foreign})
    pyramid = OmeZarrImage(images=[image])
    path = tmp_path / "image.ome.zarr"

    with pytest.raises(LossyConversionError):
        save(pyramid, path)
    save(pyramid, path, lossy=True)

    assert np.allclose(_stored(path), _values())


def test_vectors_ignore_a_time_axis(tmp_path: Path) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"]
        + [Axis("t", "time"), Axis("c", "vector")],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"]
        + [Axis("t", "time"), Axis("c", "vector")],
        name="world",
    )
    scaling = Scaling(scale=[*_SCALE, 5.0, 1.0], input=voxel, output=world)
    values = _values(shape=(*_SHAPE, 2))

    path = _write(tmp_path, SingleScaleImage(values, [scaling]))

    # The stored axes are (t, c, z, y, x).
    stored = np.asarray(abczarr.open_array(str(path / "0"))[...])
    expected = np.moveaxis((values * _SCALE)[..., ::-1], (3, 4), (0, 1))
    assert np.allclose(stored, expected.transpose(0, 1, 4, 3, 2))
    level = load(path).images[0]
    assert np.allclose(np.asarray(level.reframe(None).data), values)


# ---- tensors ---------------------------------------------------------


def _symmetric(values: np.ndarray) -> np.ndarray:
    # The symmetric layout stores 00 01 02 11 12 22.
    rows, cols = np.triu_indices(3)
    out = np.zeros((*values.shape[:-1], 3, 3))
    out[..., rows, cols] = values
    out[..., cols, rows] = values
    return out


def test_a_tensor_is_written_with_its_layout(tmp_path: Path) -> None:
    values = _values(6)
    scaling = _scaling(kind="tensor", layout="symmetric")

    path = _write(tmp_path, SingleScaleImage(values, [scaling]))

    axis = _ome(path)["coordinateSystems"][0]["axes"][0]
    assert axis == {"name": "c", "type": "tensor", "layout": "symmetric"}
    # In the stored order z, y, x, the tensor is S T S with the axes
    # reversed, so the layout lists Dzz Dzy Dzx Dyy Dyx Dxx, as in RFC-3.
    tensor = _symmetric(values) * np.outer(_SCALE, _SCALE)
    reversed_ = tensor[..., ::-1, ::-1]
    rows, cols = np.triu_indices(3)
    assert np.allclose(_stored(path), reversed_[..., rows, cols])
    level = load(path).images[0]
    assert level._array_axes()[3].layout == "symmetric"
    assert np.allclose(np.asarray(level.reframe(None).data), values)


def test_a_tensor_of_mixed_valence_keeps_it(tmp_path: Path) -> None:
    valence = ("vector", "covector")
    scaling = _scaling(kind="tensor", valence=valence, layout="C")

    path = _write(tmp_path, SingleScaleImage(_values(9), [scaling]))

    axis = _ome(path)["coordinateSystems"][0]["axes"][0]
    assert axis["valence"] == "vector,covector"
    assert axis["layout"] == "C"
    level = load(path).images[0]
    assert level._array_axes()[3].valence == (
        Valence.vector,
        Valence.covector,
    )
    assert np.allclose(np.asarray(level.reframe(None).data), _values(9))


def test_an_explicit_layout_is_refused(tmp_path: Path) -> None:
    layout = ((0, 0), (1, 1), (2, 2), (0, 1))
    scaling = _scaling(kind="tensor", layout=layout)
    image = SingleScaleImage(_values(4), [scaling])

    with pytest.raises(LossyConversionError, match="explicit layout"):
        _write(tmp_path, image)


# ---- points ----------------------------------------------------------


def test_points_are_kept_only_in_the_frame_of_the_store(
    tmp_path: Path,
) -> None:
    image = SingleScaleImage(_values(), [_scaling(kind="point")])

    # Points in voxels cannot be re-expressed without the inverse of a map.
    with pytest.raises(LossyConversionError):
        _write(tmp_path, image)
    path = _write(tmp_path, image, lossy=True)

    pyramid = load(path)
    assert pyramid.images[0].frames[3] is not None
    again = tmp_path / "again.ome.zarr"
    pyramid.save(again)
    assert np.allclose(_stored(again), _values())


# ---- fields ----------------------------------------------------------


def test_a_displacement_field_read_as_an_image_is_in_its_system(
    tmp_path: Path,
) -> None:
    values = _values()
    image = SingleScaleImage(values, [_scaling(kind="displacement")])
    path = _write(tmp_path, image)

    axis = _ome(path)["coordinateSystems"][0]["axes"][0]
    assert axis == {"name": "c", "type": "displacement", "discrete": True}
    assert isinstance(load(path), OmeZarrField)

    level = OmeZarrImage.from_store(path).images[0]
    frame = level.frames[3]
    assert np.allclose(_linear(frame)[1:, :3], np.diag(_SCALE)[::-1])
    voxels = np.asarray(level.reframe(None).data)
    assert np.allclose(voxels, values)
    # The field reader converts the vectors to voxels in the same way. It
    # keeps the stored order of the grid and of the components.
    field = OmeZarrField.from_store(path).scales[0].transformations[1]
    expected = voxels.transpose(2, 1, 0, 3)[..., ::-1]
    assert np.allclose(np.asarray(field.field), expected)


def test_a_coordinate_field_has_a_foreign_frame(tmp_path: Path) -> None:
    coarse = Scaling(
        scale=[*(2 * _SCALE), 1.0],
        input=_systems("coordinate")[0],
        output=_systems("coordinate")[1],
    )
    fine = SingleScaleImage(_values(), [_scaling(kind="coordinate")])
    small = SingleScaleImage(_values(shape=(2, 3, 3)), [coarse])

    with pytest.raises(LossyConversionError):
        _write(tmp_path, fine, small)
    path = _write(tmp_path, fine, small, lossy=True)

    pyramid = OmeZarrImage.from_store(path)
    frames = [level.frames[3] for level in pyramid.images]
    assert isinstance(frames[0], CoordinateSystem)
    assert frames[0] is frames[1]
    # The coordinates read from the store are written as they are.
    again = tmp_path / "again.ome.zarr"
    pyramid.save(again)
    assert np.allclose(_stored(again), _values())


# ---- multiscale ------------------------------------------------------


def _pyramid_levels(
    frames: bool,
) -> tx.Tuple[SingleScaleImage, SingleScaleImage]:
    fine_map = _scaling()
    coarse_map = _scaling(tuple(2 * _SCALE))
    fine = _values()
    coarse = _values(shape=(2, 3, 3))
    return (
        SingleScaleImage(
            fine, [fine_map], frames={"c": fine_map} if frames else ()
        ),
        SingleScaleImage(
            coarse, [coarse_map], frames={"c": coarse_map} if frames else ()
        ),
    )


def test_voxel_vectors_of_each_level_use_its_own_scale(
    tmp_path: Path,
) -> None:
    fine, coarse = _pyramid_levels(frames=False)

    path = _write(tmp_path, fine, coarse)

    assert np.allclose(_stored(path, 0), (fine.data * _SCALE)[..., ::-1])
    assert np.allclose(_stored(path, 1), (coarse.data * 2 * _SCALE)[..., ::-1])
    pyramid = load(path)
    for level, original in zip(pyramid.images, (fine, coarse)):
        assert np.allclose(np.asarray(level.reframe(None).data), original.data)


def test_physical_vectors_are_the_same_at_every_level(
    tmp_path: Path,
) -> None:
    fine, coarse = _pyramid_levels(frames=True)

    path = _write(tmp_path, fine, coarse)

    assert np.allclose(_stored(path, 0), fine.data[..., ::-1])
    assert np.allclose(_stored(path, 1), coarse.data[..., ::-1])
    levels = load(path).images
    # The frames of the levels differ by the scale of their voxels and
    # reach the same intrinsic system.
    assert levels[0].frames[3] is not levels[1].frames[3]
    assert levels[0].frames[3].output == levels[1].frames[3].output
    assert np.allclose(
        _linear(levels[1].frames[3])[1:, :3],
        2 * _linear(levels[0].frames[3])[1:, :3],
    )


def test_a_shared_flip_reverses_the_stored_vectors(tmp_path: Path) -> None:
    voxel, world = _systems()
    matrix = np.diag([-2.0, 3.0, 4.0, 1.0, 1.0])[:-1]
    flipped = Affine(matrix=matrix, input=voxel, output=world)
    values = _values()
    image = SingleScaleImage(values, [flipped], frames={"c": flipped})

    path = _write(tmp_path, image)

    # The flip becomes a placement of the pyramid, so the intrinsic
    # system is not flipped, and the x component changes its sign.
    expected = values * [-1.0, 1.0, 1.0]
    assert np.allclose(_stored(path), expected[..., ::-1])
    pyramid = load(path)
    level = pyramid.images[0]
    assert np.allclose(
        np.asarray(level.reframe(None).data), values * [-0.5, 1 / 3, 0.25]
    )


def test_a_rotated_level_stores_vectors_in_voxels(tmp_path: Path) -> None:
    voxel, world = _systems()
    theta = 0.3
    matrix = np.eye(5)[:-1]
    matrix[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    rotation = Affine(matrix=matrix, input=voxel, output=world)
    values = _values()
    image = SingleScaleImage(values, [rotation], frames={"c": rotation})

    path = _write(tmp_path, image)

    # The rotation becomes the placement of the pyramid, so the level
    # maps its voxels onto the intrinsic system without a change.
    voxels = values @ np.linalg.inv(matrix[:3, :3]).T
    assert np.allclose(_stored(path), voxels[..., ::-1])


# ---- axes ------------------------------------------------------------


def test_axes_round_trip_their_custom_fields() -> None:
    axes = [
        VectorAxis(name="g", valence="covector"),
        TensorAxis(name="d", layout="symmetric-lower"),
        Axis("c", "displacement"),
    ]

    back = [_to_axis(axis_to_json(axis)) for axis in axes]

    assert back[0].valence == Valence.covector
    assert back[1].layout == "symmetric-lower"
    assert back[2].discrete is True


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "v", "type": "vector", "valence": "tensor"},
        {"name": "v", "type": "vector", "valence": "vector,vector"},
        {"name": "d", "type": "tensor", "layout": "upper"},
    ],
)
def test_invalid_custom_fields_are_refused(entry: tx.Dict[str, str]) -> None:
    with pytest.raises(ValueError):
        _to_axis(entry)


def test_a_component_axis_written_as_a_channel_is_refused(
    tmp_path: Path,
) -> None:
    image = SingleScaleImage(_values(), [_scaling()])
    axes = [Axis(n, "space") for n in "xyz"] + [Axis("c", "channel")]
    pyramid = OmeZarrImage(images=[image], axes=axes)
    path = tmp_path / "image.ome.zarr"

    with pytest.raises(LossyConversionError, match="'channel'"):
        pyramid.save(path)
    pyramid.save(path, lossy=True)

    # A channel states no frame, so the voxel vectors are written as they
    # are, without a change of frame or of order.
    assert np.allclose(_stored(path), _values())


def test_every_level_is_checked_for_its_axis_types(tmp_path: Path) -> None:
    channel = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"] + [Axis("c", "channel")],
        name="voxel",
    )
    plain = Scaling(scale=[*(2 * _SCALE), 1.0], input=channel, output=channel)
    fine = SingleScaleImage(_values(), [_scaling()])
    coarse = SingleScaleImage(_values(shape=(2, 3, 3)), [plain])
    axes = [Axis(n, "space") for n in "xyz"] + [Axis("c", "channel")]
    pyramid = OmeZarrImage(images=[coarse, fine], axes=axes)

    with pytest.raises(LossyConversionError, match="'channel'"):
        pyramid.save(tmp_path / "image.ome.zarr")
