"""Tests for the vectors and tensors of NRRD images and their frames.

A NRRD header says with `kinds` which axes hold the components of a
vector, a point or a matrix, and with `measurement frame` in which
coordinate system the components are measured. The columns of the
measurement frame are the basis vectors of that system in the world
space of the header. The reader records the frame without converting
the values, and the writer records the frame of the values as a
measurement frame, or re-expresses the values when it cannot.

Headers are written by hand from the specification
(https://teem.sourceforge.net/nrrd/format.html).
"""

import pathlib

import numpy as np
import pytest
import typing_extensions as tx

import brainhops.io as io
from brainhops.datamodel._transformations.compute.multiscale import (
    _as_affine,
)
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    DisplacementField,
    Scaling,
)
from brainhops.errors import WriterError
from brainhops.io.images.nrrd import NrrdImage
from brainhops.io.images.nrrd._image import _planned_kinds

_SHAPE = (2, 3, 4)
_DIRECTIONS = np.array([[0.0, -2.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 3.0]])
_ORIGIN = np.array([1.0, 2.0, 3.0])
_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _vectors(columns: tx.Sequence[tx.Sequence[float]]) -> str:
    return " ".join(
        "(" + ",".join(repr(float(v)) for v in c) + ")" for c in columns
    )


def _write(
    path: pathlib.Path,
    kind: str,
    size: int,
    *,
    space: str = "right-anterior-superior",
    frame: tx.Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """Write a NRRD file whose first axis has `kind`, and return its values.

    The values are returned in the order of the image, with the component
    axis last.
    """
    values = np.random.default_rng(seed).random((size, *_SHAPE)).astype("f4")
    lines = [
        "NRRD0004",
        "type: float",
        "dimension: 4",
        f"space: {space}",
        f"sizes: {size} {' '.join(str(n) for n in _SHAPE)}",
        f"space directions: none {_vectors(_DIRECTIONS.T)}",
        f"kinds: {kind} domain domain domain",
        "endian: little",
        "encoding: raw",
        f"space origin: {_vectors([_ORIGIN])}",
    ]
    if frame is not None:
        lines.append(f"measurement frame: {_vectors(np.asarray(frame).T)}")
    header = ("\n".join(lines) + "\n\n").encode("ascii")
    path.write_bytes(header + values.astype("<f4").ravel(order="F").tobytes())
    return np.moveaxis(values, 0, -1)


def _systems(
    kind: str = "vector", size: int = 3, **options: tx.Any
) -> tx.Tuple[CoordinateSystem, CoordinateSystem]:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", kind, **options)],
        name="voxel",
    )
    world = CoordinateSystem(
        [
            Axis(n, "space", unit="mm", orientation=o)
            for n, o in zip(
                "xyz",
                (
                    "left-to-right",
                    "posterior-to-anterior",
                    "inferior-to-superior",
                ),
            )
        ]
        + [Axis("c", kind, **options)],
        name="RAS",
    )
    return voxel, world


def _affine(voxel: CoordinateSystem, world: CoordinateSystem) -> Affine:
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = _DIRECTIONS
    matrix[:3, -1] = _ORIGIN
    return Affine(matrix=matrix, input=voxel, output=world)


def _record(path: pathlib.Path) -> tx.Any:
    return NrrdImage.load(path).metadata.raw


def _linear(frame: tx.Any) -> np.ndarray:
    return np.asarray(_as_affine(frame).matrix)[:3, :3]


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,size,expected",
    [
        ("vector", 3, ("vector", "vector")),
        ("3-vector", 3, ("vector", "vector")),
        ("covariant-vector", 3, ("vector", "covector")),
        ("normal", 3, ("vector", "covector")),
        ("3-gradient", 3, ("vector", "covector")),
        ("3-normal", 3, ("vector", "covector")),
        ("point", 3, ("point", None)),
        ("3D-symmetric-matrix", 6, ("tensor", "symmetric")),
        ("3D-matrix", 9, ("tensor", "C")),
    ],
)
def test_each_kind_is_read_as_its_component_axis(
    tmp_path: pathlib.Path, kind: str, size: int, expected: tx.Any
) -> None:
    values = _write(tmp_path / "a.nrrd", kind, size)

    image = NrrdImage.load(tmp_path / "a.nrrd")

    axis = image.system.axes[3]
    assert axis.type == expected[0]
    if expected[0] == "vector":
        assert axis.valence == expected[1]
    if expected[0] == "tensor":
        assert axis.layout == expected[1]
        assert tuple(axis.valence) == ("vector", "vector")
    # The values are kept as the file stores them, in the world space of
    # the header, since it has no measurement frame.
    np.testing.assert_array_equal(image.data, values)
    assert image.frames[3] is image.transformation


@pytest.mark.parametrize(
    "kind,size",
    [
        ("list", 3),
        ("RGB-color", 3),
        ("3-color", 3),
        ("complex", 2),
        ("quaternion", 4),
        ("3D-masked-symmetric-matrix", 7),
        ("3D-masked-matrix", 10),
        # The components of a vector follow the three axes of the space.
        ("vector", 5),
        ("2D-symmetric-matrix", 3),
    ],
)
def test_other_kinds_are_read_as_channels(
    tmp_path: pathlib.Path, kind: str, size: int
) -> None:
    _write(tmp_path / "a.nrrd", kind, size)

    image = NrrdImage.load(tmp_path / "a.nrrd")

    assert image.system.axes[3].type == "channel"
    assert image.frames == (None,) * 4


def test_a_two_dimensional_matrix_is_read_in_a_plane(
    tmp_path: pathlib.Path,
) -> None:
    values = np.arange(3 * 4 * 5, dtype="f4").reshape((3, 4, 5), order="F")
    (tmp_path / "a.nrrd").write_bytes(
        b"NRRD0004\ntype: float\ndimension: 3\nsizes: 3 4 5\n"
        b"kinds: 2D-symmetric-matrix domain domain\nspacings: nan 2 3\n"
        b"endian: little\nencoding: raw\n\n"
        + values.ravel(order="F").tobytes()
    )

    image = NrrdImage.load(tmp_path / "a.nrrd")

    axis = image.system.axes[2]
    assert (axis.type, axis.layout) == ("tensor", "symmetric")
    # Without a world space, the frame is the physical space of the
    # spacings, so the tensors re-expressed in voxels are scaled.
    voxel = np.asarray(image.reframe(None).data)
    d = np.diag([1 / 2, 1 / 3])
    xx, xy, yy = np.moveaxis(np.asarray(image.data), -1, 0)
    t = np.stack([np.stack([xx, xy], -1), np.stack([xy, yy], -1)], -2)
    expected = d @ t @ d
    np.testing.assert_allclose(voxel[..., 0], expected[..., 0, 0], rtol=1e-6)
    np.testing.assert_allclose(voxel[..., 1], expected[..., 0, 1], rtol=1e-6)


def test_the_frame_composes_the_world_with_the_measurement_frame(
    tmp_path: pathlib.Path,
) -> None:
    values = _write(tmp_path / "a.nrrd", "vector", 3, frame=_ROTATION)

    image = NrrdImage.load(tmp_path / "a.nrrd")

    frame = image.frames[3]
    assert image.frames[3] is frame
    # The frame maps the voxels to the measurement frame: M⁻¹ D.
    np.testing.assert_allclose(
        _linear(frame), np.linalg.inv(_ROTATION) @ _DIRECTIONS
    )
    # The columns of M are the basis vectors of the measurement frame in
    # the world, so the world components are M v.
    world = np.asarray(image.reframe().data)
    np.testing.assert_allclose(world, values @ _ROTATION.T, atol=1e-6)
    voxel = np.asarray(image.reframe(None).data)
    expected = values @ (np.linalg.inv(_DIRECTIONS) @ _ROTATION).T
    np.testing.assert_allclose(voxel, expected, atol=1e-6)


def test_a_covector_follows_the_inverse_transpose(
    tmp_path: pathlib.Path,
) -> None:
    values = _write(tmp_path / "a.nrrd", "covariant-vector", 3)

    image = NrrdImage.load(tmp_path / "a.nrrd")

    voxel = np.asarray(image.reframe(None).data)
    np.testing.assert_allclose(voxel, values @ _DIRECTIONS, atol=1e-6)


def test_a_tensor_follows_the_measurement_frame_on_both_indices(
    tmp_path: pathlib.Path,
) -> None:
    values = _write(
        tmp_path / "a.nrrd", "3D-symmetric-matrix", 6, frame=_ROTATION
    )

    image = NrrdImage.load(tmp_path / "a.nrrd")

    world = np.asarray(image.reframe().data)
    xx, xy, xz, yy, yz, zz = np.moveaxis(values, -1, 0)
    t = np.stack(
        [
            np.stack([xx, xy, xz], -1),
            np.stack([xy, yy, yz], -1),
            np.stack([xz, yz, zz], -1),
        ],
        -2,
    )
    expected = _ROTATION @ t @ _ROTATION.T
    upper = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for k, (i, j) in enumerate(upper):
        np.testing.assert_allclose(
            world[..., k], expected[..., i, j], atol=1e-6
        )


@pytest.mark.parametrize(
    "frame",
    [
        "(1,0,0) (0,1,0)",
        "(1,0,0) (0,1,0) (0,0,0)",
        "(1,0,0) (0,1,0) (0,0,nan)",
    ],
)
def test_an_unreadable_measurement_frame_is_foreign(
    tmp_path: pathlib.Path, frame: str
) -> None:
    _write(tmp_path / "a.nrrd", "vector", 3)
    content = (tmp_path / "a.nrrd").read_bytes()
    content = content.replace(
        b"encoding: raw\n",
        f"encoding: raw\nmeasurement frame: {frame}\n".encode(),
    )
    (tmp_path / "a.nrrd").write_bytes(content)

    image = NrrdImage.load(tmp_path / "a.nrrd")

    slot = image.frames[3]
    assert isinstance(slot, CoordinateSystem)
    assert slot.name == "measurement"


def test_assigned_transformations_keep_the_frames_of_the_header(
    tmp_path: pathlib.Path,
) -> None:
    _write(tmp_path / "a.nrrd", "vector", 3, frame=_ROTATION)
    image = NrrdImage.load(tmp_path / "a.nrrd")
    decoded = image.transformations

    image.transformations = list(decoded)

    np.testing.assert_allclose(
        _linear(image.frames[3]), np.linalg.inv(_ROTATION) @ _DIRECTIONS
    )


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,size",
    [
        ("3-vector", 3),
        ("normal", 3),
        ("point", 3),
        ("3D-symmetric-matrix", 6),
        ("3D-matrix", 9),
    ],
)
def test_an_unchanged_file_keeps_its_kinds_and_bytes(
    tmp_path: pathlib.Path, kind: str, size: int
) -> None:
    _write(tmp_path / "a.nrrd", kind, size, frame=_ROTATION)

    NrrdImage.load(tmp_path / "a.nrrd").save(tmp_path / "b.nrrd")

    assert (tmp_path / "b.nrrd").read_bytes() == (
        tmp_path / "a.nrrd"
    ).read_bytes()


@pytest.mark.parametrize(
    "kind,options,size,written",
    [
        ("vector", {}, 3, "vector"),
        ("displacement", {}, 3, "vector"),
        ("vector", {"valence": "covector"}, 3, "covariant-vector"),
        ("tensor", {}, 6, "3D-symmetric-matrix"),
        ("tensor", {"layout": "C"}, 9, "3D-matrix"),
    ],
)
def test_component_axes_are_written_with_their_kind(
    tmp_path: pathlib.Path,
    kind: str,
    options: tx.Dict[str, tx.Any],
    size: int,
    written: str,
) -> None:
    a = _affine(*_systems(kind, **options))
    values = np.random.default_rng(1).random((*_SHAPE, size)).astype("f4")

    io.save(
        SingleScaleImage(values, [a], frames={"c": a}), tmp_path / "a.nrrd"
    )

    record = _record(tmp_path / "a.nrrd")
    assert record.kinds[3] == written.lower()
    np.testing.assert_array_equal(record.measurement_frame, np.eye(3))
    back = NrrdImage.load(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(back.data, values)
    assert back.frames[3] is back.transformation


def test_a_channel_axis_is_written_as_a_list(tmp_path: pathlib.Path) -> None:
    a = _affine(*_systems("channel"))
    values = np.zeros((*_SHAPE, 3), dtype="f4")

    io.save(SingleScaleImage(values, [a]), tmp_path / "a.nrrd")

    assert _record(tmp_path / "a.nrrd").kinds[3] == "list"
    assert NrrdImage.load(tmp_path / "a.nrrd").system.axes[3].type == "channel"


def test_vectors_in_voxels_are_recorded_with_the_directions(
    tmp_path: pathlib.Path,
) -> None:
    a = _affine(*_systems())
    values = np.random.default_rng(1).random((*_SHAPE, 3)).astype("f4")

    io.save(SingleScaleImage(values, [a]), tmp_path / "a.nrrd")

    # The array system is recorded as the measurement frame D, whose
    # columns are the space directions, and the values are kept.
    record = _record(tmp_path / "a.nrrd")
    np.testing.assert_allclose(record.measurement_frame, _DIRECTIONS)
    back = NrrdImage.load(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(back.data, values)
    np.testing.assert_allclose(back.reframe(None).data, values, atol=1e-6)


def test_a_measurement_frame_round_trips(tmp_path: pathlib.Path) -> None:
    values = _write(tmp_path / "a.nrrd", "vector", 3, frame=_ROTATION)
    image = NrrdImage.load(tmp_path / "a.nrrd")

    # Assigned frames are given frames, so the values are written in them
    # rather than as the header of the image states them.
    image.frames = image.frames
    image.save(tmp_path / "b.nrrd")

    record = _record(tmp_path / "b.nrrd")
    np.testing.assert_allclose(record.measurement_frame, _ROTATION, atol=1e-15)
    back = NrrdImage.load(tmp_path / "b.nrrd")
    np.testing.assert_array_equal(back.data, values)


@pytest.mark.parametrize(
    "kind,size",
    [("vector", 3), ("point", 3), ("3D-symmetric-matrix", 6)],
)
def test_an_lps_file_written_in_ras_records_the_flip(
    tmp_path: pathlib.Path, kind: str, size: int
) -> None:
    values = _write(
        tmp_path / "a.nrrd",
        kind,
        size,
        space="left-posterior-superior",
        frame=_ROTATION,
    )
    image = NrrdImage.load(tmp_path / "a.nrrd")

    image.save(tmp_path / "b.nrrd", space="RAS")

    record = _record(tmp_path / "b.nrrd")
    assert record.space == "right-anterior-superior"
    assert record.kinds[0] == kind.lower()
    flip = np.diag([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(record.measurement_frame, flip @ _ROTATION)
    back = NrrdImage.load(tmp_path / "b.nrrd")
    np.testing.assert_array_equal(back.data, values)
    if kind != "vector":
        return
    # The vectors are the same in the world: their RAS components are
    # the flipped LPS components.
    np.testing.assert_allclose(
        np.asarray(back.reframe().data),
        np.asarray(image.reframe().data) @ flip.T,
        atol=1e-6,
    )


def test_a_lazily_applied_rotation_is_recorded_in_the_measurement_frame(
    tmp_path: pathlib.Path,
) -> None:
    voxel, world = _systems()
    a = _affine(voxel, world)
    values = np.random.default_rng(2).random((*_SHAPE, 3)).astype("f4")
    theta = 0.3
    rotation = np.eye(5)[:-1]
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    u = Affine(matrix=rotation, input=world, output=world)
    image = SingleScaleImage(values, [a], frames={"c": a})(u)

    io.save(image, tmp_path / "a.nrrd")

    # The new world is U⁻¹ ∘ A, so the measurement frame is U⁻¹, and the
    # values are written as they are.
    record = _record(tmp_path / "a.nrrd")
    np.testing.assert_allclose(
        record.measurement_frame, np.linalg.inv(rotation[:3, :3]), atol=1e-12
    )
    back = NrrdImage.load(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(back.data, values)
    np.testing.assert_allclose(
        np.asarray(back.reframe().data),
        np.asarray(image.reframe().data),
        atol=1e-6,
    )


def test_two_axes_in_different_frames_share_one_measurement_frame(
    tmp_path: pathlib.Path,
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"]
        + [Axis("u", "vector"), Axis("v", "vector")],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space", unit="mm") for n in "xyz"], name="world"
    )
    matrix = np.zeros((3, 6))
    matrix[:, :3] = _DIRECTIONS
    matrix[:, -1] = _ORIGIN
    a = Affine(matrix=matrix, input=voxel, output=world)
    values = np.random.default_rng(3).random((*_SHAPE, 3, 3)).astype("f4")
    image = SingleScaleImage(values, [a], frames={"u": a})

    io.save(image, tmp_path / "a.nrrd")

    # The first axis is in the world, so the measurement frame is the
    # identity, and the voxel components of the second axis are
    # re-expressed in the world.
    record = _record(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(record.measurement_frame, np.eye(3))
    back = NrrdImage.load(tmp_path / "a.nrrd")
    assert [axis.type for axis in back.system.axes[3:]] == ["vector"] * 2
    np.testing.assert_allclose(back.data, values @ _DIRECTIONS.T, atol=1e-6)


@pytest.mark.parametrize(
    "layout,order",
    [
        # Each layout lists the components of a symmetric matrix, which
        # are written as Mxx Mxy Mxz Myy Myz Mzz.
        ("symmetric", [0, 1, 2, 3, 4, 5]),
        ("symmetric-lower", [0, 1, 3, 2, 4, 5]),
        ("symmetric-diagonal-first", [0, 3, 4, 1, 5, 2]),
    ],
)
def test_a_symmetric_matrix_is_written_in_the_order_of_nrrd(
    tmp_path: pathlib.Path, layout: str, order: tx.List[int]
) -> None:
    a = _affine(*_systems("tensor", layout=layout))
    values = np.random.default_rng(4).random((*_SHAPE, 6)).astype("f4")

    io.save(
        SingleScaleImage(values, [a], frames={"c": a}), tmp_path / "a.nrrd"
    )

    back = NrrdImage.load(tmp_path / "a.nrrd")
    assert back.system.axes[3].layout == "symmetric"
    np.testing.assert_array_equal(back.data, values[..., order])


def test_the_missing_components_of_a_diagonal_are_written_as_zeros(
    tmp_path: pathlib.Path,
) -> None:
    a = _affine(*_systems("tensor", layout="diagonal"))
    values = np.random.default_rng(5).random((*_SHAPE, 3)).astype("f4")

    io.save(
        SingleScaleImage(values, [a], frames={"c": a}), tmp_path / "a.nrrd"
    )

    back = NrrdImage.load(tmp_path / "a.nrrd")
    data = np.asarray(back.data)
    assert data.shape == (*_SHAPE, 6)
    np.testing.assert_array_equal(data[..., [0, 3, 5]], values)
    np.testing.assert_array_equal(data[..., [1, 2, 4]], 0)


def test_a_matrix_in_fortran_order_is_written_by_rows(
    tmp_path: pathlib.Path,
) -> None:
    a = _affine(*_systems("tensor", layout="F"))
    values = np.random.default_rng(6).random((*_SHAPE, 9)).astype("f4")

    io.save(
        SingleScaleImage(values, [a], frames={"c": a}), tmp_path / "a.nrrd"
    )

    back = NrrdImage.load(tmp_path / "a.nrrd")
    assert back.system.axes[3].layout == "C"
    np.testing.assert_array_equal(
        back.data, values[..., [0, 3, 6, 1, 4, 7, 2, 5, 8]]
    )


@pytest.mark.parametrize(
    "kind,options,size",
    [
        # NRRD does not say how a matrix changes with the frame, and reads
        # every matrix with two indices of valence "vector".
        ("tensor", {"valence": ("covector", "covector")}, 6),
        ("tensor", {"valence": ("vector", "covector"), "layout": "C"}, 9),
        # The components of a vector follow the three axes of the world.
        ("vector", {}, 2),
    ],
)
def test_values_that_nrrd_cannot_describe_are_refused(
    tmp_path: pathlib.Path,
    kind: str,
    options: tx.Dict[str, tx.Any],
    size: int,
) -> None:
    a = _affine(*_systems(kind, **options))
    values = np.ones((*_SHAPE, size), dtype="f4")
    image = SingleScaleImage(values, [a])

    with pytest.raises(WriterError, match="lossy=True"):
        io.save(image, tmp_path / "a.nrrd")

    io.save(image, tmp_path / "a.nrrd", lossy=True)
    assert _record(tmp_path / "a.nrrd").kinds[3] == "list"
    np.testing.assert_array_equal(
        NrrdImage.load(tmp_path / "a.nrrd").data, values
    )


def test_a_foreign_frame_is_refused(tmp_path: pathlib.Path) -> None:
    a = _affine(*_systems())
    foreign = CoordinateSystem([Axis(n, "space") for n in "xyz"], name="other")
    values = np.ones((*_SHAPE, 3), dtype="f4")
    image = SingleScaleImage(values, [a], frames={"c": foreign})

    with pytest.raises(WriterError, match="lossy=True"):
        io.save(image, tmp_path / "a.nrrd")
    io.save(image, tmp_path / "a.nrrd", lossy=True)


def test_points_in_voxels_are_refused(tmp_path: pathlib.Path) -> None:
    a = _affine(*_systems("point"))
    values = np.ones((*_SHAPE, 3), dtype="f4")

    with pytest.raises(WriterError, match="lossy=True"):
        io.save(SingleScaleImage(values, [a]), tmp_path / "a.nrrd")

    # Points in the world are written as they are.
    io.save(
        SingleScaleImage(values, [a], frames={"c": a}), tmp_path / "b.nrrd"
    )
    assert _record(tmp_path / "b.nrrd").kinds[3] == "point"


def test_a_frame_that_is_not_affine_is_refused(tmp_path: pathlib.Path) -> None:
    voxel, world = _systems()
    a = _affine(voxel, world)
    field = DisplacementField(
        data=np.zeros((*_SHAPE, 3)) + 0.1, input=world, output=world
    )
    values = np.ones((*_SHAPE, 3), dtype="f4")
    image = SingleScaleImage(values, [a], frames={"c": field @ a})

    with pytest.raises(WriterError, match="lossy=True"):
        io.save(image, tmp_path / "a.nrrd")


def test_without_a_world_the_values_are_written_in_physical_units(
    tmp_path: pathlib.Path,
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", "vector")], name="voxel"
    )
    physical = CoordinateSystem(
        [Axis(n, "space", unit="mm") for n in "ijk"] + [Axis("c", "vector")],
        name="physical",
    )
    scaling = Scaling(scale=[2.0, 3.0, 4.0, 1.0], input=voxel, output=physical)
    values = np.random.default_rng(7).random((*_SHAPE, 3)).astype("f4")

    io.save(SingleScaleImage(values, [scaling]), tmp_path / "a.nrrd")

    record = _record(tmp_path / "a.nrrd")
    assert "space" not in record.fields
    assert "measurement frame" not in record.fields
    back = NrrdImage.load(tmp_path / "a.nrrd")
    np.testing.assert_allclose(back.data, values * [2.0, 3.0, 4.0], rtol=1e-6)
    np.testing.assert_allclose(back.reframe(None).data, values, rtol=1e-6)


def test_without_geometry_the_values_are_kept_in_voxels(
    tmp_path: pathlib.Path,
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", "vector")], name="voxel"
    )
    unit = Scaling(scale=[1.0] * 4, input=voxel, output=voxel)
    values = np.arange(np.prod(_SHAPE) * 3, dtype="i2").reshape((*_SHAPE, 3))

    io.save(SingleScaleImage(values, [unit]), tmp_path / "a.nrrd")

    back = NrrdImage.load(tmp_path / "a.nrrd")
    assert back.system.axes[3].type == "vector"
    assert back.data.dtype == np.int16
    np.testing.assert_array_equal(back.data, values)


def test_an_indexed_image_keeps_its_measurement_frame(
    tmp_path: pathlib.Path,
) -> None:
    values = _write(tmp_path / "a.nrrd", "vector", 3, frame=_ROTATION)
    image = NrrdImage.load(tmp_path / "a.nrrd")

    part = image[1:, :, ::2]
    io.save(part, tmp_path / "b.nrrd")

    # The frame and the world both compose with the map of the voxels, so
    # the measurement frame is the same and the values are kept.
    np.testing.assert_allclose(
        _record(tmp_path / "b.nrrd").measurement_frame, _ROTATION, atol=1e-12
    )
    back = NrrdImage.load(tmp_path / "b.nrrd")
    np.testing.assert_array_equal(back.data, values[1:, :, ::2])
    np.testing.assert_allclose(
        np.asarray(back.reframe().data),
        np.asarray(image.reframe().data)[1:, :, ::2],
        atol=1e-6,
    )


def _slice(path: pathlib.Path) -> np.ndarray:
    """Write vectors of three components on a slice of a 3-D world."""
    values = np.random.default_rng(8).random((3, 4, 5)).astype("f4")
    path.write_bytes(
        b"NRRD0004\ntype: float\ndimension: 3\n"
        b"space: right-anterior-superior\nsizes: 3 4 5\n"
        b"space directions: none (1,0,0) (0,1,0)\n"
        b"kinds: vector domain domain\nendian: little\nencoding: raw\n\n"
        + values.ravel(order="F").tobytes()
    )
    return np.moveaxis(values, 0, -1)


def test_vectors_of_a_slice_in_a_volume_are_refused_when_changed(
    tmp_path: pathlib.Path,
) -> None:
    values = _slice(tmp_path / "a.nrrd")
    image = NrrdImage.load(tmp_path / "a.nrrd")
    assert image.system.axes[2].type == "vector"

    # The file is written again as it is.
    image.save(tmp_path / "b.nrrd")
    assert (tmp_path / "b.nrrd").read_bytes() == (
        tmp_path / "a.nrrd"
    ).read_bytes()

    # The world maps two axes of the array onto the three components, so
    # no measurement frame relates another frame to it.
    assigned = NrrdImage.load(tmp_path / "a.nrrd")
    assigned.frames = assigned.frames
    for changed in (image[1:], assigned):
        with pytest.raises(WriterError, match="lossy=True"):
            io.save(changed, tmp_path / "c.nrrd")
    io.save(image[1:], tmp_path / "c.nrrd", lossy=True)
    assert _record(tmp_path / "c.nrrd").kinds[2] == "list"
    np.testing.assert_array_equal(
        NrrdImage.load(tmp_path / "c.nrrd").data, values[1:]
    )


def test_an_axis_written_as_a_list_gives_no_measurement_frame(
    tmp_path: pathlib.Path,
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"]
        + [
            Axis("t", "tensor", valence=("covector", "covector")),
            Axis("v", "vector"),
        ],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space", unit="mm") for n in "xyz"], name="world"
    )
    matrix = np.zeros((3, 6))
    matrix[:, :3] = _DIRECTIONS
    matrix[:, -1] = _ORIGIN
    a = Affine(matrix=matrix, input=voxel, output=world)
    values = np.random.default_rng(9).random((*_SHAPE, 6, 3)).astype("f4")
    # The tensor is in the array system, which NRRD cannot describe, and
    # the vector is in the world.
    image = SingleScaleImage(values, [a], frames={"v": a})

    io.save(image, tmp_path / "a.nrrd", lossy=True)

    record = _record(tmp_path / "a.nrrd")
    assert record.kinds[3:] == ["list", "vector"]
    np.testing.assert_array_equal(record.measurement_frame, np.eye(3))
    back = NrrdImage.load(tmp_path / "a.nrrd")
    np.testing.assert_array_equal(back.data, values)


def test_a_nearly_singular_measurement_frame_is_foreign(
    tmp_path: pathlib.Path,
) -> None:
    _write(tmp_path / "a.nrrd", "vector", 3, frame=np.diag([1, 1, 1e-17]))

    image = NrrdImage.load(tmp_path / "a.nrrd")

    slot = image.frames[3]
    assert isinstance(slot, CoordinateSystem)
    assert slot.name == "measurement"


@pytest.mark.parametrize(
    "base,size,written",
    [
        ("3-vector", 3, "3-vector"),
        # In a plane, a vector has two entries, which `3-vector` cannot
        # hold.
        ("3-vector", 2, "vector"),
        ("2-vector", 2, "2-vector"),
    ],
)
def test_a_kind_is_kept_only_with_the_length_that_it_requires(
    base: str, size: int, written: str
) -> None:
    # The image axes x, y and c are stored in the order c, x, y.
    kinds = _planned_kinds(
        f"{base} domain domain", [1, 2, 0], {2: "vector"}, (4, 5, size)
    )

    assert kinds == f"{written} domain domain"
