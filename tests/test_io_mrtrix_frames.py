"""Tests for the frames of the vectors and tensors of MRtrix images.

MRtrix stores the components of a vector or of a tensor along the fourth
axis of an image, in scanner RAS millimetres. The header does not say
what the volumes are, so the reader types the fourth axis only from the
key that brainhops writes, or from the command that wrote the file, and
records the frame of the scanner without converting the values. The
writer re-expresses the values in the scanner and packs the
coefficients of a tensor in the order of `dwi2tensor`.

The fixtures that MRtrix commands would write are encoded independently
of the reader, by placing each voxel at the byte offset that MRtrix
computes (`core/stride.h`).
"""

import pathlib
import struct

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
)
from brainhops.errors import LossyConversionError, WriterError
from brainhops.io import save
from brainhops.io.images.mrtrix import MrtrixImage

_L = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
_SHAPE = (4, 5, 6)

# The entries of a symmetric matrix in the order of `dwi2tensor`, and in
# the layout "symmetric", which is the upper triangle row by row.
_MRTRIX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_SYMMETRIC = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _systems(
    kind: str = "displacement", **options: tx.Any
) -> tx.Tuple[CoordinateSystem, CoordinateSystem]:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"] + [Axis("c", kind, **options)],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"] + [Axis("c", kind, **options)],
        name="RAS",
    )
    return voxel, world


def _affine(voxel: CoordinateSystem, world: CoordinateSystem) -> Affine:
    matrix = np.eye(5)[:-1]
    matrix[:3, :3] = _L
    matrix[:3, -1] = [1.0, 2.0, 3.0]
    return Affine(matrix=matrix, input=voxel, output=world)


def _rotation(theta: float = 0.3) -> np.ndarray:
    rotation = np.eye(5)[:-1]
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    return rotation


def _full(values: np.ndarray, entries: tx.Sequence[tx.Tuple[int, int]]):  # noqa: ANN202
    """Unpack symmetric matrices stored along the last axis."""
    full = np.zeros((*values.shape[:-1], 3, 3), dtype=values.dtype)
    for k, (i, j) in enumerate(entries):
        full[..., i, j] = values[..., k]
        full[..., j, i] = values[..., k]
    return full


def _pack(full: np.ndarray, entries: tx.Sequence[tx.Tuple[int, int]]):  # noqa: ANN202
    return np.stack([full[..., i, j] for i, j in entries], axis=-1)


def _stored(path: pathlib.Path) -> np.ndarray:
    """Return the values of a file, read as an image of channels."""
    return np.asarray(MrtrixImage.load(path).data)


@pytest.fixture
def vectors() -> np.ndarray:
    return np.random.default_rng(0).random((*_SHAPE, 3)).astype("f4")


@pytest.fixture
def tensors() -> np.ndarray:
    """Symmetric matrices in the layout `"symmetric"`."""
    rng = np.random.default_rng(1)
    return rng.random((*_SHAPE, 6)).astype("f4")


# A reference encoder of MRtrix files, as `test_io_mrtrix.py` has one.


def _strides(dim: tx.Sequence[int], layout: str) -> tx.List[int]:
    entries = layout.split(",")
    ranks = [int(e.lstrip("+-")) for e in entries]
    signs = [-1 if e.startswith("-") else 1 for e in entries]
    strides = [0] * len(dim)
    step = 1
    for axis in sorted(range(len(dim)), key=lambda i: ranks[i]):
        strides[axis] = signs[axis] * step
        step *= dim[axis]
    return strides


def _mif(
    path: pathlib.Path,
    data: np.ndarray,
    transform: np.ndarray,
    vox: tx.Sequence[float],
    history: tx.Optional[str] = None,
    layout: tx.Optional[str] = None,
    extra: tx.Sequence[str] = (),
) -> pathlib.Path:
    """Write a `.mif` file as an MRtrix command would write it."""
    dim = data.shape
    layout = layout or ",".join(f"+{i}" for i in range(len(dim)))
    strides = _strides(dim, layout)
    start = sum(-s * (n - 1) for s, n in zip(strides, dim) if s < 0)
    values = [0.0] * data.size
    for index in np.ndindex(*dim):
        offset = start + sum(i * s for i, s in zip(index, strides))
        values[offset] = float(data[index])
    payload = b"".join(struct.pack("<f", v) for v in values)
    lines = [
        "mrtrix image",
        "dim: " + ",".join(map(str, dim)),
        "vox: " + ",".join(map(str, vox)),
        f"layout: {layout}",
        "datatype: Float32LE",
    ]
    lines += [
        "transform: " + ",".join(repr(float(v)) for v in row)
        for row in transform
    ]
    # A key of several lines is written once per line.
    for line in (history or "").split("\n") if history else ():
        lines.append(f"command_history: {line}")
    lines += list(extra)
    head = "\n".join(lines) + "\n"
    offset = 0
    while True:
        text = (head + f"file: . {offset}\nEND\n").encode()
        if len(text) == offset:
            break
        offset = len(text)
    path.write_bytes(text + payload)
    return path


_TRANSFORM = np.array(
    [
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, -20.0],
        [0.0, 0.0, 1.0, 5.0],
    ]
)
_VOX = (2.0, 2.0, 3.0, float("nan"))
# The voxel-to-scanner matrix of these files, `transform @ diag(vox)`.
_V2S = _TRANSFORM[:, :3] * np.array(_VOX[:3])


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def test_a_dwi2tensor_file_is_read_as_tensors_in_the_scanner(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    # The volumes of `dwi2tensor` are D11 D22 D33 D12 D13 D23, and the
    # strides of the file do not change the frame.
    values = _pack(_full(tensors, _SYMMETRIC), _MRTRIX)
    path = _mif(
        tmp_path / "dt.mif",
        values,
        _TRANSFORM,
        _VOX,
        history="dwi2tensor dwi.mif dt.mif  (version=3.0.4)",
        layout="-0,+1,+2,+3",
    )

    image = MrtrixImage.load(path)

    axis = image.system.axes[3]
    assert axis.type == "tensor"
    assert axis.layout == "symmetric-diagonal-first"
    frame = image.frames[3]
    assert image.frames[3] is frame
    assert image.frames[:3] == (None, None, None)
    assert image.reframe(frame) is image
    assert np.allclose(np.asarray(image.data), values)
    # In the array system, the tensor is L⁻¹ D L⁻ᵀ.
    inverse = np.linalg.inv(_V2S)
    voxel = _full(np.asarray(image.reframe(None).data), _MRTRIX)
    expected = inverse @ _full(values, _MRTRIX) @ inverse.T
    assert np.allclose(voxel, expected, atol=1e-5)
    # The frame maps the voxels to the scanner, as the preferred
    # transformation does, and ignores the fourth axis.
    assert frame.output is image.transformation.output
    assert np.allclose(np.asarray(frame.matrix)[:, :3], _V2S)
    assert np.allclose(np.asarray(frame.matrix)[:, 3], 0)
    # The frame is the preferred transformation, so the values are already
    # in the default frame and in the frame named after the scanner.
    assert frame is image.transformation
    for target in (-1, "scanner"):
        ras = np.asarray(image.reframe(target).data)
        assert np.allclose(ras, values, atol=1e-6)


@pytest.mark.parametrize(
    "history, kind",
    [
        ("warpconvert w.mif deformation2displacement d.mif", "displacement"),
        ("warpconvert w.mif displacement2deformation d.mif", "coordinate"),
        ("warpinvert w.mif i.mif -displacement", "displacement"),
        ("warpinit template.mif w.mif", "coordinate"),
    ],
)
def test_a_warp_command_types_the_fourth_axis(
    tmp_path: pathlib.Path, vectors: np.ndarray, history: str, kind: str
) -> None:
    path = _mif(tmp_path / "w.mif", vectors, _TRANSFORM, _VOX, history)

    image = io.images.load(path)

    assert isinstance(image, MrtrixImage)
    assert image.system.axes[3].type == kind
    assert image.frames[3] is image.frames[3]
    if kind == "displacement":
        voxel = np.asarray(image.reframe(None).data)
        expected = vectors @ np.linalg.inv(_V2S).T
        assert np.allclose(voxel, expected, atol=1e-5)


def test_a_warpfull_file_holds_coordinates(tmp_path: pathlib.Path) -> None:
    values = np.zeros((*_SHAPE, 3, 4), dtype="f4")
    path = _mif(
        tmp_path / "full.mif",
        values,
        _TRANSFORM,
        (*_VOX, float("nan")),
        "mrregister a.mif b.mif -nl_warp_full full.mif",
    )

    image = io.images.load(path)

    assert image.system.axes[3].type == "coordinate"
    assert image.system.axes[4].type in (None, "channel")
    assert image.frames[3] is not None


@pytest.mark.parametrize(
    "dim, history",
    [
        # Nothing says what the volumes are.
        (3, None),
        # `mrregister` also writes the moved image, whatever its volumes.
        (3, "mrregister a.mif b.mif -transformed moved.mif"),
        # The peaks of `sh2peaks` hold directions without an order of
        # their own, and the eigenvalues of `tensor2metric` have the shape
        # of its eigenvectors.
        (3, "sh2peaks fod.mif peaks.mif -num 1"),
        (3, "tensor2metric dt.mif -value eig.mif -num 1,2,3"),
        # Spherical harmonics change by a rule of their own.
        (45, "dwi2fod csd dwi.mif wm.txt wm.mif"),
        # The kurtosis tensor of `dwi2tensor` has another shape.
        (15, "dwi2tensor dwi.mif dt.mif -dkt dkt.mif"),
        # A key of brainhops under a later command may be stale.
        (3, "brainhops  (version=1)\nmrcalc v.mif -abs out.mif"),
    ],
)
def test_other_volumes_stay_channels(
    tmp_path: pathlib.Path, dim: int, history: tx.Optional[str]
) -> None:
    values = np.zeros((*_SHAPE, dim), dtype="f4")
    path = _mif(
        tmp_path / "x.mif",
        values,
        _TRANSFORM,
        _VOX,
        history,
        # Without the line of brainhops last, the key decides nothing.
        extra=["brainhops_field_type: vectors"] if dim == 3 else [],
    )

    image = MrtrixImage.load(path)

    assert image.system.axes[3].type is None
    assert all(slot is None for slot in image.frames)


def test_a_key_that_disagrees_with_the_shape_is_ignored(
    tmp_path: pathlib.Path,
) -> None:
    values = np.zeros((*_SHAPE, 4), dtype="f4")
    path = _mif(
        tmp_path / "x.mif",
        values,
        _TRANSFORM,
        _VOX,
        "brainhops  (version=1)",
        extra=["brainhops_field_type: vectors"],
    )

    assert MrtrixImage.load(path).system.axes[3].type is None


# ----------------------------------------------------------------------
#   ROUND TRIPS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, options, stored_kind",
    [
        ("displacement", {}, "displacement"),
        ("vector", {}, "vector"),
        ("vector", {"valence": "covector"}, "vector"),
    ],
)
def test_vectors_are_written_in_the_scanner_and_read_back(
    tmp_path: pathlib.Path,
    vectors: np.ndarray,
    kind: str,
    options: tx.Dict[str, tx.Any],
    stored_kind: str,
) -> None:
    image = SingleScaleImage(vectors, [_affine(*_systems(kind, **options))])
    path = tmp_path / "v.mif"

    save(image, path)

    if options.get("valence") == "covector":
        expected = vectors @ np.linalg.inv(_L)
    else:
        expected = vectors @ _L.T
    assert np.allclose(_stored(path), expected, atol=1e-5)
    back = MrtrixImage.load(path)
    axis = back.system.axes[3]
    assert axis.type == stored_kind
    assert axis.valences == image.transformation.input.axes[3].valences
    voxel = np.asarray(back.reframe(None).data)
    assert np.allclose(voxel, vectors, atol=1e-5)


def test_tensors_are_written_in_the_order_of_mrtrix(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    systems = _systems("tensor", layout="symmetric")
    image = SingleScaleImage(tensors, [_affine(*systems)])
    path = tmp_path / "dt.mif"

    save(image, path)

    full = _full(tensors, _SYMMETRIC)
    expected = _pack(_L @ full @ _L.T, _MRTRIX)
    assert np.allclose(_stored(path), expected, atol=1e-4)
    back = MrtrixImage.load(path)
    assert back.system.axes[3].layout == "symmetric-diagonal-first"
    voxel = _full(np.asarray(back.reframe(None).data), _MRTRIX)
    assert np.allclose(voxel, full, atol=1e-4)


def test_tensors_in_the_scanner_are_only_reordered(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    a = _affine(*_systems("tensor", layout="symmetric"))
    image = SingleScaleImage(tensors, [a], frames={"c": a})
    path = tmp_path / "dt.mif"

    save(image, path)

    reordered = _pack(_full(tensors, _SYMMETRIC), _MRTRIX)
    assert np.allclose(_stored(path), reordered, atol=1e-5)


def test_a_diagonal_tensor_gets_its_zeros(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    a = _affine(*_systems("tensor", layout="diagonal"))
    image = SingleScaleImage(vectors, [a], frames={"c": a})
    path = tmp_path / "dt.mif"

    save(image, path)

    stored = _stored(path)
    assert stored.shape == (*_SHAPE, 6)
    assert np.allclose(stored[..., :3], vectors, atol=1e-5)
    assert np.allclose(stored[..., 3:], 0)


def test_a_file_with_its_own_frames_is_written_as_it_is(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    path = _mif(
        tmp_path / "dt.mif",
        tensors,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif  (version=3.0.4)",
    )
    image = MrtrixImage.load(path)
    copy = tmp_path / "copy.mif"

    save(image, copy)
    assert copy.read_bytes() == path.read_bytes()

    # Data that are assigned keep the frame of the header, and so do the
    # header keys.
    image.data = np.asarray(image.data) * 2
    save(image, copy)
    written = MrtrixImage.load(copy)
    assert np.allclose(np.asarray(written.data), tensors * 2)
    assert written.metadata.raw.keyval["command_history"].startswith(
        "dwi2tensor"
    )
    assert written.system.axes[3].type == "tensor"


def test_frames_that_are_assigned_are_re_expressed(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    path = _mif(
        tmp_path / "d.mif",
        vectors,
        _TRANSFORM,
        _VOX,
        "warpconvert w.mif deformation2displacement d.mif",
    )
    image = MrtrixImage.load(path)
    # The values are said to be in the array system.
    image.frames = {3: None}
    copy = tmp_path / "copy.mif"

    save(image, copy)

    assert np.allclose(_stored(copy), vectors @ _V2S.T, atol=1e-4)
    written = MrtrixImage.load(copy)
    assert written.metadata.raw.keyval["brainhops_field_type"] == (
        "displacements"
    )
    assert written.system.axes[3].type == "displacement"


def test_a_lazy_rotation_rotates_the_written_values(
    tmp_path: pathlib.Path, vectors: np.ndarray, tensors: np.ndarray
) -> None:
    rotation = _rotation()
    inverse = np.linalg.inv(rotation[:3, :3])
    for kind, values, options in (
        ("vector", vectors, {}),
        ("tensor", tensors, {"layout": "symmetric"}),
    ):
        voxel, world = _systems(kind, **options)
        a = _affine(voxel, world)
        u = Affine(matrix=rotation, input=world, output=world)
        path = tmp_path / f"{kind}.mif"

        save(SingleScaleImage(values, [a], frames={"c": a})(u), path)

        if kind == "vector":
            expected = values @ inverse.T
        else:
            full = inverse @ _full(values, _SYMMETRIC) @ inverse.T
            expected = _pack(full, _MRTRIX)
        assert np.allclose(_stored(path), expected, atol=1e-5)


def test_a_lazy_rotation_of_a_read_file_rotates_its_values(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    values = _pack(_full(tensors, _SYMMETRIC), _MRTRIX)
    path = _mif(
        tmp_path / "dt.mif",
        values,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif  (version=3.0.4)",
    )
    image = MrtrixImage.load(path)
    world = image.transformation.output
    rotation = np.eye(4)[:-1]
    rotation[:2, :2] = _rotation()[:2, :2]
    u = Affine(matrix=rotation, input=world, output=world)
    copy = tmp_path / "copy.mif"

    save(image(u), copy)

    inverse = np.linalg.inv(rotation[:3, :3])
    full = inverse @ _full(values, _MRTRIX) @ inverse.T
    assert np.allclose(_stored(copy), _pack(full, _MRTRIX), atol=1e-5)
    assert MrtrixImage.load(copy).system.axes[3].type == "tensor"


def test_an_image_written_by_brainhops_is_not_read_as_a_warp(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    image = SingleScaleImage(vectors, [_affine(*_systems("vector"))])
    path = tmp_path / "v.mif"
    save(image, path)

    assert isinstance(io.load(path), MrtrixImage)


def test_a_stale_kind_is_removed(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    path = tmp_path / "v.mif"
    save(SingleScaleImage(vectors, [_affine(*_systems("vector"))]), path)
    image = MrtrixImage.load(path)
    # Reordered components are channels, under the header of vectors.
    reordered = image[..., ::-1]
    assert reordered.transformation.input.axes[3].type == "channel"
    copy = tmp_path / "copy.mif"

    save(reordered, copy)

    written = MrtrixImage.load(copy)
    assert "brainhops_field_type" not in written.metadata.raw.keyval
    assert written.system.axes[3].type is None

    # The line of a command that wrote tensors no longer decides either.
    dt = _mif(
        tmp_path / "dt.mif",
        np.zeros((*_SHAPE, 7), dtype="f4"),
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif -predicted_signal p.mif",
    )
    six = MrtrixImage.load(dt)[..., :6]
    save(six, copy)
    assert MrtrixImage.load(copy).system.axes[3].type is None


# ----------------------------------------------------------------------
#   LOSSES
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"valence": ("covector", "covector"), "layout": "symmetric"},
        {"layout": "C"},
    ],
)
def test_a_tensor_that_mrtrix_cannot_order_needs_lossy(
    tmp_path: pathlib.Path, options: tx.Dict[str, tx.Any]
) -> None:
    size = 9 if options.get("layout") == "C" else 6
    values = np.random.default_rng(2).random((*_SHAPE, size))
    image = SingleScaleImage(values, [_affine(*_systems("tensor", **options))])
    path = tmp_path / "t.mif"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)
    with pytest.raises(LossyConversionError):
        MrtrixImage.from_instance(image).consolidate()

    save(image, path, lossy=True)
    assert np.allclose(_stored(path), values, atol=1e-6)
    written = MrtrixImage.load(path)
    assert written.system.axes[3].type is None
    assert "brainhops_field_type" not in written.metadata.raw.keyval


def test_a_frame_that_is_not_affine_needs_lossy(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    voxel, world = _systems()
    displacements = np.zeros((*_SHAPE, 1, 4))
    displacements[..., 0, 0] = 0.1 * np.arange(_SHAPE[1])[None, :, None]
    field = DisplacementField(data=displacements, input=voxel, output=voxel)
    image = SingleScaleImage(
        vectors, [_affine(voxel, world)], frames={"c": field}
    )
    path = tmp_path / "d.mif"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)

    save(image, path, lossy=True)
    # The field is linear, so its finite differences are exact.
    shear = np.eye(3)
    shear[0, 1] = 0.1
    expected = vectors @ (_L @ np.linalg.inv(shear)).T
    assert np.allclose(_stored(path), expected, atol=1e-5)
    assert MrtrixImage.load(path).system.axes[3].type == "displacement"


def test_a_foreign_frame_needs_lossy(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    voxel, world = _systems("vector")
    foreign = CoordinateSystem([Axis(n, "space") for n in "uvw"], name="F")
    image = SingleScaleImage(
        vectors, [_affine(voxel, world)], frames={"c": foreign}
    )
    path = tmp_path / "v.mif"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)

    save(image, path, lossy=True)
    assert np.allclose(_stored(path), vectors)
    assert MrtrixImage.load(path).system.axes[3].type is None


def test_coordinates_in_another_frame_need_lossy(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    path = _mif(
        tmp_path / "w.mif",
        vectors,
        _TRANSFORM,
        _VOX,
        "warpinit template.mif w.mif",
    )
    image = MrtrixImage.load(path)
    world = image.transformation.output
    rotation = np.eye(4)[:-1]
    rotation[:2, :2] = _rotation()[:2, :2]
    u = Affine(matrix=rotation, input=world, output=world)
    copy = tmp_path / "copy.mif"

    # Unchanged coordinates are written as they are.
    save(image, copy)
    assert copy.read_bytes() == path.read_bytes()

    # Moved coordinates would need the inverse of the map.
    with pytest.raises(WriterError, match="lossy=True"):
        save(image(u), copy)
    save(image(u), copy, lossy=True)
    assert np.allclose(_stored(copy), vectors)
    assert MrtrixImage.load(copy).system.axes[3].type is None


# ----------------------------------------------------------------------
#   GEOMETRY AND DATA TYPE
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    [
        (Ellipsis, slice(1, 3)),
        (slice(0, 2),),
        (slice(None), slice(None, None, -1)),
    ],
)
def test_an_indexed_image_is_written_again(
    tmp_path: pathlib.Path, index: tx.Tuple[tx.Any, ...]
) -> None:
    values = np.random.default_rng(3).random((*_SHAPE, 7)).astype("f4")
    path = _mif(tmp_path / "dwi.mif", values, _TRANSFORM, _VOX)
    image = MrtrixImage.load(path)
    part = image[index]
    copy = tmp_path / "part.mif"

    save(part, copy)

    written = MrtrixImage.load(copy)
    assert np.allclose(np.asarray(written.data), values[index])
    point = np.array([1.0, 1.0, 1.0])
    voxel = np.asarray(_as_affine(part.transformation).matrix)
    expected = voxel[:, :3] @ point + voxel[:, -1]
    stored = np.asarray(written.transformation.matrix)
    assert np.allclose(stored[:, :3] @ point + stored[:, -1], expected)
    assert np.allclose(stored[:, 3], 0)


def test_re_expressed_values_keep_the_type_of_the_header(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    path = _mif(
        tmp_path / "dt.mif",
        tensors,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif  (version=3.0.4)",
    )
    image = MrtrixImage.load(path)
    world = image.transformation.output
    rotation = np.eye(4)[:-1]
    rotation[:2, :2] = _rotation()[:2, :2]
    moved = image(Affine(matrix=rotation, input=world, output=world))
    copy = tmp_path / "copy.mif"

    save(moved, copy)
    assert MrtrixImage.load(copy).metadata.raw.datatype == "Float32LE"

    save(moved, copy, datatype="Float64LE")
    assert MrtrixImage.load(copy).metadata.raw.datatype == "Float64LE"


# ----------------------------------------------------------------------
#   OTHER FORMATS AND OTHER LAYOUTS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "history, extra, kind",
    [
        (
            "warpconvert w.mif deformation2displacement d.mif",
            [],
            "displacement",
        ),
        (
            "brainhops  (version=1)",
            ["brainhops_field_type: vectors"],
            "vector",
        ),
    ],
)
def test_a_field_read_from_mrtrix_is_written_to_nifti_as_a_field(
    tmp_path: pathlib.Path,
    vectors: np.ndarray,
    history: str,
    extra: tx.List[str],
    kind: str,
) -> None:
    nb = pytest.importorskip("nibabel")
    from brainhops.io.images.nifti import NiftiImage

    path = _mif(
        tmp_path / "w.mif", vectors, _TRANSFORM, _VOX, history, extra=extra
    )
    image = MrtrixImage.load(path)
    assert image.system.axes[3].type == kind
    nifti = tmp_path / "w.nii"

    save(SingleScaleImage.from_instance(image), nifti)

    stored = nb.load(str(nifti))
    assert stored.shape == (*_SHAPE, 1, 3)
    assert stored.header.get_intent()[0] == "displacement vector"
    # Both files store the vectors in the same scanner RAS.
    values = np.asarray(stored.dataobj)[:, :, :, 0, :]
    assert np.allclose(values, vectors, atol=1e-6)
    back = NiftiImage.load(nifti)
    assert back.transformation.input.axes[4].type == "displacement"
    assert back.frames[4] is not None
    voxel = np.asarray(back.reframe(None).data)[:, :, :, 0, :]
    expected = vectors @ np.linalg.inv(_V2S).T
    assert np.allclose(voxel, expected, atol=1e-5)


def test_tensors_read_from_mrtrix_need_lossy_in_nifti(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    pytest.importorskip("nibabel")
    path = _mif(
        tmp_path / "dt.mif",
        tensors,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif  (version=3.0.4)",
    )
    image = SingleScaleImage.from_instance(MrtrixImage.load(path))

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, tmp_path / "dt.nii")


def test_a_field_read_from_mrtrix_is_written_to_mgh(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    pytest.importorskip("nibabel")
    path = _mif(
        tmp_path / "w.mif",
        vectors,
        _TRANSFORM,
        _VOX,
        "warpconvert w.mif deformation2displacement d.mif",
    )
    image = SingleScaleImage.from_instance(MrtrixImage.load(path))

    save(image, tmp_path / "w.mgz")

    back = io.images.load(tmp_path / "w.mgz")
    assert np.allclose(
        np.reshape(np.asarray(back.data), vectors.shape), vectors
    )


def test_a_series_of_fields_keeps_its_components_fourth(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    voxel = CoordinateSystem(
        [Axis(n, "space") for n in "ijk"]
        + [Axis("c", "vector"), Axis("t", "time")],
        name="voxel",
    )
    world = CoordinateSystem(
        [Axis(n, "space") for n in "xyz"]
        + [Axis("c", "vector"), Axis("t", "time", unit="s")],
        name="RAS",
    )
    matrix = np.eye(6)[:-1]
    matrix[:3, :3] = _L
    matrix[4, 4] = 2.0
    a = Affine(matrix=matrix, input=voxel, output=world)
    series = np.stack([vectors, 2 * vectors], axis=-1)
    path = tmp_path / "series.mif"

    save(SingleScaleImage(series, [a]), path)

    back = MrtrixImage.load(path)
    assert back.shape == (*_SHAPE, 3, 2)
    assert back.system.axes[3].type == "vector"
    assert back.metadata.raw.vox[4] == 2.0
    stored = np.asarray(back.data)
    assert np.allclose(stored[..., 0], vectors @ _L.T, atol=1e-5)
    assert np.allclose(stored[..., 1], 2 * vectors @ _L.T, atol=1e-5)


@pytest.mark.parametrize(
    "history, kind",
    [
        # The name of a file does not decide.
        (
            "warpconvert my2displacement.mif displacement2deformation o.mif",
            "coordinate",
        ),
        # The options are skipped with their values.
        (
            "warpconvert -template t.mif w.mif warpfull2displacement o.mif",
            "displacement",
        ),
        ("warpconvert w.mif -from 2 warpfull2deformation o.mif", "coordinate"),
        (
            "warpconvert -nthreads 4 w.mif deformation2displacement o.mif",
            "displacement",
        ),
    ],
)
def test_the_conversion_type_of_warpconvert_decides(
    tmp_path: pathlib.Path, vectors: np.ndarray, history: str, kind: str
) -> None:
    path = _mif(tmp_path / "w.mif", vectors, _TRANSFORM, _VOX, history)

    assert MrtrixImage.load(path).system.axes[3].type == kind


def test_a_predicted_signal_of_six_volumes_is_not_a_tensor(
    tmp_path: pathlib.Path, tensors: np.ndarray
) -> None:
    # The predicted signal keeps the gradient table of its input, and the
    # tensor image moves it to `prior_dw_scheme`.
    scheme = [f"dw_scheme: 1,0,0,{b}" for b in (1000,) * 6]
    path = _mif(
        tmp_path / "p.mif",
        tensors,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif -predicted_signal p.mif",
        extra=scheme,
    )
    assert MrtrixImage.load(path).system.axes[3].type is None

    path = _mif(
        tmp_path / "dt.mif",
        tensors,
        _TRANSFORM,
        _VOX,
        "dwi2tensor dwi.mif dt.mif -predicted_signal p.mif",
        extra=["prior_dw_scheme: 1,0,0,1000"],
    )
    assert MrtrixImage.load(path).system.axes[3].type == "tensor"


def test_a_lossy_write_removes_a_stale_key(
    tmp_path: pathlib.Path, vectors: np.ndarray
) -> None:
    path = tmp_path / "v.mif"
    save(SingleScaleImage(vectors, [_affine(*_systems("vector"))]), path)
    header = MrtrixImage.load(path).metadata
    values = np.random.default_rng(4).random((*_SHAPE, 9))
    a = _affine(*_systems("tensor", layout="C"))
    image = MrtrixImage(data=values, transformations=[a], metadata=header)
    copy = tmp_path / "t.mif"

    save(image, copy, lossy=True)

    written = MrtrixImage.load(copy)
    assert "brainhops_field_type" not in written.metadata.raw.keyval
    assert written.system.axes[3].type is None
