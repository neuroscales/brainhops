"""Tests for the frames of the vectors of NIfTI displacement images.

A DISPVECT file stores RAS vectors in the world of its best affine. The
reader records that frame without converting the values, and the writer
re-expresses the values in it (see
`spec/design/vector-and-tensor-images.md`).
"""

import pathlib

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

from brainhops.datamodel.axes import Axis  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.systems import CoordinateSystem  # noqa: E402
from brainhops.datamodel.transformations import (  # noqa: E402
    Affine,
    DisplacementField,
)
from brainhops.errors import LossyConversionError, WriterError  # noqa: E402
from brainhops.io import save  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402

_L = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
_SHAPE = (4, 5, 6)


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


def _stored(path: pathlib.Path) -> np.ndarray:
    # The vectors of a 3-D field are stored along the fifth NIfTI axis.
    return np.asarray(nb.load(str(path)).dataobj)[:, :, :, 0, :]


@pytest.fixture
def values() -> np.ndarray:
    return np.random.default_rng(0).random((*_SHAPE, 3)).astype("f4")


def test_vectors_in_voxel_units_are_written_in_ras(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    a = _affine(*_systems())
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a]), path)

    assert nb.load(str(path)).header.get_intent()[0] == "displacement vector"
    assert np.allclose(_stored(path), values @ _L.T, atol=1e-5)


def test_vectors_in_ras_are_written_as_they_are(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    a = _affine(*_systems())
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a}), path)

    assert np.allclose(_stored(path), values)


def test_a_lazy_transformation_rotates_the_written_vectors(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems()
    a = _affine(voxel, world)
    theta = 0.3
    rotation = np.eye(5)[:-1]
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    u = Affine(matrix=rotation, input=world, output=world)
    path = tmp_path / "field.nii"

    save(SingleScaleImage(values, [a], frames={"c": a})(u), path)

    inverse = np.linalg.inv(rotation[:3, :3])
    assert np.allclose(_stored(path), values @ inverse.T, atol=1e-5)


def test_the_reader_records_the_frame_of_the_file(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    path = tmp_path / "field.nii"
    save(SingleScaleImage(values, [_affine(*_systems())]), path)

    image = NiftiImage.load(path)

    assert image.transformation.input.axes[4].type == "displacement"
    assert image.frames[4] is not None
    # The values are already in the frame of the preferred transformation.
    assert image.reframe() is image
    voxel = np.asarray(image.reframe(None).data)[:, :, :, 0, :]
    assert np.allclose(voxel, values, atol=1e-5)


def test_a_consolidated_image_reads_its_frames_from_the_header(
    values: np.ndarray,
) -> None:
    a = _affine(*_systems())
    image = NiftiImage.from_instance(SingleScaleImage(values, [a]))

    consolidated = image.consolidate()

    assert consolidated.frames[4] is not None
    assert consolidated.reframe() is consolidated


def test_a_frame_that_nifti_cannot_store_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems("vector", valence="covector")
    image = SingleScaleImage(values, [_affine(voxel, world)])
    path = tmp_path / "gradient.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)
    with pytest.raises(LossyConversionError):
        NiftiImage.from_instance(image).consolidate()

    save(image, path, lossy=True)
    stored = np.asarray(nb.load(str(path)).dataobj)[:, :, :, 0, :]
    assert np.allclose(stored, values)


def test_a_frame_that_is_not_affine_needs_lossy(
    tmp_path: pathlib.Path, values: np.ndarray
) -> None:
    voxel, world = _systems()
    displacements = np.zeros((*_SHAPE, 1, 4))
    displacements[..., 0, 0] = 0.1 * np.arange(_SHAPE[1])[None, :, None]
    field = DisplacementField(data=displacements, input=voxel, output=voxel)
    image = SingleScaleImage(
        values, [_affine(voxel, world)], frames={"c": field}
    )
    path = tmp_path / "field.nii"

    with pytest.raises(WriterError, match="lossy=True"):
        save(image, path)

    save(image, path, lossy=True)
    # The field is linear, so its finite differences are exact.
    shear = np.eye(3)
    shear[0, 1] = 0.1
    expected = values @ (_L @ np.linalg.inv(shear)).T
    assert np.allclose(_stored(path), expected, atol=1e-5)
