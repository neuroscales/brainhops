"""Dense FNIRT fields keep their own sampling grid and FSL vector frame."""

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

nb = pytest.importorskip("nibabel")

from brainhops import io  # noqa: E402
from brainhops.backends import backend  # noqa: E402
from brainhops.datamodel import transformations as xforms  # noqa: E402

SHAPE = (7, 9, 5)
AFFINE = np.diag([1.1, 1.4, 1.7, 1])


def _world(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points through an affine independently of the brainhops reader."""
    return points @ affine[:3, :3].T + affine[:3, 3]


def _grid(shape: tuple) -> np.ndarray:
    """Return a coordinate grid whose components follow the array axes."""
    return np.moveaxis(np.indices(shape, dtype=float), 0, -1)


def _pull(points: np.ndarray) -> np.ndarray:
    """Return a varying physical map that exposes mirrored sampling."""
    return points + points * np.array([0.05, 0.03, 0.02]) + [0.1, -0.1, 0.15]


def _files(
    folder: Path,
    layout: str,
    step: float,
    convention: str,
    vector5d: bool = False,
) -> tuple:
    """Write one map with independently chosen field and reference layouts."""
    source, reference, field = [
        folder / name for name in ("source.nii", "reference.nii", "field.nii")
    ]
    values = (100 + _grid(SHAPE) @ [3, 7, 11]).astype(np.float32)
    nb.save(nb.Nifti1Image(values, AFFINE), source)
    nb.save(nb.Nifti1Image(np.zeros(SHAPE, np.int16), AFFINE), reference)
    shape = tuple(int((s - 1) / step) + 1 for s in SHAPE)
    field_affine = AFFINE @ np.diag([step, step, step, 1])
    geometry = nb.Nifti1Image(np.zeros(shape, np.float32), field_affine)
    orientation = nb.orientations.ornt_transform(
        nb.orientations.io_orientation(field_affine),
        nb.orientations.axcodes2ornt(tuple(layout)),
    )
    geometry = geometry.as_reoriented(orientation)
    points = _world(geometry.affine, _grid(geometry.shape))
    # FSL flips x for this positive-determinant reference. Components of
    # the vectors stay in that FSL frame when the storage axes change.
    ras2fsl = np.eye(4)
    ras2fsl[0, 0] = -1
    ras2fsl[0, 3] = (SHAPE[0] - 1) * AFFINE[0, 0]
    vectors = _world(ras2fsl, _pull(points))
    if convention == "relative":
        vectors -= _world(ras2fsl, points)
    if vector5d:
        vectors = vectors[..., None, :]
    image = nb.Nifti1Image(vectors.astype(np.float32), geometry.affine)
    image.header["intent_code"] = 2006 if convention == "relative" else 0
    nb.save(image, field)
    return source, reference, field, values


@pytest.mark.parametrize("layout", ["RAS", "LAS", "SLP"])
@pytest.mark.parametrize("step", [0.5, 1, 2])
@pytest.mark.parametrize("convention", ["relative", "absolute"])
@pytest.mark.parametrize("vector5d", [False, True])
def test_loaded_dense_field_uses_its_own_grid(
    tmp_path: Path,
    layout: str,
    step: float,
    convention: str,
    vector5d: bool,
) -> None:
    source_path, ref_path, field_path, values = _files(
        tmp_path, layout, step, convention, vector5d
    )
    with backend("numpy"):
        source = io.images.load(source_path, hint="nifti")
        reference = io.images.load(ref_path, hint="nifti")
        warp = io.load(
            field_path,
            hint="fnirt",
            moving=source,
            reference=reference,
            deformation_type=convention,
        )
        points = _world(nb.load(ref_path).affine, _grid(SHAPE))
        sequence = xforms.Sequence(
            transformations=[
                xforms.CoordinatesField(field=points),
                *warp.transformations,
            ]
        )
        actual_world = np.asarray(sequence.compute().field)
        np.testing.assert_allclose(actual_world, _pull(points), atol=3e-6)
        result = source(warp).reslice(reference, bound="nearest")
    voxels = _world(np.linalg.inv(nb.load(source_path).affine), _pull(points))
    expected = map_coordinates(
        values, np.moveaxis(voxels, -1, 0), order=1, mode="nearest"
    )
    np.testing.assert_allclose(result.data, expected, atol=3e-5)
    np.testing.assert_array_equal(source.data, values)


@pytest.mark.parametrize("convention", ["relative", "absolute"])
def test_dense_grid_placement_keeps_dask_fields_lazy(
    tmp_path: Path,
    convention: str,
) -> None:
    da = pytest.importorskip("dask.array")
    callbacks = pytest.importorskip("dask.callbacks")
    source, reference, field, _ = _files(tmp_path, "SLP", 0.5, convention)
    tasks = []
    callback = callbacks.Callback(pretask=lambda *args: tasks.append(True))
    with backend("dask"), callback:
        moving = io.images.load(source, hint="nifti")
        fixed = io.images.load(reference, hint="nifti")
        warp = io.load(
            field,
            hint="fnirt",
            moving=moving,
            reference=fixed,
            deformation_type=convention,
        )
        assert isinstance(warp.transformations[1].data, da.Array)
        assert not tasks
