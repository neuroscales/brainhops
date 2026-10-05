"""
Tests for reslicing a space-and-time (4D) NIfTI image.

A NIfTI voxel-to-world affine is three-dimensional: it maps the spatial
axes and leaves the time axis of a 4D image untouched. Reslicing such an
image applies the spatial affine to every frame, and carries each frame
over unchanged in time.
"""

import numpy as np
import pytest

nb = pytest.importorskip("nibabel")
ndi = pytest.importorskip("scipy.ndimage")

import brainhops.io as io  # noqa: E402
from brainhops.datamodel.images import SingleScaleImage  # noqa: E402
from brainhops.datamodel.transformations import Affine  # noqa: E402

SHAPE = (6, 7, 5, 4)


def _save(path: object, data: np.ndarray, affine: np.ndarray) -> str:
    """Write a 4D NIfTI with spatial units in mm and time units in s."""
    nii = nb.Nifti1Image(data, affine)
    nii.header.set_xyzt_units("mm", "sec")
    nb.save(nii, str(path))
    return str(path)


def _rotation(angle: float) -> np.ndarray:
    """A rotation about the z axis, around the center of the volume."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.eye(4)
    rotation[:2, :2] = [[c, -s], [s, c]]
    center = np.eye(4)
    center[:3, -1] = -(np.asarray(SHAPE[:3]) - 1) / 2
    return np.linalg.inv(center) @ rotation @ center


def _expected(
    data: np.ndarray, source: np.ndarray, target: np.ndarray
) -> tuple:
    """
    Reslice each frame with scipy, and say which voxels sample inside.

    Only the voxels whose coordinates fall inside the source volume are
    compared, where linear interpolation does not depend on the boundary
    condition.
    """
    ijk = np.stack(
        np.meshgrid(*[np.arange(n) for n in SHAPE[:3]], indexing="ij"), -1
    )
    vox2vox = np.linalg.inv(source) @ target
    coords = ijk @ vox2vox[:3, :3].T + vox2vox[:3, -1]
    coords = np.moveaxis(coords, -1, 0)
    expected = np.stack(
        [
            ndi.map_coordinates(data[..., t], coords, order=1)
            for t in range(SHAPE[-1])
        ],
        -1,
    )
    inside = np.all(
        [(coords[d] >= 0) & (coords[d] <= SHAPE[d] - 1) for d in range(3)],
        axis=0,
    )
    return expected, inside


def test_a_4d_nifti_resliced_onto_its_own_grid_is_unchanged(
    tmp_path,  # noqa: ANN001
) -> None:
    """Reslicing a space-and-time image onto its own grid is a no-op."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    path = _save(tmp_path / "bold.nii.gz", data, np.eye(4))

    resliced = io.load(path).reslice()

    assert resliced.shape == SHAPE
    assert np.array_equal(np.asarray(resliced), data)


def test_a_4d_nifti_with_numpy_data_resliced_onto_its_own_geometry(
    tmp_path,  # noqa: ANN001
) -> None:
    """The geometry read again from the header still cancels the affine."""
    data = np.arange(np.prod(SHAPE), dtype="float32").reshape(SHAPE)
    path = _save(tmp_path / "bold.nii.gz", data, _rotation(0.3))
    img = io.load(path)

    resliced = SingleScaleImage(
        data=np.asarray(img.data), transformations=img.transformations
    ).reslice(img.geometry)

    assert np.array_equal(np.asarray(resliced), data)


@pytest.mark.parametrize("backend", ["dask", "numpy"])
def test_a_4d_nifti_is_resliced_through_its_spatial_affine(
    tmp_path,  # noqa: ANN001
    backend: str,
) -> None:
    """
    A rotation in space is applied to every frame, and time is untouched.

    The rotation couples the x and y axes, so the reslice cannot be split
    into one-dimensional steps along them.
    """
    rng = np.random.default_rng(0)
    data = rng.random(SHAPE).astype("float32")
    source = _rotation(np.pi / 7)
    source[:3, :3] *= [1.5, 1.0, 2.0]
    target = np.diag([1.25, 1.25, 2.0, 1.0])
    img = io.load(_save(tmp_path / "source.nii.gz", data, source))
    geometry = io.load(_save(tmp_path / "target.nii.gz", data, target))
    if backend == "numpy":
        img = SingleScaleImage(
            data=np.asarray(img.data), transformations=img.transformations
        )
    else:
        pytest.importorskip("dask.array")
        assert type(img.data).__module__.startswith("dask")

    resliced = img.reslice(geometry.geometry)

    expected, inside = _expected(data, source, target)
    got = np.asarray(resliced)
    assert got.shape == SHAPE
    assert inside.any() and not inside.all()
    assert np.allclose(got[inside], expected[inside], atol=1e-5)
    # The output geometry is the target's, still three-dimensional.
    matrix = np.asarray(resliced.transformation.to(Affine).matrix)
    assert np.allclose(matrix, target[:3])
