"""Loaded reader maps must survive conversion and separable reslicing."""

from pathlib import Path

import numpy as np
import pytest

import brainhops.io as io
from brainhops.backends import backend
from brainhops.datamodel.transformations import Affine


@pytest.mark.parametrize("hint", ["niftyreg.aladin", "mrtrix.linear"])
def test_loaded_affine_survives_conversion_and_direct_image_application(
    tmp_path: Path, hint: str
) -> None:
    nib = pytest.importorskip("nibabel")
    indices = np.indices((7, 8, 9), dtype=np.float32)
    values = 100 * indices[0] + 10 * indices[1] + indices[2]
    geometry = np.diag([1.25, 1.5, 2.0, 1.0])
    for name in ("moving", "reference"):
        image = nib.Nifti1Image(values, geometry)
        image.set_sform(geometry, code=1)
        image.set_qform(geometry, code=1)
        nib.save(image, tmp_path / f"{name}.nii.gz")
    pull = np.eye(4)
    pull[0, 3] = 1.25
    path = tmp_path / "transform.txt"
    np.savetxt(path, pull)
    with backend("numpy"):
        source = io.images.load(tmp_path / "moving.nii.gz")
        reference = io.images.load(tmp_path / "reference.nii.gz")
        transform = io.load(path, hint=hint)
        # Test application before the diagnostic conversion, so a cached
        # conversion cannot hide a failure in the original loaded path.
        actual = source(transform).reslice(
            reference, degree=1, bound="constant"
        )
        np.testing.assert_allclose(
            actual.data[:-1, 1:-1, 1:-1], values[1:, 1:-1, 1:-1]
        )
        np.testing.assert_allclose(transform.to(Affine).matrix, pull[:3])
