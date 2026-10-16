"""ITK affine families using full quaternions and composed shear."""

from pathlib import Path

import numpy as np
import pytest

from brainhops import io
from brainhops.datamodel import transformations as xforms


@pytest.mark.parametrize("extension", ["tfm", "h5", "mat"])
@pytest.mark.parametrize(
    "kind", ["quaternion", "negative-w", "nonunit", "skew"]
)
def test_native_extended_affine_families(
    tmp_path: Path, extension: str, kind: str
) -> None:
    sitk = pytest.importorskip("SimpleITK")
    if extension == "h5":
        pytest.importorskip("h5py")
    if kind == "skew":
        native = sitk.ComposeScaleSkewVersor3DTransform()
        native.SetRotation([1.0, 2.0, -3.0], 0.6)
        native.SetScale([1.3, 0.8, 1.1])
        native.SetSkew([0.2, -0.1, 0.3])
        native.SetTranslation([2.0, -3.0, 4.0])
    else:
        native = sitk.Transform(3, sitk.sitkQuaternionRigid)
        q = np.array([0.2, -0.1, 0.3, 0.6])
        if kind == "negative-w":
            q[3] *= -1
        if kind != "nonunit":
            q /= np.linalg.norm(q)
        native.SetParameters([*q, 2.0, -3.0, 4.0])
    native.SetFixedParameters([4.0, -2.0, 1.0])
    path = tmp_path / f"native.{extension}"
    sitk.WriteTransform(native, str(path))
    loaded = io.transformations.load(path, hint="itk")
    points = np.array([[1.0, 2.0, -3.0], [10.0, -5.0, 8.0], [-4.0, 6.0, 2.0]])
    expected = np.array([native.TransformPoint(p) for p in points])
    matrix = np.asarray(loaded.compute().to(xforms.Affine).matrix)
    actual = points @ matrix[:, :3].T + matrix[:, 3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    # MATLAB output preserves the native class and parameterisation.
    out = tmp_path / "copy.mat"
    from brainhops.io.transformations.itk.mat import MatRaw

    MatRaw(blocks=tuple(loaded.transformations)).save(out)
    back = io.transformations.load(out, hint="itk")
    assert back[0].type == loaded[0].type
    np.testing.assert_allclose(back[0].parameters, native.GetParameters())
    np.testing.assert_allclose(back.compute().to(xforms.Affine).matrix, matrix)
