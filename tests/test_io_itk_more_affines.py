"""ITK affine families using full quaternions and composed shear."""

from pathlib import Path

import numpy as np
import pytest

from brainhops import io
from brainhops.datamodel import transformations as xforms
from brainhops.io.transformations.itk.mat import MatRaw

_CENTER = np.array([4.0, -2.0, 1.0])
_TRANSLATION = np.array([2.0, -3.0, 4.0])


def _rotation(q: np.ndarray) -> np.ndarray:
    """Return the matrix ITK builds from the quaternion `(x, y, z, w)`.

    The matrix is the one of the conjugate quaternion, and it is not
    normalised, so a nonunit quaternion also scales by its squared norm.
    """
    x, y, z, w = q
    return np.array(
        [
            [
                w * w + x * x - y * y - z * z,
                2 * (x * y - w * z),
                2 * (x * z + w * y),
            ],
            [
                2 * (x * y + w * z),
                w * w - x * x + y * y - z * z,
                2 * (y * z - w * x),
            ],
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                w * w - x * x - y * y + z * z,
            ],
        ]
    )


def _quaternion(kind: str) -> np.ndarray:
    q = np.array([0.2, -0.1, 0.3, 0.6])
    if kind == "negative-w":
        q[3] *= -1
    if kind != "nonunit":
        q /= np.linalg.norm(q)
    return q


def _skew_case() -> "tuple[list[float], np.ndarray]":
    versor = np.array([0.1, -0.2, 0.3])
    w = np.sqrt(1 - versor @ versor)
    x, y, z = versor
    rotation = np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )
    scale = [1.3, 0.8, 1.1]
    shear = np.array([[1, 0.2, -0.1], [0, 1, 0.3], [0, 0, 1]])
    matrix = rotation @ np.diag(scale) @ shear
    parameters = [*versor, *_TRANSLATION, *scale, 0.2, -0.1, 0.3]
    return parameters, matrix


def _write_tfm(path: Path, name: str, parameters: "list[float]") -> None:
    path.write_text(
        "#Insight Transform File V1.0\n"
        "#Transform 0\n"
        f"Transform: {name}_double_3_3\n"
        "Parameters: " + " ".join(repr(float(p)) for p in parameters) + "\n"
        "FixedParameters: " + " ".join(repr(float(c)) for c in _CENTER) + "\n"
    )


@pytest.mark.parametrize("output", ["tfm", "mat"])
@pytest.mark.parametrize(
    "kind", ["quaternion", "negative-w", "nonunit", "skew"]
)
def test_extended_affine_families(
    tmp_path: Path, kind: str, output: str
) -> None:
    """The two families apply the matrix ITK defines about their center."""
    if kind == "skew":
        name = "ComposeScaleSkewVersor3DTransform"
        parameters, matrix = _skew_case()
    else:
        name = "QuaternionRigidTransform"
        q = _quaternion(kind)
        parameters = [*q, *_TRANSLATION]
        matrix = _rotation(q)
    path = tmp_path / "native.tfm"
    _write_tfm(path, name, parameters)
    loaded = io.transformations.load(path, hint="itk")
    if output == "mat":
        out = tmp_path / "copy.mat"
        MatRaw(blocks=tuple(loaded.transformations)).save(out)
        loaded = io.transformations.load(out, hint="itk")
    assert loaded[0].type == name
    np.testing.assert_allclose(loaded[0].parameters, parameters)
    points = np.array([[1.0, 2.0, -3.0], [10.0, -5.0, 8.0], [-4.0, 6.0, 2.0]])
    expected = (points - _CENTER) @ matrix.T + _CENTER + _TRANSLATION
    affine = np.asarray(loaded.compute().to(xforms.Affine).matrix)
    actual = points @ affine[:, :3].T + affine[:, 3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
