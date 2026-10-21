"""Tests of the conversion of small parameters to NumPy arrays.

The small parameters of transformations, such as the matrix of an affine
transformation, are NumPy arrays. Their hints carry a converter that
moves an array of any backend to memory before NumPy checks it.
"""

import numpy as np
import pytest

from brainhops import backends
from brainhops._core.typing import ToNumpyParameter
from brainhops.datamodel.transformations import Affine, Translation

da = pytest.importorskip("dask.array")

MATRIX = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]])


def test_a_lazy_matrix_is_computed_by_the_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The converter moves a Dask matrix to memory with to_backend, rather
    # than leaving it to numpy.asanyarray.
    calls = []
    original = backends.to_backend

    def to_backend(x, backend=None):  # noqa: ANN001, ANN202
        calls.append(backend)
        return original(x, backend)

    monkeypatch.setattr(backends, "to_backend", to_backend)
    affine = Affine(da.from_array(MATRIX))
    assert isinstance(affine.matrix, np.ndarray)
    np.testing.assert_array_equal(affine.matrix, MATRIX)
    assert "numpy" in calls


def test_values_that_numpy_converts_are_kept() -> None:
    # Lists and NumPy arrays go to the NumPy converter as before, which
    # checks the number of dimensions.
    assert Affine(MATRIX.tolist()).matrix.shape == MATRIX.shape
    vector = np.arange(3.0)
    np.testing.assert_array_equal(Translation(vector).translation, vector)
    convert = ToNumpyParameter(np.ndarray)
    assert isinstance(convert([1.0, 2.0]), np.ndarray)


def test_a_cupy_matrix_is_copied_to_memory() -> None:
    # A matrix built on the GPU is copied to memory, where small
    # parameters live.
    cp = pytest.importorskip("cupy")
    affine = Affine(cp.asarray(MATRIX))
    assert isinstance(affine.matrix, np.ndarray)
    np.testing.assert_array_equal(affine.matrix, MATRIX)
