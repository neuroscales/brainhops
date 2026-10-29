"""Tests of the conversion of small parameters to NumPy arrays.

The small parameters of transformations, such as the matrix of an affine
transformation, are NumPy arrays. Their hints are converted by the NumPy
converter of bagof, which moves an array of any backend to memory and
fits it to the shape of the hint with the rules of NumPy broadcasting.
"""

import numpy as np
import pytest
import typing_extensions as tx
from bagof.converters import Converter
from bagof.converters.exceptions import ConversionError

from brainhops._core.typing import nparray, npmatrix, npvector
from brainhops.datamodel.transformations import Affine, Translation

MATRIX = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]])


def _convert(hint: tx.Any, value: tx.Any) -> np.ndarray:
    return Converter.get(hint)(value)


def test_a_lazy_matrix_is_computed_by_the_converter() -> None:
    # The converter computes a Dask matrix in memory before NumPy checks
    # it.
    da = pytest.importorskip("dask.array")
    affine = Affine(da.from_array(MATRIX))
    assert isinstance(affine.matrix, np.ndarray)
    np.testing.assert_array_equal(affine.matrix, MATRIX)


def test_values_that_numpy_converts_are_kept() -> None:
    # Lists and NumPy arrays go to the NumPy converter as before.
    assert Affine(MATRIX.tolist()).matrix.shape == MATRIX.shape
    vector = np.arange(3.0)
    np.testing.assert_array_equal(Translation(vector).translation, vector)
    assert isinstance(_convert(npvector[float], [1.0, 2.0]), np.ndarray)


def test_a_cupy_matrix_is_copied_to_memory() -> None:
    # A matrix built on the GPU is copied to memory, where small
    # parameters live.
    cp = pytest.importorskip("cupy")
    affine = Affine(cp.asarray(MATRIX))
    assert isinstance(affine.matrix, np.ndarray)
    np.testing.assert_array_equal(affine.matrix, MATRIX)


@pytest.mark.parametrize(
    "hint, value, shape",
    [
        (npvector[float], 2.0, (1,)),
        (npvector[float], [1.0, 2.0], (2,)),
        (npmatrix[float], 2.0, (1, 1)),
        (npmatrix[float], [1.0, 2.0], (1, 2)),
        (npmatrix[float], np.ones((2, 3)), (2, 3)),
        (npmatrix[float], np.ones((1, 2, 3)), (2, 3)),
        (nparray[float], np.ones((2, 3, 4)), (2, 3, 4)),
    ],
)
def test_the_converter_fits_the_array_to_the_shape_of_the_hint(
    hint: tx.Any, value: tx.Any, shape: tx.Tuple[int, ...]
) -> None:
    assert _convert(hint, value).shape == shape


@pytest.mark.parametrize(
    "hint, value",
    [
        (npvector[float], np.ones((2, 3))),
        (npmatrix[float], np.ones((2, 3, 4))),
    ],
)
def test_the_converter_refuses_extra_dimensions(
    hint: tx.Any, value: tx.Any
) -> None:
    with pytest.raises(ConversionError):
        _convert(hint, value)


def test_an_affine_refuses_a_stack_of_matrices() -> None:
    # A stack would otherwise be read as a single matrix.
    with pytest.raises(TypeError, match="could not convert"):
        Affine(np.stack([MATRIX, MATRIX]))
