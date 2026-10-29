"""Tests of the converter of the NumPy annotations of small parameters."""

import numpy as np
import pytest
import typing_extensions as tx
from bagof.converters import Converter
from bagof.converters.exceptions import ConversionError
from bagof.hints.numpy import dtype, ndarray

from brainhops._core.typing import ToNumpyParameter, npmatrix, npvector

_THREE_ROWS = tx.Annotated[
    ndarray[tx.Tuple[tx.Literal[3], int], dtype[float]], ToNumpyParameter
]
_ANY_SHAPE = tx.Annotated[
    ndarray[tx.Tuple[int, ...], dtype[float]], ToNumpyParameter
]


def _convert(hint: tx.Any, value: tx.Any) -> np.ndarray:
    return Converter.get(hint)(value)


@pytest.mark.parametrize(
    "hint, value, shape",
    [
        (npvector[float], 2.0, (1,)),
        (npvector[float], [1.0, 2.0], (2,)),
        (npmatrix[float], 2.0, (1, 1)),
        (npmatrix[float], [1.0, 2.0], (1, 2)),
        (npmatrix[float], np.ones((2, 3)), (2, 3)),
        (_THREE_ROWS, [1.0, 2.0], (3, 2)),
        (_ANY_SHAPE, np.ones((2, 3, 4)), (2, 3, 4)),
    ],
)
def test_the_converter_pads_and_broadcasts_to_the_hint(
    hint: tx.Any, value: tx.Any, shape: tx.Tuple[int, ...]
) -> None:
    assert _convert(hint, value).shape == shape


@pytest.mark.parametrize(
    "hint, value",
    [
        (npvector[float], np.ones((2, 3))),
        (npmatrix[float], np.ones((2, 3, 4))),
        (_THREE_ROWS, np.ones((2, 4))),
    ],
)
def test_the_converter_refuses_what_broadcasting_cannot_fit(
    hint: tx.Any, value: tx.Any
) -> None:
    with pytest.raises(ConversionError):
        _convert(hint, value)


def test_a_broadcast_array_is_writable() -> None:
    array = _convert(_THREE_ROWS, [1.0, 2.0])
    array[0, 0] = 5.0
    assert array[1, 0] == 1.0
