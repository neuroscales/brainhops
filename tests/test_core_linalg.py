"""Tests of the batched matrix functions in the linear algebra module."""

import numpy as np
import pytest
import typing_extensions as tx

from brainhops._core.linalg import (
    expm,
    logm,
    require_principal,
    require_square,
    sqrtm,
)
from brainhops.errors import DomainError

# A reflection has the eigenvalue -1, so it has no real principal root
# and no real principal logarithm.
_REFLECTION = np.diag([1.0, -1.0])


def test_sqrtm_error_names_the_caller_operation() -> None:
    with pytest.raises(DomainError, match="The square root of the frame"):
        sqrtm(_REFLECTION, "The square root of the frame")


def test_sqrtm_error_names_the_operation_given_by_keyword() -> None:
    with pytest.raises(DomainError, match="The square root of the frame"):
        sqrtm(_REFLECTION, what="The square root of the frame")


def test_logm_error_names_the_caller_operation() -> None:
    with pytest.raises(DomainError, match="The logarithm of the frame"):
        logm(_REFLECTION, "The logarithm of the frame")


def test_expm_error_names_the_caller_operation() -> None:
    with pytest.raises(DomainError, match="The exponential of the frame"):
        expm(np.ones((2, 3)), "The exponential of the frame")


def test_sqrtm_error_names_the_caller_operation_on_dask() -> None:
    da = pytest.importorskip("dask.array")
    matrix = da.from_array(_REFLECTION, chunks=1)
    root = sqrtm(matrix, "The square root of the frame")
    with pytest.raises(DomainError, match="The square root of the frame"):
        root.compute()


def test_batched_functions_keep_their_name_and_docstring() -> None:
    assert sqrtm.__name__ == "sqrtm"
    assert sqrtm.__doc__.startswith("Return the principal square root")


# --- stacks of matrices -----------------------------------------------


def _stack() -> np.ndarray:
    # Two by three rotations scaled by positive factors, which all have
    # a real principal root and logarithm.
    angles = np.linspace(0.1, 0.6, 6).reshape(2, 3)
    cos, sin = np.cos(angles), np.sin(angles)
    stack = np.empty((2, 3, 2, 2))
    stack[..., 0, 0], stack[..., 0, 1] = cos, -sin
    stack[..., 1, 0], stack[..., 1, 1] = sin, cos
    return stack * np.linspace(1.0, 2.0, 6).reshape(2, 3, 1, 1)


@pytest.mark.parametrize("function", [sqrtm, expm, logm])
def test_matrix_function_applies_to_each_matrix_of_a_stack(
    function: tx.Callable[[np.ndarray], np.ndarray],
) -> None:
    stack = _stack()
    result = function(stack)
    assert result.shape == stack.shape
    for index in np.ndindex(*stack.shape[:-2]):
        np.testing.assert_allclose(result[index], function(stack[index]))


def test_matrix_function_of_an_empty_stack_is_empty() -> None:
    assert sqrtm(np.empty((0, 2, 2))).shape == (0, 2, 2)


def test_matrix_function_applies_to_a_stack_on_dask() -> None:
    da = pytest.importorskip("dask.array")
    stack = _stack()
    result = sqrtm(da.from_array(stack, chunks=(1, 2, 2, 2))).compute()
    np.testing.assert_allclose(result, sqrtm(stack))


def test_require_principal_names_the_matrix_of_a_stack_that_fails() -> None:
    stack = np.stack([np.eye(2), _REFLECTION])
    with pytest.raises(DomainError, match=r"at index \(1,\)"):
        require_principal(stack, "The root")


def test_require_square_reads_the_last_two_dimensions() -> None:
    require_square(np.zeros((4, 3, 3)))
    with pytest.raises(DomainError, match="maps 3 axes to 2 axes"):
        require_square(np.zeros((4, 2, 3)))
    with pytest.raises(DomainError, match="not a matrix"):
        require_square(np.zeros(3))
