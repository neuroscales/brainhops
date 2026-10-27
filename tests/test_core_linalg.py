"""Tests of the batched matrix functions in the linear algebra module."""

import numpy as np
import pytest

from brainhops._core.linalg import expm, logm, sqrtm
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
