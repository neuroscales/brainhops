import numpy as np
import scipy.linalg
import typing_extensions as tx

from brainhops.backends import cp, get_array_backend
from brainhops.errors import DomainError

if tx.TYPE_CHECKING:
    from brainhops._core.typing import ArrayProtocol as _ArrayProtocol

ArrayProtocol: tx.TypeAlias = "_ArrayProtocol"


_RTOL = float(np.sqrt(np.finfo(np.float64).eps))
"""Relative tolerance of the eigenvalue tests.

The tolerance, the square root of the float64 machine epsilon, decides when an
eigenvalue counts as zero or as real. Principal square roots and logarithms are
ill-conditioned near the boundary of their domain, so matrices within rounding
error of the boundary are refused rather than given an inaccurate answer.
"""


def inv(matrix: "ArrayProtocol") -> "ArrayProtocol":
    """Return the inverse of a square matrix."""
    nx = get_array_backend(matrix)
    return nx.linalg.inv(matrix)


def sqrtm(
    matrix: "ArrayProtocol", what: str = "The square root of this matrix"
) -> "ArrayProtocol":
    """Return the principal square root of a square matrix.

    The root is computed on the host with SciPy and converted back to the
    backend of the input. The `what` argument names the operation in the
    [`DomainError`][] raised when the matrix is not square or has no real
    principal root.
    """
    host = to_host(matrix, dtype=np.float64)
    dtype = dtype_or_float64(matrix)
    require_square(host, what)
    require_principal(host, what)
    return to(ensure_real(scipy.linalg.sqrtm(host)), matrix, dtype=dtype)


def expm(
    tangent: ArrayProtocol, what: str = "The exponential of this matrix"
) -> ArrayProtocol:
    """Return the exponential of a square matrix, which always exists.

    The `what` argument names the operation in error messages.
    """
    host = to_host(tangent)
    require_square(host, what)
    return to(scipy.linalg.expm(host), tangent)


def logm(
    matrix: ArrayProtocol, what: str = "The logarithm of this matrix"
) -> ArrayProtocol:
    """Return the principal logarithm of a square matrix.

    The logarithm is computed on the host and checked in the same way as
    the root in [`sqrtm`][]. The `what` argument names the operation in the
    [`DomainError`][] raised when the matrix is not square or has no real
    principal logarithm.
    """
    host = to_host(matrix, dtype=np.float64)
    dtype = dtype_or_float64(matrix)
    require_square(host, what)
    require_principal(host, what)
    return to(ensure_real(scipy.linalg.logm(host)), matrix, dtype=dtype)


class _BoolWithMessage:
    def __new__(
        cls, value: bool, message: str = "", return_reason: bool = True
    ) -> "_BoolWithMessage":
        if not return_reason:
            return bool(value)
        return super().__new__(cls)

    def __init__(self, value: bool, message: str = "", *a, **k) -> None:
        self._value = value
        self._message = message

    def __str__(self) -> str:
        return str(self._message)

    def __bool__(self) -> bool:
        return self._value


def is_positive(array: ArrayProtocol) -> bool:
    """Return whether all the values of an array are strictly positive."""
    backend = get_array_backend(array)
    return not bool(backend.any(array <= 0))


def require_positive(array: ArrayProtocol, what: str = "This") -> None:
    """Raise a [`DomainError`][] unless all values are strictly positive."""
    if not is_positive(array):
        raise DomainError(
            f"{what} is not defined: its values must all be positive, "
            f"but they are {to_host(array).tolist()}."
        )


def is_square(matrix: ArrayProtocol) -> bool:
    """Return whether an array is a square two-dimensional matrix."""
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]


def require_square(matrix: ArrayProtocol, what: str = "This") -> None:
    """Raise a [`DomainError`][] unless a matrix is square."""
    if not is_square(matrix):
        rows, cols = matrix.shape
        raise DomainError(
            f"{what} is not defined: it maps {cols} axes to {rows} axes."
        )


@tx.overload
def is_principal(
    linear: ArrayProtocol, return_reason: tx.Literal[False]
) -> bool: ...


@tx.overload
def is_principal(
    linear: ArrayProtocol, return_reason: tx.Literal[True]
) -> _BoolWithMessage: ...


def is_principal(linear: ArrayProtocol, return_reason: bool = False):
    """Return whether a matrix has a real principal square root and logarithm.

    A real matrix has both when it has no eigenvalue on the closed
    negative real axis (Higham, *Functions of Matrices*, theorems 1.29 and
    1.31). Matrices that are not finite or are numerically singular are
    rejected too. With `return_reason=True`, the result is a boolean-like
    object whose string gives the reason, for use in error messages.
    """
    nx = get_array_backend(linear)
    if not bool(nx.isfinite(linear).all()):
        return _BoolWithMessage(False, "is not finite", return_reason)
    eigenvalues = nx.linalg.eigvals(linear)
    size = nx.abs(eigenvalues)
    scale = size.max(initial=0.0)
    if bool(scale == 0.0) or bool((size <= _RTOL * scale).any()):
        return _BoolWithMessage(False, "is singular", return_reason)
    negative = (eigenvalues.real < 0) & (
        nx.abs(eigenvalues.imag) <= _RTOL * size
    )
    if bool(negative.any()):
        return _BoolWithMessage(
            False,
            "has an eigenvalue on the negative real axis",
            return_reason,
        )
    return _BoolWithMessage(True, "is principal", return_reason)


def require_principal(linear: np.ndarray, what: str = "This") -> None:
    result = is_principal(linear, return_reason=True)
    if not result:
        raise DomainError(
            f"{what} is not defined: it {result}, "
            f"so it has no principal square root or logarithm."
        )


def to_host(array: ArrayProtocol, **kwargs) -> np.ndarray:
    # The matrix functions run on the host, on NumPy arrays, whatever the
    # backend of the input.
    backend = get_array_backend(array)
    if cp and backend is cp:
        array = array.get()
    return np.asarray(array, **kwargs)


def to(result: np.ndarray, like: ArrayProtocol, **kwargs) -> ArrayProtocol:
    """Convert a NumPy result to the array backend of `like`."""
    return get_array_backend(like).asarray(result, **kwargs)


def dtype_or_float64(like: ArrayProtocol) -> np.dtype:
    """Return the dtype of `like` if it is floating, and float64 otherwise."""
    nx = get_array_backend(like)
    dtype = nx.dtype(getattr(like, "dtype", nx.float64))
    if not nx.issubdtype(dtype, nx.floating):
        dtype = nx.dtype(nx.float64)
    return dtype


def ensure_real(result: ArrayProtocol) -> ArrayProtocol:
    """Return the real part of an array whose imaginary part is negligible.

    The imaginary part counts as negligible when it is below `_RTOL` times
    the larger of one and the largest magnitude in the array. Otherwise, a
    [`DomainError`][] is raised.
    """
    nx = get_array_backend(result)
    if nx.iscomplexobj(result):
        scale = max(1.0, float(nx.abs(result).max(initial=0.0)))
        if float(nx.abs(result.imag).max(initial=0.0)) > _RTOL * scale:
            raise DomainError("Matrix is not real.")
        result = result.real
    return result
