# dependencies
import numpy as np
import scipy.linalg
import typing_extensions as tx

# api
from brainhops.backends import cp, get_array_backend
from brainhops.errors import DomainError

# typing
if tx.TYPE_CHECKING:
    from brainhops._core.typing import ArrayProtocol as _ArrayProtocol

ArrayProtocol: tx.TypeAlias = "_ArrayProtocol"

# constants

_RTOL = float(np.sqrt(np.finfo(np.float64).eps))
"""
The relative tolerance below which an eigenvalue counts as zero, or as
real, and an imaginary part as rounding error. Near the boundary of their
domain the principal square root and logarithm are ill-conditioned, so a
matrix that is within rounding of it is refused rather than answered.
"""


# --- linalg -----------------------------------------------------------


def inv(matrix: "ArrayProtocol") -> "ArrayProtocol":
    """The inverse of a square matrix."""
    nx = get_array_backend(matrix)
    return nx.linalg.inv(matrix)


def sqrtm(
    matrix: "ArrayProtocol", what: str = "The square root of this matrix"
) -> "ArrayProtocol":
    """
    The principal square root of a square matrix.

    `what` names the operation in the error a refusal raises, so that a
    caller can say whose square root it was asking for.
    """
    host = to_host(matrix, dtype=np.float64)
    dtype = dtype_or_float64(matrix)
    require_square(host, what)
    require_principal(host, what)
    return to(ensure_real(scipy.linalg.sqrtm(host)), matrix, dtype=dtype)


def expm(
    tangent: ArrayProtocol, what: str = "The exponential of this matrix"
) -> ArrayProtocol:
    """The exponential of a square matrix: every real one has one."""
    host = to_host(tangent)
    require_square(host, what)
    return to(scipy.linalg.expm(host), tangent)


def logm(
    matrix: ArrayProtocol, what: str = "The logarithm of this matrix"
) -> ArrayProtocol:
    """The principal logarithm of a square matrix."""
    host = to_host(matrix, dtype=np.float64)
    dtype = dtype_or_float64(matrix)
    require_square(host, what)
    require_principal(host, what)
    return to(ensure_real(scipy.linalg.logm(host)), matrix, dtype=dtype)


# --- checks utils -----------------------------------------------------


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


# --- checks -----------------------------------------------------------


def is_positive(array: ArrayProtocol) -> bool:
    """Whether all elements of an array are positive."""
    backend = get_array_backend(array)
    return not bool(backend.any(array <= 0))


def require_positive(array: ArrayProtocol, what: str = "This") -> None:
    """Refuse values that are not all positive."""
    if not is_positive(array):
        raise DomainError(
            f"{what} is not defined: its values must all be positive, "
            f"but they are {to_host(array).tolist()}."
        )


def is_square(matrix: ArrayProtocol) -> bool:
    """Whether a matrix is square."""
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]


def require_square(matrix: ArrayProtocol, what: str = "This") -> None:
    """Refuse a matrix that is not square."""
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
    """
    Whether a linear transformation is principal.

    The principal square root and logarithm of a real matrix exist, and
    are real, exactly when it has no eigenvalue on the closed negative
    real axis (Higham, Functions of Matrices, Thms 1.29 and 1.31).
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


# --- backend utils ----------------------------------------------------


def to_host(array: ArrayProtocol, **kwargs) -> np.ndarray:
    # A small matrix or vector, as a float64 numpy array. The matrix
    # functions run on the host, whatever backend the parameter lives on.
    backend = get_array_backend(array)
    if cp and backend is cp:
        array = array.get()
    return np.asarray(array, **kwargs)


def to(result: np.ndarray, like: ArrayProtocol, **kwargs) -> ArrayProtocol:
    """
    Return `result` with the same backend and dtype as `like`.
    """
    return get_array_backend(like).asarray(result, **kwargs)


def dtype_or_float64(like: ArrayProtocol) -> np.dtype:
    """
    The dtype of `like` if it is floating, else float64.
    """
    nx = get_array_backend(like)
    dtype = nx.dtype(getattr(like, "dtype", nx.float64))
    if not nx.issubdtype(dtype, nx.floating):
        dtype = nx.dtype(nx.float64)
    return dtype


def ensure_real(result: ArrayProtocol) -> ArrayProtocol:
    """
    Ensure that the input is real (has zero imaginary part), and return
    it as a real array.
    """
    nx = get_array_backend(result)
    if nx.iscomplexobj(result):
        scale = max(1.0, float(nx.abs(result).max(initial=0.0)))
        if float(nx.abs(result.imag).max(initial=0.0)) > _RTOL * scale:
            raise DomainError("Matrix is not real.")
        result = result.real
    return result
