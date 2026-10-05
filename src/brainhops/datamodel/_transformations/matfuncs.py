"""Matrix functions: the principal square root, exponential and logarithm.

The functions run on the host, in float64, whatever backend the parameter
lives on, and return their result on that backend, in its floating dtype.
A function whose principal value does not exist, or is not real, raises
[`DomainError`][] rather than return a complex, non-principal or
approximate result.
"""

# dependencies
import numpy as np
import scipy.linalg

# core
from brainhops._core.typing import ArrayProtocol

# api
from brainhops.backends import get_array_backend

# internals
from .errors import DomainError

_RTOL = float(np.sqrt(np.finfo(np.float64).eps))
"""
The relative tolerance below which an eigenvalue counts as zero, or as
real, and an imaginary part as rounding error. Near the boundary of their
domain the principal square root and logarithm are ill-conditioned, so a
matrix that is within rounding of it is refused rather than answered.
"""


# ----------------------------------------------------------------------
#   SQUARE ROOT
# ----------------------------------------------------------------------


def sqrtm(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    """The principal square root of a square matrix."""
    host = _to_host(matrix)
    _require_square(host, what)
    _require_principal(host, what)
    return _like(_real(scipy.linalg.sqrtm(host), what), matrix)


def affine_sqrtm(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    """The principal square root of a compact `(N, N + 1)` affine."""
    host = _to_host(matrix)
    n = host.shape[0]
    _require_square(host[:, :-1], what)
    _require_principal(host[:, :-1], what)
    # The homogeneous matrix adds the eigenvalue 1, whose square root is 1,
    # so its principal square root keeps the last row `[0, ..., 0, 1]`.
    root = _real(scipy.linalg.sqrtm(_homogeneous(host)), what)
    return _like(root[:n], matrix)


# ----------------------------------------------------------------------
#   EXPONENTIAL
# ----------------------------------------------------------------------


def expm(tangent: ArrayProtocol) -> ArrayProtocol:
    """The exponential of a square matrix: every real one has one."""
    host = _to_host(tangent)
    _require_square(host, "The exponential of this tangent")
    return _like(scipy.linalg.expm(host), tangent)


def affine_expm(tangent: ArrayProtocol) -> ArrayProtocol:
    """The exponential of a compact `(N, N + 1)` affine tangent.

    The tangent `[L, l]` is the top of the homogeneous generator
    `[[L, l], [0, ..., 0]]`, whose exponential is a homogeneous affine;
    its top `N` rows are returned. `L` may be singular: a translation has
    the tangent `[0, t]`.
    """
    host = _to_host(tangent)
    n = host.shape[0]
    _require_square(host[:, :-1], "The exponential of this tangent")
    generator = np.zeros((n + 1, n + 1))
    generator[:n] = host
    return _like(scipy.linalg.expm(generator)[:n], tangent)


# ----------------------------------------------------------------------
#   LOGARITHM
# ----------------------------------------------------------------------


def logm(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    """The principal logarithm of a square matrix."""
    host = _to_host(matrix)
    _require_square(host, what)
    _require_principal(host, what)
    return _like(_real(scipy.linalg.logm(host), what), matrix)


def affine_logm(matrix: ArrayProtocol, what: str) -> ArrayProtocol:
    """The principal logarithm of a compact `(N, N + 1)` affine.

    It is the tangent `[L, l]`: the top `N` rows of the principal
    logarithm of the homogeneous matrix, whose last row is `[0, ..., 0]`.
    """
    host = _to_host(matrix)
    n = host.shape[0]
    _require_square(host[:, :-1], what)
    _require_principal(host[:, :-1], what)
    log = _real(scipy.linalg.logm(_homogeneous(host)), what)
    return _like(log[:n], matrix)


def log_scale(scale: ArrayProtocol, what: str) -> ArrayProtocol:
    """The logarithm of positive scaling factors."""
    require_positive(scale, what)
    backend = get_array_backend(scale)
    return backend.log(backend.asarray(scale, dtype=_float_dtype(scale)))


# ----------------------------------------------------------------------
#   CHECKS
# ----------------------------------------------------------------------


def require_positive(scale: ArrayProtocol, what: str) -> None:
    """Refuse scaling factors that are not all positive."""
    backend = get_array_backend(scale)
    if bool(backend.any(scale <= 0)):
        raise DomainError(
            f"{what} is not defined: its scales must all be positive, but "
            f"they are {_to_host(scale).tolist()}."
        )


def _require_square(matrix: np.ndarray, what: str) -> None:
    rows, cols = matrix.shape
    if rows != cols:
        raise DomainError(
            f"{what} is not defined: it maps {cols} axes to {rows} axes, "
            "not a space to itself."
        )


def _require_principal(linear: np.ndarray, what: str) -> None:
    # The principal square root and logarithm of a real matrix exist, and
    # are real, exactly when it has no eigenvalue on the closed negative
    # real axis (Higham, Functions of Matrices, Thms 1.29 and 1.31).
    if not np.isfinite(linear).all():
        raise DomainError(f"{what} is not defined: it is not finite.")
    eigenvalues = np.linalg.eigvals(linear)
    size = np.abs(eigenvalues)
    scale = float(size.max(initial=0.0))
    if scale == 0.0 or (size <= _RTOL * scale).any():
        raise DomainError(
            f"{what} is not defined: its linear part is singular, so it has "
            "no principal square root or logarithm."
        )
    negative = (eigenvalues.real < 0) & (
        np.abs(eigenvalues.imag) <= _RTOL * size
    )
    if negative.any():
        values = sorted(float(v) for v in eigenvalues.real[negative])
        raise DomainError(
            f"{what} is not defined: its linear part has the eigenvalue(s) "
            f"{values} on the negative real axis, so it has no real "
            "principal square root or logarithm. A reflection, or a "
            "rotation by a half turn, is such a transformation."
        )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _to_host(array: ArrayProtocol) -> np.ndarray:
    # A small matrix or vector, as a float64 numpy array. The matrix
    # functions run on the host, whatever backend the parameter lives on.
    backend = get_array_backend(array)
    if backend is not np and backend.__name__.startswith("cupy"):
        array = array.get()
    return np.asarray(array, dtype=np.float64)


def _like(result: np.ndarray, like: ArrayProtocol) -> ArrayProtocol:
    # Return a host result on the backend, and in the floating dtype, of
    # the parameter it was computed from.
    dtype = _float_dtype(like)
    return get_array_backend(like).asarray(result.astype(dtype))


def _float_dtype(like: ArrayProtocol) -> np.dtype:
    # The dtype of `like` if it is floating, else float64.
    dtype = np.dtype(getattr(like, "dtype", np.float64))
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return dtype


def _real(result: np.ndarray, what: str) -> np.ndarray:
    # The real part of a matrix function whose exact value is real, after
    # checking that the imaginary part is only rounding error.
    if np.iscomplexobj(result):
        scale = max(1.0, float(np.abs(result).max(initial=0.0)))
        if float(np.abs(result.imag).max(initial=0.0)) > _RTOL * scale:
            raise DomainError(f"{what} is not defined: it is not real.")
        result = result.real
    return result


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    # The square homogeneous matrix of a compact `(N, N + 1)` affine.
    n = matrix.shape[0]
    full = np.zeros((n + 1, n + 1))
    full[:n] = matrix
    full[n, n] = 1
    return full
