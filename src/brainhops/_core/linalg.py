import functools

import numpy as np
import scipy.linalg
import typing_extensions as tx

from brainhops.errors import DomainError

from .backends import (
    array_backend,
    is_dask,
    like,
    promote_dtypes,
    to_numpy,
    to_promoted_backend,
)

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


# --- chunking and batching -------------------------------------------


def chunk(
    function: tx.Callable[[ArrayProtocol], ArrayProtocol],
    array: ArrayProtocol,
    dims: tx.Union[int, tx.Sequence[int]],
) -> ArrayProtocol:
    """Apply a function to an array, block by block when it is a Dask array.

    Dask has no linear algebra on stacks of matrices, so a Dask array is
    rechunked to hold the dimensions in `dims` whole, and the function is
    applied to each block, which is a NumPy or CuPy array. Any other array
    is passed to the function as it is.
    """
    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(sorted(dims))
    if is_dask(array):
        whole = array.rechunk({dim: -1 for dim in dims})
        return whole.map_blocks(function, dtype=array.dtype)
    return function(array)


def chunked(dims: tx.Union[int, tx.Sequence[int]]) -> tx.Callable:
    """Return a decorator that applies a function to the blocks of an array.

    The decorated function receives a NumPy or CuPy array, as in
    [`chunk`][], whose blocks hold the dimensions in `dims` whole. The
    arguments that follow the array are passed to the function unchanged,
    on every backend.
    """

    def decorator(function: tx.Callable) -> tx.Callable:
        @functools.wraps(function)
        def wrapper(array: ArrayProtocol, *a, **k) -> ArrayProtocol:
            # The extra arguments follow the array in the call, but
            # `functools.partial` would place them before it, so a
            # closure passes them instead.
            def apply(block: ArrayProtocol) -> ArrayProtocol:
                return function(block, *a, **k)

            return chunk(apply, array, dims=dims)

        return wrapper

    return decorator


def batched(ndim: int) -> tx.Callable:
    """Return a decorator that applies a function to each element of a stack.

    The decorated function receives one element, an array of `ndim`
    dimensions such as a single matrix. The leading dimensions of a larger
    array index the elements of a stack, and the function is applied to
    each element in turn, after which the results are stacked again in the
    same order. This serves the functions, such as those of SciPy, that
    accept a stack only in recent releases. A [`DomainError`][] raised for
    one element names the index of that element in the stack. The
    arguments that follow the array are passed to the function unchanged.
    """

    def decorator(function: tx.Callable) -> tx.Callable:
        @functools.wraps(function)
        def wrapper(array: ArrayProtocol, *a, **k) -> ArrayProtocol:
            if array.ndim <= ndim:
                return function(array, *a, **k)
            batch = tuple(array.shape[:-ndim])
            results = []
            for index in np.ndindex(*batch):
                try:
                    results.append(function(array[index], *a, **k))
                except DomainError as error:
                    raise DomainError(
                        f"{error} It is the element at index {index} of "
                        f"the stack."
                    ) from error
            if not results:
                return array_backend(array).empty_like(array)
            xp = array_backend(results[0])
            stacked = xp.stack(results)
            return stacked.reshape(batch + tuple(stacked.shape[1:]))

        return wrapper

    return decorator


chunkedvector = chunked(-1)
chunkedmatrix = chunked((-2, -1))
batchedmatrix = batched(2)


# --- linalg -----------------------------------------------------------


@chunkedmatrix
def inv(matrix: "ArrayProtocol") -> "ArrayProtocol":
    """Return the inverse of a square matrix."""
    xp = array_backend(matrix)
    return xp.linalg.inv(matrix)


@chunkedmatrix
@batchedmatrix
def sqrtm(
    matrix: "ArrayProtocol", what: str = "The square root of this matrix"
) -> "ArrayProtocol":
    """Return the principal square root of a square matrix.

    The root is computed on the host with SciPy and converted back to the
    backend of the input. The `what` argument names the operation in the
    [`DomainError`][] raised when the matrix is not square or has no real
    principal root.
    """
    host = to_numpy(matrix, dtype=np.float64)
    dtype = promote_dtypes(matrix, float)
    require_square(host, what)
    require_principal(host, what)
    return like(_ensure_real(scipy.linalg.sqrtm(host)), matrix, dtype=dtype)


@chunkedmatrix
@batchedmatrix
def expm(
    tangent: ArrayProtocol, what: str = "The exponential of this matrix"
) -> ArrayProtocol:
    """Return the exponential of a square matrix, which always exists.

    The `what` argument names the operation in error messages.
    """
    host = to_numpy(tangent)
    require_square(host, what)
    return like(scipy.linalg.expm(host), tangent)


@chunkedmatrix
@batchedmatrix
def logm(
    matrix: ArrayProtocol, what: str = "The logarithm of this matrix"
) -> ArrayProtocol:
    """Return the principal logarithm of a square matrix.

    The logarithm is computed on the host and checked in the same way as
    the root in [`sqrtm`][]. The `what` argument names the operation in the
    [`DomainError`][] raised when the matrix is not square or has no real
    principal logarithm.
    """
    host = to_numpy(matrix, dtype=np.float64)
    dtype = promote_dtypes(matrix, float)
    require_square(host, what)
    require_principal(host, what)
    return like(_ensure_real(scipy.linalg.logm(host)), matrix, dtype=dtype)


def batchT(x: ArrayProtocol) -> ArrayProtocol:
    """Return the transpose of each matrix in a batch of matrices."""
    xp = array_backend(x)
    return xp.swapaxes(x, -1, -2)


def matmul(left: ArrayProtocol, right: ArrayProtocol) -> ArrayProtocol:
    """Return the matrix product of two arrays.

    The last two axes of each array are treated as matrices, and the other
    axes are broadcast.
    """
    left, right = to_promoted_backend(left, right)
    xp = array_backend(left)
    return xp.matmul(left, right)


def matvec(matrix: ArrayProtocol, vector: ArrayProtocol) -> ArrayProtocol:
    """Return the matrix-vector product of a matrix and a vector.

    The last two axes of the first array are treated as matrices, and the
    last axis of the second array is treated as a vector. The other axes
    are broadcast.
    """
    matrix, vector = to_promoted_backend(matrix, vector)
    xp = array_backend(matrix)
    return xp.matmul(matrix, vector[..., None])[..., 0]


_PolarMatrix: tx.TypeAlias = tx.Literal["U", "P"]
_PolarWhich: tx.TypeAlias = tx.Union[
    _PolarMatrix, tx.Tuple[_PolarMatrix, _PolarMatrix]
]


@chunkedmatrix
def _polar_u(matrix: ArrayProtocol) -> ArrayProtocol:
    xp = array_backend(matrix)
    u, _, vt = xp.linalg.svd(matrix)
    return xp.matmul(u, vt)


@chunkedmatrix
def _polar_p(matrix: ArrayProtocol) -> ArrayProtocol:
    xp = array_backend(matrix)
    u, s, vt = xp.linalg.svd(matrix)
    v = vt.swapaxes(-1, -2) * s[..., None, :]
    return xp.matmul(v, vt)


def polar(
    matrix: ArrayProtocol, which: _PolarWhich = ("U", "P")
) -> ArrayProtocol:
    """Return the rotation of the polar decomposition of stacked matrices.

    A matrix `J = U S Vᵀ` has the polar decomposition `J = R P` with
    `R = U Vᵀ`. `R` is orthogonal, and it includes a reflection when `J`
    has a negative determinant.
    """
    if is_dask(matrix):
        if which == "U":
            return _polar_u(matrix)
        if which == "P":
            return _polar_p(matrix)
        return ({"U": _polar_u, "P": _polar_p}[each](matrix) for each in which)

    xp = array_backend(matrix)
    u, s, vt = xp.linalg.svd(matrix)
    if any("U" in _ for _ in which):
        u = xp.matmul(u, vt)
    if any("P" in _ for _ in which):
        v = vt.swapaxes(-1, -2) * s[..., None, :]
        p = xp.matmul(v, vt)
    if which == "U":
        return u
    if which == "P":
        return p
    return ({"U": u, "P": p}[each] for each in which)


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
    """Return whether all the values of an array are strictly positive."""
    xp = array_backend(array)
    return not bool(xp.any(array <= 0))


def require_positive(array: ArrayProtocol, what: str = "This") -> None:
    """Raise a [`DomainError`][] unless all values are strictly positive."""
    if not is_positive(array):
        raise DomainError(
            f"{what} is not defined: its values must all be positive, "
            f"but they are {to_numpy(array).tolist()}."
        )


def is_square(matrix: ArrayProtocol) -> bool:
    """Return whether an array is a square matrix or a stack of them.

    The last two dimensions of the array are the rows and the columns of
    each matrix, and the other dimensions index the matrices of a stack.
    """
    return matrix.ndim >= 2 and matrix.shape[-2] == matrix.shape[-1]


def require_square(matrix: ArrayProtocol, what: str = "This") -> None:
    """Raise a [`DomainError`][] unless a matrix is square.

    A stack of matrices passes when its matrices, which span the last two
    dimensions of the array, are square.
    """
    if matrix.ndim < 2:
        raise DomainError(f"{what} is not defined: it is not a matrix.")
    if not is_square(matrix):
        rows, cols = matrix.shape[-2:]
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
    xp = array_backend(linear)
    if not bool(xp.isfinite(linear).all()):
        return _BoolWithMessage(False, "is not finite", return_reason)
    eigenvalues = xp.linalg.eigvals(linear)
    size = xp.abs(eigenvalues)
    scale = size.max(initial=0.0)
    if bool(scale == 0.0) or bool((size <= _RTOL * scale).any()):
        return _BoolWithMessage(False, "is singular", return_reason)
    negative = (eigenvalues.real < 0) & (
        xp.abs(eigenvalues.imag) <= _RTOL * size
    )
    if bool(negative.any()):
        return _BoolWithMessage(
            False,
            "has an eigenvalue on the negative real axis",
            return_reason,
        )
    return _BoolWithMessage(True, "is principal", return_reason)


def require_principal(linear: np.ndarray, what: str = "This") -> None:
    """Raise a [`DomainError`][] unless a matrix is principal.

    A matrix is principal when it has a real principal square root and
    logarithm, as tested by [`is_principal`][]. Each matrix of a stack is
    tested on its own, and the message of the error names the first one
    that fails.
    """
    for index in np.ndindex(*linear.shape[:-2]):
        result = is_principal(linear[index], return_reason=True)
        if not result:
            where = f" at index {index}" if index else ""
            raise DomainError(
                f"{what} is not defined{where}: it {result}, "
                f"so it has no principal square root or logarithm."
            )


# --- backend utils ----------------------------------------------------


def _ensure_real(result: ArrayProtocol) -> ArrayProtocol:
    """Return the real part of an array whose imaginary part is negligible.

    The imaginary part counts as negligible when it is below `_RTOL` times
    the larger of one and the largest magnitude in the array. Otherwise, a
    [`DomainError`][] is raised.
    """
    xp = array_backend(result)
    if xp.iscomplexobj(result):
        scale = max(1.0, float(xp.abs(result).max(initial=0.0)))
        if float(xp.abs(result.imag).max(initial=0.0)) > _RTOL * scale:
            raise DomainError("Matrix is not real.")
        result = result.real
    return result
