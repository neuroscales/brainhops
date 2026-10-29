"""Operations on compact affine matrices.

A compact affine is an `M x (N+1)` matrix that maps `N` input axes to `M`
output axes. It is the homogeneous matrix without its bottom row,
`[0, ..., 0, 1]`. All functions accept batches of matrices, stacked
along the leading dimensions, and return compact matrices.
"""

__all__ = [
    "to_homogeneous",
    "to_compact",
    "inv",
    "axis_scales",
    "matmul",
    "matvec",
    "lmdiv",
    "rmdiv",
    "expm",
    "logm",
    "sqrtm",
]

# stdlib
from math import prod
from types import ModuleType

# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol
from numpy import broadcast_shapes

# internal
from .backends import array_backend, promote_backends, to_backend
from .linalg import expm as _expm
from .linalg import logm as _logm
from .linalg import sqrtm as _sqrtm


def to_homogeneous(
    matrix: ArrayProtocol, tangent: bool = False
) -> ArrayProtocol:
    """Convert a compact affine to a square homogeneous matrix.

    The function appends the row `[0, ..., 0, 1]`. When `tangent` is true,
    the function appends the row `[0, ..., 0, 0]` instead, because the input
    is then a tangent, that is, the generator of an affine transform, whose
    bottom row is zero. A stack of matrices, of shape `(..., N, N+1)`,
    receives the row under each of its matrices.
    """
    xp = array_backend(matrix)
    nd = matrix.shape[-1] - 1
    corner = 0 if tangent else 1
    pad = xp.asarray([[0] * nd + [corner]])
    # A stack of matrices receives one padding row per matrix.
    pad = xp.broadcast_to(pad, tuple(matrix.shape[:-2]) + tuple(pad.shape))
    full = xp.concatenate([matrix, pad], axis=-2)
    return full


def to_compact(matrix: ArrayProtocol) -> ArrayProtocol:
    """Convert a homogeneous matrix to a compact affine.

    The bottom row, `[0, ..., 0, 1]`, is dropped.
    """
    return matrix[..., :-1, :]


def inv(
    A: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Invert compact affines.

    When the linear part is not square, its pseudo-inverse is used.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    backend : str or module, optional
        Array backend, given by name or as a module.
        By default, the backend is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact inverses, with shape `(..., N, M+1)`.
    """
    xp = array_backend(backend or A)
    A = xp.asarray(A)
    M, N = A.shape[-2:]
    N -= 1

    matinv = xp.linalg.pinv if N != M else xp.linalg.inv
    matmul = xp.matmul

    C = xp.empty(A.shape[:-2] + (N, M + 1), dtype=A.dtype)
    C[..., :-1] = matinv(A[..., :-1])
    C[..., -1:] = -matmul(C[..., :-1], A[..., -1:])

    return C


def axis_scales(
    A: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Return the scale of a compact affine along each input axis.

    The scale of an axis is the Euclidean norm of the corresponding column of
    the linear part. For a voxel-to-world affine, the scales are the voxel
    sizes, and a rotation leaves them unchanged.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    backend : str or module, optional
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The scales, with shape `(..., N)`.
    """
    xp = array_backend(backend or A)
    A = xp.asarray(A)
    linear = A[..., :-1]
    return (linear**2).sum(axis=-2) ** 0.5


def matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *Cs: ArrayProtocol,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
    zero_cancels_nan: bool = False,
) -> ArrayProtocol:
    """Multiply two or more compact affines.

    In a chain of more than two affines, the order of the products is chosen
    to keep the cost low, so that matrices without batch dimensions are
    multiplied first.

    The option `zero_cancels_nan` serves the projections that create a
    free axis, that is, an axis whose value is NaN in the translation of
    an affine. With the option, an exact zero coefficient of the left
    affine cancels a NaN in the translation of the right affine, while a
    non-zero coefficient carries the NaN into the translation of the
    product. The outputs that do not read a free axis are therefore
    unaffected by it. Without the option, NaN propagates as in an
    ordinary matrix product.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    B : ArrayProtocol
        Affines, with shape `(..., N, P+1)`.
    *Cs : ArrayProtocol
        Further affines, the first of which has shape `(..., P, Q+1)`.
    backend : str or module, optional
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.
    zero_cancels_nan : bool, default=False
        Whether an exact zero coefficient cancels a NaN in the
        translation of the affine that it multiplies.

    Returns
    -------
    ArrayProtocol
        The compact products, with shape `(..., M, P+1)` for two affines.
    """
    options = dict(backend=backend, zero_cancels_nan=zero_cancels_nan)
    if Cs:
        return _chain_matmul(A, B, *Cs, **options)
    else:
        return _matmul(A, B, **options)


def _matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
    zero_cancels_nan: bool = False,
) -> ArrayProtocol:
    backend = backend or promote_backends(A, B)
    xp = array_backend(backend)
    A = to_backend(A, backend)
    B = to_backend(B, backend)

    M, _ = A.shape[-2:]
    _, N = B.shape[-2:]
    N -= 1

    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])
    C = xp.empty(batch + (M, N + 1), dtype=A.dtype)

    Al, At = A[..., :-1], A[..., -1:]
    Bl, Bt = B[..., :-1], B[..., -1:]
    C[..., :-1] = xp.matmul(Al, Bl)

    # A Python scalar is not accepted by every backend (CuPy refuses it),
    # so the mask without NaN handling is an array of False.
    if zero_cancels_nan:
        is_nan = xp.isnan(Bt)
    else:
        is_nan = xp.zeros_like(Bt, dtype=bool)
    if not bool(xp.any(is_nan)):
        C[..., -1:] = xp.matmul(Al, Bt)
        C[..., -1:] += At
        return C

    # A free entry of the translation is replaced by zero in the product,
    # and every output whose row reads it with a non-zero coefficient is
    # set to NaN.
    finite = xp.where(is_nan, xp.zeros_like(Bt), Bt)
    reads = (Al != 0).astype(A.dtype)
    reads = xp.matmul(reads, is_nan.astype(A.dtype)) > 0
    nan = xp.full_like(At, float("nan"))
    C[..., -1:] = xp.where(reads, nan, xp.matmul(Al, finite) + At)
    return C


def _matmul_cost(A: ArrayProtocol, B: ArrayProtocol) -> int:
    # The cost ignores the size of the output. Its only purpose is to ensure
    # that the matrices without batch dimensions are multiplied first.
    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])
    return prod(batch) * B.shape[-2]


def _chain_matmul(
    *As: ArrayProtocol,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
    zero_cancels_nan: bool = False,
) -> ArrayProtocol:
    backend = backend or promote_backends(*As)
    options = dict(backend=backend, zero_cancels_nan=zero_cancels_nan)

    if not As:
        return None
    if len(As) == 1:
        return As[0]

    costs = [_matmul_cost(As[i], As[i + 1]) for i in range(len(As) - 1)]
    best_cost = min(range(len(costs)), key=lambda i: costs[i])
    A0 = As[:best_cost]
    A1 = As[best_cost:]
    return _chain_matmul(*A0, _matmul(*A1[:2], **options), *A1[2:], **options)


def matvec(
    A: ArrayProtocol,
    b: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Apply compact affines to vectors.

    The result is the product of the linear part and the vector, plus the
    translation.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    b : ArrayProtocol
        Vectors, with shape `(..., N)`.
    backend : str or module, optional
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The transformed vectors, with shape `(..., M)`.
    """
    backend = backend or promote_backends(A, b)

    xp = array_backend(backend)
    A = xp.asarray(A, dtype=A.dtype)
    b = xp.asarray(b, dtype=A.dtype)

    c = xp.matmul(A[..., :-1], b[..., None])[..., 0]
    c += A[..., -1]

    return c


def lmdiv(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    vector: bool = False,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Solve `A X = B` for compact affines, the left division of `B` by `A`.

    In homogeneous coordinates, the result equals `inv(A) @ B`. When the
    linear part of `A` is not square, the result is computed with the
    pseudo-inverse of `A`.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    B : ArrayProtocol
        Affines, with shape `(..., M, P+1)`, or vectors, with shape
        `(..., M)`.
    vector : bool, default=False
        Whether `B` holds vectors. A one-dimensional `B` is always treated
        as a vector.
    backend : str or module, optional
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact solutions, with shape `(..., N, P+1)`, or vectors with
        shape `(..., N)`.
    """
    backend = backend or promote_backends(A, B)

    if A.shape[-2] != A.shape[-1] - 1:
        # A system with a non-square linear part cannot be solved directly.
        return matmul(inv(A, backend=backend), B, backend=backend)

    xp = array_backend(backend or A)
    A = xp.asarray(A, dtype=A.dtype)
    B = xp.asarray(B, dtype=A.dtype)

    if B.ndim == 1:
        vector = True
    if vector:
        B = B[..., None]

    C = xp.linalg.solve(A[..., :-1], B)
    C[..., -1:] -= xp.linalg.solve(A[..., :-1], A[..., -1:])

    if vector:
        C = C[..., 0]

    return C


def rmdiv(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Solve `X B = A` for compact affines, the right division of `A` by `B`.

    In homogeneous coordinates, the result equals `A @ inv(B)`. When the
    linear part of `B` is not square, the result is computed with the
    pseudo-inverse of `B`.

    Parameters
    ----------
    A : ArrayProtocol
        Affines, with shape `(..., M, N+1)`.
    B : ArrayProtocol
        Affines, with shape `(..., P, N+1)`.
    backend : str or module, optional
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact solutions, with shape `(..., M, P+1)`.
    """
    backend = backend or promote_backends(A, B)

    if B.shape[-2] != B.shape[-1] - 1:
        # A system with a non-square linear part cannot be solved directly.
        return matmul(A, inv(B, backend=backend), backend=backend)

    xp = array_backend(backend)
    A = xp.asarray(A, dtype=A.dtype)
    B = xp.asarray(B, dtype=A.dtype)

    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])

    def t(X: ArrayLike) -> ArrayLike:
        return xp.swapaxes(X, -2, -1)

    M = A.shape[-2]
    P = B.shape[-2]
    C = xp.empty(batch + (M, P + 1), dtype=A.dtype)

    At = t(A[..., :-1])
    Bt = t(B[..., :-1])
    C[..., :-1] = t(xp.linalg.solve(Bt, At))
    C[..., -1:] = A[..., -1:]
    C[..., -1:] -= xp.matmul(C[..., :-1], B[..., -1:])

    return C


def expm(
    matrix: ArrayProtocol,
    what: str = "The exponential of this affine tangent",
) -> ArrayProtocol:
    """Return the exponential of a compact affine tangent.

    A tangent is the generator of an affine transform. Its compact form
    `[L, l]`, with shape `(N, N+1)`, stands for the square matrix
    `[[L, l], [0, ..., 0]]`, whose bottom row is zero. The function
    exponentiates that square matrix and returns the resulting affine in
    compact form. The linear part `L` may be singular, as in the pure
    translation `[0, t]`. The `what` argument names the operation in error
    messages.
    """
    return to_compact(_expm(to_homogeneous(matrix, tangent=True), what))


def logm(
    matrix: ArrayProtocol,
    what: str = "The logarithm of this affine",
) -> ArrayProtocol:
    """Return the principal logarithm of a compact affine.

    The input is an affine map, so it is completed with the row
    `[0, ..., 0, 1]` before the logarithm is taken. The bottom row of the
    logarithm is zero, and the function returns the remaining rows, which
    form a tangent `[L, l]`. The `what` argument names the operation in
    error messages.
    """
    return to_compact(_logm(to_homogeneous(matrix), what))


def sqrtm(
    matrix: ArrayProtocol,
    what: str = "The square root of this affine",
) -> ArrayProtocol:
    """Return the principal square root of a compact affine.

    Both the input and the root are affine maps, so their full matrices end
    with the row `[0, ..., 0, 1]`. The `what` argument names the operation
    in error messages.
    """
    return to_compact(_sqrtm(to_homogeneous(matrix), what))
