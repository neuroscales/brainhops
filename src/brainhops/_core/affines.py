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

from math import prod
from types import ModuleType

import typing_extensions as tx
from bagof.hints.array import ArrayLike, ArrayProtocol
from numpy import broadcast_shapes

from brainhops.backends import get_array_backend

from .linalg import expm as _expm
from .linalg import logm as _logm
from .linalg import sqrtm as _sqrtm


def to_homogeneous(
    matrix: ArrayProtocol, tangent: bool = False
) -> ArrayProtocol:
    """Convert a compact affine to a square homogeneous matrix.

    The row `[0, ..., 0, 1]` is appended, or the row `[0, ..., 0, 0]` when
    `tangent` is true, which is the bottom row of the generator of an
    affine transform.
    """
    nx = get_array_backend(matrix)
    nd = matrix.shape[-1] - 1
    corner = 0 if tangent else 1
    pad = nx.asarray([[0] * nd + [corner]])
    full = nx.concatenate([matrix, pad], axis=-2)
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
    A
        Affines, with shape `(..., M, N+1)`.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact inverses, with shape `(..., N, M+1)`.
    """
    backend = get_array_backend(backend or A)
    M, N = A.shape[-2:]
    N -= 1

    matinv = backend.linalg.pinv if N != M else backend.linalg.inv
    matmul = backend.matmul

    C = backend.empty(A.shape[:-2] + (N, M + 1), dtype=A.dtype)
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
    A
        Affines, with shape `(..., M, N+1)`.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The scales, with shape `(..., N)`.
    """
    backend = get_array_backend(backend or A)
    A = backend.asarray(A)
    linear = A[..., :-1]
    return (linear**2).sum(axis=-2) ** 0.5


def matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *Cs: ArrayProtocol,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """Multiply two or more compact affines.

    In a chain of more than two affines, the order of the products is chosen
    to keep the cost low, so that matrices without batch dimensions are
    multiplied first.

    Parameters
    ----------
    A
        Affines, with shape `(..., M, N+1)`.
    B
        Affines, with shape `(..., N, P+1)`.
    *Cs
        Further affines, the first of which has shape `(..., P, Q+1)`.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact products, with shape `(..., M, P+1)` for two affines.
    """
    if Cs:
        return _chain_matmul(A, B, *Cs, backend=backend)
    else:
        return _matmul(A, B, backend=backend)


def _matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    backend = get_array_backend(backend or A)
    A = backend.asarray(A, dtype=A.dtype)
    B = backend.asarray(B, dtype=A.dtype)

    M, _ = A.shape[-2:]
    _, N = B.shape[-2:]
    N -= 1

    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])
    C = backend.empty(batch + (M, N + 1), dtype=A.dtype)

    C[..., :-1] = backend.matmul(A[..., :-1], B[..., :-1])
    C[..., -1:] = backend.matmul(A[..., :-1], B[..., -1:])
    C[..., -1:] += A[..., -1:]

    return C


def _matmul_cost(A: ArrayProtocol, B: ArrayProtocol) -> int:
    # The cost ignores the size of the output; it only serves to multiply
    # the matrices without batch dimensions first.
    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])
    return prod(batch) * B.shape[-2]


def _chain_matmul(
    *As: ArrayProtocol,
    backend: tx.Optional[tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    if not As:
        return None
    if len(As) == 1:
        return As[0]

    costs = [_matmul_cost(As[i], As[i + 1]) for i in range(len(As) - 1)]
    best_cost = min(range(len(costs)), key=lambda i: costs[i])
    A0 = As[:best_cost]
    A1 = As[best_cost:]
    return _chain_matmul(
        *A0, _matmul(*A1[:2], backend=backend), *A1[2:], backend=backend
    )


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
    A
        Affines, with shape `(..., M, N+1)`.
    b
        Vectors, with shape `(..., N)`.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The transformed vectors, with shape `(..., M)`.
    """
    backend = get_array_backend(backend or A)
    A = backend.asarray(A, dtype=A.dtype)
    b = backend.asarray(b, dtype=A.dtype)

    c = backend.matmul(A[..., :-1], b[..., None])[..., 0]
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
    A
        Affines, with shape `(..., M, N+1)`.
    B
        Affines, with shape `(..., M, P+1)`, or vectors, with shape
        `(..., M)`.
    vector
        Whether `B` holds vectors. A one-dimensional `B` is always treated
        as a vector.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact solutions, with shape `(..., N, P+1)`, or vectors with
        shape `(..., N)`.
    """
    if A.shape[-2] != A.shape[-1] - 1:
        # A system with a non-square linear part cannot be solved directly.
        return matmul(inv(A, backend=backend), B, backend=backend)

    backend = get_array_backend(backend or A)
    A = backend.asarray(A, dtype=A.dtype)
    B = backend.asarray(B, dtype=A.dtype)

    if B.ndim == 1:
        vector = True
    if vector:
        B = B[..., None]

    C = backend.linalg.solve(A[..., :-1], B)
    C[..., -1:] -= backend.linalg.solve(A[..., :-1], A[..., -1:])

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
    A
        Affines, with shape `(..., M, N+1)`.
    B
        Affines, with shape `(..., P, N+1)`.
    backend
        Array backend, given by name or as a module. By default, the backend
        is inferred from `A`.

    Returns
    -------
    ArrayProtocol
        The compact solutions, with shape `(..., M, P+1)`.
    """
    if B.shape[-2] != B.shape[-1] - 1:
        # A system with a non-square linear part cannot be solved directly.
        return matmul(A, inv(B, backend=backend), backend=backend)

    backend = get_array_backend(backend or A)
    A = backend.asarray(A, dtype=A.dtype)
    B = backend.asarray(B, dtype=A.dtype)

    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])

    def t(X: ArrayLike) -> ArrayLike:
        return backend.swapaxes(X, -2, -1)

    M = A.shape[-2]
    P = B.shape[-2]
    C = backend.empty(batch + (M, P + 1), dtype=A.dtype)

    At = t(A[..., :-1])
    Bt = t(B[..., :-1])
    C[..., :-1] = t(backend.linalg.solve(Bt, At))
    C[..., -1:] = A[..., -1:]
    C[..., -1:] -= backend.matmul(C[..., :-1], B[..., -1:])

    return C


def expm(
    matrix: ArrayProtocol,
    what: str = "The exponential of this affine tangent",
) -> ArrayProtocol:
    """Return the exponential of a compact affine tangent.

    A tangent `[L, l]`, with shape `(N, N+1)`, is the top of the generator
    `[[L, l], [0, ..., 0]]`, and the result is the top of the homogeneous
    affine that the generator exponentiates to. The linear part `L` may be
    singular, as in the pure translation `[0, t]`. The `what` argument names
    the operation in error messages.
    """
    return to_compact(_expm(to_homogeneous(matrix, tangent=True), what))


def logm(
    matrix: ArrayProtocol,
    what: str = "The logarithm of this affine",
) -> ArrayProtocol:
    """Return the principal logarithm of a compact affine.

    The input is a map, so it is completed with the affine row
    `[0, ..., 0, 1]`. The result is a tangent `[L, l]`, the top of the
    logarithm, whose last row is zero. The `what` argument names the
    operation in error messages.
    """
    return to_compact(_logm(to_homogeneous(matrix), what))


def sqrtm(
    matrix: ArrayProtocol,
    what: str = "The square root of this affine",
) -> ArrayProtocol:
    """Return the principal square root of a compact affine.

    Both the input and the root are maps, completed with the affine row
    `[0, ..., 0, 1]`. The `what` argument names the operation in error
    messages.
    """
    return to_compact(_sqrtm(to_homogeneous(matrix), what))
