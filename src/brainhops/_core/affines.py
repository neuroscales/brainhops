"""
This module contains utilities for working with "compact" affine matrices,
i.e., M x (N+1) matrices that do not contain the homogeneous row.
"""

# stdlib
from math import prod
from types import ModuleType

import typing_extensions as _tx

# dependencies
from numpy import broadcast_shapes

# locals
from .backends import get_array_backend
from .typing import ArrayLike, ArrayProtocol


def inv(
    A: ArrayProtocol,
    *,
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """
    Invert a M x (N+1) affine matrix (i.e., that does not contain the
    homogeneous row), eventually batched.

    This function also returns an affine without the homogeneous row.

    Parameters
    ----------
    A : (..., M, N+1) array-like
        An M x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    backend : {"numpy", "cupy", "dask"}, optional
        The array backend to use. If `None`, the backend is inferred
        from the input array `A`.

    Returns
    -------
    C : (..., N, M+1) array-like
        The inverse of the input affine matrix `A`, without the
        homogeneous row.
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


def matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *Cs: ArrayProtocol,
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """
    Multiply two M x (N+1) affine matrices (i.e., that do not contain
    the homogeneous row), eventually batched.

    This function also returns an affine without the homogeneous row.

    Parameters
    ----------
    A : (..., M, N+1) array-like
        An M x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    B : (..., N, P+1) array-like
        An N x (P+1) affine matrix, without the homogeneous row,
        eventually batched.
    *Cs : (..., P, Q+1) array-like
        Additional affine matrices to multiply, eventually batched.
    backend : {"numpy", "cupy", "dask"}, optional
        The array backend to use. If `None`, the backend is inferred
        from the input array `A`.

    Returns
    -------
    C : (..., M, P+1) array-like
        The product of the input affine matrices `A` and `B`, without
        the homogeneous row.
    """
    if Cs:
        return _chain_matmul(A, B, *Cs, backend=backend)
    else:
        return _matmul(A, B, backend=backend)


def _matmul(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
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
    # Cost of multiplying two batched matrices, ignoring the size of
    # the output matrix.
    # This is used to determine the best order of multiplication for a
    # chain of matrices. This is a very naive implementation, but it should
    # at least ensure that we multiply non-batched matrices first.
    batch = broadcast_shapes(A.shape[:-2], B.shape[:-2])
    return prod(batch) * B.shape[-2]


def _chain_matmul(
    *As: ArrayProtocol,
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
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
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """
    Multiply an M x (N+1) affine matrix (i.e., that does not contain the
    homogeneous row) with an N-dimensional vector, eventually batched.

    Parameters
    ----------
    A : (..., M, N+1) array-like
        An M x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    b : (..., N) array-like
        An N-dimensional vector, eventually batched.
    backend : {"numpy", "cupy", "dask"}, optional
        The array backend to use. If `None`, the backend is inferred
        from the input array `A`.

    Returns
    -------
    c : (..., M) array-like
        The product of the input affine matrix `A` and vector `b`,
        without the homogeneous row.
    """
    backend = get_array_backend(backend or A)
    A = backend.asarray(A, dtype=A.dtype)
    b = backend.asarray(b, dtype=A.dtype)

    c = backend.linalg.matmul(A[..., :-1], b[..., None])[..., 0]
    c += A[..., -1]

    return c


def lmdiv(
    A: ArrayProtocol,
    B: ArrayProtocol,
    *,
    vector: bool = False,
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """
    Solve the linear system AX = B for X, where A is an M x (N+1) affine
    matrix (i.e., that does not contain the homogeneous row), and B is
    a M x (P+1) matrix, eventually batched.

    This is equivalent to `inv(A) @ B` in the homogeneous case.

    If `vector=True`, B is treated as a M-dimensional vector and the
    result will be a N-dimensional vector.

    Parameters
    ----------
    A : (..., M, N+1) array-like
        An M x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    B : (..., M, P+1) array-like | (..., M) array-like
        An M x (P+1) affine matrix, without the homogeneous row,
        eventually batched.
    vector : bool, optional
        If `True`, B is treated as a M-dimensional vector instead of a
        M x (P+1) matrix.
    backend : {"numpy", "cupy", "dask"}, optional
        The array backend to use. If `None`, the backend is inferred
        from the input array `A`.

    Returns
    -------
    C : (..., N, P+1) array-like | (..., N) array-like
        The product `inv(A) @ B`, without the homogeneous row.
    """
    if A.shape[-2] != A.shape[-1] - 1:
        # Non-square matrix -> fallback to inv (and therefore pinv)
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
    backend: _tx.Optional[_tx.Union[str, ModuleType]] = None,
) -> ArrayProtocol:
    """
    Solve the linear system XB = A for X, where A is an M x (N+1) affine
    matrix (i.e., that does not contain the homogeneous row), and B is
    a P x (N+1) affine matrix, eventually batched.

    This is equivalent to `A @ inv(B)` in the homogeneous case.

    Parameters
    ----------
    A : (..., M, N+1) array-like
        An M x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    B : (..., P, N+1) array-like
        A P x (N+1) affine matrix, without the homogeneous row,
        eventually batched.
    backend : {"numpy", "cupy", "dask"}, optional
        The array backend to use. If `None`, the backend is inferred
        from the input array `A`.

    Returns
    -------
    C : (..., M, P+1) array-like
        The product `B @ inv(A)`, without the homogeneous row.
    """
    if B.shape[-2] != B.shape[-1] - 1:
        # Non-square matrix -> fallback to inv (and therefore pinv)
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
