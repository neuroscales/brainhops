"""Unit tests for compact affine operators against homogeneous baselines."""

import numpy as np

from brainhops._core import affines


def _to_homogeneous(A: np.ndarray) -> np.ndarray:
    """Convert compact affine (..., M, N+1) to homogeneous (..., M+1, N+1)."""
    *batch, m, np1 = A.shape
    H = np.zeros((*batch, m + 1, np1), dtype=A.dtype)
    H[..., :-1, :] = A
    H[..., -1, -1] = 1
    return H


def _from_homogeneous(H: np.ndarray) -> np.ndarray:
    """Convert homogeneous (..., M+1, N+1) to compact affine (..., M, N+1)."""
    return H[..., :-1, :]


def _random_affine(rng, m: int, n: int, batch=(), dtype=np.float64) -> np.ndarray:
    """Generate random compact affine with shape (*batch, m, n+1)."""
    A = rng.normal(size=(*batch, m, n + 1)).astype(dtype)
    if m == n:
        A[..., :, :-1] += np.eye(n, dtype=dtype)
    return A


def test_inv_matches_homogeneous_baseline_square_and_rectangular():
    rng = np.random.default_rng(0)

    # Square linear part -> true inverse
    A_sq = _random_affine(rng, m=3, n=3)
    got_sq = affines.inv(A_sq, backend="numpy")
    exp_sq = _from_homogeneous(np.linalg.inv(_to_homogeneous(A_sq)))
    np.testing.assert_allclose(got_sq, exp_sq, rtol=1e-10, atol=1e-10)

    # Rectangular linear part -> pseudoinverse baseline
    A_rect = _random_affine(rng, m=2, n=3)
    got_rect = affines.inv(A_rect, backend="numpy")
    exp_rect = _from_homogeneous(np.linalg.pinv(_to_homogeneous(A_rect)))
    np.testing.assert_allclose(got_rect, exp_rect, rtol=1e-10, atol=1e-10)


def test_matmul_matches_homogeneous_baseline_with_broadcast_and_chain():
    rng = np.random.default_rng(1)
    A = _random_affine(rng, m=2, n=3, batch=(4, 1))
    B = _random_affine(rng, m=3, n=4, batch=(1, 5))
    C = _random_affine(rng, m=4, n=2, batch=(1, 1))

    got = affines.matmul(A, B, C, backend="numpy")

    Ah = _to_homogeneous(A)
    Bh = _to_homogeneous(B)
    Ch = _to_homogeneous(C)
    exp = _from_homogeneous(np.matmul(np.matmul(Ah, Bh), Ch))

    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)


def test_matvec_matches_homogeneous_baseline_with_broadcast():
    rng = np.random.default_rng(2)
    A = _random_affine(rng, m=3, n=4, batch=(2, 1))
    b = rng.normal(size=(1, 5, 4))

    got = affines.matvec(A, b, backend="numpy")

    ones = np.ones((*b.shape[:-1], 1), dtype=b.dtype)
    bh = np.concatenate([b, ones], axis=-1)
    exp = np.matmul(_to_homogeneous(A), bh[..., None])[..., :-1, 0]

    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)


def test_lmdiv_vector_matches_homogeneous_baseline():
    rng = np.random.default_rng(3)
    A = _random_affine(rng, m=3, n=3, batch=(2,))
    b = rng.normal(size=(2, 3))

    got_vec = affines.lmdiv(A, b, vector=True, backend="numpy")

    Ah = _to_homogeneous(A)
    ones = np.ones((*b.shape[:-1], 1), dtype=b.dtype)
    bh = np.concatenate([b, ones], axis=-1)

    exp_vec = np.linalg.solve(Ah, bh[..., None])[..., :-1, 0]

    np.testing.assert_allclose(got_vec, exp_vec, rtol=1e-10, atol=1e-10)


def test_lmdiv_matrix_matches_homogeneous_baseline():
    rng = np.random.default_rng(33)
    A = _random_affine(rng, m=3, n=3, batch=(2,))
    B = _random_affine(rng, m=3, n=2, batch=(2,))

    got = affines.lmdiv(A, B, backend="numpy")

    Ah = _to_homogeneous(A)
    Bh = _to_homogeneous(B)
    exp = _from_homogeneous(np.linalg.solve(Ah, Bh))

    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-10)


def test_rmdiv_matches_homogeneous_baseline():
    rng = np.random.default_rng(4)

    # Matrix form: solve X @ B = A
    A = _random_affine(rng, m=2, n=3, batch=(3,))
    B = _random_affine(rng, m=3, n=3, batch=(3,))
    got_mat = affines.rmdiv(A, B, backend="numpy")

    Ah = _to_homogeneous(A)
    Bh = _to_homogeneous(B)
    exp_mat = _from_homogeneous(np.matmul(Ah, np.linalg.inv(Bh)))
    np.testing.assert_allclose(got_mat, exp_mat, rtol=1e-10, atol=1e-10)
