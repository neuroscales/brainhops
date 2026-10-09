import numpy as np
import typing_extensions as _tx
from scipy.ndimage import gaussian_filter


def inverse2d(disp: np.ndarray) -> np.ndarray:
    """Invert a two-dimensional displacement field.

    The field is treated as a triangular mesh in which each triangle defines
    an affine transform, following Ashburner, Andersson and Friston,
    "High-Dimensional Image Registration Using Symmetric Priors", NeuroImage,
    1999 (https://www.fil.ion.ucl.ac.uk/spm/doc/papers/john_high_dim.pdf).
    The points of the output that no triangle covers are filled in by
    repeatedly smoothing the values of their neighbours.

    Parameters
    ----------
    disp : np.ndarray
        Displacements in voxels, with shape `(Nx, Ny, 2)` and the last axis
        ordered `[x, y]`.

    Returns
    -------
    np.ndarray
        The inverse field, with the same shape as `disp`.

    Raises
    ------
    ValueError
        If the last axis of `disp` does not have length 2.
    """
    disp = np.asanyarray(disp)
    out = np.full_like(disp, np.nan)

    (Nx, Ny, Nd) = disp.shape
    if Nd != 2:
        raise ValueError(
            f"Expected a 2D displacement field with shape (Nx, Ny, 2), "
            f"but got shape {disp.shape}"
        )

    src = np.meshgrid(*(np.arange(s) for s in (Nx, Ny)), indexing="ij")
    src = np.stack(src, axis=-1)

    dst = src + disp

    for src1, dst1 in zip(_yield_triangles(src), _yield_triangles(dst)):
        _process_triangle(src1, dst1, out)

    out -= src

    # Fill the points that no triangle covers by iterative Gaussian
    # smoothing, normalised by the smoothed mask.
    msk = msk0 = np.isfinite(out)
    while not msk.all():
        out[~msk] = 0
        wgt = msk.astype(np.float64)
        sigma = 1 / np.sqrt(8 * np.log(2))  # FWHM of one voxel
        sigma = (sigma, sigma, 0)
        smo = gaussian_filter(out, sigma=sigma, mode="nearest")
        wgt = gaussian_filter(wgt, sigma=sigma, mode="nearest")
        smo /= wgt
        out[~msk0] = smo[~msk0]
        msk = np.isfinite(out)

    return out


X, Y = 0, 1
BATCH_AXIS, VERTEX_AXIS, SPACE_AXIS = 0, 1, 2


def _process_triangle(
    src: np.ndarray, dst: np.ndarray, out: np.ndarray
) -> None:
    """Rasterise a batch of triangles into `out`, in place.

    The vertices `src` and `dst`, in the source and target domains, have
    shape `(N, 3, 2)`, and `out` has shape `(Nx, Ny, 2)`.
    """
    idx = np.argsort(dst[:, :, Y : Y + 1], axis=VERTEX_AXIS)
    tri = np.take_along_axis(dst, idx, axis=VERTEX_AXIS)

    # Scan the horizontal lines that cross each triangle, from the smallest
    # integer y inside it.
    y = np.ceil(tri[:, 0, Y]).astype(np.int64)
    while True:
        mask0 = (0 <= y) & (y < out.shape[Y]) & (y <= tri[:, 2, Y])
        if not mask0.any():
            break

        upp_mask = y <= tri[:, 2, Y]
        low_mask = y <= tri[:, 1, Y]
        upp_mask &= ~low_mask
        upp_mask &= mask0
        low_mask &= mask0
        del mask0

        if low_mask.any():
            ym = y[low_mask]
            srcm, dstm = src[low_mask], dst[low_mask]
            seg = _find_segment(tri[low_mask], ym)
            _process_segment(srcm, dstm, ym, seg, out)

        if upp_mask.any():
            ym = y[upp_mask]
            srcm, dstm = src[upp_mask], dst[upp_mask]
            seg = _find_segment(tri[upp_mask][:, ::-1], ym)
            _process_segment(srcm, dstm, ym, seg, out)

        y += 1


def _process_segment(
    src: np.ndarray,
    dst: np.ndarray,
    y: np.ndarray,
    seg: np.ndarray,
    out: np.ndarray,
) -> None:
    """Rasterise a batch of segments that lie on horizontal lines.

    The triangles `src` and `dst` have shape `(N, 3, 2)`. The array `y`,
    with shape `(N,)`, holds the y coordinate of the line on which each
    segment lies, and `seg`, with shape `(N, 2, 1)`, holds the x
    coordinates of the end points of each segment in the target domain.
    The output `out` has shape `(Nx, Ny, 2)` and is written in place.
    """

    idx = np.argsort(seg[:, :, X : X + 1], axis=VERTEX_AXIS)
    seg = np.take_along_axis(seg, idx, axis=VERTEX_AXIS)

    x = np.ceil(seg[:, 0, X]).astype(np.int64)
    while True:
        mask = (0 <= x) & (x < out.shape[X]) & (x <= seg[:, 1, X])
        if not mask.any():
            break

        xm, ym = x[mask], y[mask]
        vdst = np.stack((xm, ym), axis=-1)
        bary = _barycoord(vdst, dst[mask])

        # The source point is the mean of the source vertices, weighted by
        # the barycentric coordinates of the target point.
        vsrc = np.einsum("ijk,ij->ik", src[mask], bary)

        out[xm, ym] = vsrc

        x += 1


def _barycoord(x: np.ndarray, tri: np.ndarray) -> np.ndarray:
    # Compute the barycentric coordinates, of shape (N, 3), of the points
    # x, of shape (N, 2), in the triangles tri, of shape (N, 3, 2).

    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]

    v01 = v1 - v0
    v02 = v2 - v0
    v0x = x - v0

    d00 = np.einsum("ij,ij->i", v01, v01)
    d01 = np.einsum("ij,ij->i", v01, v02)
    d11 = np.einsum("ij,ij->i", v02, v02)
    d20 = np.einsum("ij,ij->i", v0x, v01)
    d21 = np.einsum("ij,ij->i", v0x, v02)
    dt = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / dt
    w = (d00 * d21 - d01 * d20) / dt
    u = 1 - v - w

    return np.stack((u, v, w), axis=-1)


def _find_segment(tri: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(tri, shape=(len(tri), 2, 1))

    p1x, p1y = tri[:, 0].T
    p2x, p2y = tri[:, 1].T
    p3x, p3y = tri[:, 2].T

    ry = (y - p1y) / (p2y - p1y)
    out[:, 0, X] = p1x + (p2x - p1x) * ry
    ry = (y - p1y) / (p3y - p1y)
    out[:, 1, X] = p1x + (p3x - p1x) * ry

    return out


def _truncate_and_stack2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Crop three vertex arrays to a common shape and stack them as triangles.

    The result has shape `(N, 3, 2)`, with one triangle per point of the
    common `(Nx, Ny)` grid.
    """
    vertices = (a, b, c)
    nx, ny = (min(x.shape[i] for x in vertices) for i in range(2))
    vertices = (vertex[:nx, :ny].reshape(-1, 2) for vertex in vertices)
    return np.stack(tuple(vertices), axis=1)


def _yield_triangles(field: np.ndarray) -> _tx.Iterator[np.ndarray]:
    """Yield the vertices of the triangles of a coordinate field.

    The triangles are yielded in batches that share the same pattern of
    vertices, as arrays of shape `(N, 3, 2)`.
    """
    # The grid is split into a red-black checkerboard of four subgrids, two
    # red and two black, so that triangles aligned on a Cartesian grid can be
    # extracted in batches by slicing.

    # =========== #
    #    R E D    #
    # =========== #

    x00 = field[0::2, 0::2]
    x01 = field[0::2, 1::2]
    x10 = field[1::2, 0::2]
    x11 = field[1::2, 1::2]

    yield from yield_red(x00, x01, x10, x11)

    x00 = field[1::2, 1::2]
    x01 = field[1::2, 2::2]
    x10 = field[2::2, 1::2]
    x11 = field[2::2, 2::2]

    yield from yield_red(x00, x01, x10, x11)

    # =========== #
    #  B L A C K  #
    # =========== #

    x00 = field[1::2, 0::2]
    x01 = field[1::2, 1::2]
    x10 = field[2::2, 0::2]
    x11 = field[2::2, 1::2]

    yield from yield_black(x00, x01, x10, x11)

    x00 = field[0::2, 1::2]
    x01 = field[0::2, 2::2]
    x10 = field[1::2, 1::2]
    x11 = field[1::2, 2::2]

    yield from yield_black(x00, x01, x10, x11)


def yield_red(
    x00: np.ndarray, x01: np.ndarray, x10: np.ndarray, x11: np.ndarray
) -> _tx.Iterator[np.ndarray]:
    # The two triangles of a red block:
    # #1  _____
    #    |     /    #2
    #    |   /    / |
    #    | /    /   |
    #         /_____|

    # Triangle 1 has its tip at x00.
    yield _truncate_and_stack2d(x00, x01, x10)

    # Triangle 2 has its tip at x11.
    yield _truncate_and_stack2d(x11, x01, x10)


def yield_black(
    x00: np.ndarray, x01: np.ndarray, x10: np.ndarray, x11: np.ndarray
) -> _tx.Iterator[np.ndarray]:
    # The two triangles of a black block:
    #  #1      _____  #2
    #  | \    \     |
    #  |   \    \   |
    #  |_____\    \ |

    # Triangle 1 has its tip at x01.
    yield _truncate_and_stack2d(x01, x00, x11)

    # Triangle 2 has its tip at x10.
    yield _truncate_and_stack2d(x10, x00, x11)


def _generate_disp_field(
    shape: tuple, magnitude: float = 1, fwhm: float = 5
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    shape = tuple(shape) + (len(shape),)
    disp = (np.random.rand(*shape) * 2 - 1) * magnitude
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = (sigma,) * (len(shape) - 1) + (0,)
    disp = gaussian_filter(disp, sigma=sigma)
    return disp


def _identity_field(shape: tuple) -> np.ndarray:
    grid = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    return np.stack(grid, axis=-1)


def _compose_fields(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    grid = _identity_field(field1.shape[:-1])
    coords = grid + field1
    coords = np.transpose(coords, (2, 0, 1))

    out = np.empty_like(field1)
    for i in range(field1.shape[-1]):
        out[..., i] = map_coordinates(field2[..., i], coords, order=1)
    out += field1

    return out


def _disp2rgb(disp: np.ndarray, max: np.ndarray = None) -> np.ndarray:
    if max is None:
        max = np.abs(disp).max()
    _disp = disp
    disp = np.zeros_like(disp, shape=disp.shape[:-1] + (3,))
    disp[..., :2] = _disp
    disp = np.clip(disp / max, -1, 1)
    disp = (disp + 1) / 2
    return (disp * 255).astype(np.uint8)


def _test_inverse2d(plot: bool = True) -> None:
    shape = (64,) * 2
    disp = _generate_disp_field(shape, magnitude=10, fwhm=16)
    inv_disp = inverse2d(disp)
    comp_disp = _compose_fields(disp, inv_disp)

    mx = np.abs(disp).max()

    if plot:
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 1)
        plt.imshow(_disp2rgb(disp, max=mx))
        plt.title("Forward")
        plt.subplot(1, 3, 2)
        plt.imshow(_disp2rgb(inv_disp, max=mx))
        plt.title("Inverse")
        plt.subplot(1, 3, 3)
        plt.imshow(_disp2rgb(comp_disp, max=mx))
        plt.title("Composition")
        plt.show()

    border = 1
    if border:
        comp_disp = comp_disp[border:-border, border:-border]

    assert np.allclose(comp_disp, 0, atol=1e-2)
