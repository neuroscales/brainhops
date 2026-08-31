import numpy as np
import typing_extensions as _tx
from scipy.ndimage import gaussian_filter


def inverse2d(disp: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a displacement field by interpreting it as a
    triangular mesh, where each triangle defines an affine transform.

    This is the method described in:
        "High-Dimensional Image Registration Using Symmetric Priors"
        Ashburner, Andersson & Friston. NeuroImage (1999).
        https://www.fil.ion.ucl.ac.uk/spm/doc/papers/john_high_dim.pdf

    Parameters
    ----------
    disp : np.ndarray
        The displacement field to invert (displacements are in voxels).
        Should be of shape (Nx, Ny, 2) .
        The last dimension should contain the displacements along each
         axis, in the same order (i.e. [x, y]).

    Returns
    -------
    np.ndarray
        The inverse displacement field, of the same shape as the input.

    """
    disp = np.asanyarray(disp)
    out = np.full_like(disp, np.nan)

    (Nx, Ny, Nd) = disp.shape
    if Nd != 2:
        raise ValueError(
            f"Expected a 2D displacement field with shape (Nx, Ny, 2), "
            f"but got shape {disp.shape}"
        )

    # generate meshgrid
    src = np.meshgrid(*(np.arange(s) for s in (Nx, Ny)), indexing='ij')
    src = np.stack(src, axis=-1)

    # Convert displacements to coordinates
    dst = src + disp

    # Extract the (batches of) thetraheda
    for src1, dst1 in zip(
        _yield_triangles(src),
        _yield_triangles(dst)
    ):
        # Batch process similar thetrahedra
        _process_triangle(src1, dst1, out)

    # Convert coordinates to displacements
    out -= src

    # Fill in missing values via smoothing
    msk = msk0 = np.isfinite(out)
    while not msk.all():
        out[~msk] = 0
        wgt = msk.astype(np.float64)
        sigma = 1 / np.sqrt(8 * np.log(2))  # FWHM = 1 voxel
        sigma = (sigma, sigma, 0)
        smo = gaussian_filter(out, sigma=sigma, mode='nearest')
        wgt = gaussian_filter(wgt, sigma=sigma, mode='nearest')
        smo /= wgt
        out[~msk0] = smo[~msk0]
        msk = np.isfinite(out)

    return out


# Constants to make reading the rest of the code easier
X, Y = 0, 1
BATCH_AXIS, VERTEX_AXIS, SPACE_AXIS = 0, 1, 2


def _process_triangle(src: np.ndarray,
                      dst: np.ndarray,
                      out: np.ndarray) -> None:
    """
    Process a batch of triangles.

    Parameters
    ----------
    src, dst : np.ndarray
        Triangles vertices in the source and target domains,
        with shape (N, 3, 2), where N is the batch size.
    out : np.ndarray
        The output array to write the results to, of shape (Nx, Ny, 2).

    """
    # sort triangle vertices along y axis
    idx = np.argsort(dst[:, :, Y:Y+1], axis=VERTEX_AXIS)
    tri = np.take_along_axis(dst, idx, axis=VERTEX_AXIS)

    # For each horizontal line, find its intersection with the triangle.
    # We start from the minimum integral y in the triangle.
    y = np.ceil(tri[:, 0, Y]).astype(np.int64)
    while True:

        mask0 = (0 <= y) & (y < out.shape[Y]) & (y <= tri[:, 2, Y])
        if not mask0.any():
            break

        upp_mask = (y <= tri[:, 2, Y])
        low_mask = (y <= tri[:, 1, Y])
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


def _process_segment(src: np.ndarray, dst: np.ndarray, y: np.ndarray,
                     seg: np.ndarray, out: np.ndarray) -> None:
    """
    Process a batch of segments in a given z plane and y coordinate.

    Parameters
    ----------
    src, dst : np.ndarray
        Thetrahedra vertices in the source and target domains,
        with shape (N, 4, 3), where N is the batch size.
    y : np.ndarray
        The y coordinate of the line being processed, with shape (N,).
    seg : np.ndarray
        The segment vertices in the target domain, with shape (N, 2, 1).
    out : np.ndarray
        The output array to write the results to, of shape (Nz, Ny, Nx, 3).

    """

    # sort segment vertices along x axis
    idx = np.argsort(seg[:, :, X:X+1], axis=VERTEX_AXIS)
    seg = np.take_along_axis(seg, idx, axis=VERTEX_AXIS)

    x = np.ceil(seg[:, 0, X]).astype(np.int64)
    while True:

        mask = (0 <= x) & (x < out.shape[X]) & (x <= seg[:, 1, X])
        if not mask.any():
            break

        # Compute the barycentric coordinate of the point being processed.
        xm, ym = x[mask], y[mask]
        vdst = np.stack((xm, ym), axis=-1)                 # (N, 2)
        bary = _barycoord(vdst, dst[mask])                 # (N, 3)

        # Compute the corresponding point in the source domain as the
        # barycentric mean of the tetrahedron vertices in the source domain.
        vsrc = np.einsum('ijk,ij->ik', src[mask], bary)    # (N, 2)

        # Assign the computed point to the output array
        out[xm, ym] = vsrc

        x += 1


def _barycoord(x: np.ndarray, tri: np.ndarray) -> np.ndarray:
    # Compute the barycentric coordinates of x with respect to the
    # triangle defined by its vertices.
    # * x is of shape (N, 2), where N is the number of voxels in the
    #   batch. The last dimension contains the (x,y).
    # * The triangle is defined by its vertices, with shape (N, 3, 2),
    #   where N is the number of triangles in the batch.
    # * The output is of shape (N, 3), where the last dimension contains
    #   the barycentric coordinates of x with respect to each vertex of
    #   the triangle.

    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]

    v01 = v1 - v0
    v02 = v2 - v0
    v0x = x - v0

    d00 = np.einsum('ij,ij->i', v01, v01)
    d01 = np.einsum('ij,ij->i', v01, v02)
    d11 = np.einsum('ij,ij->i', v02, v02)
    d20 = np.einsum('ij,ij->i', v0x, v01)
    d21 = np.einsum('ij,ij->i', v0x, v02)
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


def _truncate_and_stack2d(a: np.ndarray,
                          b: np.ndarray,
                          c: np.ndarray) -> np.ndarray:
    """
    Truncate arrays so that they have the same shape, then stack them.

    Parameters
    ----------
    a, b, c : np.ndarray
        The coordinates of the vertices of the triangles,
        with shape (Nx, Ny, 2).

    Returns
    -------
    np.ndarray
        With shape (N, 3, 2), where N=Nx*Ny is the number of
        triangles in the batch.
    """
    vertices = (a, b, c)
    nx, ny = (min(x.shape[i] for x in vertices) for i in range(2))
    vertices = (vertex[:nx, :ny].reshape(-1, 2) for vertex in vertices)
    return np.stack(tuple(vertices), axis=1)


def _yield_triangles(field: np.ndarray) -> _tx.Iterator[np.ndarray]:
    """
    Yield the vertices of the triangles defined by the displacement field,
    in batches of similar triangles (i.e. with the same pattern of vertices).

    Parameters
    ----------
    field : np.ndarray
        Coordinate field to process

    Yields
    ------
    np.ndarray
        The coordinates of the vertices of the triangles, with shape
        (N, 3, 2).
    """
    # We need to split the grid into a red-black checkerboard pattern.
    # We also want to extract triangles via slicing, which means we can
    # only batch triangles whose vertices are aligned on a cartesian grid.
    # We therefore split the input grid into 4 subgrids, and designate 2 of
    # them as "red" and the other 2 as "black".

    # =========== #
    #    R E D    #
    # =========== #

    # --- no shift

    x00 = field[0::2, 0::2]
    x01 = field[0::2, 1::2]
    x10 = field[1::2, 0::2]
    x11 = field[1::2, 1::2]

    yield from yield_red(x00, x01, x10, x11)

    # --- xy shift

    x00 = field[1::2, 1::2]
    x01 = field[1::2, 2::2]
    x10 = field[2::2, 1::2]
    x11 = field[2::2, 2::2]

    yield from yield_red(x00, x01, x10, x11)

    # =========== #
    #  B L A C K  #
    # =========== #

    # --- x shift

    x00 = field[1::2, 0::2]
    x01 = field[1::2, 1::2]
    x10 = field[2::2, 0::2]
    x11 = field[2::2, 1::2]

    yield from yield_black(x00, x01, x10, x11)

    # --- y shift

    x00 = field[0::2, 1::2]
    x01 = field[0::2, 2::2]
    x10 = field[1::2, 1::2]
    x11 = field[1::2, 2::2]

    yield from yield_black(x00, x01, x10, x11)


def yield_red(x00: np.ndarray, x01: np.ndarray,
              x10: np.ndarray, x11: np.ndarray) -> _tx.Iterator[np.ndarray]:
    # Yield the two triangles that make up a red block.
    #
    # #1  _____
    #    |     /    #2
    #    |   /    / |
    #    | /    /   |
    #         /_____|

    # tip = 00
    yield _truncate_and_stack2d(x00, x01, x10)

    # tip = 11
    yield _truncate_and_stack2d(x11, x01, x10)


def yield_black(x00: np.ndarray, x01: np.ndarray,
                x10: np.ndarray, x11: np.ndarray) -> _tx.Iterator[np.ndarray]:
    # Yield the two triangles that make up a black block.
    #
    #  #1      _____  #2
    #  | \    \     |
    #  |   \    \   |
    #  |_____\    \ |

    # tip = 01
    yield _truncate_and_stack2d(x01, x00, x11)

    # tip = 10
    yield _truncate_and_stack2d(x10, x00, x11)


def _generate_disp_field(shape: tuple,
                         magnitude: float = 1,
                         fwhm: float = 5) -> np.ndarray:
    # Generate a random displacement field of the given shape, for testing.
    from scipy.ndimage import gaussian_filter
    shape = tuple(shape) + (len(shape),)
    disp = (np.random.rand(*shape) * 2 - 1) * magnitude
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = (sigma,) * (len(shape) - 1) + (0,)
    disp = gaussian_filter(disp, sigma=sigma)
    return disp


def _identity_field(shape: tuple) -> np.ndarray:
    # Generate an identity coordinate field.
    grid = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')
    return np.stack(grid, axis=-1)


def _compose_fields(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    grid = _identity_field(field1.shape[:-1])
    coords = grid + field1
    coords = np.transpose(coords, (2, 0, 1))  # (2, Nx, Ny)

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
        plt.title('Forward')
        plt.subplot(1, 3, 2)
        plt.imshow(_disp2rgb(inv_disp, max=mx))
        plt.title('Inverse')
        plt.subplot(1, 3, 3)
        plt.imshow(_disp2rgb(comp_disp, max=mx))
        plt.title('Composition')
        plt.show()

    border = 1
    if border:
        comp_disp = comp_disp[border:-border, border:-border]

    assert np.allclose(comp_disp, 0, atol=1e-2)
