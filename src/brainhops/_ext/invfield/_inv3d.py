# This module inverts a dense displacement field.
#
# The method is explained here in two dimensions; in three dimensions,
# tetrahedra take the place of triangles. The drawing below shows a grid of
# 5 x 5 vertices, each labelled by its (i,j) index:
#
# (0,0) - (0,1) - (0,2) - (0,3) - (0,4)
#   |  \    |    /  |  \    |    /  |
#   |   \   |   /   |   \   |   /   |
#   |    \  |  /    |    \  |  /    |
# (1,0) - (1,1) - (1,2) - (1,3) - (1,4)
#   |    /  |  \    |    /  |  \    |
#   |   /   |   \   |   /   |   \   |
#   |  /    |    \  |  /    |    \  |
# (2,0) - (2,1) - (2,2) - (2,3) - (2,4)
#   |  \    |    /  |  \    |    /  |
#   |   \   |   /   |   \   |   /   |
#   |    \  |  /    |    \  |  /    |
# (3,0) - (3,1) - (3,2) - (3,3) - (3,4)
#   |    /  |  \    |    /  |  \    |
#   |   /   |   \   |   /   |   \   |
#   |  /    |    \  |  /    |    \  |
# (4,0) - (4,1) - (4,2) - (4,3) - (4,4)
#
# The grid is split into unit cells coloured red and black like a
# checkerboard. For example, the cell {(0,0),(0,1),(1,0),(1,1)} is red, and
# its neighbours {(0,1),(0,2),(1,1),(1,2)} and {(1,0),(1,1),(2,0),(2,1)}
# are black. Each cell is cut into two triangles along one of its
# diagonals, and the diagonal alternates between neighbouring cells. A red
# cell is cut from its top-left to its bottom-right corner, which gives the
# triangles {(0,0),(1,0),(1,1)} and {(0,0),(0,1),(1,1)}. A black cell is
# cut from its top-right to its bottom-left corner, which gives the
# triangles {(1,1),(0,1),(0,2)} and {(1,1),(1,2),(0,2)}.
#
# Because the pattern is regular, batches of triangles that share the same
# vertex layout can be extracted by slicing the field.
# The inverse value at a voxel inside a displaced triangle is the mean of
# the original vertices, weighted by the barycentric coordinates of the
# voxel. In three dimensions, the cells are cubes cut into five tetrahedra.
#
# Reference: J. Ashburner, J. L. R. Andersson and K. J. Friston, "Image
# Registration Using a Symmetric Prior - in Three Dimensions", Human Brain
# Mapping, 2000.
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf
import numpy as np
import typing_extensions as _tx
from scipy.ndimage import gaussian_filter


def inverse3d(disp: np.ndarray) -> np.ndarray:
    """Invert a three-dimensional displacement field.

    The voxel grid is treated as a tetrahedral mesh in which each
    tetrahedron defines an affine map, following the appendix of Ashburner,
    Andersson and Friston (Human Brain Mapping, 2000,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf).
    Voxels that no displaced tetrahedron covers are filled by smoothing
    their neighbours.

    Parameters
    ----------
    disp : (Nx, Ny, Nz, 3) np.ndarray
        Displacement field in voxels, with the last axis ordered x, y, z.

    Returns
    -------
    np.ndarray
        Inverse displacement field, with the same shape as `disp`.

    Raises
    ------
    ValueError
        If the last dimension of `disp` does not have length 3.
    """
    disp = np.asanyarray(disp)
    out = np.full_like(disp, np.nan)

    (Nx, Ny, Nz, Nd) = disp.shape
    if Nd != 3:
        raise ValueError(
            f"Expected a 3D displacement field with shape (Nx, Ny, Nz, 3), "
            f"but got shape {disp.shape}"
        )

    src = np.meshgrid(*(np.arange(s) for s in (Nx, Ny, Nz)), indexing="ij")
    src = np.stack(src, axis=-1)

    dst = src + disp

    # Tetrahedra come in batches that share the same vertex pattern,
    # so that each batch can be processed in a vectorized way.
    for src1, dst1 in zip(_yield_thetrahedra(src), _yield_thetrahedra(dst)):
        _process_thetrahedron(src1, dst1, out)

    out -= src

    # Fill the voxels that no tetrahedron covers by normalized
    # Gaussian smoothing, repeated until every voxel is finite.
    msk = msk0 = np.isfinite(out)
    while not msk.all():
        out[~msk] = 0
        wgt = msk.astype(np.float64)
        sigma = 1 / np.sqrt(8 * np.log(2))  # FWHM of one voxel
        sigma = (sigma, sigma, sigma, 0)
        smo = gaussian_filter(out, sigma=sigma, mode="nearest")
        wgt = gaussian_filter(wgt, sigma=sigma, mode="nearest")
        smo /= wgt
        out[~msk0] = smo[~msk0]
        msk = np.isfinite(out)

    return out


# Names for the spatial axes and for the axes of vertex batches.
X, Y, Z = 0, 1, 2
BATCH_AXIS, VERTEX_AXIS, SPACE_AXIS = 0, 1, 2


def _process_thetrahedron(
    src: np.ndarray, dst: np.ndarray, out: np.ndarray
) -> None:
    """Fill the voxels that a batch of displaced tetrahedra covers.

    Each voxel that lies inside a tetrahedron in the target domain receives
    the coordinates of the corresponding point in the source domain. The
    arrays `src` and `dst` have shape (N, 4, 3) and hold the vertices of
    the tetrahedra in the source and target domains, and `out` is written
    in place.
    """
    idx = np.argsort(dst[:, :, Z : Z + 1], axis=VERTEX_AXIS)
    ttr = np.take_along_axis(dst, idx, axis=VERTEX_AXIS)

    # Sweep the integer z planes upwards. Depending on which vertices
    # a plane falls between, its section through the tetrahedron is a
    # lower triangle, a quadrilateral or an upper triangle.
    z = np.ceil(ttr[:, 0, Z]).astype(np.int64)
    while True:
        mask0 = (0 <= z) & (z < out.shape[Z]) & (z <= ttr[:, 3, Z])
        if not mask0.any():
            break

        upp_mask = z <= ttr[:, 3, Z]
        mid_mask = z <= ttr[:, 2, Z]
        low_mask = z <= ttr[:, 1, Z]
        upp_mask &= ~mid_mask
        mid_mask &= ~low_mask
        upp_mask &= mask0
        mid_mask &= mask0
        low_mask &= mask0
        del mask0

        # When the plane lies between vertices 0 and 1, the section is a
        # triangle.
        if low_mask.any():
            zm, srcm, dstm = z[low_mask], src[low_mask], dst[low_mask]
            tri = _find_lower_triangle(ttr[low_mask], zm)
            _process_triangle(srcm, dstm, zm, tri, out)

        # When the plane lies between vertices 1 and 2, the section is a
        # quadrilateral, which is cut into two triangles.
        if mid_mask.any():
            zm, srcm, dstm = z[mid_mask], src[mid_mask], dst[mid_mask]
            quad = _find_quadrilateral(ttr[mid_mask], zm)
            _process_triangle(srcm, dstm, zm, quad[:, 0:3], out)
            _process_triangle(srcm, dstm, zm, quad[:, 1:4], out)

        # When the plane lies between vertices 2 and 3, the section is a
        # triangle.
        if upp_mask.any():
            zm, srcm, dstm = z[upp_mask], src[upp_mask], dst[upp_mask]
            tri = _find_upper_triangle(ttr[upp_mask], zm)
            _process_triangle(srcm, dstm, zm, tri, out)

        z += 1


def _process_triangle(
    src: np.ndarray,
    dst: np.ndarray,
    z: np.ndarray,
    tri: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill the voxels that a batch of triangles in the planes `z` covers.

    The array `tri` has shape (N, 3, 2) and holds the (x, y) coordinates of
    the vertices of the triangles in the target domain. The other arguments
    are the same as in `_process_thetrahedron`.
    """

    idx = np.argsort(tri[:, :, Y : Y + 1], axis=VERTEX_AXIS)
    tri = np.take_along_axis(tri, idx, axis=VERTEX_AXIS)

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
            zm, ym = z[low_mask], y[low_mask]
            srcm, dstm = src[low_mask], dst[low_mask]
            seg = _find_segment(tri[low_mask], ym)
            _process_segment(srcm, dstm, zm, ym, seg, out)

        if upp_mask.any():
            zm, ym = (
                z[upp_mask],
                y[upp_mask],
            )
            srcm, dstm = src[upp_mask], dst[upp_mask]
            seg = _find_segment(tri[upp_mask][:, ::-1], ym)
            _process_segment(srcm, dstm, zm, ym, seg, out)

        y += 1


def _process_segment(
    src: np.ndarray,
    dst: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    seg: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill the voxels that a batch of segments on the rows `y` covers.

    The array `seg` has shape (N, 2, 1) and holds the x coordinates of the
    endpoints in the target domain. The other arguments are the same as in
    `_process_triangle`.
    """

    idx = np.argsort(seg[:, :, X : X + 1], axis=VERTEX_AXIS)
    seg = np.take_along_axis(seg, idx, axis=VERTEX_AXIS)

    x = np.ceil(seg[:, 0, X]).astype(np.int64)
    while True:
        mask = (0 <= x) & (x < out.shape[X]) & (x <= seg[:, 1, X])
        if not mask.any():
            break

        # Compute the barycentric coordinates of each voxel in its
        # tetrahedron.
        xm, ym, zm = x[mask], y[mask], z[mask]
        vdst = np.stack((xm, ym, zm), axis=-1)  # (N, 3)
        bary = _barycoord(vdst, dst[mask])  # (N, 4)

        # The source-domain point is the mean of the source vertices,
        # weighted by the barycentric coordinates.
        vsrc = np.einsum("ijk,ij->ik", src[mask], bary)  # (N, 3)

        out[xm, ym, zm] = vsrc

        x += 1


def _barycoord(x: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    # Return the barycentric coordinates, of shape (N, 4), of the points
    # `x`, of shape (N, 3), with respect to the tetrahedra `tetra`, of
    # shape (N, 4, 3).

    v0 = tetra[:, 0]
    v1 = tetra[:, 1]
    v2 = tetra[:, 2]
    v3 = tetra[:, 3]

    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0
    v0x = x - v0

    dt = np.einsum("ij,ij->i", v01, np.cross(v02, v03))
    b1 = np.einsum("ij,ij->i", v0x, np.cross(v02, v03)) / dt
    b2 = np.einsum("ij,ij->i", v01, np.cross(v0x, v03)) / dt
    b3 = np.einsum("ij,ij->i", v01, np.cross(v02, v0x)) / dt
    b0 = 1.0 - b1 - b2 - b3
    bb = np.stack((b0, b1, b2, b3), axis=-1)
    return bb


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


def _find_lower_triangle(dst: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Return the triangle in which the plane `z` cuts each tetrahedron,
    # when the plane lies between the lowest vertex and the next one. The
    # corners of the triangle lie on the three edges that leave the lowest
    # vertex.

    out = np.empty_like(dst, shape=(len(dst), 3, 2))

    v0x, v0y, v0z = dst[:, 0].T
    v1x, v1y, v1z = dst[:, 1].T
    v2x, v2y, v2z = dst[:, 2].T
    v3x, v3y, v3z = dst[:, 3].T

    rz = (z - v0z) / (v1z - v0z)
    out[:, 0, X] = v0x + (v1x - v0x) * rz
    out[:, 0, Y] = v0y + (v1y - v0y) * rz
    rz = (z - v0z) / (v2z - v0z)
    out[:, 1, X] = v0x + (v2x - v0x) * rz
    out[:, 1, Y] = v0y + (v2y - v0y) * rz
    rz = (z - v0z) / (v3z - v0z)
    out[:, 2, X] = v0x + (v3x - v0x) * rz
    out[:, 2, Y] = v0y + (v3y - v0y) * rz

    return out


def _find_upper_triangle(dst: np.ndarray, z: np.ndarray) -> np.ndarray:
    # This function is the counterpart of `_find_lower_triangle` for a
    # plane that lies between the two highest vertices. The corners of
    # the triangle lie on the three edges that leave the highest vertex.

    out = np.empty_like(dst, shape=(len(dst), 3, 2))

    v0x, v0y, v0z = dst[:, 0].T
    v1x, v1y, v1z = dst[:, 1].T
    v2x, v2y, v2z = dst[:, 2].T
    v3x, v3y, v3z = dst[:, 3].T

    rz = (z - v3z) / (v0z - v3z)
    out[:, 0, X] = v3x + (v0x - v3x) * rz
    out[:, 0, Y] = v3y + (v0y - v3y) * rz
    rz = (z - v3z) / (v1z - v3z)
    out[:, 1, X] = v3x + (v1x - v3x) * rz
    out[:, 1, Y] = v3y + (v1y - v3y) * rz
    rz = (z - v3z) / (v2z - v3z)
    out[:, 2, X] = v3x + (v2x - v3x) * rz
    out[:, 2, Y] = v3y + (v2y - v3y) * rz

    return out


def _find_quadrilateral(tetra: np.ndarray, z: np.ndarray) -> np.ndarray:

    out = np.empty_like(tetra, shape=(len(tetra), 4, 2))

    v0x, v0y, v0z = tetra[:, 0].T
    v1x, v1y, v1z = tetra[:, 1].T
    v2x, v2y, v2z = tetra[:, 2].T
    v3x, v3y, v3z = tetra[:, 3].T

    rz = (z - v2z) / (v0z - v2z)
    out[:, 0, X] = v2x + (v0x - v2x) * rz
    out[:, 0, Y] = v2y + (v0y - v2y) * rz
    rz = (z - v3z) / (v1z - v3z)
    out[:, 3, X] = v3x + (v1x - v3x) * rz
    out[:, 3, Y] = v3y + (v1y - v3y) * rz
    rz = (z - v0z) / (v3z - v0z)
    out[:, 1, X] = v0x + (v3x - v0x) * rz
    out[:, 1, Y] = v0y + (v3y - v0y) * rz
    rz = (z - v1z) / (v2z - v1z)
    out[:, 2, X] = v1x + (v2x - v1x) * rz
    out[:, 2, Y] = v1y + (v2y - v1y) * rz

    return out


def _truncate_and_stack3d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Truncate four vertex arrays to a common shape and stack them.

    Each input has shape (Nx, Ny, Nz, 3). The result has shape (N, 4, 3),
    where N is the number of voxels in the truncated shape.
    """
    vertices = (a, b, c, d)
    nx, ny, nz = (min(x.shape[i] for x in vertices) for i in range(3))
    vertices = (vertex[:nx, :ny, :nz].reshape(-1, 3) for vertex in vertices)
    return np.stack(tuple(vertices), axis=1)


def _yield_thetrahedra(field: np.ndarray) -> _tx.Generator:
    """Yield batches of tetrahedra from a field of vertex coordinates.

    All the tetrahedra in a batch share the same vertex pattern, and each
    batch is an array of shape (N, 4, 3).
    """
    # Cubes can be batched by slicing only when they lie on a common
    # subgrid, so the grid is split into four red and four black subgrids
    # according to the parity of their offsets.

    # =========== #
    #    R E D    #
    # =========== #

    # No shift

    # --- no shift

    x000 = field[0::2, 0::2, 0::2]
    x001 = field[0::2, 0::2, 1::2]
    x010 = field[0::2, 1::2, 0::2]
    x011 = field[0::2, 1::2, 1::2]
    x100 = field[1::2, 0::2, 0::2]
    x101 = field[1::2, 0::2, 1::2]
    x110 = field[1::2, 1::2, 0::2]
    x111 = field[1::2, 1::2, 1::2]

    yield from yield_red(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in x and y

    # --- xy shift

    x000 = field[1::2, 1::2, 0::2]
    x001 = field[1::2, 1::2, 1::2]
    x010 = field[1::2, 2::2, 0::2]
    x011 = field[1::2, 2::2, 1::2]
    x100 = field[2::2, 1::2, 0::2]
    x101 = field[2::2, 1::2, 1::2]
    x110 = field[2::2, 2::2, 0::2]
    x111 = field[2::2, 2::2, 1::2]

    yield from yield_red(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in y and z

    # --- yz shift

    x000 = field[0::2, 1::2, 1::2]
    x001 = field[0::2, 1::2, 2::2]
    x010 = field[0::2, 2::2, 1::2]
    x011 = field[0::2, 2::2, 2::2]
    x100 = field[1::2, 1::2, 1::2]
    x101 = field[1::2, 1::2, 2::2]
    x110 = field[1::2, 2::2, 1::2]
    x111 = field[1::2, 2::2, 2::2]

    yield from yield_red(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in x and z

    # --- xz shift

    x000 = field[1::2, 0::2, 1::2]
    x001 = field[1::2, 0::2, 2::2]
    x010 = field[1::2, 1::2, 1::2]
    x011 = field[1::2, 1::2, 2::2]
    x100 = field[2::2, 0::2, 1::2]
    x101 = field[2::2, 0::2, 2::2]
    x110 = field[2::2, 1::2, 1::2]
    x111 = field[2::2, 1::2, 2::2]

    yield from yield_red(x000, x001, x010, x011, x100, x101, x110, x111)

    # =========== #
    #  B L A C K  #
    # =========== #

    # Shift in x

    # --- x shift

    x000 = field[1::2, 0::2, 0::2]
    x001 = field[1::2, 0::2, 1::2]
    x010 = field[1::2, 1::2, 0::2]
    x011 = field[1::2, 1::2, 1::2]
    x100 = field[2::2, 0::2, 0::2]
    x101 = field[2::2, 0::2, 1::2]
    x110 = field[2::2, 1::2, 0::2]
    x111 = field[2::2, 1::2, 1::2]

    yield from yield_black(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in y

    # --- y shift

    x000 = field[0::2, 1::2, 0::2]
    x001 = field[0::2, 1::2, 1::2]
    x010 = field[0::2, 2::2, 0::2]
    x011 = field[0::2, 2::2, 1::2]
    x100 = field[1::2, 1::2, 0::2]
    x101 = field[1::2, 1::2, 1::2]
    x110 = field[1::2, 2::2, 0::2]
    x111 = field[1::2, 2::2, 1::2]

    yield from yield_black(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in z

    # --- z shift

    x000 = field[0::2, 0::2, 1::2]
    x001 = field[0::2, 0::2, 2::2]
    x010 = field[0::2, 1::2, 1::2]
    x011 = field[0::2, 1::2, 2::2]
    x100 = field[1::2, 0::2, 1::2]
    x101 = field[1::2, 0::2, 2::2]
    x110 = field[1::2, 1::2, 1::2]
    x111 = field[1::2, 1::2, 2::2]

    yield from yield_black(x000, x001, x010, x011, x100, x101, x110, x111)

    # Shift in x, y and z

    # --- xyz shift

    x000 = field[1::2, 1::2, 1::2]
    x001 = field[1::2, 1::2, 2::2]
    x010 = field[1::2, 2::2, 1::2]
    x011 = field[1::2, 2::2, 2::2]
    x100 = field[2::2, 1::2, 1::2]
    x101 = field[2::2, 1::2, 2::2]
    x110 = field[2::2, 2::2, 1::2]
    x111 = field[2::2, 2::2, 2::2]

    yield from yield_black(x000, x001, x010, x011, x100, x101, x110, x111)


def yield_red(
    x000: np.ndarray,
    x001: np.ndarray,
    x010: np.ndarray,
    x011: np.ndarray,
    x100: np.ndarray,
    x101: np.ndarray,
    x110: np.ndarray,
    x111: np.ndarray,
) -> _tx.Generator:
    # A red cube is cut into five tetrahedra. Four of them are
    # trirectangular tetrahedra, which have three right angles at their tip
    # vertex (https://en.wikipedia.org/wiki/Trirectangular_tetrahedron).
    # Their tips are at 000, 011, 101 and 110, which are two opposite
    # corners of the top face of the cube and two opposite corners of its
    # bottom face.
    # The drawing below sketches these four tetrahedra, numbered #1 to #4:
    #
    #            _______  #2
    #      /           /|         |
    # #1  /________   / |         |
    #    |              |      #3 |_______   |
    #    |                       /           | /
    #    |                      /    ________|/
    #                                         #4
    #
    # The fifth tetrahedron is regular. Its vertices are the four corners
    # of the cube that are not the tip of any of the other four tetrahedra.

    # Tip at 000
    yield _truncate_and_stack3d(x000, x001, x010, x100)

    # Tip at 011
    yield _truncate_and_stack3d(x011, x010, x001, x111)

    # Tip at 101
    yield _truncate_and_stack3d(x101, x100, x001, x111)

    # Tip at 110
    yield _truncate_and_stack3d(x110, x100, x010, x111)

    # Regular tetrahedron
    yield _truncate_and_stack3d(x111, x001, x010, x100)


def yield_black(
    x000: np.ndarray,
    x001: np.ndarray,
    x010: np.ndarray,
    x011: np.ndarray,
    x100: np.ndarray,
    x101: np.ndarray,
    x110: np.ndarray,
    x111: np.ndarray,
) -> _tx.Generator:
    # A black cube is also cut into five tetrahedra. Four of them are
    # trirectangular tetrahedra whose tips are at 010, 001, 100 and 111,
    # which are the four corners that are not tips in a red cube. The
    # drawing below sketches these four tetrahedra, numbered #1 to #4:
    #
    #    #1  ________
    #      /|          /                       |
    #     / |  _______/                        |
    #       |         | #2      |      ________| #4
    #                 |         | /           /
    #                 |         |/________   /
    #                        #3
    #
    # As in a red cube, the fifth tetrahedron is regular. Its vertices are
    # the four corners of the cube that are not the tip of any of the other
    # four tetrahedra.

    # Tip at 010
    yield _truncate_and_stack3d(x010, x011, x000, x110)

    # Tip at 001
    yield _truncate_and_stack3d(x001, x000, x011, x101)

    # Tip at 100
    yield _truncate_and_stack3d(x100, x000, x110, x101)

    # Tip at 111
    yield _truncate_and_stack3d(x111, x011, x101, x110)

    # Regular tetrahedron
    yield _truncate_and_stack3d(x000, x011, x110, x101)


def _generate_disp_field(
    shape: _tx.Sequence[int], magnitude: float = 1, fwhm: float = 5
) -> np.ndarray:
    # Generate a random smooth displacement field for testing.
    from scipy.ndimage import gaussian_filter

    shape = tuple(shape) + (len(shape),)
    disp = (np.random.rand(*shape) * 2 - 1) * magnitude
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = (sigma,) * (len(shape) - 1) + (0,)
    disp = gaussian_filter(disp, sigma=sigma)
    return disp


def _identity_field(shape: _tx.Sequence[int]) -> np.ndarray:
    grid = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    return np.stack(grid, axis=-1)


def _compose_fields(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    grid = _identity_field(field1.shape[:-1])
    coords = grid + field1
    coords = np.transpose(coords, (3, 0, 1, 2))  # channels first

    out = np.empty_like(field1)
    for i in range(field1.shape[-1]):
        out[..., i] = map_coordinates(field2[..., i], coords, order=1)
    out += field1

    return out


def _disp2rgb(
    disp: np.ndarray, max: _tx.Optional[np.ndarray] = None
) -> np.ndarray:
    if max is None:
        max = np.abs(disp).max()
    disp = np.clip(disp / max, -1, 1)
    disp = (disp + 1) / 2
    return (disp * 255).astype(np.uint8)


def _test_inverse3d(plot: bool = True) -> None:
    shape = (64,) * 3
    disp = _generate_disp_field(shape, magnitude=32, fwhm=16)
    inv_disp = inverse3d(disp)
    comp_disp = _compose_fields(disp, inv_disp)

    mx = np.abs(disp).max()

    if plot:
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 1)
        plt.imshow(_disp2rgb(disp[:, :, 32], max=mx))
        plt.title("Forward")
        plt.subplot(1, 3, 2)
        plt.imshow(_disp2rgb(inv_disp[:, :, 32], max=mx))
        plt.title("Inverse")
        plt.subplot(1, 3, 3)
        plt.imshow(_disp2rgb(comp_disp[:, :, 32], max=mx))
        plt.title("Composition")
        plt.show()

    border = 1
    if border:
        comp_disp = comp_disp[border:-border, border:-border, border:-border]

    assert np.allclose(comp_disp, 0, atol=1e-2)
