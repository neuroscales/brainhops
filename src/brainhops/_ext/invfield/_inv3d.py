# METHOD
# Described in 2D for simplicity, but the same applies to 3D, with
# thetrahedra instead of triangles.
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
# The grid can be split in a checkerboard pattern of "red" and "black"
# blocks, where each block.
# Examples of "red" blocks are:
#   - {(0,0),(0,1),(1,0),(1,1)}
#   - {(2,0),(2,1),(3,0),(3,1)}
#   - {(1,1),(1,2),(2,1),(2,2)}
# Examples of "black" blocks are:
#   - {(0,1),(0,2),(1,1),(1,2)}
#   - {(1,0),(1,1),(2,0),(2,1)}
#   - {(2,1),(2,2),(3,1),(3,2)}
# Each block is split in two triangles, for example:
#  - {(0,0),(1,0),(1,1)} and {(0,0),(0,1),(1,1)} (red)
#  - {(1,1),(0,1),(0,2)} and {(1,1),(1,2),(0,2)} (black)
#
# Thanks to the regularity of the pattern, we can extract batches of
# triangles by slicing the displacement field.
#
# Inverting the displacement field, consists of finding the voxels
# that fall in each (displaced) triangle, and computing the barycentric
# mean of the corresponding (original) vertices.
#
# In 3D, red and black blocks are cubes that are split into 5 thetrahedra.
# See:
#   "Image Registration Using a Symmetric Prior — in Three Dimensions"
#   Ashburner, Andersson & Friston. Human Brain Mapping (2000).
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf
import numpy as np
import typing_extensions as _tx
from scipy.ndimage import gaussian_filter


def inverse3d(disp: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a displacement field by interpreting it as a
    thetrahedral mesh, where each tetrahedron defines an affine transform.

    This is the method described in the appendix of:
        "Image Registration Using a Symmetric Prior — in Three Dimensions"
        Ashburner, Andersson & Friston. Human Brain Mapping (2000).
        https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf

    Parameters
    ----------
    disp : np.ndarray
        The displacement field to invert (displacements are in voxels).
        Should be of shape (Nx, Ny, Nz, 3).
        The last dimension should contain the displacements along each
        axis, in the same order (i.e. [x, y, z]).

    Returns
    -------
    np.ndarray
        The inverse displacement field, of the same shape as the input.

    """
    disp = np.asanyarray(disp)
    out = np.full_like(disp, np.nan)

    (Nx, Ny, Nz, Nd) = disp.shape
    if Nd != 3:
        raise ValueError(
            f"Expected a 3D displacement field with shape (Nx, Ny, Nz, 3), "
            f"but got shape {disp.shape}"
        )

    # generate meshgrid
    src = np.meshgrid(*(np.arange(s) for s in (Nx, Ny, Nz)), indexing='ij')
    src = np.stack(src, axis=-1)

    # Convert displacements to coordinates
    dst = src + disp

    # Extract the (batches of) thetraheda
    for src1, dst1 in zip(
        _yield_thetrahedra(src),
        _yield_thetrahedra(dst)
    ):
        # Batch process similar thetrahedra
        _process_thetrahedron(src1, dst1, out)

    # Convert coordinates to displacements
    out -= src

    # Fill in missing values via smoothing
    msk = msk0 = np.isfinite(out)
    while not msk.all():
        out[~msk] = 0
        wgt = msk.astype(np.float64)
        sigma = 1 / np.sqrt(8 * np.log(2))  # FWHM = 1 voxel
        sigma = (sigma, sigma, sigma, 0)
        smo = gaussian_filter(out, sigma=sigma, mode='nearest')
        wgt = gaussian_filter(wgt, sigma=sigma, mode='nearest')
        smo /= wgt
        out[~msk0] = smo[~msk0]
        msk = np.isfinite(out)

    return out


# Constants to make reading the rest of the code easier
X, Y, Z = 0, 1, 2
BATCH_AXIS, VERTEX_AXIS, SPACE_AXIS = 0, 1, 2


def _process_thetrahedron(src: np.ndarray,
                          dst: np.ndarray,
                          out: np.ndarray
                          ) -> None:
    """
    Process a batch of thetrahedra.

    Parameters
    ----------
    src, dst : np.ndarray
        Thetrahedra vertices in the source and target domains,
        with shape (N, 4, 3), where N is the batch size.
    out : np.ndarray
        The output array to write the results to, of shape (Nx, Ny, Nz, 3).

    """
    # sort tetrahedron vertices along z axis
    idx = np.argsort(dst[:, :, Z:Z+1], axis=VERTEX_AXIS)
    ttr = np.take_along_axis(dst, idx, axis=VERTEX_AXIS)

    # For each horizontal plane, find its intersection with the tetrahedron.
    # We start from the minimum integral z in the tetrahedron.
    z = np.ceil(ttr[:, 0, Z]).astype(np.int64)
    while True:

        mask0 = (0 <= z) & (z < out.shape[Z]) & (z <= ttr[:, 3, Z])
        if not mask0.any():
            break

        upp_mask = (z <= ttr[:, 3, Z])
        mid_mask = (z <= ttr[:, 2, Z])
        low_mask = (z <= ttr[:, 1, Z])
        upp_mask &= ~mid_mask
        mid_mask &= ~low_mask
        upp_mask &= mask0
        mid_mask &= mask0
        low_mask &= mask0
        del mask0

        # Lower triangle
        if low_mask.any():
            zm, srcm, dstm = z[low_mask], src[low_mask], dst[low_mask]
            tri = _find_lower_triangle(ttr[low_mask], zm)
            _process_triangle(srcm, dstm, zm, tri, out)

        # Middle quadrilateral (split into two triangles)
        if mid_mask.any():
            zm, srcm, dstm = z[mid_mask], src[mid_mask], dst[mid_mask]
            quad = _find_quadrilateral(ttr[mid_mask], zm)
            _process_triangle(srcm, dstm, zm, quad[:, 0:3], out)
            _process_triangle(srcm, dstm, zm, quad[:, 1:4], out)

        # Upper triangle
        if upp_mask.any():
            zm, srcm, dstm = z[upp_mask], src[upp_mask], dst[upp_mask]
            tri = _find_upper_triangle(ttr[upp_mask], zm)
            _process_triangle(srcm, dstm, zm, tri, out)

        z += 1


def _process_triangle(src: np.ndarray,
                      dst: np.ndarray,
                      z: np.ndarray,
                      tri: np.ndarray,
                      out: np.ndarray
                      ) -> None:
    """
    Process a batch of triangles in a given z plane.

    Parameters
    ----------
    src, dst : np.ndarray
        Thetrahedra vertices in the source and target domains,
        with shape (N, 4, 3), where N is the batch size.
    z : np.ndarray
        The z coordinate of the plane being processed, with shape (N,).
    tri : np.ndarray
        The triangle vertices in the target domain, with shape (N, 3, 2).
    out : np.ndarray
        The output array to write the results to, of shape (Nx, Ny, Nz, 3).

    """

    # sort triangle vertices along y axis
    idx = np.argsort(tri[:, :, Y:Y+1], axis=VERTEX_AXIS)
    tri = np.take_along_axis(tri, idx, axis=VERTEX_AXIS)

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
            zm, ym = z[low_mask], y[low_mask]
            srcm, dstm = src[low_mask], dst[low_mask]
            seg = _find_segment(tri[low_mask], ym)
            _process_segment(srcm, dstm, zm, ym, seg, out)

        if upp_mask.any():
            zm, ym = z[upp_mask], y[upp_mask],
            srcm, dstm = src[upp_mask], dst[upp_mask]
            seg = _find_segment(tri[upp_mask][:, ::-1], ym)
            _process_segment(srcm, dstm, zm, ym, seg, out)

        y += 1


def _process_segment(src: np.ndarray,
                     dst: np.ndarray,
                     z: np.ndarray,
                     y: np.ndarray,
                     seg: np.ndarray,
                     out: np.ndarray
                     ) -> None:
    """
    Process a batch of segments in a given z plane and y coordinate.

    Parameters
    ----------
    src, dst : np.ndarray
        Thetrahedra vertices in the source and target domains,
        with shape (N, 4, 3), where N is the batch size.
    z : np.ndarray
        The z coordinate of the plane being processed, with shape (N,).
    y : np.ndarray
        The y coordinate of the line being processed, with shape (N,).
    seg : np.ndarray
        The segment vertices in the target domain, with shape (N, 2, 1).
    out : np.ndarray
        The output array to write the results to, of shape (Nx, Ny, Nz, 3).

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
        xm, ym, zm = x[mask], y[mask], z[mask]
        vdst = np.stack((xm, ym, zm), axis=-1)                 # (N, 3)
        bary = _barycoord(vdst, dst[mask])                     # (N, 4)

        # Compute the corresponding point in the source domain as the
        # barycentric mean of the tetrahedron vertices in the source domain.
        vsrc = np.einsum('ijk,ij->ik', src[mask], bary)        # (N, 3)

        # Assign the computed point to the output array
        out[xm, ym, zm] = vsrc

        x += 1


def _barycoord(x: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    # Compute the barycentric coordinates of x with respect to the
    # tetrahedron defined by its vertices.
    # * x is of shape (N, 13), where N is the number of voxels in the
    #   batch. The last dimension contains the (x,y,z).
    # * The tetrahedron is defined by its vertices, with shape (N, 4, 3),
    #   where N is the number of thetrahedra in the batch.
    # * The output is of shape (N, 4), where the last dimension contains
    #   the barycentric coordinates of x with respect to each vertex of
    #   the tetrahedron.

    v0 = tetra[:, 0]
    v1 = tetra[:, 1]
    v2 = tetra[:, 2]
    v3 = tetra[:, 3]

    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0
    v0x = x - v0

    dt = np.einsum('ij,ij->i', v01, np.cross(v02, v03))
    b1 = np.einsum('ij,ij->i', v0x, np.cross(v02, v03)) / dt
    b2 = np.einsum('ij,ij->i', v01, np.cross(v0x, v03)) / dt
    b3 = np.einsum('ij,ij->i', v01, np.cross(v02, v0x)) / dt
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
    # Compute intersection of the "infinite" tetrahedron (no base)
    # and a horizontal plane. Vertices are sorted by increasing z.
    # The first vertex is the "tip" of the tetrahedron.
    #
    # The resulting intersection is a triangle. We return the (x,y)
    # coordinates of its vertices.

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
    # Compute intersection of the "infinite" tetrahedron (no base)
    # and a horizontal plane. Vertices are sorted by increasing z.
    # The last vertex is the "tip" of the tetrahedron.
    #
    # The resulting intersection is a triangle. We return the (x,y)
    # coordinates of its vertices.

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


def _truncate_and_stack3d(a: np.ndarray,
                          b: np.ndarray,
                          c: np.ndarray,
                          d: np.ndarray
                          ) -> np.ndarray:
    """
    Truncate arrays so that they have the same shape, then stack them.

    Parameters
    ----------
    a, b, c, d : np.ndarray
        The coordinates of the vertices of the thetrahedra,
        with shape (Nx, Ny, Nz, 3).

    Returns
    -------
    np.ndarray
        With shape (N, 4, 3), where N=Nx*Ny*Nz is the number of
        thetrahedra in the batch.
    """
    vertices = (a, b, c, d)
    nx, ny, nz = (min(x.shape[i] for x in vertices) for i in range(3))
    vertices = (vertex[:nx, :ny, :nz].reshape(-1, 3) for vertex in vertices)
    return np.stack(tuple(vertices), axis=1)


def _yield_thetrahedra(field: np.ndarray) -> _tx.Generator:
    """
    Yield the vertices of the thetrahedra defined by the displacement field,
    in batches of similar thetrahedra (i.e. with the same pattern of vertices).

    Parameters
    ----------
    field : np.ndarray
        Coordinate field to process

    Yields
    ------
    np.ndarray
        The coordinates of the vertices of the thetrahedra, with shape
        (N, 4, 3).
    """
    # We need to split the grid into a red-black checkerboard pattern.
    # We also want to extract thetrahedra via slicing, which means we can
    # only batch thetrahedra whose vertices are aligned on a cartesian grid.
    # We therefore split the input grid into 8 subgrids, and designate 4 of
    # them as "red" and the other 4 as "black".

    # =========== #
    #    R E D    #
    # =========== #

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


def yield_red(x000: np.ndarray,
              x001: np.ndarray,
              x010: np.ndarray,
              x011: np.ndarray,
              x100: np.ndarray,
              x101: np.ndarray,
              x110: np.ndarray,
              x111: np.ndarray
              ) -> _tx.Generator:
    # Yield the five thetrahedra that make up a red block.
    #
    # Four of them are all trirectangular thetrahedra
    # (i.e. with three right angles at the tip vertex,
    # https://en.wikipedia.org/wiki/Trirectangular_tetrahedron).
    # Their tips are two opposing vertices on the top face of the cube,
    # and the other two opposing vertices on the bottom face of the cube.
    #
    #            _______  #2
    #      /           /|         |
    # #1  /________   / |         |
    #    |              |      #3 |_______   |
    #    |                       /           | /
    #    |                      /    ________|/
    #                                         #4
    #
    # The fifth tetrahedron is a regular one, whose vertices are the
    # four vertices that were not tips in the other four thetrahedra.

    # tip = 000
    yield _truncate_and_stack3d(x000, x001, x010, x100)

    # tip = 011
    yield _truncate_and_stack3d(x011, x010, x001, x111)

    # tip = 101
    yield _truncate_and_stack3d(x101, x100, x001, x111)

    # tip = 110
    yield _truncate_and_stack3d(x110, x100, x010, x111)

    # regular one
    yield _truncate_and_stack3d(x111, x001, x010, x100)


def yield_black(x000: np.ndarray,
                x001: np.ndarray,
                x010: np.ndarray,
                x011: np.ndarray,
                x100: np.ndarray,
                x101: np.ndarray,
                x110: np.ndarray,
                x111: np.ndarray
                ) -> _tx.Generator:
    # Yield the five thetrahedra that make up a black block.
    #
    # Four of them are also trirectangular thetrahedra, whose tips are
    # the four vertices that were not tips in "red" blocks.
    #
    #    #1  ________
    #      /|          /                       |
    #     / |  _______/                        |
    #       |         | #2      |      ________| #4
    #                 |         | /           /
    #                 |         |/________   /
    #                        #3
    #
    # Similarly to the "red" blocks, the fifth tetrahedron is made of the
    # four vertices that were not tips in the other four thetrahedra.

    # tip = 010
    yield _truncate_and_stack3d(x010, x011, x000, x110)

    # tip = 001
    yield _truncate_and_stack3d(x001, x000, x011, x101)

    # tip = 100
    yield _truncate_and_stack3d(x100, x000, x110, x101)

    # tip = 111
    yield _truncate_and_stack3d(x111, x011, x101, x110)

    # regular one
    yield _truncate_and_stack3d(x000, x011, x110, x101)


def _generate_disp_field(shape: _tx.Sequence[int],
                         magnitude: float = 1,
                         fwhm: float = 5
                         ) -> np.ndarray:
    # Generate a random displacement field of the given shape, for testing.
    from scipy.ndimage import gaussian_filter
    shape = tuple(shape) + (len(shape),)
    disp = (np.random.rand(*shape) * 2 - 1) * magnitude
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = (sigma,) * (len(shape) - 1) + (0,)
    disp = gaussian_filter(disp, sigma=sigma)
    return disp


def _identity_field(shape: _tx.Sequence[int]) -> np.ndarray:
    # Generate an identity coordinate field.
    grid = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')
    return np.stack(grid, axis=-1)


def _compose_fields(field1: np.ndarray, field2: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    grid = _identity_field(field1.shape[:-1])
    coords = grid + field1
    coords = np.transpose(coords, (3, 0, 1, 2))  # (3, Nx, Ny, Nz)

    out = np.empty_like(field1)
    for i in range(field1.shape[-1]):
        out[..., i] = map_coordinates(field2[..., i], coords, order=1)
    out += field1

    return out


def _disp2rgb(disp: np.ndarray,
              max: _tx.Optional[np.ndarray] = None) -> np.ndarray:
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
        plt.title('Forward')
        plt.subplot(1, 3, 2)
        plt.imshow(_disp2rgb(inv_disp[:, :, 32], max=mx))
        plt.title('Inverse')
        plt.subplot(1, 3, 3)
        plt.imshow(_disp2rgb(comp_disp[:, :, 32], max=mx))
        plt.title('Composition')
        plt.show()

    border = 1
    if border:
        comp_disp = comp_disp[border:-border, border:-border, border:-border]

    assert np.allclose(comp_disp, 0, atol=1e-2)
