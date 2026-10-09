"""
Conversion of a bare matrix into an affine transformation.

The conventions stated by the caller are applied in this order: a row-vector
matrix is transposed, the matrix is completed to homogeneous form, an inverse
matrix is inverted, 1-based voxel indices are shifted to 0-based ones, and a
voxel endpoint that comes with an image is mapped to the world space of that
image.
"""

import numpy as np
import typing_extensions as tx

from brainhops.datamodel import systems as _systems

_SpaceLike = tx.Union[str, _systems.CoordinateSystem, None]
_BaseLike = tx.Union[int, tx.Tuple[int, int]]

# (3, 3) is either 3-D linear or 2-D homogeneous, so ndim decides.
_SHAPES = {(2, 3): 2, (3, 4): 3, (4, 4): 3, (3, 3): None}

VECTORS = ("column", "row")
DIRECTIONS = ("forward", "inverse")
SPACES = ("voxel", "pixel", "index", "ras", "lps", "unnamed")


# ----------------------------------------------------------------------
#   VALIDATION
# ----------------------------------------------------------------------


def check_vector(vector: str) -> str:
    """Normalize a vector convention to `"column"` or `"row"`."""
    vector = str(vector).lower()
    if vector in ("col", "columns"):
        vector = "column"
    if vector == "rows":
        vector = "row"
    if vector not in VECTORS:
        raise ValueError(f"vector must be one of {VECTORS}, got {vector!r}.")
    return vector


def check_direction(direction: str) -> str:
    """Normalize a direction to `"forward"` or `"inverse"`."""
    direction = str(direction).lower()
    if direction in ("fwd",):
        direction = "forward"
    if direction in ("inv", "backward"):
        direction = "inverse"
    if direction not in DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DIRECTIONS}, got {direction!r}."
        )
    return direction


def check_index_base(index_base: _BaseLike) -> tx.Tuple[int, int]:
    """Normalize an index base to an `(input, output)` pair of 0 or 1."""
    if isinstance(index_base, (tuple, list)):
        bases = tuple(int(b) for b in index_base)
    else:
        bases = (int(index_base), int(index_base))
    if len(bases) != 2 or any(b not in (0, 1) for b in bases):
        raise ValueError(
            f"index_base must be 0, 1, or an (input, output) pair of them, "
            f"got {index_base!r}."
        )
    return bases


def column_matrix(raw: np.ndarray, vector: str = "column") -> np.ndarray:
    """Return a 2-D matrix in the column-vector convention."""
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {matrix.shape}.")
    if check_vector(vector) == "row":
        matrix = matrix.T
    return matrix


def spatial_ndim(shape: tx.Tuple[int, ...], ndim: tx.Optional[int]) -> int:
    """Return the number of spatial dimensions that a matrix acts on."""
    shape = tuple(shape)
    if shape not in _SHAPES:
        raise ValueError(
            f"A plain affine matrix must have shape (3, 3), (3, 4) or "
            f"(4, 4) in 3-D, or (2, 3) or (3, 3) in 2-D (transposed for "
            f"the row-vector convention), got {shape}."
        )
    known = _SHAPES[shape]
    if known is None:
        known = 3 if ndim is None else int(ndim)
        if known not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim!r}.")
    elif ndim is not None and int(ndim) != known:
        raise ValueError(
            f"A matrix of shape {shape} is {known}-D, but ndim={ndim}."
        )
    return known


def homogeneous(matrix: np.ndarray, ndim: tx.Optional[int]) -> np.ndarray:
    """Complete a column-vector matrix to homogeneous `(N + 1, N + 1)` form."""
    n = spatial_ndim(matrix.shape, ndim)
    rows, cols = matrix.shape
    out = np.eye(n + 1)
    if (rows, cols) == (n, n):
        out[:n, :n] = matrix
    elif (rows, cols) == (n, n + 1):
        out[:n] = matrix
    else:
        last = np.zeros(n + 1)
        last[-1] = 1
        if not np.allclose(matrix[-1], last, atol=1e-6):
            raise ValueError(
                f"The last row of a homogeneous affine must be {last}, "
                f"got {matrix[-1]} (a projective matrix is not an affine)."
            )
        out[:n] = matrix[:n]
    return out


def is_affine_matrix(
    raw: np.ndarray, vector: str = "column", ndim: tx.Optional[int] = None
) -> bool:
    """Return whether `raw` reads as a finite affine matrix."""
    try:
        if not np.all(np.isfinite(raw)):
            return False
        homogeneous(column_matrix(raw, vector), ndim)
    except (ValueError, TypeError):
        return False
    return True


# ----------------------------------------------------------------------
#   SPACES
# ----------------------------------------------------------------------


def make_system(space: _SpaceLike, ndim: int) -> _systems.CoordinateSystem:
    """Return the `ndim`-dimensional coordinate system named by `space`."""
    if isinstance(space, _systems.CoordinateSystem):
        return space
    name = "unnamed" if space is None else str(space).lower()
    if name in ("voxel", "pixel", "index"):
        if ndim == 2:
            return _systems.PixelCoordinateSystem()
        return _systems.VoxelCoordinateSystem()
    if name in ("ras", "lps"):
        if ndim != 3:
            raise ValueError(f"A {name.upper()} space is 3-D, not {ndim}-D.")
        if name == "ras":
            return _systems.RASCoordinateSystem()
        return _systems.LPSCoordinateSystem()
    if name in ("unnamed", "none", ""):
        if ndim == 2:
            return _systems.CoordinateSystem2D()
        return _systems.CoordinateSystem3D()
    raise ValueError(
        f"Unknown space {space!r}: expected one of {SPACES} or a "
        f"CoordinateSystem."
    )


def is_index_system(system: _systems.CoordinateSystem) -> bool:
    """Return whether a coordinate system is a voxel or pixel space."""
    return isinstance(system, _systems.ArrayCoordinateSystem)


def _shift(ndim: int, offset: float) -> np.ndarray:
    out = np.eye(ndim + 1)
    out[:ndim, -1] = offset
    return out


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def image_to_world(
    image: tx.Any,
) -> tx.Tuple[np.ndarray, _systems.CoordinateSystem]:
    """
    Return the voxel-to-world matrix of an image and its world space.

    For a `nibabel` image or header, the world space is RAS. For a brainhops
    image, the transformation must be an affine, and the world space is its
    output.
    """
    xform = getattr(image, "transformations", None)
    if xform is not None and not hasattr(image, "get_best_affine"):
        xform = image.transformation
        matrix = getattr(xform, "homogeneous_matrix", None)
        if matrix is None:
            raise ValueError(
                "The image's voxel-to-world transformation is not an "
                "affine, so it cannot place a plain matrix in world space."
            )
        output = xform.output or _systems.CoordinateSystem3D()
        return np.asarray(matrix, dtype=np.float64), output
    if hasattr(image, "get_best_affine"):
        matrix = image.get_best_affine()
    elif getattr(image, "header", None) is not None and hasattr(
        image.header, "get_best_affine"
    ):
        matrix = image.header.get_best_affine()
    elif getattr(image, "affine", None) is not None:
        matrix = image.affine
    else:
        raise ValueError(
            "Cannot find the voxel-to-world affine of the image: pass a "
            "nibabel image or header, or a brainhops image."
        )
    return np.asarray(matrix, dtype=np.float64), _systems.RASCoordinateSystem()


# ----------------------------------------------------------------------
#   CONVERSION
# ----------------------------------------------------------------------


def to_affine(
    raw: np.ndarray,
    *,
    input: _SpaceLike = None,
    output: _SpaceLike = None,
    source: tx.Any = None,
    target: tx.Any = None,
    index_base: _BaseLike = 0,
    direction: str = "forward",
    vector: str = "column",
    ndim: tx.Optional[int] = None,
) -> tx.Tuple[
    np.ndarray,
    _systems.CoordinateSystem,
    _systems.CoordinateSystem,
    tx.Tuple[int, int],
]:
    """
    Apply the conventions stated by the caller to a raw matrix.

    The conventions are described in
    [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine].

    Returns
    -------
    matrix : np.ndarray
        The `(N, N + 1)` column-vector matrix from input to output, with
        0-based voxel indices.
    input, output : CoordinateSystem
        The spaces that the matrix maps between.
    index_base : tuple of int
        The index base applied to the input and to the output.
    """
    homog = homogeneous(column_matrix(raw, vector), ndim)
    n = homog.shape[0] - 1

    if check_direction(direction) == "inverse":
        homog = np.linalg.inv(homog)

    # Passing an image implies that its endpoint is a voxel space.
    if input is None and source is not None:
        input = "voxel"
    if output is None and target is not None:
        output = "voxel"
    in_system = make_system(input, n)
    out_system = make_system(output, n)

    in_base, out_base = check_index_base(index_base)
    scalar = not isinstance(index_base, (tuple, list))
    in_index, out_index = map(is_index_system, (in_system, out_system))
    if scalar and (in_base or out_base):
        if not (in_index or out_index):
            raise ValueError(
                "index_base=1 shifts voxel indices, but neither the input "
                "nor the output is a voxel space. Pass input='voxel' "
                "and/or output='voxel'."
            )
        in_base, out_base = in_base * in_index, out_base * out_index
    for base, index, end in (
        (in_base, in_index, "input"),
        (out_base, out_index, "output"),
    ):
        if base and not index:
            raise ValueError(
                f"index_base is 1 for the {end}, which is not a voxel space."
            )
    # x1 = x0 + 1, so a 1-based M1 is S(-1) @ M1 @ S(+1) in 0-based indices.
    if in_base:
        homog = homog @ _shift(n, 1.0)
    if out_base:
        homog = _shift(n, -1.0) @ homog

    for image, end in ((source, "input"), (target, "output")):
        if image is None:
            continue
        system = in_system if end == "input" else out_system
        if not is_index_system(system):
            raise ValueError(
                f"A {'source' if end == 'input' else 'target'} image places "
                f"a voxel {end} in world space, but the {end} is "
                f"{getattr(system, 'name', None) or 'not a voxel space'}."
            )
        vox2world, world = image_to_world(image)
        if vox2world.shape != (n + 1, n + 1):
            raise ValueError(
                f"The {end} image is {vox2world.shape[0] - 1}-D, but the "
                f"matrix is {n}-D."
            )
        if end == "input":
            homog = homog @ np.linalg.inv(vox2world)
            in_system = world
        else:
            homog = vox2world @ homog
            out_system = world

    return homog[:-1], in_system, out_system, (in_base, out_base)
