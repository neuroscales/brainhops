import numpy as np

from ._inv2d import inverse2d
from ._inv3d import inverse3d


def inverse(field: np.ndarray) -> np.ndarray:
    """Invert a displacement field.

    The grid of the field is treated as a mesh in which each simplex defines
    an affine transform, following Ashburner, Andersson and Friston,
    "High-Dimensional Image Registration Using Symmetric Priors", NeuroImage,
    1999 (in 2D, https://www.fil.ion.ucl.ac.uk/spm/doc/papers/john_high_dim.pdf),
    and "Image Registration Using a Symmetric Prior - in Three Dimensions",
    Human Brain Mapping, 2000 (in 3D,
    https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf).

    Only the displacements at the nodes of the grid are used. They define a
    piecewise-affine map, with one affine transform per simplex of the mesh,
    which is inverted exactly. A field that is meant to be interpolated
    with higher-degree splines is therefore inverted as its piecewise-linear
    interpolant. The result is approximate between nodes, and more so near
    the border, where the mesh is clipped by the field of view.

    Parameters
    ----------
    field : np.ndarray
        Displacements in voxels, with shape `(Nx, Ny, Nz, 3)` or
        `(Nx, Ny, 2)`. The last axis is ordered `[x, y, z]` or `[x, y]`.

    Returns
    -------
    np.ndarray
        The inverse field, with the same shape as `field`.

    Raises
    ------
    NotImplementedError
        If the field is neither two- nor three-dimensional.
    """
    if field.shape[-1] == 2:
        return inverse2d(field)
    elif field.shape[-1] == 3:
        return inverse3d(field)
    else:
        raise NotImplementedError(
            f"Displcement field inversion is only implemented for 2D "
            f"and 3D fields, but got field with shape {field.shape}."
        )
