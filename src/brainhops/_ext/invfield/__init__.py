import numpy as np

from ._inv2d import inverse2d
from ._inv3d import inverse3d


def inverse(field: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a displacement field by interpreting it as a
    mesh, where each cell defines an affine transform.

    This is the method described in:
        "High-Dimensional Image Registration Using Symmetric Priors"
        Ashburner, Andersson & Friston. NeuroImage (1999).
        https://www.fil.ion.ucl.ac.uk/spm/doc/papers/john_high_dim.pdf
    for the 2D version, and in:
        "Image Registration Using a Symmetric Prior — in Three Dimensions"
        Ashburner, Andersson & Friston. Human Brain Mapping (2000).
        https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf
    for the 3D version.

    Parameters
    ----------
    disp : np.ndarray
        The displacement field to invert (displacements are in voxels). 
        Should be of shape (Nx, Ny, Nz, 3) or (Nx, Ny, 2).
        The last dimension should contain the displacements along each
        axis, in the same order (i.e. [x, y, z] or [x, y]).

    Returns
    -------
    np.ndarray
        The inverse displacement field, of the same shape as the input.

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
