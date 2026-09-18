# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

from ._systems import FSLCoordinateSystem

# ----------------------------------------------------------------------
#   AFFINE TYPES
# ----------------------------------------------------------------------


class VoxelToScaledMM(_xforms.Affine):
    """Affine transformation from voxel space to FSL scaled-mm space."""

    _input: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()
    _output: _systems.CoordinateSystem = FSLCoordinateSystem()


class ScaledMMToVoxel(_xforms.Affine):
    """Affine transformation from FSL scaled-mm space to voxel space."""

    _input: _systems.CoordinateSystem = FSLCoordinateSystem()
    _output: _systems.CoordinateSystem = _systems.VoxelCoordinateSystem()


class ScaledMMToScaledMM(_xforms.Affine):
    """Affine transformation between two FSL scaled-mm spaces.

    A FLIRT matrix is an affine between the scaled-mm coordinates of the
    reference image and the scaled-mm coordinates of the moving image.
    """

    _input: _systems.CoordinateSystem = FSLCoordinateSystem()
    _output: _systems.CoordinateSystem = FSLCoordinateSystem()


# ----------------------------------------------------------------------
#   IMAGE GEOMETRY
# ----------------------------------------------------------------------


def _best_affine(obj: tx.Any) -> np.ndarray:
    """The voxel-to-world (RAS) affine of a nibabel image or header."""
    if hasattr(obj, "get_best_affine"):
        return np.asarray(obj.get_best_affine(), dtype=np.float64)
    if hasattr(obj, "header") and obj.header is not None:
        return _best_affine(obj.header)
    if getattr(obj, "affine", None) is not None:
        return np.asarray(obj.affine, dtype=np.float64)
    raise ValueError(
        "Cannot determine the voxel-to-world affine of the image. Pass a "
        "nibabel image or header, or an object that exposes one."
    )


def _shape(obj: tx.Any) -> tx.Tuple[int, ...]:
    """The spatial shape of a nibabel image or header."""
    if hasattr(obj, "get_data_shape"):
        return tuple(int(s) for s in obj.get_data_shape())
    if getattr(obj, "shape", None) is not None:
        return tuple(int(s) for s in obj.shape)
    if hasattr(obj, "header") and obj.header is not None:
        return _shape(obj.header)
    raise ValueError("Cannot determine the shape of the image.")


def _pixdim(obj: tx.Any) -> np.ndarray:
    """The positive pixel sizes (magnitudes) of a nibabel image or header.

    A pixel size that is missing, zero, or not finite is replaced with
    one, so a header with an incomplete pixdim still yields an invertible
    scaled-mm affine rather than a singular one.
    """
    header = obj
    if not hasattr(header, "get_zooms") and hasattr(obj, "header"):
        header = obj.header
    if hasattr(header, "get_zooms"):
        zooms = np.asarray(header.get_zooms(), dtype=np.float64)
    else:
        raise ValueError("Cannot determine the pixel sizes of the image.")
    pixdim = np.ones(3, dtype=np.float64)
    pixdim[: min(3, zooms.size)] = np.abs(zooms[:3])
    pixdim[~np.isfinite(pixdim) | (pixdim == 0.0)] = 1.0
    return pixdim


class _ImageGeometry:
    """The affines that place one image in the FSL coordinate systems.

    Given a nibabel image or header, this holds the affines between the
    voxel, FSL scaled-mm, and world (RAS) coordinate systems of that
    image. The scaled-mm affine scales voxel indices by the pixel sizes
    and flips the x-axis when the voxel-to-world affine has a positive
    determinant, following FSL's convention.
    """

    def __init__(self, image: tx.Any) -> None:
        vox2ras = _best_affine(image)
        shape = _shape(image)
        pixdim = _pixdim(image)

        # A header with a missing or degenerate sform can carry a
        # non-finite voxel-to-world affine. Fall back to a plain pixel-size
        # scaling so the geometry is still usable.
        if not np.all(np.isfinite(vox2ras)):
            vox2ras = np.diag(np.concatenate([pixdim, [1.0]]))

        self.vox2ras = vox2ras
        self.shape = shape
        self.pixdim = pixdim

        vox2fsl = np.diag(np.concatenate([pixdim, [1.0]]))
        if np.linalg.det(vox2ras[:3, :3]) > 0:
            flip = np.eye(4)
            flip[0, 0] = -1.0
            flip[0, 3] = (shape[0] - 1) * pixdim[0]
            vox2fsl = flip @ vox2fsl
        self.vox2fsl = vox2fsl

    @property
    def is_neurological(self) -> bool:
        """Whether the voxel-to-world affine has a positive determinant."""
        return bool(np.linalg.det(self.vox2ras[:3, :3]) > 0)

    @property
    def ras2vox(self) -> np.ndarray:
        """The RAS-to-voxel affine (the inverse of the sform/qform)."""
        return np.linalg.inv(self.vox2ras)

    @property
    def fsl2vox(self) -> np.ndarray:
        """The scaled-mm-to-voxel affine."""
        return np.linalg.inv(self.vox2fsl)

    @property
    def fsl2ras(self) -> np.ndarray:
        """The scaled-mm-to-RAS affine."""
        return self.vox2ras @ self.fsl2vox

    @property
    def ras2fsl(self) -> np.ndarray:
        """The RAS-to-scaled-mm affine."""
        return self.vox2fsl @ self.ras2vox
