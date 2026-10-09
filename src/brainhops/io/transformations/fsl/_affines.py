import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly

from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

from ._systems import FslCoordinateSystem

# ----------------------------------------------------------------------
#   AFFINE TYPES
# ----------------------------------------------------------------------


class VoxelToScaledMm(_xforms.Affine):
    """Affine transformation from voxel coordinates to FSL scaled-mm."""

    _input: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )
    _output: KwOnly[_systems.CoordinateSystem] = FslCoordinateSystem()


class ScaledMmToVoxel(_xforms.Affine):
    """Affine transformation from FSL scaled-mm to voxel coordinates."""

    _input: KwOnly[_systems.CoordinateSystem] = FslCoordinateSystem()
    _output: KwOnly[_systems.CoordinateSystem] = (
        _systems.VoxelCoordinateSystem()
    )


class ScaledMmToScaledMm(_xforms.Affine):
    """Affine transformation between two FSL scaled-mm spaces.

    A FLIRT matrix is such an affine, from the scaled-mm space of the
    reference image to that of the moving image.
    """

    _input: KwOnly[_systems.CoordinateSystem] = FslCoordinateSystem()
    _output: KwOnly[_systems.CoordinateSystem] = FslCoordinateSystem()


# ----------------------------------------------------------------------
#   IMAGE GEOMETRY
# ----------------------------------------------------------------------


def _best_affine(obj: tx.Any) -> np.ndarray:
    """Return the voxel-to-world (RAS) affine of a nibabel image or header."""
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
    """Return the spatial shape of a nibabel image or header."""
    if hasattr(obj, "get_data_shape"):
        return tuple(int(s) for s in obj.get_data_shape())
    if getattr(obj, "shape", None) is not None:
        return tuple(int(s) for s in obj.shape)
    if hasattr(obj, "header") and obj.header is not None:
        return _shape(obj.header)
    raise ValueError("Cannot determine the shape of the image.")


def _pixdim(obj: tx.Any) -> np.ndarray:
    """Return the pixel sizes, as positive magnitudes.

    Missing, zero or non-finite sizes are replaced with one, so that an
    incomplete `pixdim` still gives an invertible scaled-mm affine.
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
    """Affines that place one image in the FSL coordinate systems.

    Built from a nibabel image or header, the geometry holds the affines
    between voxel, FSL scaled-mm and world (RAS) coordinates. Following the
    FSL convention, the scaled-mm affine scales voxel indices by the pixel
    sizes and flips x when the voxel-to-world determinant is positive.
    """

    def __init__(self, image: tx.Any) -> None:
        vox2ras = _best_affine(image)
        shape = _shape(image)
        pixdim = _pixdim(image)

        # A header with a missing or degenerate sform can give a non-finite
        # voxel-to-world affine. Plain pixel-size scaling keeps the geometry
        # usable.
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
        """Whether the voxel-to-world determinant is positive."""
        return bool(np.linalg.det(self.vox2ras[:3, :3]) > 0)

    @property
    def ras2vox(self) -> np.ndarray:
        """The RAS-to-voxel affine, the inverse of the sform or qform."""
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
