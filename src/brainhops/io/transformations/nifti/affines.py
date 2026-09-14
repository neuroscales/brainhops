# dependencies
import numpy as np
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import _NiftiObject
from brainhops.io.base.parsers import Confidence
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation


class _NiftiAffine(NiftiBasedTransformation):
    """Shared scoring for affines derived from a NIfTI header."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        float a NIfTI header as a bare affine.

        *Every* NIfTI carries an affine, so this always matches -- which
        is exactly why it must score low. Asking to load a `.nii` almost
        never means "give me its voxel-to-RAS matrix"; that is reached
        through the image. Scoring `WEAK` keeps these reachable as a last
        resort without letting them outrank an image or a field.
        """
        return Confidence.WEAK


class NiftiRASToVoxel(RASToVoxel, _NiftiAffine):
    """
    Affine transformation from RAS space to voxel space, derived from a
    NIfTI header.

    !!! note "Not a registered format"
        A NIfTI header encodes voxel-to-RAS; RAS-to-voxel is its
        inverse, computed rather than stored. The two are
        indistinguishable by content -- same container, same extension,
        same confidence -- so registering both would make every `.nii`
        an ambiguity. Reach this one through
        `NiftiVoxelToRAS.inverse()`.
    """

    @property
    def matrix(self) -> tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        if self.header is not None:
            return np.linalg.inv(self.header.get_best_affine())[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value

    def inverse(self) -> VoxelToRAS:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiVoxelToRAS(image=self.image, header=self.header)
        return super().inverse().to(VoxelToRAS)


@register_format
class NiftiVoxelToRAS(VoxelToRAS, _NiftiAffine):
    """
    Affine transformation from voxel space to RAS space, derived from a
    NIfTI header.
    """

    @property
    def matrix(self) -> tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        if self.header is not None:
            return self.header.get_best_affine()[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value

    def inverse(self) -> RASToVoxel:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiRASToVoxel(image=self.image, header=self.header)
        return super().inverse().to(RASToVoxel)
