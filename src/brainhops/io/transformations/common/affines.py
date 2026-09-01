# dependencies
import numpy as np
import typing_extensions as _tx

# io
from brainhops.io.base.nifti import NiftiBasedParser
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS


class NiftiRASToVoxel(RASToVoxel, NiftiBasedParser):
    """
    Affine transformation from RAS space to voxel space, derived from a
    NIfTI header.
    """

    @property
    def matrix(self) -> _tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if self.header is not None:
            return np.linalg.inverse(self.header.get_best_affine())[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray):
        self._matrix = value

    def inverse(self) -> VoxelToRAS:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiVoxelToRAS(image=self.image, header=self.header)
        return super().inverse().to(VoxelToRAS)


class NiftiVoxelToRAS(VoxelToRAS, NiftiBasedParser):
    """
    Affine transformation from voxel space to RAS space, derived from a
    NIfTI header.
    """

    @property
    def matrix(self) -> _tx.Optional[np.ndarray]:
        """The affine matrix of the transformation."""
        if self.header is not None:
            return self.header.get_best_affine()[:-1]
        return None

    @matrix.setter
    def matrix(self, value: np.ndarray):
        self._matrix = value

    def inverse(self) -> RASToVoxel:
        """The inverse transformation, from RAS space to voxel space."""
        if getattr(self, "_matrix", None) is None:
            return NiftiRASToVoxel(image=self.image, header=self.header)
        return super().inverse().to(RASToVoxel)
