# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _apply_like,
    _new_nifti,
    _NiftiObject,
    _strip_bad_extensions,
    _voxel_to_ras,
)
from brainhops.io.base.parsers import Confidence
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation


class _NiftiAffine(NiftiBasedTransformation):
    """Shared scoring and writing for affines derived from a NIfTI header."""

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

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        """The `(4, 4)` voxel-to-RAS matrix this affine encodes."""
        raise NotImplementedError

    def _xform_code(self) -> int:
        """The xform code to store, taken from the source header."""
        header = self.header
        if header is not None:
            _, code = header.get_sform(coded=True)
            if code:
                return int(code)
            _, code = header.get_qform(coded=True)
            if code:
                return int(code)
        return 2

    def to_nibabel(self, like: tx.Any = None, **kwargs) -> nb.Nifti1Image:
        """
        Build a `nibabel` image whose affine is this transformation.

        NIfTI stores an affine only as the geometry of a data array, so a
        single-voxel placeholder volume carries it. The voxel-to-RAS
        matrix becomes both the sform and the qform, under the code the
        source header recorded.

        When `like` is given, non-geometry header fields are copied from it.
        The geometry always comes from this transformation.
        """
        matrix = self._voxel_to_ras_matrix()
        data = np.zeros((1, 1, 1), dtype="float32")
        code = self._xform_code()
        image = _new_nifti(data, matrix)
        _apply_like(image, like)
        image.header.set_sform(matrix, code=code)
        image.header.set_qform(matrix, code=code)
        _strip_bad_extensions(image)
        return image


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

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        # This transformation maps RAS to voxel, so its inverse maps
        # voxel to RAS, which is what NIfTI stores.
        return _voxel_to_ras(self.inverse())


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

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        return _voxel_to_ras(self)
