"""Affines between voxel and RAS space, derived from a NIfTI header."""

import nibabel as nb
import numpy as np
import typing_extensions as tx

from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common.nifti._geometry import _voxel_to_ras
from brainhops.io.common.nifti._header import (
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _NiftiObject,
)
from brainhops.io.transformations.base import AffineTransformationFormat
from brainhops.io.transformations.base._conversions import (
    convert_instance,
    converts_to,
)
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation


class _NiftiAffine(AffineTransformationFormat, NiftiBasedTransformation):
    """Shared scoring and writing for the affines derived from a header."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        Score a header as a bare affine, always `Confidence.WEAK`.

        Every NIfTI file carries an affine, but loading a `.nii` file rarely
        means loading its voxel-to-RAS matrix, which is reached through the
        image instead. The weak score keeps these readers as a last resort that
        never outranks an image or a field.
        """
        return Confidence.WEAK

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        """Return the (4, 4) voxel-to-RAS matrix that this affine encodes."""
        raise NotImplementedError

    def _xform_code(self) -> int:
        """
        Return the xform code of the source header, the sform code first, or 2.
        """
        header = self.header
        if header is not None:
            _, code = header.get_sform(coded=True)
            if code:
                return int(code)
            _, code = header.get_qform(coded=True)
            if code:
                return int(code)
        return 2

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build a `nibabel` image whose affine is this transformation.

        NIfTI stores an affine only as the geometry of a data array, so a
        single-voxel placeholder volume carries it. The voxel-to-RAS matrix
        becomes both the sform and the qform, under the code recorded by the
        source header. Non-encoding header fields are copied from `like`, and
        `overrides` are applied last, but the geometry always comes from this
        transformation.
        """
        matrix = self._voxel_to_ras_matrix()
        data = np.zeros((1, 1, 1), dtype="float32")
        code = self._xform_code()
        image = _new_nifti(data, matrix)
        _apply_like(image, like)
        image.header.set_sform(matrix, code=code)
        image.header.set_qform(matrix, code=code)
        _apply_overrides(image, overrides)
        return image


class NiftiRASToVoxel(RASToVoxel, _NiftiAffine):
    """
    Affine from RAS to voxel space, derived from a NIfTI header.

    !!! note "Not a registered format"
        The header encodes the voxel-to-RAS affine, and the RAS-to-voxel affine
        is its computed inverse. The two cannot be told apart by content, since
        they share the container, the extension and the confidence, so
        registering both would make every `.nii` file ambiguous. This affine is
        reached through `NiftiVoxelToRAS.inverse()`.
    """

    @property
    def data(self) -> tx.Optional[np.ndarray]:
        """
        The stored affine matrix, which the `matrix` view reads.

        The matrix is derived from the header unless it is set explicitly. It
        is named like the image `data` of the NIfTI parser, but it is stored
        separately, so the image read by the parser never stands in for an
        explicit matrix.
        """
        if getattr(self, "_explicit_matrix", None) is not None:
            return self._explicit_matrix
        if self.header is not None:
            return np.linalg.inv(self.header.get_best_affine())[:-1]
        return None

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._explicit_matrix = value

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        # The constructor stores data= in _data, where the NIfTI parser also
        # caches its image, so the value is moved to the slot of the matrix.
        data = arguments.get("data")
        if data is not None:
            self._data = None
            self._explicit_matrix = data

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> VoxelToRAS:
        """Return the inverse transformation, from voxel to RAS space."""
        if getattr(self, "_explicit_matrix", None) is None:
            return NiftiVoxelToRAS(image=self.image, header=self.header)
        return super().inverse(compute=compute, **kwargs).to(VoxelToRAS)

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        # NIfTI stores the voxel-to-RAS map, which is the inverse of this one.
        return _voxel_to_ras(self.inverse())


@register_format
class NiftiVoxelToRAS(VoxelToRAS, _NiftiAffine):
    """Affine from voxel to RAS space, derived from a NIfTI header."""

    HINTS = ("nifti",)

    @property
    def data(self) -> tx.Optional[np.ndarray]:
        """
        The stored affine matrix, which the `matrix` view reads.

        See
        [`NiftiRASToVoxel.data`][brainhops.io.transformations.nifti.affines.NiftiRASToVoxel.data].
        """
        if getattr(self, "_explicit_matrix", None) is not None:
            return self._explicit_matrix
        if self.header is not None:
            return self.header.get_best_affine()[:-1]
        return None

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._explicit_matrix = value

    def __post_init__(self, arguments: tx.Any) -> None:
        super().__post_init__(arguments)
        data = arguments.get("data")
        if data is not None:
            self._data = None
            self._explicit_matrix = data

    # --- conversions --------------------------------------------------
    # Another transformation is converted, as `t.to(cls)` converts it;
    # anything else is read or copied as the bases do.

    @classmethod
    def from_any(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_any(other, *args, **kwargs)

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        if converts_to(cls, other):
            return convert_instance(cls, other, *args, **kwargs)
        return super().from_instance(other, *args, **kwargs)

    def inverse(self, compute: bool = False, **kwargs) -> RASToVoxel:
        """Return the inverse transformation, from RAS to voxel space."""
        if getattr(self, "_explicit_matrix", None) is None:
            return NiftiRASToVoxel(image=self.image, header=self.header)
        return super().inverse(compute=compute, **kwargs).to(RASToVoxel)

    def _voxel_to_ras_matrix(self) -> np.ndarray:
        return _voxel_to_ras(self)
