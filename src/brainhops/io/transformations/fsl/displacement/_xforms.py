import typing as _tx
from os import PathLike

import nibabel as nb

from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS
from brainhops.io.transformations.base.fields import RASDisplacementField
from brainhops.io.transformations.common.affines import (
    NiftiRASToVoxel,
)
from brainhops.io.transformations.common.base import NiftiBasedTransformation
from brainhops.io.transformations.fsl.displacement._field import (
    FSLRASDisplacementField,
)

_NiftiLike = _tx.Union[
    nb.Nifti1Header,
    nb.Nifti1Image,
    str,
    PathLike,
    _tx.BinaryIO
]


class FslDisplacementTransformation(_xforms.Sequence, NiftiBasedTransformation):
    """
    A NIfTI-based nonlinear transformation that may be stored either as
    dense displacement fields or as sparse B-spline coefficients (FNIRT
    `--cout` style). Displacement data is reconstructed from spline
    coefficients lazily, only for the slice actually requested.
    """

    _transformations: _tx.Optional[_tx.Tuple[
        _tx.Optional[RASToVoxel],
        _tx.Optional[RASDisplacementField],
        _tx.Optional[VoxelToRAS]
    ]] = None
    _reference_image: _tx.Optional[nb.Nifti1Image] = None
    _target_shape: _tx.Optional[_tx.Tuple[int, int, int]] = None

    @classmethod
    def from_(
        cls,
        other: _NiftiLike,
        reference: _tx.Optional[_NiftiLike] = None,
    ) -> _tx.Self:
        """Create an FslDisplacementTransformation from a NIfTI image."""
        return cls.from_nifti(other, reference=reference)

    @classmethod
    def from_file(
        cls,
        fileobj: _tx.Union[str, PathLike],
        reference: _tx.Optional[_NiftiLike] = None,
    ) -> _tx.Self:
        """Create an FslDisplacementTransformation from a NIfTI file."""
        return cls.from_nifti(nb.load(fileobj), reference=reference)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        reference: _tx.Optional[_NiftiLike] = None,
    ) -> _tx.Self:
        """Create an FslDisplacementTransformation from a NIfTI file in bytes."""
        return cls.from_nifti(nb.Nifti1Image.from_bytes(data), reference=reference)

    @classmethod
    def from_nifti(
        cls,
        nifti: _NiftiLike,
        reference: _tx.Optional[_NiftiLike] = None,
    ) -> _tx.Self:
        """
        Create an FslDisplacementTransformation from a NIfTI image.

        Parameters
        ----------
        nifti : header, image, path, or bytes-like
            The coefficient or displacement field file itself.
        reference : header, image, path, or bytes-like, optional
            A reference image, equivalent to FNIRT's `--ref` parameter.
            Required when the underlying file stores spline coefficients.
            Both the target shape and the affine used to build this
            sequence's RAS<->voxel transforms are derived from this image
            — never from the coefficient file's own affine, which
            describes knot spacing, not the dense field's actual voxel
            grid.
        """
        if isinstance(nifti, nb.Nifti1Header):
            obj = cls(header=nifti)
        elif isinstance(nifti, nb.Nifti1Image):
            obj = cls(image=nifti)
        else:
            return cls.from_nifti(nb.load(nifti), reference=reference)

        obj._apply_reference(reference=reference)
        return obj

    def _apply_reference(self, reference: _tx.Optional[_NiftiLike]) -> None:
        """
        Resolve and store `_reference_image`/`_target_shape` from the
        constructor-overload `reference` argument. Kept as a separate
        method (rather than living in `__init__`) so it can be reused
        across every construction path without touching the base class's
        own `__init__`.
        """
        if reference is None:
            return

        if isinstance(reference, nb.Nifti1Header):
            # Headers carry shape/affine info directly; no pixel data
            # needed for our purposes, so use it as-is rather than forcing
            # it through nb.load.
            ref_shape = tuple(int(d) for d in reference.get_data_shape()[:3])
            self._reference_image = reference
            self._target_shape = ref_shape
            return

        if isinstance(reference, nb.Nifti1Image):
            self._reference_image = reference
            self._target_shape = tuple(int(d) for d in reference.shape[:3])
            return

        if isinstance(reference, (str, PathLike)):
            loaded = nb.load(reference)
        else:
            loaded = nb.Nifti1Image.from_bytes(reference.read())

        self._reference_image = loaded
        self._target_shape = tuple(int(d) for d in loaded.shape[:3])

    @property
    def transformations(self) -> _tx.Tuple[
        _tx.Optional[RASToVoxel],
        _tx.Optional[RASDisplacementField],
        _tx.Optional[VoxelToRAS]
    ]:
        """The transformations that make up the sequence."""
        _transformations = getattr(self, "_transformations", None)
        if _transformations is not None:
            return _transformations

        displacement_field = FSLRASDisplacementField(
            image=self.image, header=self.header
        )

        if displacement_field.is_spline_coefficients and self._reference_image is None:
            raise ValueError(
                "A reference image is required to build RAS<->voxel "
                "transforms for spline-coefficient FNIRT files — "
            )
        elif not displacement_field.is_spline_coefficients and self._reference_image is not None:
            raise ValueError(
                "Reference files should not be provided for dense displancement fields"
            )

        affine_source_image = self._reference_image or self.image
        affine_source_header = (
            self._reference_image.header if self._reference_image else self.header
        )

        if self._target_shape is not None:
            displacement_field._target_shape = self._target_shape

        self._transformations = (
            NiftiRASToVoxel(image=affine_source_image,
                            header=affine_source_header),
            displacement_field,
            NiftiRASToVoxel(image=affine_source_image,
                            header=affine_source_header).inverse(),
        )
        return self._transformations

    @classmethod
    def sniff_file(cls, fileobj: _tx.Union[str, PathLike], reference: _tx.Optional[_NiftiLike] = None) -> bool:
        """Check whether a file matches this transformation's expected format."""
        try:
            obj = cls.from_nifti(nb.load(fileobj), reference=reference)
            _ = obj.transformations[1].is_spline_coefficients
            return True
        except (ValueError, NotImplementedError):
            return False
