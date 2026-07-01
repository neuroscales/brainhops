import typing as _tx
from os import PathLike

import nibabel as nb

from brainhops.datamodel import transformations as _xforms
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

    reference_image: _tx.Optional[nb.Nifti1Image] = None
    target_shape: _tx.Optional[_tx.Tuple[int, int, int]] = None

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
        """
        if isinstance(nifti, nb.Nifti1Header):
            obj = cls(header=nifti)
        elif isinstance(nifti, nb.Nifti1Image):
            obj = cls(image=nifti)
        else:
            return cls.from_nifti(nb.load(nifti), reference=reference)

        obj._apply_reference(reference=reference)
        obj._ensure_transformations()
        return obj

    def _apply_reference(self, reference: _tx.Optional[_NiftiLike]) -> None:
        """
        Resolve and store `reference_image`/`target_shape` from the
        constructor-overload `reference` argument.
        """
        if reference is None:
            return

        if isinstance(reference, nb.Nifti1Header):
            ref_shape = tuple(int(d) for d in reference.get_data_shape()[:3])
            self.reference_image = reference
            self.target_shape = ref_shape
            return

        if isinstance(reference, nb.Nifti1Image):
            self.reference_image = reference
            self.target_shape = tuple(int(d) for d in reference.shape[:3])
            return

        if isinstance(reference, (str, PathLike)):
            loaded = nb.load(reference)
        else:
            loaded = nb.Nifti1Image.from_bytes(reference.read())

        self.reference_image = loaded
        self.target_shape = tuple(int(d) for d in loaded.shape[:3])

    def _ensure_transformations(self) -> None:
        """
        Lazily populate the inherited `transformations` field (from
        `Sequence`) the first time it's needed
        """
        if self.transformations is not None:
            return

        displacement_field = FSLRASDisplacementField(
            image=self.image, header=self.header
        )

        if displacement_field.is_spline_coefficients and self.reference_image is None:
            raise ValueError(
                "A reference image is required to build RAS<->voxel "
                "transforms for spline-coefficient FNIRT files — "
            )
        elif not displacement_field.is_spline_coefficients and self.reference_image is not None:
            raise ValueError(
                "Reference files should not be provided for dense displacement fields"
            )

        affine_source_image = self.reference_image or self.image
        affine_source_header = (
            self.reference_image.header if self.reference_image else self.header
        )

        if self.target_shape is not None:
            displacement_field._target_shape = self.target_shape

        self.transformations = [
            NiftiRASToVoxel(image=affine_source_image,
                            header=affine_source_header),
            displacement_field,
            NiftiRASToVoxel(image=affine_source_image,
                            header=affine_source_header).inverse(),
        ]

    @classmethod
    def sniff_file(cls, fileobj: _tx.Union[str, PathLike], reference: _tx.Optional[_NiftiLike] = None) -> bool:
        """Check whether a file matches this transformation's expected format."""
        try:
            obj = cls.from_nifti(nb.load(fileobj), reference=reference)
            _ = obj.transformations[1].is_spline_coefficients
            return True
        except (ValueError, NotImplementedError):
            return False
