import typing as tx
from os import PathLike

# optionals
if tx.TYPE_CHECKING:
    import nibabel as nb
else:
    try:
        import nibabel as nb
    except ImportError:

        class nb:
            Nifti1Header = None
            Nifti1Image = None


from brainhops.datamodel import transformations as _xforms
from brainhops.io.transformations.common.affines import (
    NiftiRASToVoxel,
)
from brainhops.io.transformations.common.base import NiftiBasedTransformation
from brainhops.io.transformations.fsl.displacement._field import (
    FSLDisplacementField,
)


class FslDisplacementTransformation(
    _xforms.Sequence, NiftiBasedTransformation
):
    """
    A NIfTI-based nonlinear transformation that may be stored either as
    dense displacement fields or as sparse B-spline coefficients
    """

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        if getattr(self, "_transformations", None) is not None:
            return self._transformations

        displacement_field = FSLDisplacementField(
            image=self.image, header=self.header
        )
        _ = displacement_field.is_spline_coefficients

        self._transformations = [
            NiftiRASToVoxel(image=self.image, header=self.header),
            displacement_field,
            NiftiRASToVoxel(image=self.image, header=self.header).inverse(),
        ]
        return self._transformations

    @transformations.setter
    def transformations(
        self, value: tx.Tuple[_xforms.Transformation, ...]
    ) -> None:
        self._transformations = value

    @classmethod
    def sniff_file(cls, fileobj: tx.Union[str, PathLike]) -> bool:
        """
        Check whether a file matches this transformation's expected format.
        """
        try:
            obj = cls.from_nifti(nb.load(fileobj))
            _ = obj.transformations[1].is_spline_coefficients
            return True
        except (ValueError, NotImplementedError):
            return False
