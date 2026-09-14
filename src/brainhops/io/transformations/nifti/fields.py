# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_FIELD_INTENTS,
    _nifti_intent,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.base.parsers import Confidence
from brainhops.io.transformations.base.fields import RASCoordinatesField
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation


@register_format
class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """
        float a NIfTI header as a deformation field.

        This is the case the intent code was made for: a file that
        declares itself a displacement or vector field is certainly one,
        and nothing else in the registry should outrank it.
        """
        if _nifti_intent(header) in _NIFTI_FIELD_INTENTS:
            return Confidence.CERTAIN
        # The intent code is often left unset -- SPM writes none -- so
        # fall back to the shape: a field of RAS coordinates carries a
        # trailing axis of length 3. A plain image has no such axis, and
        # scores zero rather than competing with the affine readers.
        shape = _nifti_shape(header)
        if shape and len(shape) >= 4 and shape[-1] == 3:
            return Confidence.MAYBE
        return Confidence.NO

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The field of RAS coordinates."""
        return self.data

    @field.setter
    def field(self, value: tx.Optional[ArrayProtocol]) -> None:
        # `field` is this format's name for the image data, so the two
        # must stay one value. Without a setter the struct's generated
        # `__init__` cannot assign the inherited `field` at all.
        self.data = value
