# dependencies
import nibabel as nb
import numpy as np
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# io
from brainhops.backends import get_array_backend
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import (
    _NIFTI_FIELD_INTENTS,
    _NIFTI_INTENT_DISPVECT,
    _apply_like,
    _apply_overrides,
    _new_nifti,
    _nifti_intent,
    _nifti_shape,
    _NiftiObject,
)
from brainhops.io.base.parsers import Confidence, WriterError
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

    def to_nibabel(
        self, like: tx.Any = None, **overrides
    ) -> tx.Union[nb.Nifti1Image, nb.Nifti2Image]:
        """
        Build the `nibabel` image that encodes this field of RAS coordinates.

        The field array becomes the NIfTI data array, and the header
        carries the displacement-vector intent code that marks the file as
        a field rather than a plain image. The voxel-to-RAS affine of the
        grid is taken from the source header when the field was read from
        one, and is the identity otherwise.

        The field array keeps its own array backend. A `cupy` or `dask`
        array is passed through rather than coerced into `numpy`. When
        `like` is given, non-encoding header fields are copied from it.
        Keyword arguments override header fields last, so an explicit value
        wins.
        """
        field = self.field
        if field is None:
            raise WriterError(
                "This field has no coordinates, so there is nothing to write."
            )
        backend = get_array_backend(field)
        field = backend.asarray(field)
        if field.ndim == 4:
            # NIfTI stores a vector field as a five-dimensional array,
            # with the components in the fifth axis and a singleton axis
            # before them. A four-dimensional array would put the
            # components in the time axis, which the reader misreads.
            field = backend.expand_dims(field, axis=3)
        if self.header is not None:
            affine = self.header.get_best_affine()
        else:
            affine = np.eye(4)
        image = _new_nifti(field, affine)
        image.header.set_intent(_NIFTI_INTENT_DISPVECT)
        _apply_like(image, like)
        _apply_overrides(image, overrides)
        return image
