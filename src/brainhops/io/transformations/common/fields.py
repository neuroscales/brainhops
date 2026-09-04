# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# core
from brainhops._core.backends import da

# io
from brainhops.io.transformations.base.fields import RASCoordinatesField
from brainhops.io.transformations.common.base import NiftiBasedTransformation


class NiftiRASCoordinatesField(RASCoordinatesField, NiftiBasedTransformation):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The field of RAS coordinates."""
        field = None
        if self.image is not None:
            field = self.image.dataobj
            if da:
                field = da.from_array(field, fancy=False, name=self.image)
        return field
