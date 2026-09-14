# dependencies
import typing_extensions as tx
from bagof.hints.array import ArrayProtocol

# io
from brainhops.io.base.nifti import NiftiParser
from brainhops.io.transformations.base.fields import RASCoordinatesField


class NiftiRASCoordinatesField(RASCoordinatesField, NiftiParser):
    """
    Field of RAS coordinates, stored in a NIfTI file.
    """

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The field of RAS coordinates."""
        return self.data
