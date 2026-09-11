# dependencies
import typing_extensions as tx

from brainhops._core.backends import get_backend

# core
from brainhops._core.typing import ArrayProtocol
from brainhops.io.transformations.base.fields import VoxelDisplacementField
from brainhops.io.transformations.common.base import NiftiBasedTransformation

# constants retrieved from fslpy on Jun 29th 2026
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009
FSL_FNIRT_DISPLACEMENT_FIELD = 2006


class FSLDisplacementField(VoxelDisplacementField, NiftiBasedTransformation):
    """
    Field of RAS displacements, stored in a NIfTI file.
    """

    @property
    def is_spline_coefficients(self) -> bool:
        header = self.header
        if header is None:
            raise ValueError(
                "No header/image available to determine transform format."
            )

        intent_code = int(header.get("intent_code", 0))
        if intent_code == FSL_CUBIC_SPLINE_COEFFICIENTS:
            self.coeff = True
            self.order = 3
        elif intent_code == FSL_QUADRATIC_SPLINE_COEFFICIENTS:
            self.coeff = True
            self.order = 2
        elif intent_code == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "DCT-basis FNIRT coefficient fields are not supported."
            )
        elif intent_code == FSL_FNIRT_DISPLACEMENT_FIELD:
            self.coeff = False
        else:
            raise ValueError(f"Unrecognized intent code: {intent_code}")

        return self.coeff

    @property
    def field(self) -> tx.Optional[ArrayProtocol]:
        """The field of RAS coordinates."""
        if getattr(self, "_field", None) is None:
            if self.image is not None:
                self._field = get_backend().from_array(self.image.dataobj)
            else:
                self._field = None
        return self._field

    @field.setter
    def field(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._field = value
