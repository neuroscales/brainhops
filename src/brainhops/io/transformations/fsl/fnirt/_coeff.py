# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import _nifti_intent, _NiftiObject
from brainhops.io.base.parsers import Confidence

from ._base import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSL_DCT_COEFFICIENTS,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS,
    FNIRTTransformation,
)

_COEFFICIENT_INTENTS = frozenset(
    {
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
    }
)

# The spline order that each coefficient intent code names.
_SPLINE_ORDER = {
    FSL_CUBIC_SPLINE_COEFFICIENTS: 3,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS: 2,
}


@register_format
class FNIRTCoefficientField(
    FNIRTTransformation,
    _xforms.Sequence,
    mapping=HIDE_IF_NONE,
):
    """A FNIRT coefficient field stored in a NIfTI file.

    A coefficient field holds the coefficients of a B-spline basis on a
    coarse grid overlaid on the reference image. Evaluating the basis
    produces the displacements of a warp field. The knot spacing is
    stored in the pixel sizes, the reference pixel sizes in the intent
    parameters, and the initial linear registration in the sform.

    The reader recognizes coefficient fields and exposes their
    parameters, but does not yet evaluate the spline basis, so requesting
    the world-space transformations raises an error.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a FNIRT coefficient field."""
        if _nifti_intent(header) in _COEFFICIENT_INTENTS:
            return Confidence.CERTAIN
        return Confidence.NO

    @property
    def spline_order(self) -> tx.Optional[int]:
        """The B-spline order, `3` for cubic, `2` for quadratic."""
        if self.header is None:
            return None
        return _SPLINE_ORDER.get(_nifti_intent(self.header))

    @property
    def knot_spacing(self) -> tx.Optional[np.ndarray]:
        """The knot spacing, in reference voxels, stored in the pixdims."""
        if self.header is None:
            return None
        zooms = self.header.get_zooms()[:3]
        return np.abs(np.asarray(zooms, dtype=np.float64))

    @property
    def reference_pixdim(self) -> tx.Optional[np.ndarray]:
        """The reference pixel sizes, stored in the intent parameters."""
        header = self.header
        if header is None:
            return None
        return np.array(
            [
                float(header["intent_p1"]),
                float(header["intent_p2"]),
                float(header["intent_p3"]),
            ],
            dtype=np.float64,
        )

    @property
    def initial_affine(self) -> tx.Optional[np.ndarray]:
        """The initial FLIRT src-to-ref affine, stored in the sform."""
        if self.header is None:
            return None
        return np.asarray(self.header.get_sform(), dtype=np.float64)

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """Not implemented: evaluating the spline basis is not supported."""
        if getattr(self, "_transformations", None) is not None:
            return self._transformations
        if _nifti_intent(self.header) == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "FNIRT discrete-cosine-transform coefficient fields are not "
                "supported."
            )
        raise NotImplementedError(
            "Reading a FNIRT coefficient field requires evaluating its "
            "B-spline basis on the knot grid and folding in the initial "
            "linear registration, which is not yet implemented. The spline "
            "order, knot spacing, reference pixel sizes and initial affine "
            "are available as attributes. Convert the coefficient field to "
            "a deformation field with FSL (`fnirtfileutils`) and read that "
            "instead."
        )

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else list(value)
