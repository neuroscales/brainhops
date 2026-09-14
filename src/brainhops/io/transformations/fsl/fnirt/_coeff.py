# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# core
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import _nifti_intent, _NiftiObject
from brainhops.io.base.parsers import Confidence

from .._affines import ImageGeometry
from .._repr import stored_repr
from ._base import (
    FSL_CUBIC_SPLINE_COEFFICIENTS,
    FSL_DCT_COEFFICIENTS,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS,
    SPLINE_ORDER,
    FNIRTWarpField,
)

_COEFFICIENT_INTENTS = frozenset(
    {
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
    }
)


@register_format
class FNIRTCoefficientField(FNIRTWarpField, mapping=HIDE_IF_NONE):
    """A FNIRT coefficient field stored in a NIfTI file.

    A coefficient field holds the coefficients of a B-spline basis on a
    coarse knot grid overlaid on the reference image. The knot spacing is
    stored in the pixel sizes, the reference pixel sizes in the intent
    parameters, and the initial linear registration in the sform.

    A cubic coefficient field is a third-order B-spline field, and a
    quadratic one is a second-order field. The reader keeps the
    coefficients on the knot grid and returns a sequence of
    transformations that maps reference-image world (RAS) coordinates to
    moving-image world (RAS) coordinates. The B-spline basis is evaluated
    when the sequence is computed, so the coefficients are never expanded
    onto the reference grid at read time. The reference and moving images
    must both be supplied, because the coefficient file is defined on the
    knot grid and carries neither image's geometry.

    A discrete-cosine-transform coefficient field is not supported.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a FNIRT coefficient field."""
        if _nifti_intent(header) in _COEFFICIENT_INTENTS:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- exposed parameters -------------------------------------------

    @property
    def spline_order(self) -> tx.Optional[int]:
        """The B-spline order, `3` for cubic, `2` for quadratic."""
        if self.header is None:
            return None
        return SPLINE_ORDER.get(_nifti_intent(self.header))

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

    # --- warp pieces --------------------------------------------------

    def _spline_order(self) -> int:
        intent = _nifti_intent(self.header)
        if intent == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "FNIRT discrete-cosine-transform coefficient fields are not "
                "supported. Convert the field to a deformation field with "
                "FSL (`fnirtfileutils`) and read that instead."
            )
        order = SPLINE_ORDER.get(intent)
        if order is None:
            raise NotImplementedError(
                f"Unsupported FNIRT coefficient intent code {intent!r}."
            )
        return order

    def _field_coeff(self) -> bool:
        return True

    def _field_bound(self) -> tx.Union[str, float]:
        # FSL evaluates the basis with out-of-grid coefficients taken as
        # zero. The knot grid extends beyond the reference field of view,
        # so within the reference image no evaluation reaches the grid
        # edge and this choice never affects the result.
        return "constant"

    def _field_array(self, ref: ImageGeometry) -> np.ndarray:
        backend = get_array_backend()
        field = backend.asarray(self.data)
        if field.ndim == 5 and field.shape[3] == 1:
            field = field[:, :, :, 0, :]
        return np.asarray(field, dtype=np.float64)

    def _ref_to_field(self, ref: ImageGeometry) -> np.ndarray:
        # The knot spacing is stored in the coefficient file's pixdims, in
        # units of the reference voxels used when the warp was generated.
        # A reference of a different resolution is rescaled by the ratio of
        # the original reference pixel sizes to the current ones.
        knot_spacing = self.knot_spacing
        ratio = self.reference_pixdim / ref.pixdim
        field_to_ref = knot_spacing * ratio
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.diag(1.0 / field_to_ref)
        return matrix

    def _initial_affine(self) -> np.ndarray:
        sform = self.initial_affine
        if sform is None or not np.all(np.isfinite(sform)):
            return np.eye(4, dtype=np.float64)
        # The sform is the source-to-reference affine. The chain applies
        # its inverse, the reference-to-source affine, to the base.
        return np.linalg.inv(sform)

    def _reference_geometry(self) -> ImageGeometry:
        if self.reference is None:
            raise ValueError(
                "A FNIRT coefficient field is stored on a coarse knot grid, "
                "not on the reference grid, so the reference image is needed "
                "to place the warp in world coordinates. Pass reference=... "
                "when loading, or set it on the transformation."
            )
        return ImageGeometry(self.reference)

    def _resolvable(self) -> bool:
        if self.header is None:
            return False
        if self.moving is None or self.reference is None:
            return False
        return _nifti_intent(self.header) != FSL_DCT_COEFFICIENTS

    def __repr__(self) -> str:
        return stored_repr(
            self, ("spline_order", "knot_spacing", "moving", "reference")
        )

    def __len__(self) -> int:
        return len(self._inspect())

    def __iter__(self) -> tx.Iterator[_xforms.Transformation]:
        return iter(self._inspect())

    def __getitem__(
        self, index: tx.Union[int, slice]
    ) -> _xforms.Transformation:
        return self._inspect()[index]
