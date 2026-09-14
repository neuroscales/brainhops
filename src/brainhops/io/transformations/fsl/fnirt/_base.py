# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import path

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base.nifti import _NiftiObject
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

from .._affines import ImageGeometry
from .._fields import RASToWarpField, WarpFieldToRAS

# FNIRT NIfTI intent codes (see $FSLDIR/src/fnirt/fnirt_file_writer.cpp).
FSL_FNIRT_DISPLACEMENT_FIELD = 2006
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009

# The B-spline order that each FNIRT intent code names. A dense
# deformation field is a first-order (linear) B-spline field sampled on
# the reference grid. A cubic and a quadratic coefficient field are
# third- and second-order B-spline fields sampled on a coarse knot grid.
SPLINE_ORDER = {
    FSL_FNIRT_DISPLACEMENT_FIELD: 1,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS: 2,
    FSL_CUBIC_SPLINE_COEFFICIENTS: 3,
}

# A FNIRT reader claims an FSL-intent NIfTI over the generic RAS
# coordinate field reader, which also scores such files as certain. The
# two are the same container and score identically, so an explicit
# priority is what separates them.
_FNIRT_PRIORITY = 10


class FNIRTTransformation(NiftiBasedTransformation):
    """Base for FNIRT non-linear transformations stored in a NIfTI file.

    A FNIRT transformation maps reference-image coordinates to
    moving-image coordinates, both in FSL scaled-mm coordinates. Turning
    it into a world-space transformation needs the moving image, and the
    reference image. The reference defaults to the geometry of the FNIRT
    file itself, because the field is defined on the reference grid. The
    moving image must be supplied.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete FNIRT transformations inherit from
    it and register themselves.
    """

    moving: tx.Optional[tx.Any] = None
    """The moving (source) image, a nibabel image or header."""

    reference: tx.Optional[tx.Any] = None
    """The reference image. Defaults to the FNIRT file's own geometry."""

    PRIORITY: tx.ClassVar[int] = _FNIRT_PRIORITY

    # --- image keyword handling ---------------------------------------
    #
    # `NiftiParser.from_file` forwards its keyword arguments to
    # `nibabel.load`, which rejects `moving=`/`reference=`. These
    # overrides peel the image keywords off before delegating, then set
    # them on the parsed object.

    @classmethod
    def _pop_images(cls, kwargs: dict) -> tx.Tuple[tx.Any, tx.Any]:
        moving = kwargs.pop("moving", None)
        if moving is None:
            moving = kwargs.pop("src", None)
        reference = kwargs.pop("reference", None)
        if reference is None:
            reference = kwargs.pop("ref", None)
        return moving, reference

    @classmethod
    def _with_images(
        cls, obj: tx.Self, moving: tx.Any, reference: tx.Any
    ) -> tx.Self:
        if moving is not None:
            obj.moving = moving
        if reference is not None:
            obj.reference = reference
        return obj

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_file(file, **kwargs)
        return cls._with_images(obj, moving, reference)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_fileobj(fileobj, **kwargs)
        return cls._with_images(obj, moving, reference)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_bytes(data, **kwargs)
        return cls._with_images(obj, moving, reference)


class FNIRTWarpField(FNIRTTransformation, _xforms.Sequence):
    """A FNIRT warp represented as a B-spline field evaluated lazily.

    A FNIRT warp is a B-spline field on a regular grid. A dense
    deformation field is a first-order field on the reference voxel grid.
    A cubic or quadratic coefficient field is a third- or second-order
    field on a coarse knot grid. This base treats all of them the same
    way: the field is kept on its own grid, and the sequence it produces
    maps reference world (RAS) coordinates to moving world (RAS)
    coordinates through three steps.

    1. An affine carries reference world coordinates onto the warp grid.
    2. A [`DisplacementField`][] holds the field on its own grid, with the
       spline `order` and the `coeff` flag that the intent code names. The
       displacement it adds is evaluated from the B-spline basis when the
       sequence is computed, so a coefficient field is never expanded at
       read time.
    3. An affine carries the warped position into moving world
       coordinates, folding in the moving image scaled-mm geometry and the
       initial FLIRT affine.

    A concrete subclass supplies the intent scoring and the four pieces
    that distinguish one warp from another: the spline order, the warp
    grid, the field array, and the initial affine.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        raise NotImplementedError

    # --- pieces a subclass supplies -----------------------------------

    def _spline_order(self) -> int:
        """The B-spline order of the field."""
        raise NotImplementedError

    def _field_coeff(self) -> bool:
        """Whether the field array holds spline coefficients."""
        raise NotImplementedError

    def _field_bound(self) -> tx.Union[str, float]:
        """The boundary condition for sampling the field."""
        raise NotImplementedError

    def _field_array(self, ref: ImageGeometry) -> np.ndarray:
        """The field array, as a relative displacement in scaled-mm.

        The array has shape `(*grid, 3)`, on the warp grid. For a dense
        deformation field the displacement is relative to the reference
        scaled-mm coordinate of the voxel. For a coefficient field the
        stored coefficients evaluate, through the spline basis, to that
        same relative displacement.
        """
        raise NotImplementedError

    def _ref_to_field(self, ref: ImageGeometry) -> np.ndarray:
        """The affine from reference voxels to warp-grid voxels."""
        raise NotImplementedError

    def _initial_affine(self) -> np.ndarray:
        """The initial FLIRT affine, as a reference-to-source scaled-mm
        matrix.

        A dense deformation field already has the initial affine folded
        in, so this is the identity for one. A coefficient field stores it
        separately, so it is applied on top of the spline displacement.
        """
        raise NotImplementedError

    def _reference_geometry(self) -> ImageGeometry:
        """The geometry that places the warp in reference world space."""
        raise NotImplementedError

    # --- the chain ----------------------------------------------------

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The transformations mapping reference RAS to moving RAS.

        Reading this property resolves the chain from the warp data and
        the image geometries. It raises when a required image is missing.
        The chain is recomputed on each access, so a later change to an
        input image is reflected.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if self.header is None:
            return []
        if self.moving is None:
            raise ValueError(
                "A FNIRT warp maps to the moving image, whose geometry the "
                "warp file does not contain, so the moving image is needed "
                "to place the warp in world coordinates. Pass moving=... "
                "when loading, or set it on the transformation."
            )

        ref = self._reference_geometry()
        mov = ImageGeometry(self.moving)
        order = self._spline_order()
        field = self._field_array(ref)
        ref_to_field = self._ref_to_field(ref)
        ref_to_source = self._initial_affine()

        return _warp_chain(
            field=field,
            order=order,
            coeff=self._field_coeff(),
            bound=self._field_bound(),
            ref_to_field=ref_to_field,
            ref=ref,
            mov=mov,
            ref_to_source=ref_to_source,
        )

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else list(value)

    def _resolvable(self) -> bool:
        """Whether the chain can be resolved without raising."""
        raise NotImplementedError

    def _inspect(self) -> tx.List[_xforms.Transformation]:
        """The transformations for repr, length and iteration.

        Returns the resolved chain when it can be resolved, an explicitly
        assigned list when one was set, and an empty list otherwise.
        Inspecting an incompletely specified warp therefore does not
        raise.

        The collection methods that call this (`__len__`, `__iter__`,
        `__getitem__`) are defined on each concrete subclass rather than
        here, because the builder regenerates a class's own collection
        methods and a subclass would otherwise not inherit them.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if not self._resolvable():
            return []
        return self.transformations


# ----------------------------------------------------------------------
#   CHAIN CONSTRUCTION
# ----------------------------------------------------------------------


def _warp_chain(
    field: np.ndarray,
    order: int,
    coeff: bool,
    bound: tx.Union[str, float],
    ref_to_field: np.ndarray,
    ref: ImageGeometry,
    mov: ImageGeometry,
    ref_to_source: np.ndarray,
) -> tx.List[_xforms.Transformation]:
    """Build the lazy reference-RAS to moving-RAS chain for a warp field.

    The field stays on its own grid. The two affines place that grid in
    world coordinates and let the spline evaluation happen during reslice.

    The B-spline order fixes two grid conventions that differ between FSL
    and the resampler the data model uses. FSL anchors the basis on the
    left of each cell, while the resampler centres it, which is a shift of
    `(order - 1) / 2` grid units. And the displacement the spline produces
    is expressed in scaled-mm, while a [`DisplacementField`][] adds it in
    units of its own grid. Both are absorbed into the affines and the
    field values, so the produced chain reproduces FSL's warp.
    """
    order = int(order)
    offset = (order - 1) / 2.0
    offsets = np.array([offset, offset, offset], dtype=np.float64)

    field_to_ref = np.linalg.inv(ref_to_field)
    # `grid_to_fsl` carries a warp-grid coordinate to reference scaled-mm.
    grid_to_fsl_lin = ref.vox2fsl[:3, :3] @ field_to_ref[:3, :3]
    grid_to_fsl_off = (
        ref.vox2fsl[:3, :3] @ field_to_ref[:3, 3] + ref.vox2fsl[:3, 3]
    )
    src_lin = ref_to_source[:3, :3]
    src_off = ref_to_source[:3, 3]

    # The displacement is stored on the grid pre-scaled into grid units in
    # the source-aligned frame, so the field's own add-of-grid composes to
    # the warped position with the initial affine applied to the base.
    combined = src_lin @ grid_to_fsl_lin
    prescale = np.linalg.inv(combined)
    prescaled = np.asarray(field, dtype=np.float64) @ prescale.T

    # ref RAS -> warp-grid voxel, with the spline anchoring offset added.
    ras_to_grid = ref_to_field @ ref.ras2vox
    ras_to_grid[:3, 3] += offsets

    # warp-grid frame -> moving RAS. The initial affine acts on the base
    # position, the moving scaled-mm geometry carries it into world space,
    # and the anchoring offset added on the input side is removed here.
    fsl_to_ras = mov.fsl2ras
    grid_to_ras_lin = fsl_to_ras[:3, :3] @ src_lin @ grid_to_fsl_lin
    grid_to_ras_off = (
        fsl_to_ras[:3, :3] @ (src_lin @ grid_to_fsl_off + src_off)
        + fsl_to_ras[:3, 3]
        - grid_to_ras_lin @ offsets
    )
    grid_to_ras = np.eye(4, dtype=np.float64)
    grid_to_ras[:3, :3] = grid_to_ras_lin
    grid_to_ras[:3, 3] = grid_to_ras_off

    return [
        RASToWarpField(matrix=ras_to_grid[:-1]),
        _xforms.DisplacementField(
            field=prescaled, order=order, bound=bound, coeff=coeff
        ),
        WarpFieldToRAS(matrix=grid_to_ras[:-1]),
    ]
