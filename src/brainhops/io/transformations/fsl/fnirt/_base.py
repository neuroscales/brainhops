# dependencies
import numpy as np
import typing_extensions as tx

# externals

# core
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.images import Image

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import _nifti_intent, _NiftiObject
from brainhops.io.base.parsers import Confidence
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

from .._affines import _ImageGeometry
from .._fields import RASToWarpField, WarpFieldToRAS
from .._repr import stored_repr

# FNIRT NIfTI intent codes. These constants are defined in `nifti1.h`.
FSL_FNIRT_DISPLACEMENT_FIELD = 2006
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009

_FNIRT_INTENTS = frozenset(
    {
        FSL_FNIRT_DISPLACEMENT_FIELD,
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
    }
)

_COEFFICIENT_INTENTS = frozenset(
    {
        FSL_CUBIC_SPLINE_COEFFICIENTS,
        FSL_QUADRATIC_SPLINE_COEFFICIENTS,
        FSL_DCT_COEFFICIENTS,
    }
)

# The B-spline order that each FNIRT intent code names. A dense
# deformation field is a first-order (linear) B-spline field sampled on
# the reference grid. A cubic and a quadratic coefficient field are
# third- and second-order B-spline fields sampled on a coarse knot grid.
_SPLINE_ORDER = {
    FSL_FNIRT_DISPLACEMENT_FIELD: 1,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS: 2,
    FSL_CUBIC_SPLINE_COEFFICIENTS: 3,
}

# The moving and reference images may be a nibabel header or image, or a
# brainhops image.
_ImageLike = tx.Union[_NiftiObject, Image]


@register_format
class FNIRTWarpField(NiftiBasedTransformation, _xforms.Sequence):
    """A FNIRT non-linear transformation stored in a NIfTI file.

    FNIRT writes its non-linear registration as one of two things, which a
    NIfTI intent code tells apart. A *deformation field* stores, per
    reference voxel, the moving location that the voxel maps to, in FSL
    scaled-mm coordinates. A *coefficient field* stores the coefficients
    of a B-spline basis on a coarse knot grid overlaid on the reference
    image. A single reader handles both, because both are B-spline fields
    on a regular grid and differ only in the spline order, in whether the
    grid holds coefficients or sampled values, and in where the grid sits.

    | Intent | Kind | Order | Grid |
    | ------ | ---- | ----- | ---- |
    | 2006 | deformation | 1 | reference voxels |
    | 2007 | cubic coefficients | 3 | knot grid |
    | 2009 | quadratic coefficients | 2 | knot grid |

    The reader keeps the field on its own grid and returns a sequence of
    transformations that maps reference-image world (RAS) coordinates to
    moving-image world (RAS) coordinates. The B-spline basis is evaluated
    only when the sequence is computed, so a coefficient field is never
    expanded onto the reference grid at read time.

    A deformation field carries the reference geometry itself, so only the
    moving image is required. A coefficient field carries neither image's
    geometry, so both the reference and the moving image are required. A
    discrete-cosine-transform coefficient field (intent 2008) is
    recognized but not supported.
    """

    moving: tx.Optional[_ImageLike] = None
    """The moving (source) image, a nibabel image or header, or a
    brainhops image."""

    reference: tx.Optional[_ImageLike] = None
    """The reference image. For a deformation field this defaults to the
    warp file's own geometry."""

    deformation_type: tx.Optional[str] = None
    """For a deformation field, either `"absolute"`, `"relative"`, or
    `None` to infer it from the data. It has no effect on a coefficient
    field."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a FNIRT warp field."""
        if _nifti_intent(header) in _FNIRT_INTENTS:
            return Confidence.CERTAIN
        return Confidence.NO

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
    def from_file(cls, file: tx.Any, **kwargs) -> tx.Self:
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

    # --- exposed parameters -------------------------------------------

    @property
    def order(self) -> tx.Optional[int]:
        """The B-spline order: `1` dense, `3` cubic, `2` quadratic."""
        if self.header is None:
            return None
        return _SPLINE_ORDER.get(_nifti_intent(self.header))

    @property
    def coeff(self) -> tx.Optional[bool]:
        """Whether the field holds spline coefficients rather than values."""
        if self.header is None:
            return None
        return _nifti_intent(self.header) in _COEFFICIENT_INTENTS

    # --- intent-driven behaviour --------------------------------------

    def _intent(self) -> tx.Optional[int]:
        if self.header is None:
            return None
        return _nifti_intent(self.header)

    def _is_coeff(self) -> bool:
        return self._intent() in _COEFFICIENT_INTENTS

    def _order(self) -> int:
        intent = self._intent()
        if intent == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "FNIRT discrete-cosine-transform coefficient fields are not "
                "supported. Convert the field to a deformation field with "
                "FSL (`fnirtfileutils`) and read that instead."
            )
        order = _SPLINE_ORDER.get(intent)
        if order is None:
            raise NotImplementedError(
                f"Unsupported FNIRT intent code {intent!r}."
            )
        return order

    def _bound(self) -> tx.Union[str, float]:
        # A coefficient field is evaluated with out-of-grid coefficients
        # taken as zero. A dense field is sampled at grid points and never
        # reaches outside, so the nearest-edge condition is harmless.
        return "constant" if self._is_coeff() else "nearest"

    def _stored_knot_spacing(self) -> np.ndarray:
        """The knot spacing in reference voxels, stored in the pixdims."""
        zooms = self.header.get_zooms()[:3]
        return np.abs(np.asarray(zooms, dtype=np.float64))

    def _reference_pixdim(self) -> np.ndarray:
        """The reference pixel sizes stored in the intent parameters."""
        header = self.header
        return np.array(
            [
                float(header["intent_p1"]),
                float(header["intent_p2"]),
                float(header["intent_p3"]),
            ],
            dtype=np.float64,
        )

    def _knot_spacing(self, ref: _ImageGeometry) -> np.ndarray:
        """The knot spacing per axis, in reference voxels.

        A dense field has one grid point per reference voxel, so its
        spacing is one. A coefficient field stores the spacing in its
        pixdims, in units of the reference voxels used when the warp was
        generated, and it is rescaled when the current reference has a
        different resolution.
        """
        if not self._is_coeff():
            return np.ones(3, dtype=np.float64)
        ratio = self._reference_pixdim() / ref.pixdim
        return self._stored_knot_spacing() * ratio

    def _ref_to_field(self, ref: _ImageGeometry) -> np.ndarray:
        """The affine from reference voxels to warp-grid voxels."""
        spacing = self._knot_spacing(ref)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.diag(1.0 / spacing)
        return matrix

    def _ref_to_source(self) -> np.ndarray:
        """The initial FLIRT affine as a reference-to-source scaled-mm
        matrix.

        A deformation field already has the initial affine folded into its
        displacements, so this is the identity. A coefficient field stores
        the source-to-reference affine in its sform, so its inverse is
        applied on top of the spline displacement.
        """
        if not self._is_coeff():
            return np.eye(4, dtype=np.float64)
        sform = np.asarray(self.header.get_sform(), dtype=np.float64)
        if not np.all(np.isfinite(sform)):
            return np.eye(4, dtype=np.float64)
        return np.linalg.inv(sform)

    def _reference_geometry(self) -> _ImageGeometry:
        reference = self.reference
        if reference is None:
            if self._is_coeff():
                raise ValueError(
                    "A FNIRT coefficient field is stored on a coarse knot "
                    "grid, not on the reference grid, so the reference image "
                    "is needed to place the warp in world coordinates. Pass "
                    "reference=... when loading, or set it on the "
                    "transformation."
                )
            reference = self.header
        return _ImageGeometry(reference)

    def _raw_field(self) -> tx.Any:
        backend = get_array_backend(self.data)
        field = backend.asarray(self.data)
        # A NIfTI vector field is often five-dimensional, with a singleton
        # axis before the three components. Collapse it to `(*grid, 3)`.
        if field.ndim == 5 and field.shape[3] == 1:
            field = field[:, :, :, 0, :]
        return field

    def _field_array(self, ref: _ImageGeometry) -> tx.Any:
        """The field array on its own grid, as scaled-mm displacements.

        A coefficient field returns its stored coefficients unchanged. A
        deformation field returns a relative displacement, converting from
        absolute coordinates when it is stored that way. The array keeps
        the backend of the file data, so a non-NumPy backend is preserved.
        """
        field = self._raw_field()
        backend = get_array_backend(field)
        if self._is_coeff():
            return field

        shape = tuple(int(s) for s in field.shape[:3])
        ref_scaled = _voxel_grid_in_scaled_mm(shape, ref.vox2fsl, backend)

        deformation_type = self.deformation_type
        if deformation_type is None:
            deformation_type = _detect_deformation_type(field, ref_scaled)
        if deformation_type == "absolute":
            return field - ref_scaled
        if deformation_type == "relative":
            return field
        raise ValueError(
            'deformation_type must be "absolute", "relative" or None, '
            f"not {deformation_type!r}."
        )

    # --- the chain ----------------------------------------------------

    def _resolvable(self) -> bool:
        """Whether the chain can be resolved without raising."""
        if self.header is None or self.moving is None:
            return False
        if self._intent() == FSL_DCT_COEFFICIENTS:
            return False
        if self._is_coeff() and self.reference is None:
            return False
        if self.deformation_type not in (None, "absolute", "relative"):
            return False
        return True

    def _build_chain(self) -> tx.List[_xforms.Transformation]:
        if self.moving is None:
            raise ValueError(
                "A FNIRT warp maps to the moving image, whose geometry the "
                "warp file does not contain, so the moving image is needed "
                "to place the warp in world coordinates. Pass moving=... "
                "when loading, or set it on the transformation."
            )
        ref = self._reference_geometry()
        mov = _ImageGeometry(self.moving)
        return _warp_chain(
            field=self._field_array(ref),
            order=self._order(),
            coeff=self._is_coeff(),
            bound=self._bound(),
            ref_to_field=self._ref_to_field(ref),
            knot_spacing=self._knot_spacing(ref),
            ref=ref,
            mov=mov,
            ref_to_source=self._ref_to_source(),
        )

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The transformations mapping reference RAS to moving RAS.

        Reading this property resolves the chain from the warp data and
        the image geometries. It raises when a required image is missing.
        The resolved chain is cached, and the cache is rebuilt when the
        moving image, the reference image, or the deformation type changes.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if self.header is None:
            return []
        key = self._chain_key()
        cached = getattr(self, "_chain_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        chain = self._build_chain()
        self._chain_cache = (key, chain)
        return chain

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else list(value)

    def _chain_key(self) -> tuple:
        return (
            id(self.moving),
            id(self.reference),
            id(self.header),
            self.deformation_type,
        )

    def _cached_transformations(self) -> tx.List[_xforms.Transformation]:
        """The transformations for repr, length and iteration.

        This returns the resolved chain when it can be resolved, an
        explicitly assigned list when one was set, and an empty list
        otherwise, so inspecting an incompletely specified warp does not
        raise. It backs `__len__`, `__iter__` and `__getitem__`.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if not self._resolvable():
            return []
        return self.transformations

    def __repr__(self) -> str:
        return stored_repr(
            self,
            ("order", "coeff", "deformation_type", "moving", "reference"),
        )

    def __len__(self) -> int:
        return len(self._cached_transformations())

    def __iter__(self) -> tx.Iterator[_xforms.Transformation]:
        return iter(self._cached_transformations())

    def __getitem__(
        self, index: tx.Union[int, slice]
    ) -> _xforms.Transformation:
        return self._cached_transformations()[index]


# ----------------------------------------------------------------------
#   CHAIN CONSTRUCTION
# ----------------------------------------------------------------------


def _warp_chain(
    field: np.ndarray,
    order: int,
    coeff: bool,
    bound: tx.Union[str, float],
    ref_to_field: np.ndarray,
    knot_spacing: np.ndarray,
    ref: _ImageGeometry,
    mov: _ImageGeometry,
    ref_to_source: np.ndarray,
) -> tx.List[_xforms.Transformation]:
    """Build the lazy reference-RAS to moving-RAS chain for a warp field.

    The field stays on its own grid. The two affines place that grid in
    world coordinates and let the spline evaluation happen during reslice.

    The B-spline order fixes a grid convention that differs between FSL
    and the resampler the data model uses. FSL anchors the coarse knot
    basis half a cell in from the resampler's centred basis, a shift of
    `order // 2` grid units, and no shift on an axis whose knot spacing is
    one, where the grid is not coarse. The displacement the spline
    produces is in scaled-mm, while a [`DisplacementField`][] adds it in
    units of its own grid. Both are absorbed into the affines and the
    field values, so the produced chain reproduces FSL's warp.
    """
    order = int(order)
    offsets = _anchor_offsets(order, knot_spacing)

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
    # the warped position with the initial affine applied to the base. The
    # field keeps its array backend through this scaling.
    combined = src_lin @ grid_to_fsl_lin
    prescale = np.linalg.inv(combined)
    backend = get_array_backend(field)
    prescaled = backend.asarray(field) @ backend.asarray(prescale.T)

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


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _anchor_offsets(order: int, knot_spacing: np.ndarray) -> np.ndarray:
    """The per-axis shift between FSL's and the resampler's spline anchor.

    FSL anchors the coarse knot basis `order // 2` grid units in from the
    resampler's centred basis, the same shift for a cubic and a quadratic
    field. An axis whose knot spacing is one has no coarse grid, so its
    shift is zero.
    """
    spacing = np.asarray(knot_spacing, dtype=np.float64)
    step = float(int(order) // 2)
    return np.where(spacing == 1.0, 0.0, step).astype(np.float64)


def _voxel_grid_in_scaled_mm(
    shape: tx.Tuple[int, ...], vox2fsl: np.ndarray, backend: tx.Any
) -> tx.Any:
    """The scaled-mm coordinate of every voxel of a grid.

    Returns an array of shape `(nx, ny, nz, 3)` whose entry at a voxel is
    the scaled-mm coordinate of that voxel under `vox2fsl`.
    """
    grid = backend.stack(
        backend.meshgrid(*[backend.arange(s) for s in shape], indexing="ij"),
        -1,
    )
    grid = backend.asarray(grid, dtype=vox2fsl.dtype)
    rotation = backend.asarray(vox2fsl[:3, :3], dtype=vox2fsl.dtype)
    offset = backend.asarray(vox2fsl[:3, 3], dtype=vox2fsl.dtype)
    return backend.matmul(rotation, grid[..., None])[..., 0] + offset


def _detect_deformation_type(field: tx.Any, ref_scaled: tx.Any) -> str:
    """Infer whether a warp field stores absolute or relative coordinates.

    Absolute coordinates grow with position, so their spread across the
    volume is large. Relative displacements are small offsets, so their
    spread is small. The type whose interpretation gives the larger spread
    is the one the field is stored in. This is the heuristic FSL itself
    uses.
    """
    backend = get_array_backend(field)
    axes = (0, 1, 2)
    std_absolute = float(backend.sum(backend.std(field, axis=axes)))
    std_relative = float(
        backend.sum(backend.std(field - ref_scaled, axis=axes))
    )
    return "absolute" if std_absolute > std_relative else "relative"
