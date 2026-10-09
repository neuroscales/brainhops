import numpy as np
import typing_extensions as tx
from bagof.magic import Alias

from brainhops.backends import get_array_backend
from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel.enums import StoreEnum
from brainhops.datamodel.images import Image
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, WriterNotImplementedError
from brainhops.io.common.nifti import NiftiParser
from brainhops.io.common.nifti._header import (
    _nifti_intent,
    _nifti_vector_field,
    _NiftiObject,
)
from brainhops.io.transformations.base import FileBasedTransformation
from brainhops.io.transformations.base.fields import voxel_grid_coordinates

from .._affines import _ImageGeometry
from .._fields import RASToWarpField, WarpFieldToRAS
from .._formats import FslTransformationFormat
from .._repr import stored_repr

# Intent codes from `nifti1.h`.
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

# Deformation fields are linear B-splines on the reference grid; coefficient
# fields are quadratic or cubic on a coarse knot grid.
_SPLINE_DEGREE = {
    FSL_FNIRT_DISPLACEMENT_FIELD: 1,
    FSL_QUADRATIC_SPLINE_COEFFICIENTS: 2,
    FSL_CUBIC_SPLINE_COEFFICIENTS: 3,
}

# A nibabel image or header, or a brainhops image.
_ImageLike = tx.Union[_NiftiObject, Image]


class _ReadOnlyNifti(FileBasedTransformation, NiftiParser):
    """
    A transformation read from a NIfTI file, and not written.

    The read-only counterpart of `NiftiBasedTransformation`. It is a class
    of its own rather than two bases of `FnirtWarpField`, because no order
    of those bases keeps the constructor's positional parameters (`moving`,
    `reference`, `deformation_type`, `image`, `header`, `transformations`).
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".nii", ".nii.gz")


@register_format
class FnirtWarpField(
    FslTransformationFormat,
    _ReadOnlyNifti,
    _xforms.ImmutableSequence,
):
    """Non-linear transformation stored in an FSL FNIRT NIfTI file.

    A deformation field stores, for every reference voxel, the matching moving
    location in scaled millimetres. A coefficient field stores B-spline
    coefficients on a coarse knot grid over the reference. Both are B-spline
    fields on a regular grid, so one reader handles them.

    | Intent | Kind | Degree | Grid |
    | ------ | ---- | ------ | ---- |
    | 2006 | deformation | 1 | reference voxels |
    | 2007 | cubic coefficients | 3 | knot grid |
    | 2009 | quadratic coefficients | 2 | knot grid |

    The transformation is an
    [`ImmutableSequence`][brainhops.datamodel.transformations.ImmutableSequence]
    of three transformations from reference RAS to moving RAS: an affine to
    warp-grid voxels, the field, and an affine to moving RAS. The field stays
    on its own grid and the spline is evaluated only when the sequence is
    computed.

    A deformation field carries the reference geometry itself, so only the
    moving image is required. A coefficient field carries neither image's
    geometry, so both the reference and the moving image are required. A
    discrete-cosine-transform coefficient field (intent 2008) is
    recognized but not supported.

    !!! note "Read, not written"
        The format is read only, and `save` does not offer it. A FNIRT
        file does not hold everything its map depends on: the moving
        image, whose geometry places the warped points in the world, is
        given separately, and so is whether a deformation field holds
        absolute or relative positions, which is otherwise guessed from
        the values. A file written from a general transformation would
        only be read back as the same map if both were given again, and
        a coefficient field also needs a knot grid, a spline anchoring
        and an initial affine that a general field does not have. Save
        a FNIRT warp as a NIfTI displacement field instead:
        `NiftiRASDisplacementField.from_any(warp).save(path)`.
    """

    HINTS = ("fnirt",)

    moving: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("moving", "mov", "src"))
    ] = None
    """The moving (source) image."""

    reference: tx.Annotated[
        tx.Optional[_ImageLike], Alias(("reference", "ref"))
    ] = None
    """The reference image, by default the geometry of a deformation field."""

    deformation_type: tx.Optional[str] = None
    """How a deformation field stores its values.

    `"absolute"` for positions, `"relative"` for displacements, or `None` to
    infer it from the data. Coefficient fields ignore it.
    """

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        if _nifti_intent(header) in _FNIRT_INTENTS:
            return Confidence.CERTAIN
        return Confidence.NO

    # `NiftiParser.from_file` forwards keyword arguments to `nibabel.load`,
    # which rejects `moving=` and `reference=`, so these are set after parsing.

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

    def to_nibabel(self, **kwargs) -> tx.NoReturn:
        """FNIRT warps are read, not written (see the class notes)."""
        raise WriterNotImplementedError(
            "A FNIRT warp is read, not written: its file does not hold the "
            "moving image its map depends on. Save it as a NIfTI "
            "displacement field instead: "
            "NiftiRASDisplacementField.from_any(warp).save(path)."
        )

    # --- exposed parameters -------------------------------------------

    @property
    def degree(self) -> tx.Optional[int]:
        """The B-spline degree, or `None` for DCT fields."""
        if self.header is None:
            return None
        return _SPLINE_DEGREE.get(_nifti_intent(self.header))

    @property
    def store(self) -> tx.Optional[StoreEnum]:
        """Whether the field stores spline coefficients or values."""
        if self.header is None:
            return None
        return StoreEnum.from_coefficients(
            _nifti_intent(self.header) in _COEFFICIENT_INTENTS
        )

    def _intent(self) -> tx.Optional[int]:
        if self.header is None:
            return None
        return _nifti_intent(self.header)

    def _is_coeff(self) -> bool:
        return self._intent() in _COEFFICIENT_INTENTS

    def _degree(self) -> int:
        intent = self._intent()
        if intent == FSL_DCT_COEFFICIENTS:
            raise NotImplementedError(
                "FNIRT discrete-cosine-transform coefficient fields are not "
                "supported. Convert the field to a deformation field with "
                "FSL (`fnirtfileutils`) and read that instead."
            )
        degree = _SPLINE_DEGREE.get(intent)
        if degree is None:
            raise NotImplementedError(
                f"Unsupported FNIRT intent code {intent!r}."
            )
        return degree

    def _bound(self) -> tx.Union[str, float]:
        # Coefficients are zero outside the grid; a dense field is never
        # sampled outside, so nearest is harmless.
        return "constant" if self._is_coeff() else "nearest"

    def _stored_knot_spacing(self) -> np.ndarray:
        """Knot spacing as stored in the pixdims."""
        zooms = self.header.get_zooms()[:3]
        return np.abs(np.asarray(zooms, dtype=np.float64))

    def _reference_pixdim(self) -> np.ndarray:
        """Reference voxel sizes, stored in the intent parameters."""
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
        """Per-axis knot spacing in reference voxels.

        The spacing is one for a deformation field. A coefficient field stores
        it in voxels of the reference used at registration, so it is rescaled
        to the current reference resolution.
        """
        if not self._is_coeff():
            return np.ones(3, dtype=np.float64)
        ratio = self._reference_pixdim() / ref.pixdim
        return self._stored_knot_spacing() * ratio

    def _ref_to_field(self, ref: _ImageGeometry) -> np.ndarray:
        """Affine from reference voxels to warp-grid voxels."""
        spacing = self._knot_spacing(ref)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.diag(1.0 / spacing)
        return matrix

    def _ref_to_source(self) -> np.ndarray:
        """The initial affine, from reference to source scaled millimetres.

        A deformation field already includes it, so it is the identity. A
        coefficient field stores its inverse in the sform.
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
        # Collapse 5-D vector fields to `(*grid, 3)`.
        return _nifti_vector_field(backend.asarray(self.data))

    def _field_array(self, ref: _ImageGeometry) -> tx.Any:
        """The field on its own grid, as scaled millimetre displacements.

        Coefficients are returned unchanged and absolute deformations are made
        relative. The array backend is preserved.
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

    def _resolvable(self) -> bool:
        if self.header is None or self.moving is None:
            return False
        if self._intent() == FSL_DCT_COEFFICIENTS:
            return False
        if self._is_coeff() and self.reference is None:
            return False
        if self.deformation_type not in (None, "absolute", "relative"):
            return False
        return True

    def _build_chain(self) -> tx.Tuple[_xforms.Transformation, ...]:
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
            degree=self._degree(),
            store=StoreEnum.from_coefficients(self._is_coeff()),
            bound=self._bound(),
            ref_to_field=self._ref_to_field(ref),
            knot_spacing=self._knot_spacing(ref),
            ref=ref,
            mov=mov,
            ref_to_source=self._ref_to_source(),
        )

    @property
    def transformations(self) -> tx.Tuple[_xforms.Transformation, ...]:
        """The transformations from reference RAS to moving RAS.

        The chain is built from the warp and both geometries, and raises when a
        required image is missing. It is cached until the images, the header or
        `deformation_type` change, and it is a tuple so that the cache cannot
        be edited in place.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if self.header is None:
            return ()
        key = self._chain_key()
        cached = getattr(self, "_chain_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        chain = self._build_chain()
        self._chain_cache = (key, chain)
        return chain

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.Sequence[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else tuple(value)

    def _chain_key(self) -> tuple:
        return (
            id(self.moving),
            id(self.reference),
            id(self.header),
            self.deformation_type,
        )

    def _cached_transformations(
        self,
    ) -> tx.Tuple[_xforms.Transformation, ...]:
        """The chain used by `len`, iteration and indexing.

        An assigned chain is returned first, otherwise the resolved chain, or
        an empty tuple when it cannot be built, so that inspection never
        raises.
        """
        explicit = getattr(self, "_transformations", None)
        if explicit is not None:
            return explicit
        if not self._resolvable():
            return ()
        return self.transformations

    def __repr__(self) -> str:
        return stored_repr(
            self,
            ("degree", "store", "deformation_type", "moving", "reference"),
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
    degree: int,
    store: tx.Any,
    bound: tx.Union[str, float],
    ref_to_field: np.ndarray,
    knot_spacing: np.ndarray,
    ref: _ImageGeometry,
    mov: _ImageGeometry,
    ref_to_source: np.ndarray,
) -> tx.Tuple[_xforms.Transformation, ...]:
    """Build the lazy reference-RAS to moving-RAS chain of a warp field.

    Two affines place the field grid in world coordinates. They also absorb two
    differences between FSL and the data model: FSL anchors coarse knots
    `degree // 2` grid units away from the resampler, and its displacements are
    in scaled millimetres rather than grid units.
    """
    degree = int(degree)
    offsets = _anchor_offsets(degree, knot_spacing)

    field_to_ref = np.linalg.inv(ref_to_field)
    # Warp-grid coordinates to reference scaled millimetres.
    grid_to_fsl_lin = ref.vox2fsl[:3, :3] @ field_to_ref[:3, :3]
    grid_to_fsl_off = (
        ref.vox2fsl[:3, :3] @ field_to_ref[:3, 3] + ref.vox2fsl[:3, 3]
    )
    src_lin = ref_to_source[:3, :3]
    src_off = ref_to_source[:3, 3]

    # Displacements in grid units of the source-aligned frame, so that the
    # initial affine applies to the base position.
    combined = src_lin @ grid_to_fsl_lin
    prescale = np.linalg.inv(combined)
    backend = get_array_backend(field)
    prescaled = backend.asarray(field) @ backend.asarray(prescale.T)

    ras_to_grid = ref_to_field @ ref.ras2vox
    ras_to_grid[:3, 3] += offsets

    # The anchor offset added on the input side is removed here.
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

    return (
        RASToWarpField(matrix=ras_to_grid[:-1]),
        _xforms.DisplacementField(
            data=prescaled, degree=degree, bound=bound, store=store
        ),
        WarpFieldToRAS(matrix=grid_to_ras[:-1]),
    )


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _anchor_offsets(degree: int, knot_spacing: np.ndarray) -> np.ndarray:
    """Per-axis shift between the FSL and resampler spline anchors.

    The shift is `degree // 2` grid units, or zero where the knot spacing is
    one.
    """
    spacing = np.asarray(knot_spacing, dtype=np.float64)
    step = float(int(degree) // 2)
    return np.where(spacing == 1.0, 0.0, step).astype(np.float64)


def _voxel_grid_in_scaled_mm(
    shape: tx.Tuple[int, ...], vox2fsl: np.ndarray, backend: tx.Any
) -> tx.Any:
    """Scaled millimetre coordinates of every voxel of a grid."""
    return voxel_grid_coordinates(shape, vox2fsl, backend)


def _detect_deformation_type(field: tx.Any, ref_scaled: tx.Any) -> str:
    """Infer whether a deformation field stores positions or displacements.

    Positions grow with the voxel position and spread widely. The field is
    taken as absolute when it spreads more than its difference from the voxel
    coordinates.
    """
    backend = get_array_backend(field)
    axes = (0, 1, 2)
    std_absolute = float(backend.sum(backend.std(field, axis=axes)))
    std_relative = float(
        backend.sum(backend.std(field - ref_scaled, axis=axes))
    )
    return "absolute" if std_absolute > std_relative else "relative"
