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
from ._base import FSL_FNIRT_DISPLACEMENT_FIELD, FNIRTWarpField


@register_format
class FNIRTDeformationField(FNIRTWarpField, mapping=HIDE_IF_NONE):
    """A FNIRT deformation (warp) field stored in a NIfTI file.

    The field is defined on the reference image grid, and each voxel
    holds the moving-image location, in FSL scaled-mm coordinates, that
    the reference voxel maps to. The field may store those locations
    either as absolute coordinates or as relative displacements from the
    reference scaled-mm coordinate of the voxel. FNIRT does not record
    which, so the storage type is inferred from the data, and can be
    overridden with `deformation_type`.

    A deformation field is a first-order B-spline field on the reference
    grid. The reader represents it as a displacement field, and returns a
    sequence of transformations that maps reference-image world (RAS)
    coordinates to moving-image world (RAS) coordinates.
    """

    deformation_type: tx.Optional[str] = None
    """Either `"absolute"`, `"relative"`, or `None` to infer it."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a FNIRT deformation field."""
        if _nifti_intent(header) == FSL_FNIRT_DISPLACEMENT_FIELD:
            return Confidence.CERTAIN
        return Confidence.NO

    # --- warp pieces --------------------------------------------------

    def _spline_order(self) -> int:
        return 1

    def _field_coeff(self) -> bool:
        return False

    def _field_bound(self) -> tx.Union[str, float]:
        return "nearest"

    def _ref_to_field(self, ref: ImageGeometry) -> np.ndarray:
        # The field is stored on the reference grid, so the warp grid is
        # the reference grid and the two coincide.
        return np.eye(4, dtype=np.float64)

    def _initial_affine(self) -> np.ndarray:
        # A deformation field already has the initial FLIRT affine folded
        # into its displacements, so nothing more is applied here.
        return np.eye(4, dtype=np.float64)

    def _reference_geometry(self) -> ImageGeometry:
        reference = self.reference
        if reference is None:
            reference = self.header
        return ImageGeometry(reference)

    def _raw_field(self) -> np.ndarray:
        backend = get_array_backend()
        field = backend.asarray(self.data)
        # A NIfTI vector field is often five-dimensional, with a singleton
        # axis before the three components. Collapse it to `(nx, ny, nz, 3)`.
        if field.ndim == 5 and field.shape[3] == 1:
            field = field[:, :, :, 0, :]
        return field

    def _field_array(self, ref: ImageGeometry) -> np.ndarray:
        backend = get_array_backend()
        field = self._raw_field()
        shape = tuple(int(s) for s in field.shape[:3])
        ref_scaled = _voxel_grid_in_scaled_mm(shape, ref.vox2fsl, backend)

        deformation_type = self.deformation_type
        if deformation_type is None:
            deformation_type = _detect_deformation_type(field, ref_scaled)

        if deformation_type == "absolute":
            return np.asarray(field - ref_scaled, dtype=np.float64)
        if deformation_type == "relative":
            return np.asarray(field, dtype=np.float64)
        raise ValueError(
            'deformation_type must be "absolute", "relative" or None, '
            f"not {deformation_type!r}."
        )

    def _resolvable(self) -> bool:
        if self.header is None or self.moving is None:
            return False
        return self.deformation_type in (None, "absolute", "relative")

    def __repr__(self) -> str:
        return stored_repr(self, ("deformation_type", "moving", "reference"))

    def __len__(self) -> int:
        return len(self._inspect())

    def __iter__(self) -> tx.Iterator[_xforms.Transformation]:
        return iter(self._inspect())

    def __getitem__(
        self, index: tx.Union[int, slice]
    ) -> _xforms.Transformation:
        return self._inspect()[index]


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


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
    spread is small. The type whose interpretation gives the larger
    spread is the one the field is stored in. This is the heuristic FSL
    itself uses.
    """
    backend = get_array_backend(field)
    axes = (0, 1, 2)
    std_absolute = float(backend.sum(backend.std(field, axis=axes)))
    std_relative = float(
        backend.sum(backend.std(field - ref_scaled, axis=axes))
    )
    return "absolute" if std_absolute > std_relative else "relative"
