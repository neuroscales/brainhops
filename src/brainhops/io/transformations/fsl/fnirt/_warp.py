# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# core
from brainhops.backends import get_array_backend

# datamodel
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.nifti import _nifti_intent, _NiftiObject
from brainhops.io.base.parsers import Confidence

# io
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

from .._affines import ImageGeometry, ScaledMMToVoxel
from .._fields import ScaledMMCoordinatesField
from ._base import FSL_FNIRT_DISPLACEMENT_FIELD, FNIRTTransformation


@register_format
class FNIRTDeformationField(
    FNIRTTransformation,
    _xforms.Sequence,
    mapping=HIDE_IF_NONE,
):
    """A FNIRT deformation (warp) field stored in a NIfTI file.

    The field is defined on the reference image grid, and each voxel
    holds the moving-image location, in FSL scaled-mm coordinates, that
    the reference voxel maps to. The field may store those locations
    either as absolute coordinates or as relative displacements from the
    reference scaled-mm coordinate of the voxel. FNIRT does not record
    which, so the storage type is inferred from the data, and can be
    overridden with `deformation_type`.

    The reader returns a sequence of transformations that maps
    reference-image world (RAS) coordinates to moving-image world (RAS)
    coordinates.
    """

    deformation_type: tx.Optional[str] = None
    """Either `"absolute"`, `"relative"`, or `None` to infer it."""

    @classmethod
    def _score_nibabel(cls, header: _NiftiObject) -> float:
        """Score a NIfTI header as a FNIRT deformation field."""
        if _nifti_intent(header) == FSL_FNIRT_DISPLACEMENT_FIELD:
            return Confidence.CERTAIN
        return Confidence.NO

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The transformations mapping reference RAS to moving RAS."""
        if getattr(self, "_transformations", None) is not None:
            return self._transformations

        if self.header is None:
            return []
        if self.moving is None:
            raise ValueError(
                "A FNIRT warp maps to the moving image, whose geometry the "
                "warp file does not contain, so the moving image is needed "
                "to place the warp in world coordinates. Pass moving=... "
                "when loading, or set it on the transformation."
            )

        reference = self.reference
        if reference is None:
            reference = self.header
        ref = ImageGeometry(reference)
        mov = ImageGeometry(self.moving)

        backend = get_array_backend()
        field = backend.asarray(self.data)
        # A NIfTI vector field is often five-dimensional, with a singleton
        # axis before the three components. Collapse it to `(nx, ny, nz, 3)`.
        if field.ndim == 5 and field.shape[3] == 1:
            field = field[:, :, :, 0, :]

        shape = tuple(int(s) for s in field.shape[:3])
        ref_scaled = _voxel_grid_in_scaled_mm(shape, ref.vox2fsl, backend)

        deformation_type = self.deformation_type
        if deformation_type is None:
            deformation_type = _detect_deformation_type(field, ref_scaled)

        if deformation_type == "absolute":
            absolute = field
        elif deformation_type == "relative":
            absolute = field + ref_scaled
        else:
            raise ValueError(
                'deformation_type must be "absolute", "relative" or None, '
                f"not {deformation_type!r}."
            )

        self._transformations = [
            RASToVoxel(matrix=ref.ras2vox[:-1]),
            ScaledMMCoordinatesField(field=absolute),
            ScaledMMToVoxel(matrix=mov.fsl2vox[:-1]),
            VoxelToRAS(matrix=mov.vox2ras[:-1]),
        ]
        return self._transformations

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else list(value)


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
