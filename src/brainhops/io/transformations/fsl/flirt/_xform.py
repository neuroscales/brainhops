# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import FileBasedTransformation
from brainhops.io.transformations.base.affines import RASToVoxel, VoxelToRAS

from .._affines import (
    ImageGeometry,
    ScaledMMToScaledMM,
    ScaledMMToVoxel,
    VoxelToScaledMM,
)
from ._parser import FLIRTMatrixParser


@register_format
class FLIRTTransform(
    FLIRTMatrixParser,
    _xforms.Sequence,
    FileBasedTransformation,
    mapping=HIDE_IF_NONE,
):
    """A linear transformation stored in a FLIRT `.mat` file.

    A FLIRT matrix maps moving-image scaled-mm coordinates to
    reference-image scaled-mm coordinates. This reader adapts it into a
    sequence of affines that maps reference-image world (RAS) coordinates
    to moving-image world (RAS) coordinates, which is the direction the
    data model uses to resample a moving image onto a reference.

    The reference and moving images must be supplied, because the `.mat`
    file carries no image geometry. They may be passed as keyword
    arguments to `load` or `from_file` (`reference=`, `moving=`), or set
    on the object before its `transformations` are read.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)

    @property
    def transformations(self) -> tx.List[_xforms.Transformation]:
        """The affines that map reference RAS to moving RAS, in order."""
        if getattr(self, "_transformations", None) is not None:
            return self._transformations

        if self.matrix is None:
            return []
        if self.reference is None or self.moving is None:
            raise ValueError(
                "A FLIRT matrix carries no image geometry, so both the "
                "reference and the moving image are needed to place it in "
                "world coordinates. Pass reference=... and moving=... when "
                "loading the matrix, or set them on the transformation."
            )

        ref = ImageGeometry(self.reference)
        mov = ImageGeometry(self.moving)
        flirt = np.asarray(self.matrix, dtype=np.float64)
        flirt_inv = np.linalg.inv(flirt)

        self._transformations = [
            RASToVoxel(matrix=ref.ras2vox[:-1]),
            VoxelToScaledMM(matrix=ref.vox2fsl[:-1]),
            ScaledMMToScaledMM(matrix=flirt_inv[:-1]),
            ScaledMMToVoxel(matrix=mov.fsl2vox[:-1]),
            VoxelToRAS(matrix=mov.vox2ras[:-1]),
        ]
        return self._transformations

    @transformations.setter
    def transformations(
        self, value: tx.Optional[tx.List[_xforms.Transformation]]
    ) -> None:
        self._transformations = None if value is None else list(value)
