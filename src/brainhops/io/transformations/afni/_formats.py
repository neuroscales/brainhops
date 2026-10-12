"""The families of the AFNI transformation formats, which define their
hints."""

# dependencies
import typing_extensions as tx

# internals
from brainhops.io.common.afni import AfniFormat
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class AfniTransformationFormat(TransformationFormat, AfniFormat):
    """Marker of transformations stored in an AFNI format.

    The marker answers to the hint `"afni"`, which it inherits from
    [`AfniFormat`][brainhops.io.common.afni.AfniFormat], and to
    `"xform.afni"`.
    """


class AfniAffineFormat(AfniTransformationFormat, AffineTransformationFormat):
    """Marker of affine transformations stored in an AFNI format."""


class AfniWarp(AfniTransformationFormat):
    """A nonlinear warp stored in an AFNI format, as `3dQwarp` writes it.

    This class is the family of the AFNI warps and answers to the hint
    `"afni.warp"`. It is not a format itself: [`AfniBrikWarp`][] reads and
    writes a warp stored as an AFNI dataset, and [`AfniNiftiWarp`][] a warp
    stored as a NIfTI file.

    A warp has three sub-bricks of displacements, in DICOM (LPS)
    millimetres. The value at a point `x` of the grid of the base is the
    displacement that takes it to the source, so that
    `x_source = x + u(x)` in DICOM coordinates. The grid is the
    *cardinal* grid of the dataset, the one that AFNI computes on
    (`IW3D_from_dataset`, in `mri_nwarp.c`).

    A warp is read as the chain that the other formats of RAS
    displacements share
    ([`ras_displacement_chain`][brainhops.io.transformations.base.fields.ras_displacement_chain]),
    with the DICOM displacements and grid flipped into RAS, which is an
    exact change of frame.

    | Slot           | Transformation                              |
    | -------------- | ------------------------------------------- |
    | `ras2voxel`    | RAS world coordinates to the warp's voxels  |
    | `displacement` | the displacement field, in voxel units      |
    | `voxel2ras`    | the warp's voxels back to RAS world         |

    The displacements are interpolated linearly. Outside its grid, AFNI
    extrapolates a warp linearly from each face of its box, and the data
    model has no such boundary condition, so the nearest value is used.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("warp",)
