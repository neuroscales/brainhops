"""Readers of FSL transformation formats.

FSL expresses transformations in scaled-mm coordinates, which are voxel
indices scaled by the pixel size, with x flipped when the voxel-to-world
affine has a positive determinant. FLIRT stores a linear transformation
as a `.mat` matrix, and FNIRT stores a non-linear transformation as a
NIfTI warp field or coefficient field.
"""

__all__ = [
    "FslCoordinateSystem",
    "flirt",
    "fnirt",
    "FlirtTransform",
    "FnirtWarpField",
]

from brainhops._core.lazy import lazy_exports

from . import flirt, fnirt

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "FslCoordinateSystem": "._systems",
        "FlirtTransform": ".flirt",
        "FnirtWarpField": ".fnirt",
    },
)
