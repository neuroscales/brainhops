"""Format families that select NiftyReg readers by hint."""

from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class NiftyRegTransformationFormat(TransformationFormat):
    """Transformation stored in a NiftyReg format."""

    HINTS = ("niftyreg",)


class NiftyRegAffineFormat(
    NiftyRegTransformationFormat, AffineTransformationFormat
):
    """Affine stored in a NiftyReg format."""
