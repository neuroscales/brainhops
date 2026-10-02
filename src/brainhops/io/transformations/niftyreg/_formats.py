"""Format-family markers used for NiftyReg dispatch hints."""

from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class NiftyRegTransformationFormat(TransformationFormat):
    """A transformation stored in a NiftyReg format."""

    HINTS = ("niftyreg",)


class NiftyRegAffineFormat(
    NiftyRegTransformationFormat, AffineTransformationFormat
):
    """An affine transformation stored in a NiftyReg format."""
