"""Format-family markers used for FSL dispatch hints."""

from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class FslTransformationFormat(TransformationFormat):
    """A transformation stored in an FSL format."""

    HINTS = ("fsl",)


class FslAffineFormat(FslTransformationFormat, AffineTransformationFormat):
    """An affine transformation stored in an FSL format."""
