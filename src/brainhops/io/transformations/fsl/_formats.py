"""Format-family markers used for FSL dispatch hints."""

from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class FSLTransformationFormat(TransformationFormat):
    """A transformation stored in an FSL format."""

    HINTS = ("fsl",)


class FSLAffineFormat(FSLTransformationFormat, AffineTransformationFormat):
    """An affine transformation stored in an FSL format."""
