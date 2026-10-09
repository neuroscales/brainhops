"""Format families of FSL transformations, used for dispatch hints."""

from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class FslTransformationFormat(TransformationFormat):
    """Marker of transformations stored in an FSL format."""

    HINTS = ("fsl",)


class FslAffineFormat(FslTransformationFormat, AffineTransformationFormat):
    """Marker of affine transformations stored in an FSL format."""
