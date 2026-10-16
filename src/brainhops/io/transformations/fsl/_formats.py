"""Format families of FSL transformations, which define their hints."""

from brainhops.io.base._base import format_registry
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


@format_registry
class FslTransformationFormat(TransformationFormat):
    """Family of transformations stored in an FSL format."""

    HINTS = ("fsl",)


@format_registry
class FslAffineFormat(FslTransformationFormat, AffineTransformationFormat):
    """Family of affine transformations stored in an FSL format."""
