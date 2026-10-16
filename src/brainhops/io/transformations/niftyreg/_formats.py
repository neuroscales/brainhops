"""Format families that select NiftyReg readers by hint."""

from brainhops.io.base._base import format_registry
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


@format_registry
class NiftyRegTransformationFormat(TransformationFormat):
    """Family of transformations stored in a NiftyReg format."""

    HINTS = ("niftyreg",)


@format_registry
class NiftyRegAffineFormat(
    NiftyRegTransformationFormat, AffineTransformationFormat
):
    """Family of affine transformations stored in a NiftyReg format."""
