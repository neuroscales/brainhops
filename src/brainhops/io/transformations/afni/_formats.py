"""Format-family markers used for AFNI dispatch hints."""

from brainhops.io.base.afni import AfniFormat
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    TransformationFormat,
)


class AfniTransformationFormat(AfniFormat, TransformationFormat):
    """A transformation stored in an AFNI format (hint `"afni"`)."""


class AfniAffineFormat(AfniTransformationFormat, AffineTransformationFormat):
    """An affine transformation stored in an AFNI format."""


class AfniWarpFormat(AfniTransformationFormat):
    """A nonlinear warp stored in an AFNI format (hint `"afni.warp"`)."""

    HINTS = ("warp",)
