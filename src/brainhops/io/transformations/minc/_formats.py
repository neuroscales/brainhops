"""Format family of MNI transform files, which defines their hints."""

__all__ = ["XfmFormat"]

from brainhops.io.base._base import format_registry
from brainhops.io.transformations.base import TransformationFormat


@format_registry
class XfmFormat(TransformationFormat):
    """Family of transformations stored in an MNI transform file (`.xfm`)."""

    HINTS = ("minc", "xfm")
