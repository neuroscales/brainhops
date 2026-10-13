"""Format family of MNI transform files, which defines their hints."""

__all__ = ["XfmFormat"]

from brainhops.io.transformations.base import TransformationFormat


class XfmFormat(TransformationFormat):
    """Marker of transformations stored in an MNI transform file (`.xfm`)."""

    HINTS = ("minc", "xfm")
