"""Format-family marker used for MRtrix transformation dispatch hints."""

from brainhops.io.transformations.base import TransformationFormat


class MrtrixTransformationFormat(TransformationFormat):
    """A transformation stored in an MRtrix format."""

    # The hint of `MrtrixParser` as well, declared here so that the text
    # formats (which are not MRtrix images) answer to it too, and so that
    # it is qualified by the transformation family: `"xform.mrtrix"`.
    HINTS = ("mrtrix",)
