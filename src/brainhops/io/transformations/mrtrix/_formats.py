"""The family of the MRtrix transformation formats, which defines their
hints."""

from brainhops.io.transformations.base import TransformationFormat


class MrtrixTransformationFormat(TransformationFormat):
    """Marker of transformations stored in an MRtrix format."""

    # The hint of the MRtrix image is declared again so that the text
    # formats, which are not MRtrix images, answer to it too, and so that
    # it is also qualified by the transformation family, as
    # "xform.mrtrix".
    HINTS = ("mrtrix",)
