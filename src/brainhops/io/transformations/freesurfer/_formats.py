"""Format-family markers used for FreeSurfer dispatch hints."""

from brainhops.io.transformations.base import TransformationFormat


class FreesurferTransformationFormat(TransformationFormat):
    """A transformation stored in a FreeSurfer format."""

    HINTS = ("freesurfer",)
