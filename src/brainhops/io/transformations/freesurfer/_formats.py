"""Format-family markers used for FreeSurfer dispatch hints."""

from brainhops.io.base.freesurfer import FreesurferFormat
from brainhops.io.transformations.base import TransformationFormat


class FreesurferTransformationFormat(FreesurferFormat, TransformationFormat):
    """A transformation stored in a FreeSurfer format."""

    # Declared again (it is `FreesurferFormat`'s hint) so that it is also
    # qualified by the transformation family: `"xform.freesurfer"`.
    HINTS = ("freesurfer",)
