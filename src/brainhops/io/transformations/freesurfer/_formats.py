"""Format family of FreeSurfer transformations, used for dispatch hints."""

from brainhops.io.common.freesurfer import FreesurferFormat
from brainhops.io.transformations.base import TransformationFormat


class FreesurferTransformationFormat(FreesurferFormat, TransformationFormat):
    """Marker of transformations stored in a FreeSurfer format."""

    # The hint of `FreesurferFormat` is declared again so that it is also
    # qualified by the transformation family, as "xform.freesurfer".
    HINTS = ("freesurfer",)
