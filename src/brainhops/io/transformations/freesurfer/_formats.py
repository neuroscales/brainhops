"""Format family of FreeSurfer transformations, which defines their hints."""

from brainhops.io.base._base import format_registry
from brainhops.io.common.freesurfer import FreesurferFormat
from brainhops.io.transformations.base import TransformationFormat


@format_registry
class FreesurferTransformationFormat(FreesurferFormat, TransformationFormat):
    """Family of transformations stored in a FreeSurfer format."""

    # The hint of `FreesurferFormat` is declared again so that it is also
    # qualified by the transformation family, as "xform.freesurfer".
    HINTS = ("freesurfer",)
