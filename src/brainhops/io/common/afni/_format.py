"""The family of the AFNI formats."""

from brainhops.io.base._base import FileFormat, format_registry

# ----------------------------------------------------------------------
#   FORMAT FAMILY
# ----------------------------------------------------------------------


@format_registry
class AfniFormat(FileFormat):
    """Family of the AFNI formats, images and transformations alike.

    The family owns a format registry, so that its `load` and `sniff*`
    methods choose among every AFNI format. It answers to the hint
    `"afni"`, and the hints of its formats, such as `"brik"`, can also be
    reached as `"afni.brik"`.
    """

    HINTS = ("afni",)
