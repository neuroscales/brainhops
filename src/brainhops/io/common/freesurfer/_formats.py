"""The family of the FreeSurfer formats."""

from brainhops.io.base._base import FileFormat, format_registry


@format_registry
class FreesurferFormat(FileFormat):
    """Family of the FreeSurfer formats.

    The family holds the MGH and MGZ images, and the LTA, M3Z and
    `register.dat` transformations. It owns a format registry, so that
    its `load` and `sniff*` methods choose among these formats. The hint
    `"freesurfer"` selects them all, and the hints of its formats, such
    as `"mgh"`, `"lta"`, `"m3z"` or `"regdat"`, can also be reached as
    `"freesurfer.mgh"`.
    """

    HINTS = ("freesurfer",)
