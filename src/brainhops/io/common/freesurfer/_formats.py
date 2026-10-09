"""The base class shared by every FreeSurfer format."""


class FreesurferFormat:
    """
    The base format of the FreeSurfer family: MGH and MGZ images, and LTA
    and M3Z transformations.

    The hint `"freesurfer"` selects them all. Subclass hints such as
    `"mgh"`, `"lta"` or `"m3z"` can also be reached as `"freesurfer.mgh"`.
    """

    HINTS = ("freesurfer",)
