"""The base class shared by every FreeSurfer format."""


class FreesurferFormat:
    """
    The base format of the FreeSurfer family: MGH and MGZ images, and LTA,
    M3Z and `register.dat` transformations.

    The hint `"freesurfer"` selects them all. Subclass hints such as
    `"mgh"`, `"lta"`, `"m3z"` or `"regdat"` can also be reached as
    `"freesurfer.mgh"`.
    """

    HINTS = ("freesurfer",)
