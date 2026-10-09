"""The base class shared by every FreeSurfer format."""


class FreesurferFormat:
    """
    A format of the FreeSurfer family, whatever it stores.

    It is the shared base of the FreeSurfer image formats (MGH/MGZ) and
    transformation formats (LTA, M3Z), and carries the `"freesurfer"` hint
    they all answer to. Each format adds its own hints (`"mgh"`,
    `"lta"`, `"m3z"`, ...), which are then also reachable as
    `"freesurfer.mgh"`, `"freesurfer.lta"`, ...
    """

    HINTS = ("freesurfer",)
