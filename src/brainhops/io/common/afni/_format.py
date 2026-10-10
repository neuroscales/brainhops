"""The family of the AFNI formats."""

# ----------------------------------------------------------------------
#   FORMAT FAMILY
# ----------------------------------------------------------------------


class AfniFormat:
    """
    The base format of the AFNI family, images and transformations.

    Subclass hints such as `"brik"` can also be reached as `"afni.brik"`.
    """

    HINTS = ("afni",)
