"""Format-family markers used for additive FSL dispatch hints."""


class FSLTransformationFormat:
    """A transformation stored in an FSL format."""

    FORMAT_HINTS = ("fsl",)


class FSLAffineFormat(FSLTransformationFormat):
    """An affine transformation stored in an FSL format."""

    FORMAT_HINTS = ("affine",)
