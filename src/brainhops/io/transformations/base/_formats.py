"""Semantic format families used to derive qualified dispatch hints."""

__all__ = ["TransformationFormat", "AffineTransformationFormat"]


class TransformationFormat:
    """A stored transformation format."""

    HINTS = ("xform",)


class AffineTransformationFormat(TransformationFormat):
    """A format that stores an affine transformation."""

    HINTS = ("affine",)
