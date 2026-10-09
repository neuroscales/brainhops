"""Semantic families of transformation formats.

Qualified dispatch hints of the form `"xform.<family>"` are derived from
these marker classes.
"""

__all__ = ["TransformationFormat", "AffineTransformationFormat"]


class TransformationFormat:
    """Marker of any stored transformation format."""

    HINTS = ("xform",)


class AffineTransformationFormat(TransformationFormat):
    """Marker of formats that store an affine transformation."""

    HINTS = ("affine",)
