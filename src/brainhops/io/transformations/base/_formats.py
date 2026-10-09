"""Marker classes for the families of transformation formats.

A hint is a name that can be passed to `load` as `hint=` to restrict the
candidate formats. A format answers to the hints of the marker classes
that it derives from. The hints of nested families are also joined with
dots, so that an affine format answers to `"affine"` and also to
`"xform.affine"`.
"""

__all__ = ["TransformationFormat", "AffineTransformationFormat"]


class TransformationFormat:
    """Marker of any stored transformation format."""

    HINTS = ("xform",)


class AffineTransformationFormat(TransformationFormat):
    """Marker of formats that store an affine transformation."""

    HINTS = ("affine",)
