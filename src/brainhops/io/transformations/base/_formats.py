"""The classes of the families of transformation formats.

A family, such as `AffineTransformationFormat`, owns a format registry,
so that its `load` and `sniff*` methods choose among the formats of the
family, as those of `TransformationFormat` choose among all formats.

A hint is a name that can be passed to `load` as `hint=` to restrict the
candidate formats. A format answers to the hints of the families that
it derives from. The hints of nested families are also joined with
dots, so that an affine format answers to `"affine"` and also to
`"xform.affine"`.
"""

__all__ = ["AffineTransformationFormat"]

from brainhops.io.base._base import format_registry

from ._base import TransformationFormat


@format_registry
class AffineTransformationFormat(TransformationFormat):
    """Family of formats that store an affine transformation."""

    HINTS = ("affine",)
