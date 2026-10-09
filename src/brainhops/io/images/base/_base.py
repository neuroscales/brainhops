__all__ = ["ImageFormat"]

import typing_extensions as tx

from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import Image
from brainhops.io.base._base import (
    Format,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.base.specs import register_parser


@register_parser(Image)
@format_registry
class ImageFormat(_FileBasedModelMixin, DataModelBase, Format, eq=False):
    """Format dispatcher and common base for stored images.

    [`ImageFormat`][] is the common base of the image formats, and it
    chooses which format reads a given file. A concrete reader inherits
    from this class and registers itself with `@register_format`, so that
    [`load`][brainhops.io.images.load] finds the reader without a
    hand-maintained table of formats.
    """

    PRIORITY: tx.ClassVar[int] = 10
    """Precedence of images over other kinds of object, used to break ties.

    When a file could be read as several kinds of object, the readers that
    score their input, for example by the NIfTI intent code, normally
    decide first, and this priority matters only when nothing else does.
    A NIfTI file, for instance, is both an image and a set of affines, and
    it is read as an image when nothing else separates the two.
    """
