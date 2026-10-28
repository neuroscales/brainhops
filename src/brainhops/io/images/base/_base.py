__all__ = ["ImageFormat"]

import typing_extensions as tx

from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.images import Image
from brainhops.io.base._base import (
    FileFormat,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.base.specs import register_parser


@register_parser(Image)
@format_registry
class ImageFormat(
    _FileBasedModelMixin, DataModelBase, FileFormat, eq=False, kw_only=True
):
    """Format dispatcher and common base for stored images.

    [`ImageFormat`][] is the common base of the image formats, and it
    chooses which format reads a given file. A concrete reader inherits
    from this class and registers itself with `@register_format`, so that
    [`load`][brainhops.io.images.load] finds the reader without a
    hand-maintained table of formats.

    The `consolidate` method of an image format, and each method that
    writes a file, take an error policy as their `on_error=` and
    `on_loss=` keywords (see
    [`ErrorPolicy`][brainhops.errors.ErrorPolicy]), which decides what
    happens to what the file cannot hold. A format overrides the hook
    `_consolidate`, around which the base class applies the policy, and
    puts the decorator `takes_error_policy` above each writer that it
    defines, which applies the policy around the writer. Neither the
    hook nor a writer declares these keywords.
    """

    PRIORITY: tx.ClassVar[int] = 10
    """Precedence of images over other kinds of object, used to break ties.

    When a file could be read as several kinds of object, the readers that
    score their input, for example by the NIfTI intent code, normally
    decide first, and this priority matters only when nothing else does.
    A NIfTI file, for instance, is both an image and a set of affines, and
    it is read as an image when nothing else separates the two.
    """
