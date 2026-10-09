__all__ = ["FileBasedImage", "WritableFileBasedImage"]

import typing_extensions as tx

from brainhops.datamodel.images import Image
from brainhops.io.base._base import (
    FileBasedObject,
    WritableFileBasedObject,
    _FileBasedModelMixin,
    format_registry,
)
from brainhops.io.base.specs import register_parser


@register_parser(Image)
@format_registry
class FileBasedImage(_FileBasedModelMixin, Image, FileBasedObject):
    """An image stored in a file.

    [`FileBasedImage`][] dispatches between the image formats. A concrete
    reader inherits from it and opts in with `@register_format`, which
    makes [`load`][brainhops.io.images.load] find the reader without a
    hand-maintained table.
    """

    PRIORITY: tx.ClassVar[int] = 10
    """Precedence of the image kind, used only to break ties.

    Sniffers that score their input, for example by NIfTI intent codes,
    normally decide first. A NIfTI file is both an image and a set of
    affines, and the image wins when nothing else separates the two.
    """


@format_registry
class WritableFileBasedImage(FileBasedImage, WritableFileBasedObject):
    """A file-based image that can also be written to disk."""
