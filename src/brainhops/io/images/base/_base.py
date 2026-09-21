__all__ = ["FileBasedImage", "WritableFileBasedImage"]

# dependencies
import typing_extensions as tx

# internals
from brainhops.datamodel.images import Image
from brainhops.io.base._base import (
    FileBasedObject,
    WritableFileBasedObject,
    format_registry,
)
from brainhops.io.base.specs import register_parser


@format_registry
class FileBasedImage(Image, FileBasedObject):
    """
    An image that is stored in a file.

    This is the dispatcher for image formats: concrete readers inherit
    from it and opt in with `@register_format`, which is what makes
    `brainhops.io.images.load` work without a hand-maintained table.
    """

    PRIORITY: tx.ClassVar[int] = 10
    """
    Kind precedence, used only to break ties that confidence could not.

    A NIfTI file is legitimately both an image and a set of affines, so
    when nothing else separates them the image wins. Scoring sniffers
    (e.g. NIfTI intent codes) normally decide well before this matters.
    """


@format_registry
class WritableFileBasedImage(FileBasedImage, WritableFileBasedObject):
    """An image that is stored in a file and can be written to disk."""


# A field typed as the data-model Image automatically uses the image
# dispatcher. Concrete Image subclasses inherit this registration.
register_parser(Image, FileBasedImage)
