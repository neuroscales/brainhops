# dependencies
import typing_extensions as tx

# internals
from brainhops._core.path import FileOrContentLike
from brainhops.io.base.specs import ImageSpec

from ._base import FileBasedImage


def load(
    filelike: tx.Union[FileOrContentLike, ImageSpec],
    brute: bool = False,
    **kwargs,
) -> FileBasedImage:
    """
    Read an image from a file, whatever its format.

    Parameters
    ----------
    filelike : FileOrContentLike | ImageSpec
        Input file, its content, or a structured image source.
    brute : bool
        If no format recognizes the input, try every registered reader.
    **kwargs
        Format-specific options.

    Returns
    -------
    image : FileBasedImage
        The image that was read.
    """
    return FileBasedImage.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """
    Identify which image format would read this file.

    The file is only inspected, never parsed, so this is a prediction of
    what `load` would reach for rather than a guarantee.

    Parameters
    ----------
    filelike : FileOrContentLike
        Input file, or its content.
    **kwargs
        Format-specific options.

    Returns
    -------
    format : type | None
        The best match among registered image formats, or `None` if no
        single one stands out.
    """
    return FileBasedImage.sniff(filelike, **kwargs)
