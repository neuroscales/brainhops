import typing_extensions as tx

from brainhops._core.path import FileOrContentLike
from brainhops.io.base.specs import ImageSpec

from ._base import ImageFormat


def load(
    filelike: tx.Union[FileOrContentLike, ImageSpec],
    brute: bool = False,
    **kwargs,
) -> ImageFormat:
    """Read an image from a file, whatever its format.

    Parameters
    ----------
    filelike : file, content or ImageSpec
        File to read, its content, or a structured image source.
    brute : bool, default=False
        Whether to try every registered reader when no format recognizes
        the input.
    **kwargs : Any
        Format-specific options.

    Returns
    -------
    ImageFormat
        The image that was read.
    """
    return ImageFormat.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """Predict which image format would read a file.

    The file is inspected but never parsed, so the result predicts what
    [`load`][] would try first rather than guaranteeing success.

    Parameters
    ----------
    filelike : file or content
        File to inspect, or its content.
    **kwargs : Any
        Format-specific options.

    Returns
    -------
    type or None
        The best-matching registered format, or `None` if no single
        format stands out.
    """
    return ImageFormat.sniff(filelike, **kwargs)
