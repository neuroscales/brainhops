__all__ = ["load", "sniff"]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.path import FileOrContentLike

from ._base import FileBasedTransformation


def load(
    filelike: FileOrContentLike,
    brute: bool = False,
    **kwargs,
) -> FileBasedTransformation:
    """
    Read a transformation from a file, whatever its format.

    Parameters
    ----------
    filelike : FileOrContentLike
        Input file, or its content.
    brute : bool
        If no format recognizes the input, try every registered reader.
    **kwargs
        Format-specific options.

    Returns
    -------
    transformation : FileBasedTransformation
        The transformation that was read.
    """
    return FileBasedTransformation.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """
    Identify which transformation format would read this file.

    The file is not parsed, only inspected, so this is a prediction of
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
        The best match among registered transformation formats, or
        `None` if no single one stands out.
    """
    return FileBasedTransformation.sniff(filelike, **kwargs)
