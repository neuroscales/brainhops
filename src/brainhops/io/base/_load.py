__all__ = ["load", "sniff"]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.path import FileOrContentLike

from ._base import FileBasedObject


def load(
    filelike: FileOrContentLike,
    brute: bool = False,
    **kwargs,
) -> FileBasedObject:
    """
    Read an object from a file, whatever its kind and format.

    Every registered format is a candidate, whether it builds an image,
    a transformation or anything else. Formats score how well the
    content matches them, so a NIfTI whose intent code says
    "displacement field" is read as a transformation, while a plain one
    is read as an image.

    !!! tip "Prefer the scoped entry points when you know the kind"
        `images.load` and `transformations.load` only consider formats of
        the kind you asked for, which is both faster and unambiguous.

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
    obj : FileBasedObject
        The object that was read.
    """
    return FileBasedObject.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """
    Identify which format would read this file, without reading it.

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
        The best match among all registered formats, or `None` if no
        single one stands out.
    """
    return FileBasedObject.sniff(filelike, **kwargs)
