__all__ = ["load", "sniff"]

import typing_extensions as tx

from brainhops._core.path import FileOrContentLike

from ._base import FileBasedObject


def load(
    filelike: FileOrContentLike,
    brute: bool = False,
    **kwargs,
) -> FileBasedObject:
    """Read an object from a file, whatever its kind and format.

    The registered format that best matches the content is chosen: a NIfTI file
    holding a displacement field is read as a transformation, a plain one as an
    image.

    !!! tip "Prefer the scoped entry points when the kind is known"
        [`images.load`][brainhops.io.images.load] and
        [`transformations.load`][brainhops.io.transformations.load] consider
        only their own kind, which is faster and unambiguous.

    Parameters
    ----------
    filelike : FileOrContentLike
        The file or its content.
    brute : bool
        If no format recognizes the input, try every registered reader.
    """
    return FileBasedObject.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """Predict which format [`load`][] would use, without reading the file.

    Returns
    -------
    type or None
        The best match among all registered formats, or `None` if no single
        format stands out.
    """
    return FileBasedObject.sniff(filelike, **kwargs)
