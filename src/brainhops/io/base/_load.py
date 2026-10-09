__all__ = ["load", "sniff"]

import typing_extensions as tx

from brainhops._core.path import FileOrContentLike

from ._base import Format


def load(
    filelike: FileOrContentLike,
    brute: bool = False,
    **kwargs,
) -> Format:
    """Read an object from a file, whatever its kind and format.

    The file is read by the registered format that best matches its content.
    For example, a NIfTI file that holds a displacement field is read as a
    transformation, whereas a plain NIfTI file is read as an image.

    !!! tip "Prefer the scoped entry points when the kind is known"
        [`images.load`][brainhops.io.images.load] and
        [`transformations.load`][brainhops.io.transformations.load] consider
        only their own kind, which is faster and unambiguous.

    Parameters
    ----------
    filelike : FileOrContentLike
        The file or its content.
    brute : bool, default=False
        Whether to try every registered reader when no format recognizes the
        input.
    """
    return Format.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """Predict which format [`load`][] would use, without reading the file.

    Returns
    -------
    type or None
        The best match among all registered formats, or `None` if no single
        format stands out.
    """
    return Format.sniff(filelike, **kwargs)
