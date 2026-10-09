__all__ = ["load", "sniff"]

import typing_extensions as tx

from brainhops._core.path import FileOrContentLike

from ._base import FileBasedTransformation


def load(
    filelike: FileOrContentLike,
    brute: bool = False,
    **kwargs,
) -> FileBasedTransformation:
    """Read a transformation from a file, detecting its format.

    Parameters
    ----------
    filelike : FileOrContentLike
        The file, or its content.
    brute : bool, default=False
        If no format recognises the input, try every registered reader.
    **kwargs : Any
        Format-specific options.

    Returns
    -------
    FileBasedTransformation
        The transformation read from the file.
    """
    return FileBasedTransformation.load(filelike, brute=brute, **kwargs)


def sniff(filelike: FileOrContentLike, **kwargs) -> tx.Optional[type]:
    """Predict which format class would read a file, without parsing it.

    The result only predicts the class that [`load`][] would choose, and it
    does not guarantee that reading succeeds.

    Parameters
    ----------
    filelike : FileOrContentLike
        The file, or its content.
    **kwargs : Any
        Format-specific options, as in [`load`][].

    Returns
    -------
    type or None
        The format class, or `None` when no single format is the best match.
    """
    return FileBasedTransformation.sniff(filelike, **kwargs)
