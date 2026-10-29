"""Helpers shared by the sniffers and the readers of the file formats.

The sniffer of a format reads the start of a file and says how likely it
is that the file holds that format. When the file does not, the sniffer
either returns `Confidence.NO` or raises the error that its caller asks
for. The readers of most formats also take no option, and refuse any
that they are given. The functions of this module do this work for every
format, which only gives the words of the error messages.
"""

import typing_extensions as tx

from brainhops.errors import SnifferContentError

from .parsers import Confidence

_Raw = tx.TypeVar("_Raw")


def no_options(kwargs: tx.Mapping[str, tx.Any], owner: str) -> None:
    """Refuse the options given to a reader or a writer that takes none.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        The options that were given.
    owner : str
        What refuses the options, as the error message names it, such as
        `"AFNI reader"`.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(f"Unknown {owner} option(s): {', '.join(kwargs)}")


def declined(
    error: tx.Union[bool, tx.Type[Exception]],
    message: str,
    failure: tx.Optional[Exception] = None,
) -> float:
    """Decline sniffed content, and raise the requested error if any.

    Parameters
    ----------
    error : bool or type[Exception]
        What the caller of the sniffer asks for: `False` to get
        `Confidence.NO`, `True` to get a `SnifferContentError`, or the
        class of the error to raise.
    message : str
        The message of the error.
    failure : Exception, optional
        The error that made the sniffer decline, which becomes the cause
        of the raised error.

    Returns
    -------
    float
        `Confidence.NO`, when `error` is `False`.

    Raises
    ------
    SnifferContentError
        If `error` is `True`.
    """
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message) from failure
    return Confidence.NO


def checked_score(
    score: float, error: tx.Union[bool, tx.Type[Exception]], message: str
) -> float:
    """Return the score of sniffed content, or raise if the score is zero.

    Parameters
    ----------
    score : float
        The score of the content.
    error : bool or type[Exception]
        What the caller of the sniffer asks for when the score is zero:
        `False` to get the score, `True` to get a `SnifferContentError`,
        or the class of the error to raise.
    message : str
        The message of the error.

    Returns
    -------
    float
        The score.

    Raises
    ------
    SnifferContentError
        If the score is zero and `error` is `True`.
    """
    if score or not error:
        return score
    if error is True:
        error = SnifferContentError
    raise error(message)


def sniffed_raw(
    read: tx.Callable[[tx.IO], _Raw], file: tx.IO
) -> tx.Tuple[tx.Optional[_Raw], tx.Optional[Exception]]:
    """Read the record of a stream for a sniffer, keeping any error.

    A sniffer that reads a whole record needs the record when the read
    succeeds, and the error when it fails, so that it can give the error
    as the cause of its own.

    Parameters
    ----------
    read : Callable[[IO], Raw]
        The function that reads the record of the stream. It restores the
        position of the stream.
    file : IO
        The stream.

    Returns
    -------
    raw : Raw or None
        The record, or `None` if it cannot be read.
    failure : Exception or None
        The error that refused the stream, or `None` if the record is read.
    """
    try:
        return read(file), None
    except Exception as e:  # noqa: BLE001
        return None, e
