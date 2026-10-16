"""The record of an ITK text `.tfm` file, which holds its blocks."""

__all__ = ["TfmRaw"]

import re
from warnings import warn

import numpy as np
import typing_extensions as tx

from brainhops._core.peek import peekable_lines
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileReader,
)

from .._chains import _check_composite, _read_only
from .._common import _TRANSFORM_TYPE_RE, ItkStruct, _type_fields

_TRANSFORM_RE = re.compile(r"^Transform:\s*" + _TRANSFORM_TYPE_RE.pattern[1:])
_PARAMETERS_RE = re.compile(r"^Parameters:\s*(?P<values>.*)$")
_FIXEDPARAMETERS_RE = re.compile(r"^FixedParameters:\s*(?P<values>.*)$")


class TfmRaw(Raw, TextFileReader):
    """Record of an ITK text `.tfm` file, which holds its blocks as read.

    A `.tfm` file is short and is read in one pass, so the record holds
    every block of the file, in the order of the file, as an
    [`ItkStruct`][brainhops.io.transformations.itk.ItkStruct]. A file that
    holds a composite transformation starts with a `CompositeTransform`
    header, which the record keeps as its first block. The metadata of a
    `.tfm` file,
    [`TfmMetadata`][brainhops.io.transformations.itk.tfm.TfmMetadata],
    holds a record, and a
    [`TfmTransform`][brainhops.io.transformations.itk.tfm.TfmTransform]
    decodes its chain from the record of its metadata.

    The blocks of a record are also the elements of the chain that a
    transformation decodes, so their parameters are read-only, and a
    record is never changed in place. brainhops does not write `.tfm`
    files, so the record has no writer and does not keep the text of the
    file.
    """

    blocks: tx.Tuple[ItkStruct, ...] = ()
    """Every block of the file, in the order of the file.

    A `CompositeTransform` header can only be the first block.
    """

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the confidence that a line starts a `.tfm` block.

        The line scores `Confidence.CERTAIN` when it is a `Transform:`
        line and `Confidence.NO` otherwise.

        Parameters
        ----------
        line : str
            The first line of the file that is not a comment.
        error : bool or type[Exception], default=False
            Whether to raise an error when the line does not start a
            block. With `True`, the error is a `SnifferContentError`.
        **kwargs : Any
            Options of the reader, which are not used.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        # The version header is a comment, which peekable_lines has already
        # dropped, so the first line must be the Transform: line. A file
        # that holds only the header yields an end sentinel that is not a
        # str.
        if isinstance(line, str) and _TRANSFORM_RE.match(line.strip()):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Not an ITK transform block: {line!r}")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> "TfmRaw":
        """Read the blocks of a `.tfm` file from its lines.

        Each block is a `Transform:` line, then a `Parameters:` line and a
        `FixedParameters:` line, either of which may be missing and is
        then read as empty. Lines that start with `#` are comments.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        **kwargs : Any
            Options of the reader. There are none, so any option is
            refused.

        Returns
        -------
        TfmRaw
            The record, which holds every block of the file.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If a composite header is not the first block.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__}.from_lines() takes no options, but was "
                f"given {', '.join(sorted(kwargs))}."
            )
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)
        blocks = []
        while lines.peek():
            line = lines.next()
            transform = _TRANSFORM_RE.match(line)
            if not transform:
                warn(f"Unexpected line: {line}", stacklevel=2)
                break
            parameters = _PARAMETERS_RE.match(lines.peek())
            if parameters:
                lines.next()
            fixed_parameters = _FIXEDPARAMETERS_RE.match(lines.peek())
            if fixed_parameters:
                lines.next()
            blocks.append(
                ItkStruct(
                    **_type_fields(transform),
                    parameters=_read_vector(parameters),
                    fixed_parameters=_read_vector(fixed_parameters),
                )
            )
        _check_composite(blocks)
        return cls(blocks=tuple(blocks))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_vector(match: tx.Optional[re.Match]) -> np.ndarray:
    """Return the values of a parameter line, or an empty vector."""
    text = None if match is None else match.group("values")
    values = list(map(float, text.split())) if text else []
    return _read_only(np.array(values))
