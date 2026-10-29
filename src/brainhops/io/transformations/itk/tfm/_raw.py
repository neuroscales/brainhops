"""The record of an ITK text `.tfm` file, which holds its blocks."""

__all__ = ["TfmRaw"]

import re
from warnings import warn

import numpy as np
import typing_extensions as tx
from bagof.magic import NoRepr

from brainhops._core._internal_api import peekable_lines
from brainhops.datamodel.metadata._internal_api import takes_error_policy
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileReader,
    TextFileWriter,
)

from .._chains import _check_composite, _is_composite, _read_only, _type_name
from .._common import (
    _TRANSFORM_TYPE_RE,
    ItkPrecision,
    ItkStruct,
    _type_fields,
)

_TRANSFORM_RE = re.compile(r"^Transform:\s*" + _TRANSFORM_TYPE_RE.pattern[1:])
_PARAMETERS_RE = re.compile(r"^Parameters:\s*(?P<values>.*)$")
_FIXEDPARAMETERS_RE = re.compile(r"^FixedParameters:\s*(?P<values>.*)$")


class TfmRaw(Raw, TextFileReader, TextFileWriter):
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
    record is never changed in place. The record also keeps the lines of
    the file, so that a file that is read and written again keeps its
    bytes, and a record whose blocks were encoded again is written as
    ITK writes them (see [`to_lines`][]).
    """

    blocks: tx.Tuple[ItkStruct, ...] = ()
    """Every block of the file, in the order of the file.

    A `CompositeTransform` header can only be the first block.
    """

    lines: NoRepr[tx.Optional[tx.Tuple[str, ...]]] = None
    """The lines of the file that the blocks were read from.

    The lines, comments included, are written back as they are while
    the record holds the blocks that were read from them. They are
    `None` for a record that was encoded from a chain.
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
        if isinstance(lines, peekable_lines):
            lines = list(lines)
        text = []
        for line in lines:
            if not isinstance(line, str):
                break
            text.append(line.rstrip("\r\n"))
        lines = peekable_lines(text)
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
        return cls(blocks=tuple(blocks), lines=tuple(text))

    # --- to -----------------------------------------------------------

    @takes_error_policy
    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """Yield the lines of the file, without their line breaks.

        The lines that the record was read from are returned while the
        record holds the blocks that were read from them. Otherwise, the
        file is written as `itk::TxtTransformIOTemplate::Write` writes
        it: the line `#Insight Transform File V1.0`, then, for
        each block, a comment that numbers it, its `Transform:` line,
        and its `Parameters:` and `FixedParameters:` lines. A
        `CompositeTransform` header has no parameter lines. Each number
        is written with the fewest digits that read back as the same
        number in the precision of its block.

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Yields
        ------
        str
            The lines of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}.to_lines() takes no options, but "
                f"was given {', '.join(sorted(kwargs))}."
            )
        if self.lines is not None and _still_holds(self):
            return iter(self.lines)
        return _lines(self.blocks)

    @takes_error_policy
    def to_text(self, **kwargs) -> str:
        """Return the text of the file, ending with a newline.

        The text is made of the lines of [`to_lines`][].

        Parameters
        ----------
        **kwargs : Any
            Options of the writer. There are none, so any option is
            refused.

        Returns
        -------
        str
            The content of the file.

        Raises
        ------
        TypeError
            If an option is given.
        """
        return "".join(line + "\n" for line in self.to_lines(**kwargs))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_vector(match: tx.Optional[re.Match]) -> np.ndarray:
    """Return the values of a parameter line, or an empty vector."""
    text = None if match is None else match.group("values")
    values = list(map(float, text.split())) if text else []
    return _read_only(np.array(values))


def _still_holds(record: TfmRaw) -> bool:
    """Tell whether the lines of a record hold its blocks."""
    read = TfmRaw.from_lines(record.lines).blocks
    return len(read) == len(record.blocks) and all(
        _same_block(a, b) for a, b in zip(read, record.blocks)
    )


def _same_block(first: ItkStruct, second: ItkStruct) -> bool:
    """Tell whether two blocks store the same values."""
    return _type_name(first) == _type_name(second) and all(
        np.array_equal(
            np.asarray(getattr(first, name), dtype=np.float64),
            np.asarray(getattr(second, name), dtype=np.float64),
        )
        for name in ("parameters", "fixed_parameters")
    )


def _lines(blocks: tx.Sequence[ItkStruct]) -> tx.Iterator[str]:
    """Yield the lines of a file that holds blocks, as ITK writes them."""
    yield "#Insight Transform File V1.0"
    for index, block in enumerate(blocks):
        yield f"#Transform {index}"
        yield f"Transform: {_type_name(block)}"
        if _is_composite(block):
            continue
        precision = ItkPrecision(block.precision)
        yield _vector_line("Parameters:", block.parameters, precision)
        yield _vector_line(
            "FixedParameters:", block.fixed_parameters, ItkPrecision.Double
        )


def _vector_line(tag: str, values: tx.Any, precision: ItkPrecision) -> str:
    """Return a parameter line, with its values separated by spaces."""
    dtype = np.float32 if precision == ItkPrecision.Float else np.float64
    values = np.asarray(values, dtype=np.float64).reshape(-1).astype(dtype)
    return " ".join([tag, *(_format_number(value) for value in values)])


def _format_number(value: np.floating) -> str:
    """Return the shortest text that reads back as the same number.

    An integral value is written without a decimal point, as ITK writes
    it, so that `1.0` is written `1`.
    """
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text
