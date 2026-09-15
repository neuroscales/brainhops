# stdlib
import re
from warnings import warn

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# core
from brainhops._core.peek import peekable_lines

# io
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParser,
)

from .._common import ITKStruct, ITKTransformClass

# constants
_HEADER = "#Insight Transform File V1.0"
_TRANSFORM_RE = re.compile(
    r"^Transform:\s*"
    r"(?P<type>\w+)_"
    r"(?P<precision>float|double)_"
    r"(?P<input_dim>\d+)_"
    r"(?P<output_dim>\d+)$"
)
_PARAMETERS_RE = re.compile(r"^Parameters:\s*(?P<values>.*)$")
_FIXEDPARAMETERS_RE = re.compile(r"^FixedParameters:\s*(?P<values>.*)$")


class TFMTransformParser(
    Magic,
    TextFileParser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK text (`.tfm`) transform file into a chain of
    transform blocks."""

    transform_group: tx.List[ITKStruct] = Factory(list)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score how confident the parser is that a line starts a `.tfm`
        transform block."""
        # The first (non-comment) line should be
        # "Transform: {ClassName}_{Precision}_{InputDim}_{OutputDim}".
        # The version header is a comment, and `peekable_lines` has
        # already dropped it, so the first line seen here is the block.
        # A file that is only a header has no such line: `peekable_lines`
        # yields its end sentinel, which is not a string.
        if isinstance(line, str) and _TRANSFORM_RE.match(line.strip()):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Not an ITK transform block: {line!r}")
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Build the transform chain from an iterable over lines of a
        `.tfm` file."""

        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        obj = cls()

        while True:
            if not lines.peek():
                break

            # Parse transform type
            line = lines.next()
            transform = _TRANSFORM_RE.match(line)
            if not transform:
                warn(f"Unexpected line: {line}", stacklevel=1)
                break

            transform_type = transform.group("type")
            precision = transform.group("precision")
            input_dim = int(transform.group("input_dim"))
            output_dim = int(transform.group("output_dim"))

            # Parse parameters
            line = lines.peek()
            parameters = _PARAMETERS_RE.match(line)
            if parameters:
                lines.next()  # consume the line
                parameters = _read_vector(parameters.group("values"))
            else:
                parameters = []

            # Parse fixed parameters
            line = lines.peek()
            fixed_parameters = _FIXEDPARAMETERS_RE.match(line)
            if fixed_parameters:
                lines.next()  # consume the line
                fixed_parameters = _read_vector(
                    fixed_parameters.group("values")
                )
            else:
                fixed_parameters = []

            if transform_type == "CompositeTransform":
                # skip composite transforms, they just point to the
                # following transforms.
                continue

            transform_type = ITKTransformClass(transform_type)

            obj.transform_group.append(
                ITKStruct(
                    type=transform_type,
                    precision=precision,
                    ndim_input=input_dim,
                    ndim_output=output_dim,
                    parameters=np.array(parameters),
                    fixed_parameters=np.array(fixed_parameters),
                )
            )

        return obj


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_vector(text: tx.Optional[str]) -> tx.List[float]:
    if not text:
        return []
    return list(map(float, text.split()))
