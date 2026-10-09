import re
from warnings import warn

import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Magic

from brainhops._core.peek import peekable_lines
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParser,
)

from .._common import ItkStruct, ItkTransformClass, _application_order

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


class TfmTransformParser(
    Magic,
    TextFileParser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """
    Parser that reads an ITK text `.tfm` file into a chain of transformations.

    The parser is a mixin for a sequence of transformations. Each block of the
    file becomes a brainhops transformation and is stored in the
    `transformations` of that sequence. The blocks of a composite
    transformation are listed in application order, which is the reverse of
    their order in the file, because ITK applies the last block first.
    """

    @classmethod
    def sniff_line(
        cls,
        line: str,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score the confidence that a line starts a `.tfm` transformation block.

        The line scores `Confidence.CERTAIN` when it is a `Transform:` line and
        `Confidence.NO` otherwise. When `error` is set, a line that does not
        match raises instead, either `SnifferContentError` or the exception
        class given.
        """
        # The version header is a comment, which peekable_lines has already
        # dropped, so the first line must be the Transform: line. A file that
        # holds
        # only the header yields an end sentinel that is not a str.
        if isinstance(line, str) and _TRANSFORM_RE.match(line.strip()):
            return Confidence.CERTAIN
        if error:
            if error is True:
                error = SnifferContentError
            raise error(f"Not an ITK transform block: {line!r}")
        return Confidence.NO

    @classmethod
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Build the chain of transformations from the lines of a `.tfm` file.

        Parameters
        ----------
        lines : iterable of str
            The lines of the file.
        position : int, optional
            The top-level transformation to read. A file that starts with a
            `CompositeTransform` header holds a single composite
            transformation, while any other file holds one transformation per
            block. By default, the first transformation is read, with a warning
            if the file holds several.
        """

        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        obj = cls()
        blocks = []
        composites = []
        index = 0

        while True:
            if not lines.peek():
                break

            line = lines.next()
            transform = _TRANSFORM_RE.match(line)
            if not transform:
                warn(f"Unexpected line: {line}", stacklevel=1)
                break

            transform_type = transform.group("type")
            precision = transform.group("precision")
            input_dim = int(transform.group("input_dim"))
            output_dim = int(transform.group("output_dim"))

            line = lines.peek()
            parameters = _PARAMETERS_RE.match(line)
            if parameters:
                lines.next()
                parameters = _read_vector(parameters.group("values"))
            else:
                parameters = []

            line = lines.peek()
            fixed_parameters = _FIXEDPARAMETERS_RE.match(line)
            if fixed_parameters:
                lines.next()
                fixed_parameters = _read_vector(
                    fixed_parameters.group("values")
                )
            else:
                fixed_parameters = []

            index += 1
            if transform_type == "CompositeTransform":
                # A composite header has no parameters of its own: its queue is
                # made
                # of the blocks that follow it.
                composites.append(index - 1)
                continue

            transform_type = ItkTransformClass(transform_type)

            blocks.append(
                ItkStruct(
                    type=transform_type,
                    precision=precision,
                    ndim_input=input_dim,
                    ndim_output=output_dim,
                    parameters=np.array(parameters),
                    fixed_parameters=np.array(fixed_parameters),
                )
            )

        obj.transformations = _application_order(blocks, composites, position)
        return obj


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_vector(text: tx.Optional[str]) -> tx.List[float]:
    if not text:
        return []
    return list(map(float, text.split()))
