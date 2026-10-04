# stdlib
import re
from warnings import warn

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

from brainhops._core.peek import peekable_lines

# core
from brainhops.datamodel.metadata import MetadataField

# io
from brainhops.io.base.parsers import (
    Confidence,
    SnifferContentError,
    TextFileParser,
)

from .._common import ItkStruct, ItkTransformClass, _application_order
from .._metadata import ItkMetadata

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


class TfmTransformParser(
    Magic,
    TextFileParser,
    convert=True,
    repr=HIDE_IF_NONE,
):
    """Parses an ITK text (`.tfm`) transform file into a chain of
    transform blocks.

    The blocks of a `CompositeTransform` are listed in the order they
    apply to points, which is the reverse of their order in the file
    (ITK applies the last block of a composite first).

    Each block is itself a brainhops transformation, so the parsed blocks
    are stored straight into the `transformations` of the sequence that
    this parser is mixed into.
    """

    metadata: MetadataField[
        ItkMetadata,
        Factory(),
        tx.Doc(
            """
            None: an ITK `.tfm` file stores no metadata, so every field is
            unsupported. See
            [`ItkMetadata`][brainhops.io.transformations.itk.ItkMetadata].
            """
        ),
    ]

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
    def from_lines(
        cls,
        lines: tx.Iterable[str],
        position: tx.Optional[int] = None,
        **kwargs,
    ) -> tx.Self:
        """Build the transform chain from an iterable over lines of a
        `.tfm` file.

        Parameters
        ----------
        lines : iterable of str
            Lines of the file.
        position : int, optional
            Which top-level transform of the file to read: the
            composite, if the file starts with a `CompositeTransform`
            header, else one of its blocks. By default, the first one,
            with a warning if the file holds several.
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

            index += 1
            if transform_type == "CompositeTransform":
                # A composite header has no parameters of its own: its
                # queue is the blocks that follow it.
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
