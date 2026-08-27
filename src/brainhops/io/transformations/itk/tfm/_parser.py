# stdlib
import re
from warnings import warn

# dependencies
import typing_extensions as _tx

# core
from brainhops._core.peek import peekable_lines

# externals
from bagof.magic import HIDE_IF_NONE, Factory, Magic

# io
from brainhops.io.base.parsers import TextFileParser

from .._common import ITKStruct

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
    TextFileParser,
    Magic,
    convert=True,
    mapping=HIDE_IF_NONE,
    repr=HIDE_IF_NONE,
):
    transform_group: _tx.List[ITKStruct] = Factory(list)

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_line(cls, line: str) -> bool:
        # The first (non-comment) line should be
        # "Transform: {ClassName}_{Precision}_{InputDim}_{OutputDim}"
        # Note that I am not checking for the header comment.
        return _TRANSFORM_RE.match(line.strip()) is not None

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:

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
                    fixed_parameters.group("values"))
            else:
                fixed_parameters = []

            if transform_type == "CompositeTransform":
                # skip composite transforms, they just point to the
                # following transforms.
                continue

            obj.transform_group.append(
                ITKStruct(
                    type=transform_type,
                    precision=precision,
                    ndim_input=input_dim,
                    ndim_output=output_dim,
                    parameters=parameters,
                    fixed_parameters=fixed_parameters,
                )
            )

        return obj


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_vector(text: _tx.Optional[str]) -> _tx.List[float]:
    if not text:
        return []
    return list(map(float, text.split()))
