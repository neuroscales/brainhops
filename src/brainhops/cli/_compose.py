"""The ``compose`` command: combine transformations into one.

The command is intended to read several transformations, combine them
into a single transformation, and write the result to a file. The
combination follows the composition operator: given transformations ``A``
and ``B``, the composite maps a coordinate through ``B`` first and then
through ``A``, matching ``A @ B`` in the data model.

Each operand may carry a per-operand operation before it enters the
composition. The planned operations are inversion, and the matrix square,
square root and exponential of a transformation. The surface that spells
these operations on the command line is still being decided, so the
command parses its arguments and reports that the operation is not
implemented rather than combining anything.

The command depends on a writable transformation format, which is not
part of the library yet, and on the chosen operation surface. It is
registered so that ``brainhops compose --help`` describes the intended
behaviour, and so that the argument names can settle before the operation
is built.
"""

from __future__ import annotations

import argparse

from ._errors import CliError


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``compose`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "compose",
        help="Combine transformations into a single transformation.",
        description=(
            "Combine several transformations into one and write the "
            "result. Not implemented yet."
        ),
    )
    parser.add_argument(
        "transforms",
        nargs="*",
        metavar="FILE",
        help=(
            "Transformation files to combine, in composition order. The "
            "last file listed is applied first."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Path to write the combined transformation to.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the ``compose`` command from parsed arguments."""
    raise CliError(
        "'brainhops compose' is not implemented yet. It depends on a "
        "writable transformation format (issue #41) and on the choice of "
        "operation surface for per-operand operations such as inversion "
        "and the matrix square root."
    )
