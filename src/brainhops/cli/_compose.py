"""The `compose` command, which is not implemented yet.

The command is meant to combine several transformations into one and
write the result. For transformations A and B, the composite applies B
first and then A, as `A @ B` does in the data model, so the last file
listed is applied first. Operands may also be inverted, squared or
otherwise transformed before composition, but the command-line spelling
of these operations is not settled. Writing also requires a writable
transformation format (issue #41). Until then, the command only parses
its arguments, so that `--help` documents the intended interface.
"""

from __future__ import annotations

import argparse

import typing_extensions as tx

from ._errors import CliError


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the `compose` subcommand."""
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


def run(args: argparse.Namespace) -> tx.NoReturn:
    """Run the `compose` command, which raises a [`CliError`][]."""
    raise CliError(
        "'brainhops compose' is not implemented yet. It depends on a "
        "writable transformation format (issue #41) and on the choice of "
        "operation surface for per-operand operations such as inversion "
        "and the matrix square root."
    )
