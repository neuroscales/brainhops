"""The command-line entry point and its subcommand dispatcher.

`build_parser` assembles the top-level parser and registers every
subcommand. `main` parses the arguments, calls the selected command and
turns a `CliError` into a message on standard error and a non-zero exit
code.
"""

from __future__ import annotations

import argparse
import sys

import typing_extensions as tx

from . import _compose, _convert, _reslice
from ._errors import CliError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every subcommand registered."""
    parser = argparse.ArgumentParser(
        prog="brainhops",
        description="Scalable spatial transforms.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
    )
    _reslice.add_parser(subparsers)
    _compose.add_parser(subparsers)
    _convert.add_parser(subparsers)
    return parser


def main(argv: tx.Optional[tx.Sequence[str]] = None) -> int:
    """Run the command line and return the process exit code.

    Arguments are read from `argv`, or from the process arguments when
    `argv` is `None`. With no command, the help text is printed and the
    exit code is `1`. A command that raises `CliError` prints its message
    to standard error and returns the error's exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except CliError as error:
        print(f"brainhops: {error}", file=sys.stderr)
        return error.exit_code
