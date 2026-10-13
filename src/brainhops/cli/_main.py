"""Entry point and dispatcher of the command-line interface."""

from __future__ import annotations

import argparse
import sys

import typing_extensions as tx

from . import _compose, _convert, _reslice
from ._errors import CliError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with all the subcommands."""
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
    """Run the command-line program and return its exit code.

    `argv` defaults to the arguments of the process. Without a command, the
    help is printed and 1 is returned. A [`CliError`][] is reported on
    standard error, and its exit code is returned.
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
