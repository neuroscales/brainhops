"""The command-line entry point and its subcommand dispatcher.

`build_parser` assembles the top-level parser and registers every
subcommand, from the descriptions in `_commands`. `main` parses the
arguments, calls the selected command and turns a `CliError` into a
message on standard error and a non-zero exit code. The module that runs
a command is imported only once that command is selected.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import typing_extensions as tx

from ._commands import COMMANDS, Command
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
    for command in COMMANDS:
        _add_command(subparsers, command)
    return parser


def _add_command(
    subparsers: argparse._SubParsersAction,
    command: Command,
) -> argparse.ArgumentParser:
    """Register a subcommand, which imports its module only to run."""
    parser = subparsers.add_parser(
        command.name,
        help=command.help,
        description=command.description,
    )
    for argument in command.arguments:
        parser.add_argument(*argument.flags, **argument.options)

    def run(args: argparse.Namespace) -> int:
        module = importlib.import_module(command.module, __package__)
        return module.run(args)

    parser.set_defaults(func=run)
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
