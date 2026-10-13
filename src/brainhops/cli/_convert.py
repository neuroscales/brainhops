"""The `convert` command, which is not implemented yet.

The command is meant to rewrite an object in another file format, for
example an ITK transformation as a NIfTI displacement field, with the
target format selected by the output file name. It requires writable
formats (issue #41) and a policy for features that the target format
cannot represent. Until then, the command only parses its arguments, so
that `--help` documents the intended interface.
"""

from __future__ import annotations

import argparse

import typing_extensions as tx

from ._errors import CliError


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the `convert` subcommand."""
    parser = subparsers.add_parser(
        "convert",
        help="Rewrite an object in another file format.",
        description=(
            "Read an object from one format and write it back in another. "
            "Not implemented yet."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to the object to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Path to write the converted object to. Its extension "
        "selects the target format.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> tx.NoReturn:
    """Run the `convert` command, which raises a [`CliError`][]."""
    raise CliError(
        "'brainhops convert' is not implemented yet. It depends on "
        "writable formats (issue #41) and on the policy for features that "
        "the source records but the target format cannot represent."
    )
