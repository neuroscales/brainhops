"""The ``convert`` command: rewrite an object in another format.

The command is intended to read an object from one file format and write
it back in another, for example an ITK transform to a NIfTI displacement
field. The kind of object is detected from the input, and the target
format is chosen from the output file.

Formats differ in what they can represent. A feature the source records
may have no place in the target format. How such a difference is handled
is still being decided, and is tied to how a writer behaves when it is
handed a transformation it cannot represent. The command parses its
arguments and reports that the conversion is not implemented rather than
writing anything.

The command depends on writable formats, which are not part of the
library yet, and on the chosen policy for unrepresentable features. It is
registered so that ``brainhops convert --help`` describes the intended
behaviour, and so that the argument names can settle before the
conversion is built.
"""

from __future__ import annotations

import argparse

from ._errors import CliError


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``convert`` subcommand and its arguments."""
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


def run(args: argparse.Namespace) -> int:
    """Run the ``convert`` command from parsed arguments."""
    raise CliError(
        "'brainhops convert' is not implemented yet. It depends on "
        "writable formats (issue #41) and on the policy for features that "
        "the source records but the target format cannot represent."
    )
