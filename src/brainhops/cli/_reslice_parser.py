"""The arguments of the ``reslice`` command.

They are kept apart from `_reslice`, which imports the data model and
the file formats, so that building the parser (``brainhops --help``)
does not import them: `_reslice` is imported only when the command runs.
"""

from __future__ import annotations

import argparse


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``reslice`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "reslice",
        help="Resample an image onto a reference grid.",
        description=(
            "Resample an image onto the grid of a reference image, after "
            "applying a chain of transformations to it. Each transform is "
            "a forward (push) map that moves the input image, applied in "
            "the order given (the first -t is applied first). The "
            "resampler pulls internally, inverting the composed map to "
            "sample the reference grid. A warp stored as a pull map "
            "(reference to moving), as ANTs, SPM and FSL warps usually "
            "are, must be inverted first with the '|inv' operator."
        ),
    )
    parser.add_argument(
        "input",
        metavar="SOURCE",
        help=(
            "Image to resample, as a source specification: a path followed "
            "by optional pipe-separated format hints and named options. "
            "For example, 'input.dat|nifti'. Quote values containing '|' "
            "when invoking the command from a shell."
        ),
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        metavar="SOURCE",
        help=(
            "Reference image whose geometry defines the output grid and "
            "its placement in world space, using the same source-"
            "specification syntax as the input image."
        ),
    )
    parser.add_argument(
        "-t",
        "--transform",
        action="append",
        default=[],
        metavar="SOURCE",
        dest="transforms",
        help=(
            "A forward (push) transformation to apply to the input image. "
            "Repeat the option to apply several, in the order given. The "
            "value is a source specification: a path followed by optional "
            "pipe-separated format hints, named options, and operators. "
            "For example, 'affine.mat|flirt|reference:[ref.nii.gz]|inv'. "
            "Bracketed option values are nested source specifications, so "
            "they may carry their own hints and options. The '|inv' "
            "operator inverts the "
            "transform, which is what a pull-convention warp needs. The "
            "'|' usually needs shell quoting; encode a literal pipe in a "
            "path as '%%7C'. The operators 'sqrt', "
            "'square', 'exp' and 'log' are recognised but not implemented "
            "yet (tracked in issue #47)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="IMAGE",
        help="Path to write the resampled image to.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Spline degree (0=nearest, 1=linear, 3=cubic). Default 1.",
    )
    parser.add_argument(
        "--bound",
        default="reflect",
        help="Boundary condition used outside the field of view. Default "
        "'reflect'.",
    )
    parser.set_defaults(func=_run)
    return parser


def _run(args: argparse.Namespace) -> int:
    """Run the ``reslice`` command, importing it only now."""
    from ._reslice import run

    return run(args)
