"""The subcommands, described without importing them.

Each subcommand is described here by its name, its help text and its
arguments, which is all the parser needs. The module that runs it, and
everything that module imports (the data model, every file format), is
imported only when the command actually runs, so that ``brainhops
--help`` and argument errors are answered at once.

The module of each command holds its ``run`` function, and its
docstring describes the command in more depth.
"""

from __future__ import annotations

import typing_extensions as tx


class Argument(tx.NamedTuple):
    """The arguments of one `ArgumentParser.add_argument` call."""

    flags: tx.Tuple[str, ...]
    options: tx.Dict[str, tx.Any]


class Command(tx.NamedTuple):
    """A subcommand: its parser, and the module that runs it."""

    name: str
    module: str
    help: str
    description: str
    arguments: tx.Tuple[Argument, ...]


def _argument(*flags: str, **options: tx.Any) -> Argument:
    return Argument(flags, options)


RESLICE = Command(
    name="reslice",
    module="._reslice",
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
    arguments=(
        _argument(
            "input",
            metavar="SOURCE",
            help=(
                "Image to resample, as a source specification: a path "
                "followed by optional pipe-separated format hints and "
                "named options. For example, 'input.dat|nifti'. Quote "
                "values containing '|' when invoking the command from a "
                "shell."
            ),
        ),
        _argument(
            "-r",
            "--reference",
            required=True,
            metavar="SOURCE",
            help=(
                "Reference image whose geometry defines the output grid "
                "and its placement in world space, using the same source-"
                "specification syntax as the input image."
            ),
        ),
        _argument(
            "-t",
            "--transform",
            action="append",
            default=[],
            metavar="SOURCE",
            dest="transforms",
            help=(
                "A forward (push) transformation to apply to the input "
                "image. Repeat the option to apply several, in the order "
                "given. The value is a source specification: a path "
                "followed by optional pipe-separated format hints, named "
                "options, and operators. For example, "
                "'affine.mat|flirt|reference:[ref.nii.gz]|inv'. Bracketed "
                "option values are nested source specifications, so they "
                "may carry their own hints and options. The '|inv' "
                "operator inverts the transform, which is what a "
                "pull-convention warp needs. The '|' usually needs shell "
                "quoting; encode a literal pipe in a path as '%%7C'. The "
                "operators 'sqrt', 'square', 'exp' and 'log' are "
                "recognised but not implemented yet (tracked in issue "
                "#47)."
            ),
        ),
        _argument(
            "-o",
            "--output",
            required=True,
            metavar="IMAGE",
            help="Path to write the resampled image to.",
        ),
        _argument(
            "--degree",
            type=int,
            default=1,
            help="Spline degree (0=nearest, 1=linear, 3=cubic). Default 1.",
        ),
        _argument(
            "--bound",
            default="reflect",
            help="Boundary condition used outside the field of view. "
            "Default 'reflect'.",
        ),
    ),
)

COMPOSE = Command(
    name="compose",
    module="._compose",
    help="Combine transformations into a single transformation.",
    description=(
        "Combine several transformations into one and write the "
        "result. Not implemented yet."
    ),
    arguments=(
        _argument(
            "transforms",
            nargs="*",
            metavar="FILE",
            help=(
                "Transformation files to combine, in composition order. "
                "The last file listed is applied first."
            ),
        ),
        _argument(
            "-o",
            "--output",
            metavar="FILE",
            help="Path to write the combined transformation to.",
        ),
    ),
)

CONVERT = Command(
    name="convert",
    module="._convert",
    help="Rewrite an object in another file format.",
    description=(
        "Read an object from one format and write it back in another. "
        "Not implemented yet."
    ),
    arguments=(
        _argument(
            "input",
            help="Path to the object to convert.",
        ),
        _argument(
            "-o",
            "--output",
            required=True,
            metavar="FILE",
            help="Path to write the converted object to. Its extension "
            "selects the target format.",
        ),
    ),
)

# In the order they are listed in `brainhops --help`.
COMMANDS = (RESLICE, COMPOSE, CONVERT)
