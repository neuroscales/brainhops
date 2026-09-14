"""The ``reslice`` command: resample an image onto a reference grid.

The command reads an input image, applies a chain of transformations to
it, and resamples the result onto the grid of a reference image. Each
transformation is read from a file named on the command line and applied
in the order it is given.

The target grid comes from a reference image. Its geometry defines both
the sampling grid and the world placement of the output, so the output
occupies the same space as the reference.
"""

from __future__ import annotations

import argparse

from brainhops.datamodel.images import Image

from ._io import load_image, load_transform, save_image


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``reslice`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "reslice",
        help="Resample an image onto a reference grid.",
        description=(
            "Resample an image onto the grid of a reference image, after "
            "applying a chain of transformations to it."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to the image to resample.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        metavar="IMAGE",
        help=(
            "Reference image whose geometry defines the output grid and "
            "its placement in world space."
        ),
    )
    parser.add_argument(
        "-t",
        "--transform",
        action="append",
        default=[],
        metavar="FILE",
        dest="transforms",
        help=(
            "A transformation file to apply to the input image. Repeat "
            "the option to apply several, in the order given."
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
        "--order",
        type=int,
        default=1,
        help="Spline interpolation order (0=nearest, 1=linear). Default 1.",
    )
    parser.add_argument(
        "--bound",
        default="reflect",
        help="Boundary condition used outside the field of view. Default "
        "'reflect'.",
    )
    parser.set_defaults(func=run)
    return parser


def reslice_image(
    input_path: str,
    reference_path: str,
    transform_paths: list,
    order: int = 1,
    bound: str = "reflect",
) -> Image:
    """Resample an image onto a reference grid and return it.

    The input image is read, each transformation in `transform_paths` is
    read and applied in order, and the result is resampled onto the grid
    of the reference image. The returned image lives on the reference
    grid and in the reference world space.

    This function performs no file writing, so it can be exercised on its
    own. The command wraps it with the step that writes the result to
    disk.
    """
    image = load_image(input_path)
    for transform_path in transform_paths:
        transform = load_transform(transform_path)
        image = image(transform)
    reference = load_image(reference_path)
    return image.reslice(reference, order=order, bound=bound)


def run(args: argparse.Namespace) -> int:
    """Run the ``reslice`` command from parsed arguments."""
    resliced = reslice_image(
        args.input,
        args.reference,
        args.transforms,
        order=args.order,
        bound=args.bound,
    )
    save_image(resliced, args.output)
    print(f"Wrote {args.output}")
    return 0
