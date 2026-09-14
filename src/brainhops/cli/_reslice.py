"""The ``reslice`` command: resample an image onto a reference grid.

The command reads an input image, applies a chain of transformations to
it, and resamples the result onto the grid of a reference image. Each
transformation is read from a file named on the command line and applied
in the order it is given.

The target grid comes from a reference image. Its geometry defines both
the sampling grid and the world placement of the output, so the output
occupies the same space as the reference.

Transformations follow the push convention. Each ``-t/--transform`` is a
forward map that moves the input image through world space, in the same
direction the image itself travels, and matches the library's
``image(T)`` operation. The transformations are applied in the order
given, so the first ``-t`` is applied first. The resampler pulls
internally: it inverts the composed map to sample the reference grid from
the input. A user therefore reasons about the forward motion of the
image, while the sampling is done in the opposite direction on its
behalf.

Many warp files are stored the other way round, as pull maps that run
from the reference to the moving image. ANTs, SPM and FSL warps are
usually of this kind. Such a file must be inverted before it can be used
as a push transform here. The ``inv:`` prefix on a transform value does
exactly that.
"""

from __future__ import annotations

import argparse

from brainhops.datamodel.images import Image

from ._io import load_image, load_transform, save_image

_INVERSE_PREFIX = "inv:"


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
            "are, must be inverted first with the 'inv:' prefix."
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
            "A forward (push) transformation file to apply to the input "
            "image. Repeat the option to apply several, in the order "
            "given. Prefix the value with 'inv:' (for example "
            "'inv:warp.nii.gz') to apply the inverse of the transform "
            "instead, which is what a pull-convention warp needs."
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


def _load_push_transform(spec: str) -> Image:
    """Read one transform value, honouring an ``inv:`` prefix.

    A value such as ``inv:warp.nii.gz`` reads ``warp.nii.gz`` and returns
    its inverse. A plain value returns the transform unchanged. The
    returned transform is a forward (push) map, ready to compose.
    """
    inverted = spec.startswith(_INVERSE_PREFIX)
    path = spec[len(_INVERSE_PREFIX) :] if inverted else spec
    transform = load_transform(path)
    if inverted:
        transform = transform.inverse()
    return transform


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

    Transformations follow the push convention. Each entry is a forward
    map that moves the input image through world space, the same
    direction the image travels, matching the library's `image(T)`
    operation. The entries are applied in order, so the first is applied
    first. Resampling then pulls: the composed forward map is inverted to
    sample the reference grid from the input, so the caller reasons
    forward while the sampling runs the other way.

    A warp stored as a pull map, running from the reference to the moving
    image, is the opposite direction and must be inverted before use.
    ANTs, SPM and FSL warps are usually of this kind. An entry may be
    prefixed with `inv:` (for example `inv:warp.nii.gz`) to apply the
    inverse of that transform, which is what such a warp needs.

    This function performs no file writing, so it can be exercised on its
    own. The command wraps it with the step that writes the result to
    disk.
    """
    image = load_image(input_path)
    for spec in transform_paths:
        transform = _load_push_transform(spec)
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
