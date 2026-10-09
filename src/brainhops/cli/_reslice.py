"""The `reslice` command, which resamples an image onto a reference grid.

The command reads an input image, applies a chain of transformations and
resamples the result onto the grid of a reference image, whose geometry
also places the output in world space.

Each `-t` or `--transform` option names a forward (push) map that moves
the input image, as `image(T)` does in the library, and the
transformations are applied in the order given. Internally, the resampler
works in the pull direction: it inverts the composed map to find, for
each voxel of the reference grid, the point of the input image to sample.
Warps written by ANTs, SPM and FSL are usually pull maps, which go from
the reference to the moving image, so they must be inverted with the
`inv` operator:

    path/to/warp.nii.gz|inv
"""

from __future__ import annotations

import argparse

import typing_extensions as tx

from brainhops.datamodel.images import Image
from brainhops.io.base import ImageSpec, OperationSpec, TransformationSpec

from ._errors import CliError
from ._io import (
    load_image,
    load_transform,
    save_image,
)


class _UnimplementedOperation(OperationSpec, frozen=True):
    """Operation that is recognised but not implemented yet (issue #47)."""

    def apply(self, value: tx.Any) -> tx.NoReturn:  # noqa: ARG002
        raise CliError(
            f"Transform operator '{self.name}' is not implemented yet; "
            "tracked in issue #47."
        )


# The exponential and the logarithm are not registered as operators,
# because a file that holds a velocity field declares it with an option,
# such as `|svf` or `|displacements|log:true`.
for _operation_name in ("sqrt", "square"):
    TransformationSpec.register_operation(_operation_name)(
        _UnimplementedOperation
    )


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the `reslice` subcommand."""
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
    parser.set_defaults(func=run)
    return parser


def _split_transform_spec(spec: str) -> TransformationSpec:
    """Parse a transformation source specification."""
    try:
        return TransformationSpec.from_arg(spec)
    except ValueError as exc:
        raise CliError(
            f"Invalid transformation source {spec!r}: {exc}"
        ) from exc


def _split_image_spec(spec: str) -> ImageSpec:
    """Parse an image source specification."""
    try:
        return ImageSpec.from_arg(spec)
    except ValueError as exc:
        raise CliError(f"Invalid image source {spec!r}: {exc}") from exc


def _load_push_transform(spec: str) -> Image:
    """Load a transformation and apply its chain of operators.

    The operators are applied in the order written, so that `warp|a|b`
    applies `a` to the loaded transformation and then applies `b` to the
    result. The returned transformation is a forward (push) map.
    """
    source = _split_transform_spec(spec)
    transform = load_transform(source)
    return source.apply_operations(transform)


def reslice_image(
    input_path: tx.Union[str, ImageSpec],
    reference_path: tx.Union[str, ImageSpec],
    transform_paths: list,
    degree: int = 1,
    bound: str = "reflect",
) -> Image:
    """Resample an image onto a reference grid after a chain of transforms.

    This function implements the `reslice` command, apart from writing the
    result. Each entry of `transform_paths` is a source specification of a
    forward (push) transformation, as described in the module
    documentation, and the entries are applied in order. The operators
    `sqrt` and `square` are recognised but not implemented yet (issue #47).
    `degree` is the spline degree (0 for nearest neighbour, 1 for linear and
    3 for cubic), and `bound` is the boundary condition outside the field of
    view.
    """
    input_spec = (
        input_path
        if isinstance(input_path, ImageSpec)
        else _split_image_spec(input_path)
    )
    image = load_image(input_spec)
    for spec in transform_paths:
        transform = _load_push_transform(spec)
        image = image(transform)
    reference_spec = (
        reference_path
        if isinstance(reference_path, ImageSpec)
        else _split_image_spec(reference_path)
    )
    reference = load_image(reference_spec)
    return image.reslice(reference, degree=degree, bound=bound)


def run(args: argparse.Namespace) -> int:
    """Run the `reslice` command and write its result."""
    resliced = reslice_image(
        args.input,
        args.reference,
        args.transforms,
        degree=args.degree,
        bound=args.bound,
    )
    save_image(resliced, args.output)
    print(f"Wrote {args.output}")
    return 0
