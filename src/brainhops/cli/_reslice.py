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
as a push transform here. A pipe-separated operator on the transform
value does exactly that: ``path/to/warp.nii.gz|inv``.

Its arguments are declared in `brainhops.cli._commands`, so that the
parser is built without importing this module.
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
    """A reserved transformation operation tracked in issue #47."""

    def apply(self, value: tx.Any) -> tx.NoReturn:  # noqa: ARG002
        raise CliError(
            f"Transform operator '{self.name}' is not implemented yet; "
            "tracked in issue #47."
        )


for _operation_name in ("sqrt", "square", "exp", "log"):
    TransformationSpec.register_operation(_operation_name)(
        _UnimplementedOperation
    )


def _split_transform_spec(spec: str) -> TransformationSpec:
    """Parse a transform source, including nested options and operations."""
    try:
        return TransformationSpec.from_arg(spec)
    except ValueError as exc:
        raise CliError(
            f"Invalid transformation source {spec!r}: {exc}"
        ) from exc


def _split_image_spec(spec: str) -> ImageSpec:
    """Parse an image source, including hints and nested options."""
    try:
        return ImageSpec.from_arg(spec)
    except ValueError as exc:
        raise CliError(f"Invalid image source {spec!r}: {exc}") from exc


def _load_push_transform(spec: str) -> Image:
    """Read one transform value, honouring its operator chain.

    A value such as `warp.nii.gz|inv` reads `warp.nii.gz` and returns its
    inverse. Operators are applied in written order, as function
    composition over the loaded transform, so `warp|a|b` is `b(a(load))`.
    The returned transform is a forward (push) map, ready to compose.
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
    ANTs, SPM and FSL warps are usually of this kind. An entry may carry
    pipe-separated operators after its path, applied in written order.
    The `inv` operator inverts the transform, so `warp.nii.gz|inv` is
    what such a warp needs. The `|` usually needs shell quoting. The
    operators `sqrt`, `square`, `exp` and `log` are recognised but not
    implemented yet, and are tracked in issue #47.

    This function performs no file writing, so it can be exercised on its
    own. The command wraps it with the step that writes the result to
    disk.
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
    """Run the ``reslice`` command from parsed arguments."""
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
