"""The `convert` command, which rewrites an object in another format.

The command reads an image or a transformation, and writes it in the
format that the name of the output file selects, for example an ITK
transform as a NIfTI displacement field. The kind of object is detected
from the input file, as [`brainhops.io.load`][brainhops.io.load]
detects it, unless `--kind` names it.

The input is a source specification, written `path|hint|key:value`. Its
hints select the format that reads the file, and its options give the
reader what the file does not hold, such as the images of an FSL warp,
as in `warp.nii.gz|fnirt|moving:[source.nii.gz]`. A transformation may
also be followed by operators, such as `|inv`, which are applied before
it is written.

The target format may need images too: an FSL warp or a FLIRT matrix
is written for a reference and a moving image, which `--reference` and
`--moving` give. A format that does not need them ignores them. A
transformation is written in a format that holds it only approximately,
such as a field interpolated with cubic splines written as the linearly
interpolated values of a NIfTI file, only with `--lossy`.
"""

from __future__ import annotations

import argparse

import typing_extensions as tx

from brainhops import io
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base import ImageSpec, TransformationSpec
from brainhops.io.base.parsers import AmbiguousFormatError, WriterError

from ._errors import CliError, WritingUnavailable
from ._io import load_image, load_transform

KINDS = ("auto", "image", "transform")
"""The values of `--kind`: detect the kind of object, or name it."""


def add_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the `convert` subcommand."""
    parser = subparsers.add_parser(
        "convert",
        help="Rewrite an object in another file format.",
        description=(
            "Read an image or a transformation from one format and write "
            "it in the format that the output file name selects. The input "
            "is a source specification, 'path|hint|key:value', whose hints "
            "select the reader and whose options give the reader what the "
            "file does not hold, such as 'warp.nii.gz|fnirt|moving:[src.nii"
            ".gz]'. A transformation may be followed by operators, such as "
            "'|inv'."
        ),
    )
    parser.add_argument(
        "input",
        metavar="SOURCE",
        help="Source specification of the object to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Path to write the converted object to. Its extension "
        "selects the target format.",
    )
    parser.add_argument(
        "--to",
        dest="hint",
        metavar="HINT",
        help="Format hint of the target, such as 'itk' or 'niftyreg', for "
        "a name that several formats claim or that none claims.",
    )
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default="auto",
        help="Kind of the input object. By default, it is detected from "
        "the file.",
    )
    parser.add_argument(
        "--reference",
        metavar="IMAGE",
        help="Reference (fixed) image, for a target format that needs its "
        "geometry, such as an FSL warp or a FLIRT matrix.",
    )
    parser.add_argument(
        "--moving",
        metavar="IMAGE",
        help="Moving (source) image, for a target format that needs its "
        "geometry, such as an FSL warp or a FLIRT matrix.",
    )
    parser.add_argument(
        "--lossy",
        action="store_true",
        help="Allow a target format that holds the transformation only "
        "approximately, such as a cubic B-spline written as linearly "
        "interpolated values. A format that holds it exactly is still "
        "preferred.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the `convert` command, and return its exit code."""
    obj = load_object(args.input, args.kind)
    conversion: tx.Dict[str, tx.Any] = {}
    if args.reference is not None:
        conversion["reference"] = load_image(args.reference)
    if args.moving is not None:
        conversion["moving"] = load_image(args.moving)
    save_object(
        obj, args.output, hint=args.hint, lossy=args.lossy, **conversion
    )
    return 0


def load_object(source: str, kind: str = "auto") -> tx.Any:
    """
    Read the object that a source specification names.

    Parameters
    ----------
    source : str
        The source specification, `path|hint|key:value`, which may end
        with transformation operators such as `|inv`.
    kind : {"auto", "image", "transform"}, default="auto"
        The kind of object. `"auto"` detects it from the file.

    Returns
    -------
    Any
        The image or the transformation, with its operators applied.

    Raises
    ------
    CliError
        If the specification is invalid, or the object cannot be read.
    """
    if kind == "image":
        return load_image(source)
    try:
        spec = TransformationSpec.from_arg(source)
    except ValueError as exc:
        raise CliError(f"Invalid source {source!r}: {exc}") from exc
    if kind == "transform":
        return spec.apply_operations(load_transform(spec))
    if spec.operations:
        # Operators apply to transformations only.
        return spec.apply_operations(load_transform(spec))
    try:
        obj = io.load(spec)
    except FileNotFoundError as exc:
        raise CliError(f"Input not found: {spec.path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"Could not read {spec.path!r}: {exc}") from exc
    if not isinstance(obj, Transformation):
        # An image is read again by the readers of images, which take the
        # options of an image specification.
        return load_image(
            ImageSpec(path=spec.path, hints=spec.hints, options=spec.options)
        )
    return obj


def save_object(
    obj: tx.Any,
    output: str,
    hint: tx.Optional[str] = None,
    lossy: bool = False,
    **conversion: tx.Any,
) -> None:
    """
    Write an object in the format that the output name selects.

    Parameters
    ----------
    obj : Any
        The image or the transformation to write.
    output : str
        The path of the output file.
    hint : str, optional
        A format hint that selects the target format.
    lossy : bool, default=False
        Whether a transformation may be written in a format that holds
        it only approximately.
    **conversion : Any
        Options of the conversion of a transformation, such as the
        `reference` and `moving` images of an FSL warp.

    Raises
    ------
    WritingUnavailable
        If no format can write the object to `output`.
    """
    options: tx.Dict[str, tx.Any] = {}
    if isinstance(obj, Transformation):
        options = {"lossy": lossy, "conversion": conversion}
    try:
        io.save(obj, output, hint=hint, **options)
    except (AmbiguousFormatError, WriterError) as exc:
        raise WritingUnavailable(f"Could not write {output!r}: {exc}") from exc
