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

The output is a target specification, written in the same syntax. Its
hints select the target format, for a name that several formats claim
or that none claims, and its options are options of the conversion. An
option whose value is a nested specification in brackets is read as an
image, which gives the target format a geometry that the object does
not hold, as in `warp.nii.gz|fnirt|reference:[ref.nii.gz]|moving:[src.nii.gz]`
for an FSL warp. A format that does not take an option ignores it. A
transformation is written in a format that holds it only approximately,
such as a field interpolated with cubic splines written as the linearly
interpolated values of a NIfTI file, only with `--lossy`.
"""

from __future__ import annotations

import argparse

import typing_extensions as tx

from brainhops import io
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base import ImageSpec, SourceSpec, TransformationSpec
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
            "'|inv'. The output is written in the same syntax: its hints "
            "select the target format and its options are options of the "
            "conversion, such as the images an FSL warp is written for, "
            "'out.nii.gz|fnirt|reference:[ref.nii.gz]|moving:[src.nii.gz]'."
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
        metavar="TARGET",
        help="Target specification, 'path|hint|key:value', of the file to "
        "write. The name and the hints select the target format, and the "
        "options are options of the conversion, such as "
        "'reference:[ref.nii.gz]' and 'moving:[src.nii.gz]' for an FSL "
        "warp.",
    )
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default="auto",
        help="Kind of the input object. By default, it is detected from "
        "the file.",
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
    path, hints, conversion = parse_target(args.output)
    save_object(obj, path, hints=hints, lossy=args.lossy, **conversion)
    return 0


def parse_target(
    target: str,
) -> tx.Tuple[str, tx.Tuple[str, ...], tx.Dict[str, tx.Any]]:
    """
    Read a target specification into a path, hints and options.

    The target is written `path|hint|key:value`, as a source is. An
    option whose value is a nested specification in brackets, as in
    `reference:[ref.nii.gz]`, is read as an image. The values `true` and
    `false` are booleans, a number is an integer or a float, and any
    other value is kept as a string.

    Parameters
    ----------
    target : str
        The target specification.

    Returns
    -------
    path : str
        The path of the file to write.
    hints : tuple of str
        The format hints of the target.
    options : dict
        The options of the conversion.

    Raises
    ------
    CliError
        If the specification is invalid, or an image cannot be read.
    """
    try:
        spec = SourceSpec.from_arg(target)
    except ValueError as exc:
        raise CliError(f"Invalid target {target!r}: {exc}") from exc
    options = {}
    for key, value in spec.options.items():
        if isinstance(value, SourceSpec):
            image = ImageSpec(
                path=value.path, hints=value.hints, options=value.options
            )
            options[key] = load_image(image)
        else:
            options[key] = _option_value(value)
    return str(spec.path), tuple(spec.hints), options


def _option_value(text: str) -> tx.Any:
    # A boolean, a number, or the text itself.
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


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
    hints: tx.Sequence[str] = (),
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
    hints : sequence of str, default=()
        Format hints that select the target format.
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
        io.save(obj, output, hint=tuple(hints) or None, **options)
    except (AmbiguousFormatError, WriterError) as exc:
        raise WritingUnavailable(f"Could not write {output!r}: {exc}") from exc
