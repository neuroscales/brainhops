"""Loading and saving helpers shared by the commands.

These wrap the `brainhops.io` entry points with the error handling the
command line needs. A missing input file, or an output format that has no
writer yet, becomes a `CliError` with a message a user can act on rather
than a traceback.
"""

from __future__ import annotations

import typing_extensions as tx

from brainhops import io
from brainhops.datamodel.images import Image
from brainhops.io.base import ImageSpec, TransformationSpec, format_hints
from brainhops.io.base.parsers import AmbiguousFormatError, WriterError

from ._errors import CliError, WritingUnavailable


def _image_formats_by_hint() -> tx.Dict[str, tx.Set[type]]:
    """The image readers available under each registered hint."""
    from brainhops.io.images.base import FileBasedImage

    result: tx.Dict[str, tx.Set[type]] = {}
    for fmt in getattr(FileBasedImage, "_REGISTRY", set()):
        for hint in format_hints(fmt):
            result.setdefault(hint, set()).add(fmt)
    return result


def image_format_hints() -> tx.Set[str]:
    """The format hints recognized after an image path in the CLI."""
    return set(_image_formats_by_hint())


def load_image(source: tx.Union[str, ImageSpec]) -> Image:
    """Read an image from a file, or raise a `CliError`.

    Strings are parsed as image source specifications. The format is
    detected unless the specification supplies hints, and any registered
    image format is accepted. Invalid specifications, missing paths and
    unreadable formats are reported as `CliError` instances.
    """
    try:
        spec = (
            source
            if isinstance(source, ImageSpec)
            else ImageSpec.from_arg(source)
        )
    except ValueError as exc:
        raise CliError(f"Invalid image source {source!r}: {exc}") from exc
    try:
        formats = image_format_hints()
        unknown = set(spec.hints) - formats
        if unknown:
            available = ", ".join(sorted(formats)) or "none"
            raise CliError(
                f"Unknown or unavailable image format hint "
                f"{sorted(unknown)!r}. Available hints: {available}."
            )
        return io.images.load(spec)
    except FileNotFoundError as exc:
        raise CliError(f"Input image not found: {spec.path}") from exc
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"Could not read image {spec.path!r}: {exc}") from exc


def _transform_formats_by_hint() -> tx.Dict[str, tx.Set[type]]:
    """The transformation readers available under each registered hint."""
    from brainhops.io.transformations.base import FileBasedTransformation

    result: tx.Dict[str, tx.Set[type]] = {}
    for fmt in getattr(FileBasedTransformation, "_REGISTRY", set()):
        for hint in format_hints(fmt):
            result.setdefault(hint, set()).add(fmt)
    return result


def transform_format_hints() -> tx.Set[str]:
    """The format hints recognized after a transform path in the CLI."""
    return set(_transform_formats_by_hint())


def load_transform(
    source: tx.Union[str, TransformationSpec],
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
) -> tx.Any:
    """Read a transformation from a file, or raise a `CliError`.

    The format is detected from the file unless `hint` names a specific
    reader. A path that does not exist, an unknown hint, or content that
    the selected reader cannot parse is reported as a `CliError`.
    """
    spec = (
        source
        if isinstance(source, TransformationSpec)
        else TransformationSpec(path=source)
    )
    if hint is not None:
        if spec.hints:
            raise CliError(
                "Format hints were supplied both in the spec and API."
            )
        requested = (hint,) if isinstance(hint, str) else tuple(hint)
        spec = TransformationSpec(
            path=spec.path,
            hints=tuple(str(item).lower() for item in requested),
            options=spec.options,
            operations=spec.operations,
        )
    try:
        formats = transform_format_hints()
        unknown = set(spec.hints) - formats
        if unknown:
            available = ", ".join(sorted(formats)) or "none"
            raise CliError(
                f"Unknown or unavailable transformation format hint "
                f"{sorted(unknown)!r}. Available hints: {available}."
            )
        return io.transformations.load(spec)
    except FileNotFoundError as exc:
        raise CliError(f"Transformation not found: {spec.path}") from exc
    except CliError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CliError(
            f"Could not read transformation {spec.path!r}: {exc}"
        ) from exc


def save_image(image: Image, output: str) -> None:
    """Write an image to a file, or raise `WritingUnavailable`.

    The output format is chosen from the file name by `brainhops.io.save`.
    When no image format is registered for the name, none of them stands
    out, or the one chosen cannot write this image, the image is left
    unwritten and
    `WritingUnavailable` is raised. The computation that produced the
    image has already succeeded at that point, so the error says that
    the image could not be saved, and why, rather than reading as a
    crash.
    """
    try:
        io.save(image, output)
    except (AmbiguousFormatError, WriterError) as exc:
        raise WritingUnavailable(
            f"The image was computed but could not be saved to "
            f"{output!r}: {exc}"
        ) from exc
