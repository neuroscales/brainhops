"""Loading and saving helpers shared by the commands.

The helpers wrap [`brainhops.io`][brainhops.io] so that failures become a
[`CliError`][] with an actionable message instead of a traceback.
"""

from __future__ import annotations

import typing_extensions as tx

from brainhops import io
from brainhops.datamodel.images import Image
from brainhops.io.base import ImageSpec, TransformationSpec, format_hints
from brainhops.io.base.parsers import AmbiguousFormatError, WriterError

from ._errors import CliError, WritingUnavailable


def _image_formats_by_hint() -> tx.Dict[str, tx.Set[type]]:
    """Map each image format hint to the registered image formats."""
    from brainhops.io.images.base import ImageFormat

    result: tx.Dict[str, tx.Set[type]] = {}
    for fmt in getattr(ImageFormat, "_REGISTRY", set()):
        for hint in format_hints(fmt):
            result.setdefault(hint, set()).add(fmt)
    return result


def image_format_hints() -> tx.Set[str]:
    """Return the format hints that may follow an image path."""
    return set(_image_formats_by_hint())


def load_image(source: tx.Union[str, ImageSpec]) -> Image:
    """Read an image from a path or a source specification.

    The format is detected from the file unless the specification supplies
    format hints. A [`CliError`][] is raised if the specification or one of
    its hints is invalid, if the file does not exist, or if the image cannot
    be read.
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
    """Map each transformation format hint to the registered formats."""
    from brainhops.io.transformations.base import TransformationFormat

    result: tx.Dict[str, tx.Set[type]] = {}
    for fmt in getattr(TransformationFormat, "_REGISTRY", set()):
        for hint in format_hints(fmt):
            result.setdefault(hint, set()).add(fmt)
    return result


def transform_format_hints() -> tx.Set[str]:
    """Return the format hints that may follow a transformation path."""
    return set(_transform_formats_by_hint())


def load_transform(
    source: tx.Union[str, TransformationSpec],
    hint: tx.Optional[tx.Union[str, tx.Iterable[str]]] = None,
) -> tx.Any:
    """Read a transformation from a path or a source specification.

    The format is detected from the file unless format hints are supplied,
    either in the specification or with `hint`, but not both. A
    [`CliError`][] is raised if the hints are invalid, if the file does not
    exist, or if the transformation cannot be read.
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
    """Write an image, with a format chosen from the file name.

    A [`WritingUnavailable`][] error is raised if no single format is able
    to write the image to `output`.
    """
    try:
        io.save(image, output)
    except (AmbiguousFormatError, WriterError) as exc:
        raise WritingUnavailable(
            f"The image was computed but could not be saved to "
            f"{output!r}: {exc}"
        ) from exc
