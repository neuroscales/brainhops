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

from ._errors import CliError, WritingUnavailable


def load_image(source: str) -> Image:
    """Read an image from a file, or raise a `CliError`.

    The format is detected from the file, so any registered image format
    is accepted. A path that does not exist or that no format recognises
    is reported as a `CliError` rather than an exception from the reader.
    """
    try:
        return io.images.load(source)
    except FileNotFoundError as exc:
        raise CliError(f"Input image not found: {source}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CliError(f"Could not read image {source!r}: {exc}") from exc


def load_transform(source: str) -> tx.Any:
    """Read a transformation from a file, or raise a `CliError`.

    The format is detected from the file. A path that does not exist or
    that no transformation format recognises is reported as a `CliError`.
    """
    try:
        return io.transformations.load(source)
    except FileNotFoundError as exc:
        raise CliError(f"Transformation not found: {source}") from exc
    except Exception as exc:  # noqa: BLE001
        raise CliError(
            f"Could not read transformation {source!r}: {exc}"
        ) from exc


def _writable_image_formats() -> tx.Set[type]:
    """The registered image formats that can be written to disk.

    The set is empty while no image writer has been added. A caller uses
    it to tell a user that writing is not available yet, rather than
    failing part way through.
    """
    from brainhops.io.images.base import WritableFileBasedImage

    return set(getattr(WritableFileBasedImage, "_REGISTRY", set()))


def _match_by_extension(
    output: str, formats: tx.Iterable[type]
) -> tx.Optional[type]:
    """The format whose declared extension matches `output`.

    When several formats match, the one with the longest matching
    extension wins, so `.nii.gz` is preferred over `.gz`. `None` means no
    registered format claims the extension.
    """
    lowered = output.lower()
    best: tx.Optional[type] = None
    best_length = -1
    for fmt in formats:
        for extension in getattr(fmt, "EXTENSIONS", ()):
            if lowered.endswith(extension.lower()):
                if len(extension) > best_length:
                    best_length = len(extension)
                    best = fmt
    return best


def save_image(image: Image, output: str) -> None:
    """Write an image to a file, or raise `WritingUnavailable`.

    The output format is chosen from the file extension. When no image
    writer is registered for that extension, the image is left unwritten
    and `WritingUnavailable` is raised. The computation that produced the
    image has already succeeded at that point, so the error names the
    format and points at the open work rather than reading as a crash.

    Image writing is not part of the library yet. This function is
    written against the writable-format registry so that it starts
    working the moment a writer is registered, with no change here.
    """
    formats = _writable_image_formats()
    fmt = _match_by_extension(output, formats)
    if fmt is None:
        raise WritingUnavailable(
            "Writing images is not available yet: no image writer is "
            f"registered for {output!r}. The image was resliced "
            "successfully but could not be saved. NIfTI image writing is "
            "tracked in issue #41."
        )
    # A writer exists: build it from the resampled image and write it.
    writer = fmt(data=image.data, transformations=list(image.transformations))
    writer.save(output)
