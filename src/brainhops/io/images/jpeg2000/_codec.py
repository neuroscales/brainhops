"""Encoding of JPEG 2000 images with Pillow."""

# stdlib
import math

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops.io.base.parsers import WriterError
from brainhops.io.images.pillow._utils import array_to_pillow, encode_pillow

_DEFAULT_RESOLUTIONS = 6
"""The number of resolution levels that OpenJPEG writes by default."""

_OPTIONS = frozenset(
    {
        "offset",
        "tile_offset",
        "tile_size",
        "quality_mode",
        "quality_layers",
        "codeblock_size",
        "precinct_size",
        "irreversible",
        "progression",
        "cinema_mode",
        "mct",
        "plt",
    }
)
"""The options of the JPEG 2000 writer of Pillow that are passed on."""

_DECIDED = {
    "signed": "the type of the data decides whether samples are signed",
    "no_jp2": "the container is chosen with container=",
    "dpi": "the transformations decide the pixel size",
}
"""The options of Pillow that the image decides, and why."""


def encoder_options(options: tx.Mapping[str, tx.Any]) -> tx.Dict[str, tx.Any]:
    """Return the options of the encoder, refusing the others.

    Parameters
    ----------
    options : mapping of str to Any
        Options of the JPEG 2000 writer of Pillow.

    Returns
    -------
    dict of str to Any
        The options.

    Raises
    ------
    WriterError
        If an option sets something that the image decides.
    TypeError
        If an option is unknown.
    """
    decided = sorted(set(options) & set(_DECIDED))
    if decided:
        reasons = "; ".join(_DECIDED[name] for name in decided)
        raise WriterError(
            f"The options {decided} cannot be given to the JPEG 2000 "
            f"writer: {reasons}."
        )
    unknown = sorted(set(options) - _OPTIONS)
    if unknown:
        raise TypeError(
            f"The JPEG 2000 writer does not take the option(s) "
            f"{', '.join(unknown)}."
        )
    return dict(options)


def default_resolutions(shape: tx.Sequence[int]) -> int:
    """Return the most resolution levels that an image of a shape can have.

    The coarsest level must have a pixel, and there are at most six
    levels, as OpenJPEG writes by default.

    Parameters
    ----------
    shape : sequence of int
        The number of rows and columns.

    Returns
    -------
    int
        The number of resolution levels.
    """
    smallest = max(1, min(int(n) for n in shape))
    return max(1, min(_DEFAULT_RESOLUTIONS, int(math.log2(smallest)) + 1))


def encode_jpeg2000(
    storage: np.ndarray,
    container: str,
    num_resolutions: tx.Optional[int],
    options: tx.Mapping[str, tx.Any],
) -> bytes:
    """Encode samples as a JP2 file or a codestream with Pillow.

    The samples are encoded losslessly, with the reversible 5-3 wavelet
    and one quality layer, unless the options ask otherwise.

    Parameters
    ----------
    storage : numpy.ndarray
        The samples, `(rows, columns)` or `(rows, columns, samples)`.
    container : {"jp2", "j2k"}
        The container to write.
    num_resolutions : int, optional
        The number of resolution levels, which is the most that the image
        can have, up to six, by default.
    options : mapping of str to Any
        Options of the JPEG 2000 writer of Pillow, as returned by
        [`encoder_options`][].

    Returns
    -------
    bytes
        The file.

    Raises
    ------
    WriterError
        If the number of resolution levels is out of range, or Pillow
        cannot store the samples: it stores 8-bit grey, grey and alpha,
        RGB and RGBA images, and 16-bit grey images.
    """
    rows, columns = storage.shape[:2]
    most = default_resolutions((rows, columns))
    if num_resolutions is None:
        num_resolutions = most
    elif not 1 <= int(num_resolutions) <= most:
        raise WriterError(
            f"An image of {columns} x {rows} pixels has between 1 and {most} "
            f"resolution levels, not {num_resolutions}."
        )
    options = dict(options, num_resolutions=int(num_resolutions))
    if container == "j2k":
        options["no_jp2"] = True
    im = array_to_pillow(np.ascontiguousarray(np.asarray(storage)))
    return encode_pillow(im, "JPEG2000", **options)
