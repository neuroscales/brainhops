"""The two files of an Analyze image, and how they are found."""

# stdlib
import re

# dependencies
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops.io.base.parsers import ParserContentError

# The header and the image of a pair share their name, and differ only in
# their extension, whose case is kept. Both files may be compressed with
# gzip, as nibabel writes them.
_PAIR = re.compile(r"\.(hdr|img|HDR|IMG)(\.gz)?$")


def analyze_files(
    filename: path.FilenameLike,
) -> tx.Tuple[path.Path, path.Path]:
    """Find the header file and the image file of an Analyze image.

    The path may name either file of the pair. The other file has the
    same name, the same compression and the same case, with the other
    extension: `brain.hdr` goes with `brain.img`, and `brain.IMG.gz` with
    `brain.HDR.gz`.

    Parameters
    ----------
    filename : path-like
        The path of the header file or of the image file.

    Returns
    -------
    header : Path
        The path of the header file, which may not exist.
    image : Path
        The path of the image file, which may not exist.

    Raises
    ------
    ParserContentError
        If the path has neither the extension of a header file nor that
        of an image file.
    """
    filename = path.Path(filename)
    match = _PAIR.search(filename.name)
    if match is None:
        raise ParserContentError(
            f"An Analyze image is a header file (.hdr) and an image file "
            f"(.img), and {filename.name!r} is neither."
        )
    stem = filename.name[: match.start()]
    upper = match.group(1).isupper()
    suffix = match.group(2) or ""
    header = stem + (".HDR" if upper else ".hdr") + suffix
    image = stem + (".IMG" if upper else ".img") + suffix
    return filename.parent / header, filename.parent / image
