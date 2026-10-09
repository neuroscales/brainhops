"""Type hints for files, paths and file contents.

!!! warning "Internal module"
    `brainhops._core.path` is an internal module. It is documented only to
    explain the type hints that appear in the public interface, and it
    should not be imported or relied upon. It may change or disappear
    without notice.
"""

__all__ = [
    "PathLike",
    "Path",
    "LocalPath",
    "FilenameLike",
    "BinaryFileLike",
    "TextFileLike",
    "FileLike",
    "BinaryContentLike",
    "TextContentLike",
    "ContentLike",
    "FileOrContentLike",
    "TextFileOrContentLike",
    "BinaryFileOrContentLike",
    "exists",
]

import errno
from os import PathLike
from pathlib import Path as LocalPath

import typing_extensions as tx
from bagof.paths import Path

FilenameLike = tx.Union[PathLike, str]
"""Name of a file, given as a string or a path-like object."""

BinaryFileLike = tx.Union[FilenameLike, tx.BinaryIO]
"""Name of a file, or a file object opened in binary mode."""

TextFileLike = tx.Union[FilenameLike, tx.TextIO]
"""Name of a file, or a file object opened in text mode."""

FileLike = tx.Union[BinaryFileLike, TextFileLike]
"""Name of a file, or a file object opened in binary or text mode."""

BinaryContentLike = tx.Union[bytes, bytearray, tx.Iterable[bytes]]
"""Binary content, given as bytes, a bytearray or an iterable of bytes."""

TextContentLike = tx.Union[str, tx.Iterable[str]]
"""Text content, given as a string or an iterable of strings."""

ContentLike = tx.Union[BinaryContentLike, TextContentLike]
"""Binary or text content."""

FileOrContentLike = tx.Union[FileLike, ContentLike]
"""Name of a file, file object, or content."""

TextFileOrContentLike = tx.Union[TextFileLike, TextContentLike]
"""Name of a file, text file object, or text content."""

BinaryFileOrContentLike = tx.Union[BinaryFileLike, BinaryContentLike]
"""Name of a file, binary file object, or binary content."""


def exists(filename: FilenameLike) -> bool:
    """Return whether a file exists.

    A name that is too long for the operating system (`ENAMETOOLONG`) is
    reported as not existing instead of raising an error. Such a name is
    usually file content that was passed where a path was expected. Other
    errors raised by the file system are propagated.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    try:
        return filename.exists()
    except OSError as e:
        if e.errno == errno.ENAMETOOLONG:
            return False
        raise
