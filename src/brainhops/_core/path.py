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
]

# stdlib
from os import PathLike
from pathlib import Path as LocalPath

# dependencies
import typing_extensions as tx
from bagof.paths import Path

# typing

FilenameLike = tx.Union[PathLike, str]
"""A [`PathLike`][os.PathLike] or [`str`][str] that represents a filename."""

BinaryFileLike = tx.Union[FilenameLike, tx.BinaryIO]
"""A path to a file or a binary file object."""

TextFileLike = tx.Union[FilenameLike, tx.TextIO]
"""A path to a file or a text file object."""

FileLike = tx.Union[BinaryFileLike, TextFileLike]
"""A path to a file or a file object."""

BinaryContentLike = tx.Union[bytes, bytearray, tx.Iterable[bytes]]
"""
The content of a binary file:
a [`bytes`][], [`bytearray`][] or an iterable of [`bytes`][].
"""

TextContentLike = tx.Union[str, tx.Iterable[str]]
"""
The content of a text file:
a [`str`][str] or an iterable of [`str`][str].
"""

ContentLike = tx.Union[BinaryContentLike, TextContentLike]
"""The content of a text or binary file."""

FileOrContentLike = tx.Union[FileLike, ContentLike]
"""A path to a file, a file object, or the content of a file."""

TextFileOrContentLike = tx.Union[TextFileLike, TextContentLike]
"""A path to a text file, a text file object, or the content of a text file."""

BinaryFileOrContentLike = tx.Union[BinaryFileLike, BinaryContentLike]
"""
A path to a binary file, a binary file object, or the content of a
binary file.
"""
