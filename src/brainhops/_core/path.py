__all__ = ["PathLike", "Path", "LocalPath", "UPath", "AnyPath"]

# stdlib
from os import PathLike
from pathlib import Path as LocalPath

# dependencies
import typing_extensions as tx

# optionals
try:
    from upath import UPath
except ImportError:
    UPath = None

try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None

Path = UPath or AnyPath or LocalPath

# typing
FilenameLike = tx.Union[PathLike, str]
BinaryFileLike = tx.Union[FilenameLike, tx.BinaryIO]
TextFileLike = tx.Union[FilenameLike, tx.TextIO]
FileLike = tx.Union[BinaryFileLike, TextFileLike]
BinaryContentLike = tx.Union[bytes, bytearray, tx.Iterable[bytes]]
TextContentLike = tx.Union[str, tx.Iterable[str]]
ContentLike = tx.Union[BinaryContentLike, TextContentLike]
FileOrContentLike = tx.Union[FileLike, ContentLike]
TextFileOrContentLike = tx.Union[TextFileLike, TextContentLike]
BinaryFileOrContentLike = tx.Union[BinaryFileLike, BinaryContentLike]
