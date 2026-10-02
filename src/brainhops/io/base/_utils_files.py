"""Helpers for formats whose header and data may live in separate files."""

__all__ = ["local_path", "open_path", "sibling"]

# stdlib
import os

# dependencies
import typing_extensions as tx

# core
from brainhops._core import path


def local_path(file: tx.Any) -> tx.Optional[str]:
    """The local file system path of `file`, or `None` if it has none."""
    try:
        local = os.fspath(file)
    except Exception:
        return None
    if isinstance(local, bytes):
        local = os.fsdecode(local)
    return local if os.path.isfile(local) else None


def open_path(file: tx.Any) -> tx.BinaryIO:
    """Open `file` for binary reading, locally or through its `open`."""
    local = local_path(file)
    if local is not None:
        return open(local, "rb")
    return file.open("rb")


def sibling(filename: tx.Any, name: str) -> tx.Any:
    """`name`, relative to the directory of `filename` unless absolute."""
    if os.path.isabs(name):
        return path.Path(name)
    if not isinstance(filename, path.PathLike):
        filename = path.Path(filename)
    return filename.parent / name
