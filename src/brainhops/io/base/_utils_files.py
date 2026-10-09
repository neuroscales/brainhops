"""Helpers for formats that store header and data in separate files."""
# TODO: the rest of the code uses Path objects; use them here as well and
# move generic utilities to brainhops._core.path.

__all__ = ["local_path", "open_path", "sibling"]

import os

import typing_extensions as tx

from brainhops._core import path


def local_path(file: tx.Any) -> tx.Optional[str]:
    """Return the local path of an existing regular file, or `None`."""
    try:
        local = os.fspath(file)
    except Exception:
        return None
    if isinstance(local, bytes):
        local = os.fsdecode(local)
    return local if os.path.isfile(local) else None


def open_path(file: tx.Any) -> tx.BinaryIO:
    """Open a file for binary reading, locally or through its `open` method."""
    local = local_path(file)
    if local is not None:
        return open(local, "rb")
    return file.open("rb")


def sibling(filename: tx.Any, name: str) -> tx.Any:
    """Resolve `name` against the directory of `filename`, unless absolute.

    An absolute `name` is returned as a path of its own.
    """
    if os.path.isabs(name):
        return path.Path(name)
    if not isinstance(filename, path.PathLike):
        filename = path.Path(filename)
    return filename.parent / name
