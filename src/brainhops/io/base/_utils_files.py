"""Helpers for formats that store header and data in separate files."""
# TODO: the rest of the code uses Path objects; use them here as well and
# move generic utilities to brainhops._core.path.

__all__ = [
    "LOCAL_PROTOCOLS",
    "is_local",
    "local_path",
    "open_path",
    "replace_file",
    "same_file",
    "sibling",
    "write_file",
]

import os
import uuid

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


LOCAL_PROTOCOLS = frozenset({"", "file", "local"})
"""The protocols of paths on the local file system."""


def is_local(file: tx.Any) -> bool:
    """Return whether a path names a file on the local file system."""
    return path.Path(file).protocol.lower() in LOCAL_PROTOCOLS


def same_file(first: tx.Any, second: tx.Any) -> bool:
    """Return whether two local paths name the same file.

    The paths are compared once their symbolic links are resolved. A
    stream, a remote path or any other object is never the same file.
    """
    names = []
    for file in (first, second):
        if not isinstance(file, (str, os.PathLike)):
            return False
        try:
            if not is_local(file):
                return False
            names.append(os.path.realpath(os.fspath(file)))
        except Exception:  # noqa: BLE001
            return False
    return names[0] == names[1]


def replace_file(filename: tx.Any, write: tx.Callable[[str], None]) -> None:
    """Write a local file next to its target, then move it over the target.

    The function `write` receives the name of a new file in the directory
    of the target, whose name ends like the name of the target, so that a
    library that chooses its encoding from the extension, as nibabel
    does, chooses the same. Once the file is written, it replaces the
    target at once, following a symbolic link to the file that it names.

    A writer that reads the old file while it writes the new one, such as
    nibabel copying the voxels of a lazy image, therefore reads the old
    content, and a memory map of the old file keeps it. A lazy array that
    opens the file by name reads the new file. The new file gets the
    permissions that the umask gives a new file.

    Raises
    ------
    OSError
        If the file cannot be written, with the name of the target.
    """
    target = os.path.realpath(os.fspath(filename))
    directory, name = os.path.split(target)
    temporary = os.path.join(directory, f".{uuid.uuid4().hex}.{name}")
    try:
        write(temporary)
        os.replace(temporary, target)
    except OSError as error:
        if os.path.exists(temporary):
            os.remove(temporary)
        if error.filename == temporary:
            raise type(error)(error.errno, error.strerror, target) from None
        raise
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def write_file(filename: tx.Any, content: bytes) -> None:
    """Write the bytes of a file, replacing a local file at once.

    A local file is written with [`replace_file`][], so that a lazy array
    or a memory map that reads the old file is not disturbed while the
    new file is written. Other paths are written in place.

    Raises
    ------
    OSError
        If the file cannot be written, with the name of the target.
    """
    if isinstance(filename, str):
        filename = path.Path(filename)
    if not is_local(filename):
        with filename.open("wb") as f:
            f.write(content)
        return

    def write(name: str) -> None:
        with open(name, "xb") as f:
            f.write(content)

    replace_file(filename, write)
