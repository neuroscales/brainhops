"""Type hints for files and paths, and helpers that find and replace files.

The type hints describe the files, paths and contents that the readers
and writers accept. The helpers work on any path that [`Path`][] wraps,
local or remote. The formats that store a header and its data in
separate files find one from the other with [`sibling`][]. A writer replaces a local file at once with
[`write_file`][] or [`replace_file`][], so that a lazy array that reads
the old file is never left reading a file that is half written. A
writer replaces a local directory, such as the store of a pyramid, with
[`replace_directory`][] for the same reason.

!!! warning "Internal module"
    `brainhops._core.path` is an internal module. It is documented to
    explain the type hints that appear in the public interface and to
    help contributors who write a format, and it should not be imported
    or relied upon outside brainhops. It may change or disappear
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
    "LOCAL_PROTOCOLS",
    "is_local",
    "local_path",
    "open_path",
    "replace_directory",
    "replace_file",
    "same_file",
    "sibling",
    "write_file",
]

import errno
import os
import uuid
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


# ----------------------------------------------------------------------
#   LOCAL AND REMOTE PATHS
# ----------------------------------------------------------------------


LOCAL_PROTOCOLS = frozenset({"", "file", "local"})
"""The protocols of paths on the local file system."""


def is_local(file: FilenameLike) -> bool:
    """Return whether a path names a file on the local file system.

    Raises
    ------
    TypeError
        If `file` is not a path, for example an open file.
    """
    return Path(file).protocol.lower() in LOCAL_PROTOCOLS


def local_path(file: tx.Any) -> tx.Optional[Path]:
    """Return the path of an existing local regular file, or `None`.

    A library that opens a file by name only, or a memory map, needs a
    file on the local file system. A remote path, an open file, a missing
    file and a directory all give `None`, and the caller then reads the
    file through a stream instead.
    """
    if isinstance(file, bytes):
        file = os.fsdecode(file)
    if not isinstance(file, (str, PathLike)):
        return None
    try:
        file = Path(file)
        if not is_local(file):
            return None
        return file if file.is_file() else None
    except (TypeError, ValueError, OSError):
        return None


def open_path(file: FilenameLike) -> tx.BinaryIO:
    """Open a local or remote file for binary reading.

    A string is a path, and any other path is opened with its own `open`
    method, so a remote path is read through its file system.
    """
    if not hasattr(file, "open"):
        file = Path(file)
    return file.open("rb")


def sibling(filename: FilenameLike, name: str) -> Path:
    """Return the path of a file named relative to the directory of another.

    A header that stores its data in another file names the data file
    relative to the directory of the header. This function resolves such
    a name against the directory of `filename`, which may be remote. An
    absolute `name` names a local file of its own and is returned as it
    is.
    """
    if LocalPath(name).is_absolute():
        return Path(name)
    return Path(filename).parent / name


def same_file(first: tx.Any, second: tx.Any) -> bool:
    """Return whether two local paths name the same file.

    The paths are compared once their symbolic links are resolved, so the
    files do not need to exist. A stream, a remote path or any other
    object is never the same file.
    """
    names = []
    for file in (first, second):
        if not isinstance(file, (str, PathLike)):
            return False
        try:
            file = Path(file)
            if not is_local(file):
                return False
            names.append(file.resolve())
        except Exception:  # noqa: BLE001
            return False
    return names[0] == names[1]


# ----------------------------------------------------------------------
#   REPLACING FILES
# ----------------------------------------------------------------------


def replace_file(
    filename: FilenameLike, write: tx.Callable[[Path], None]
) -> None:
    """Write a local file next to its target, then move it over the target.

    The function `write` receives the path of a new file in the directory
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
    target = Path(filename).resolve()
    temporary = target.with_name(f".{uuid.uuid4().hex}.{target.name}")
    try:
        write(temporary)
        temporary.replace(target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        if error.filename is not None and str(error.filename) == str(
            temporary
        ):
            raise type(error)(
                error.errno, error.strerror, str(target)
            ) from None
        raise
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def replace_directory(
    dirname: FilenameLike, write: tx.Callable[[Path], None]
) -> None:
    """Write a local directory next to its target, then move it into place.

    A format whose container is a directory, such as the store of a
    pyramid, writes it with this function, so that a lazy array that
    reads the files of the old directory never reads a directory that is
    half written, and so that a failed write leaves the old directory as
    it was. The function `write` receives the path of a new, empty
    directory beside the target, whose name ends like the name of the
    target, as for [`replace_file`][], and fills it. Once it is filled,
    the target, if it exists, is moved aside, the new directory is moved
    to the name of the target, following a symbolic link to the
    directory that it names, and the old directory is removed.

    A writer may therefore read the files of the old directory while it
    fills the new one, for example to copy the files of the levels that
    did not change. A lazy array that opens its file by name then reads
    the file of the same name in the new directory. The directory cannot
    be swapped in a single step, so a reader that opens the target
    between the two moves finds no directory, but never a half-written
    one. If the second move fails, the old directory is moved back.

    Only a local path is accepted. A remote store cannot be renamed, so
    a writer of a remote store writes in place, and such a save is not
    atomic.

    Parameters
    ----------
    dirname : str or PathLike
        The local path of the directory to write.
    write : callable
        The function that fills the new directory, given its path.

    Raises
    ------
    ValueError
        If `dirname` is not a local path.
    OSError
        If the directory cannot be written or moved.
    """
    if not is_local(dirname):
        raise ValueError(f"Only a local directory can be replaced: {dirname}")
    target = Path(dirname).resolve()
    token = uuid.uuid4().hex
    temporary = target.with_name(f".{token}.{target.name}")
    old = target.with_name(f".{token}.old.{target.name}")
    temporary.mkdir()
    try:
        write(temporary)
    except BaseException:
        _remove_tree(temporary)
        raise
    if not (target.is_symlink() or target.exists()):
        temporary.replace(target)
        return
    target.replace(old)
    try:
        temporary.replace(target)
    except BaseException:
        old.replace(target)
        _remove_tree(temporary)
        raise
    if old.is_dir() and not old.is_symlink():
        old.rmtree()
    else:
        old.unlink()


def _remove_tree(directory: Path) -> None:
    """Remove a directory that a failed write left, ignoring errors."""
    try:
        directory.rmtree()
    except OSError:
        pass


def write_file(filename: FilenameLike, content: bytes) -> None:
    """Write the bytes of a file, replacing a local file at once.

    A local file is written with [`replace_file`][], so that a lazy array
    or a memory map that reads the old file is not disturbed while the
    new file is written. Other paths are written in place.

    Raises
    ------
    OSError
        If the file cannot be written, with the name of the target.
    """
    filename = Path(filename)
    if not is_local(filename):
        filename.write_bytes(content)
        return

    def write(name: Path) -> None:
        with name.open("xb") as f:
            f.write(content)

    replace_file(filename, write)
