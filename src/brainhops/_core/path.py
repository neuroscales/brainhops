"""Type hints for files and paths, and helpers that find and replace files.

The type hints describe the files, paths and contents that the readers
and writers accept. The helpers work on any path that [`Path`][] wraps,
local or remote. A writer replaces a local file at once with
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
    "base_name",
    "local_path",
    "replace_directory",
    "replace_file",
    "is_same_file",
    "write_file",
]

import errno
import os
import uuid
from os import DirEntry, PathLike, sep
from os.path import basename
from pathlib import Path as LocalPath
from urllib.parse import urlsplit

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


def base_name(other: tx.Any) -> tx.Optional[str]:
    """Return the base name of the file that a path or an open file names.

    A format reads the extension of a file from this name, and an open
    file is named by its `name` attribute. The name is taken from the
    text of the path without touching storage,
    because `os.fspath` raises on remote paths and downloads cloud paths.
    For a URL, the name is the last segment of its path, without query or
    fragment, so `https://host/x.nii.gz?token=...` names `x.nii.gz`. For an
    fsspec chain such as `simplecache::s3://...`, the name comes from the
    last link.
    """
    if isinstance(other, DirEntry):
        # str() of a DirEntry gives its repr.
        text = other.path
    elif isinstance(other, (str, PathLike)):
        text = str(other)
    else:
        text = getattr(other, "name", None)
        if not isinstance(text, str):
            return None
    return _base_name_of_text(text)


def _base_name_of_text(text: str) -> str:
    """Return the last component of a local path or of the path of a URL."""
    if "::" in text and "://" in text:
        # In an fsspec chain, the last link is the file, and the other links
        # are layers such as caches or archives.
        text = text.rsplit("::", 1)[-1]
    if _has_scheme(text):
        # Only the path of the URL is kept, because the query and the
        # fragment may contain slashes and dots.
        text = urlsplit(text).path
        # Trailing slashes are stripped so that a directory store, such as a
        # .zarr folder, still gets a name.
        return text.rstrip("/").rsplit("/", 1)[-1]
    return basename(text.rstrip("/" + sep))


def _has_scheme(text: str) -> bool:
    """Return whether `text` starts with a URL scheme such as `s3:`.

    A scheme has at least two characters, so a drive such as `C:` is not one.
    """
    scheme, colon, _ = text.partition(":")
    return (
        bool(colon)
        and len(scheme) > 1
        and scheme.isascii()
        and scheme[0].isalpha()
        and all(c.isalnum() or c in "+-." for c in scheme)
    )


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


def is_same_file(first: tx.Any, second: tx.Any) -> bool:
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
