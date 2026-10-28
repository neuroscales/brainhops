"""Type hints for files and paths, and helpers to name and replace files.

The type hints describe the files, paths and contents that the readers
and writers accept. A reader or a writer converts a file name to a
[`Path`][] once, where it receives it, with [`as_path_or_stream`][],
and keeps the [`Path`][] from then on. The helpers work on any path
that [`Path`][] wraps, local or remote, and cover what [`Path`][] does
not do by itself. A reader asks [`Path`][] directly whether a file
exists, with its method `exists`, and whether a path names a file on
the local file system, with its method `is_local`. A writer replaces
a local file at once with [`atomic_write_bytes`][] or
[`replace_file`][], so that a lazy array that reads the old file is
never left reading a file that is half written. A writer replaces a
local directory, such as the store of a pyramid, with
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
    "as_path_or_stream",
    "base_name",
    "is_local_file",
    "replace_directory",
    "replace_file",
    "is_same_file",
    "atomic_write_bytes",
]

import uuid
from os import PathLike
from pathlib import Path as LocalPath

import typing_extensions as tx
from bagof.paths import Path, PurePath, UnsupportedPathOperation

_T = tx.TypeVar("_T")

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


# ----------------------------------------------------------------------
#   LOCAL AND REMOTE PATHS
# ----------------------------------------------------------------------


@tx.overload
def as_path_or_stream(other: FilenameLike) -> Path: ...


@tx.overload
def as_path_or_stream(other: _T) -> _T: ...


def as_path_or_stream(other: tx.Any) -> tx.Any:
    """Return a file name as a [`Path`][], and any other value unchanged.

    A reader or a writer that accepts either the name of a file or an
    open file calls this function once, where it receives its argument,
    and keeps the result. A string or a path-like object becomes a
    [`Path`][], and a [`Path`][] is returned as it is. Any other value,
    such as an open file, bytes or `None`, is returned unchanged, so
    that the caller can tell a path from a stream with `isinstance(x,
    Path)`. The function reads only its argument and never touches
    storage.

    Parameters
    ----------
    other : Any
        The name of a file, or any other value.

    Returns
    -------
    Any
        The name as a [`Path`][], or `other` itself.
    """
    if isinstance(other, Path):
        return other
    if isinstance(other, (str, PathLike)):
        return Path(other)
    return other


def base_name(other: tx.Any) -> tx.Optional[str]:
    """Return the base name of the file that a path or an open file names.

    A format reads the extension of a file from this name, and an open
    file is named by its `name` attribute. The name is read from the
    text of the path with [`PurePath`][], without touching storage. The
    query of a URL is not part of the name, and neither is the fragment
    of an HTTP URL, so `https://host/x.nii.gz?token=...` names
    `x.nii.gz`. For an fsspec chain such as `simplecache::s3://...`, the
    name comes from the last link. For content, such as bytes, or for an
    open file without a name, the function returns `None`.

    A name given as text is not converted with [`as_path_or_stream`][],
    because building a [`Path`][] chooses a driver for the scheme of a
    URL, and fails when no installed driver reads that scheme, while the
    name can still be read from the text.
    """
    if not isinstance(other, (str, PathLike)):
        other = getattr(other, "name", None)
        if not isinstance(other, str):
            return None
    try:
        return PurePath(other).name
    except UnsupportedPathOperation:
        # A path object that no driver knows is read from its text.
        return PurePath(str(other)).name


def is_local_file(file: tx.Any) -> bool:
    """Return whether a value is a path to an existing file on this machine.

    A library that opens a file by name only, or a memory map, needs a
    file on the local file system. A remote path, a missing file and a
    directory are not local files. A value that is not a [`Path`][],
    such as an open file, is not a local file either, so that a caller
    that holds a path or a stream can ask without telling them apart
    first.
    """
    return isinstance(file, Path) and file.is_local() and file.is_file()


def is_same_file(first: Path, second: Path) -> bool:
    """Return whether two local paths name the same file.

    The paths are compared once their symbolic links are resolved, so the
    files do not need to exist. A remote path is never the same file as
    another path.
    """
    if not (first.is_local() and second.is_local()):
        return False
    return first.resolve() == second.resolve()


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
        if error.filename is not None and str(error.filename) == str(
            temporary
        ):
            raise type(error)(
                error.errno, error.strerror, str(target)
            ) from None
        raise
    finally:
        # The new file is gone once it replaced the target, and is
        # removed if the write failed.
        temporary.unlink(missing_ok=True)


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
    if not Path(dirname).is_local():
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


def atomic_write_bytes(filename: FilenameLike, content: bytes) -> None:
    """Write the bytes of a file, replacing a local file at once.

    A local file is written with [`replace_file`][], so that a lazy array
    or a memory map that reads the old file is not disturbed while the
    new file is written. A remote file cannot be replaced in one step,
    so it is written in place, once the whole content is known.

    Raises
    ------
    OSError
        If the file cannot be written, with the name of the target.
    """
    filename = Path(filename)
    if not filename.is_local():
        filename.write_bytes(content)
        return

    def write(name: Path) -> None:
        with name.open("xb") as f:
            f.write(content)

    replace_file(filename, write)
