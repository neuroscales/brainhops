"""Type hints for files and paths, and helpers to name and replace files.

The type hints describe the files, paths and contents that the readers
and writers accept. A reader or a writer converts a file name to a
[`Path`][] once, where it receives it, with [`as_path_or_stream`][],
and keeps the [`Path`][] from then on. The helpers work on any path
that [`Path`][] wraps, local or remote, and cover what [`Path`][] does
not do by itself. A reader asks [`Path`][] directly whether a file
exists, with its method `exists`, and whether a path names a file on
the local file system, with its method `is_local`. A writer that
holds the whole content of a local file replaces the file at once
with the method `write_bytes` of [`Path`][] and its option
`atomic=True`, so that a lazy array that reads the old file is never
left reading a file that is half written. A remote file cannot be
replaced in one step, so it is written in place, which the writer
asks for with `atomic=filename.is_local()`. A writer whose library
writes a file by name uses [`replace_file`][] for the same reason,
and a writer replaces a local directory, such as the store of a
pyramid, with [`replace_directory`][]. Both helpers are built on the
method `staged` of [`Path`][].

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
    "replace_directory",
    "replace_file",
    "is_same_file",
]

import errno
import os
from os import PathLike
from pathlib import Path as LocalPath

import typing_extensions as tx
from bagof.paths import (
    CommitError,
    Path,
    PurePath,
    UnsupportedPathOperation,
)

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


def is_same_file(first: Path, second: Path) -> bool:
    """Return whether two local paths name the same file.

    A writer asks this question before it replaces a file, to learn
    whether a lazy array reads the file that it is about to write. The
    files do not need to exist. The answer comes from the method
    `samefile` of [`Path`][] with `strict=False`, which resolves
    symbolic links and `..`, and counts a hard link to an existing file
    as the same file. A remote path is never the same file as another
    path, even an equal one.
    """
    return first.is_local() and first.samefile(second, strict=False)


# ----------------------------------------------------------------------
#   REPLACING FILES
# ----------------------------------------------------------------------


def replace_file(
    filename: FilenameLike, write: tx.Callable[[Path], None]
) -> None:
    """Write a local file next to its target, then move it over the target.

    A format whose library writes a file by name, such as h5py or
    tifffile, writes it with this function, so that a lazy array that
    reads the old file never reads a file that is half written. The
    function `write` receives a path in a hidden staging directory
    beside the target, with the same name as the target, so that a
    library that chooses its encoding from the extension, as nibabel
    does, chooses the same. Once the file is written, it replaces the
    target at once, following a symbolic link to the file that it
    names. The staging is done by the method `staged` of [`Path`][].

    A writer that reads the old file while it writes the new one
    therefore reads the old content, and a memory map of the old file
    keeps it. A lazy array that opens the file by name reads the new
    file. The new file keeps the permissions of the file that it
    replaces. If `write` fails, or if the file cannot be moved, the
    target is left as it was and the staging directory is removed.

    The function adds three rules to the method `staged`. The target
    must not be a directory. The function `write` must create the file,
    because a writer that writes nothing has failed. An error names the
    target rather than the staged file, which the caller never sees.

    Parameters
    ----------
    filename : str or PathLike
        The local path of the file to write.
    write : callable
        The function that writes the new file, given its path.

    Raises
    ------
    IsADirectoryError
        If the target is a directory.
    FileNotFoundError
        If `write` did not create the file, or if the directory of the
        target does not exist.
    OSError
        If the file cannot be written, with the name of the target.
    """
    target = Path(filename)
    staged = None
    try:
        with target.staged() as staged:
            write(staged)
            if not staged.exists():
                code = errno.ENOENT
                raise FileNotFoundError(code, os.strerror(code), str(target))
    except OSError as error:
        # A failed commit chains the error of the move, which names the
        # staged file, or the target when the target is a directory.
        cause = error.__cause__ if isinstance(error, CommitError) else error
        if isinstance(cause, OSError):
            named = _naming_target(cause, staged, target)
            if named is not error:
                raise named from None
        raise


def _naming_target(
    error: OSError, staged: tx.Optional[Path], target: Path
) -> OSError:
    """Return the error with the target in place of the staged file."""
    if staged is None or error.errno is None:
        return error
    directory = str(staged.parent)
    name = error.filename
    if not isinstance(name, str) or not (
        name == directory or name.startswith(directory + os.sep)
    ):
        return error
    return type(error)(error.errno, error.strerror, str(target))


def replace_directory(
    dirname: FilenameLike, write: tx.Callable[[Path], None]
) -> None:
    """Write a local directory next to its target, then move it into place.

    A format whose container is a directory, such as the store of a
    pyramid, writes it with this function, so that a lazy array that
    reads the files of the old directory never reads a directory that is
    half written, and so that a failed write leaves the old directory as
    it was. The function `write` receives the path of a new, empty
    directory in a hidden staging directory beside the target, with the
    same name as the target, and fills it. Once it is filled, the
    target, if it exists, is moved aside, the new directory is moved to
    the name of the target, following a symbolic link to the directory
    that it names, and the old directory is removed. The staging and the
    moves are done by the method `staged` of [`Path`][].

    A writer may therefore read the files of the old directory while it
    fills the new one, for example to copy the files of the levels that
    did not change. A lazy array that opens its file by name then reads
    the file of the same name in the new directory. The directory cannot
    be swapped in a single step, so a reader that opens the target
    between the two moves finds no directory, but never a half-written
    one. If `write` fails, the target is left as it was and the new
    directory is removed. If the second move fails, the old directory is
    moved back. If even that fails, nothing is removed: the old directory
    stays in a hidden directory beside the target, and the error, a
    [`CommitError`][], names it in its attribute `previous`.

    The function adds three rules to the method `staged`. Only a local
    path is accepted, because a remote store cannot be renamed, so a
    writer of a remote store writes in place, and such a save is not
    atomic. A directory never replaces a file, and the function refuses
    it before `write` is called rather than after. The new directory
    exists, empty, when `write` receives it, because the writers of a
    store expect it.

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
    NotADirectoryError
        If the target exists and is not a directory.
    OSError
        If the directory cannot be written or moved.
    """
    target = Path(dirname)
    if not target.is_local():
        raise ValueError(f"Only a local directory can be replaced: {dirname}")
    # A file may still appear while the directory is written, and the
    # method `staged` then refuses to replace it.
    if target.exists() and not target.is_dir():
        code = errno.ENOTDIR
        raise NotADirectoryError(code, os.strerror(code), str(dirname))
    with target.staged() as staged:
        staged.mkdir()
        write(staged)
