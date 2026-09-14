"""
Utilities for reading from streams without consuming them.

TODO: merge or combine this module with `_core.peek`. The two solve the
same problem from opposite ends -- `peekable`/`peekable_lines` let a
sniffer look at the next line without eating it, and `preserve_position`
lets it read a whole stream and put it back. A parser usually needs both,
and importing them from two places obscures that they are one concern.
Folding `peek` in here (or both into a shared module) would make the
"look, don't consume" contract explicit in one place. Left separate for
now because `peek` has callers that would all need updating.
"""

__all__ = [
    "preserve_position",
    "open_compressed",
    "COMPRESSORS",
    "HAS_INDEXED_GZIP",
]

# stdlib
import gzip
from contextlib import contextmanager

# dependencies
import typing_extensions as tx


@contextmanager
def preserve_position(file: tx.IO) -> tx.Generator[tx.IO, None, None]:
    """
    Restore a stream to the position it held on entry.

    Sniffing and parsing both consume the stream, and dispatch may try
    several parsers on the same one, so every consumer must leave the
    stream where it found it -- including when it raises.

    The stream is restored to its *entry* position, not to zero: it may
    have been positioned deliberately, e.g. when a parser is handed a
    sub-stream embedded in a larger file.

    Non-seekable streams (pipes, sockets, stdin) are yielded untouched.

    Parameters
    ----------
    file : IO
        The stream to protect.

    Yields
    ------
    file : IO
        The same stream, unchanged.
    """
    pos = None
    if hasattr(file, "tell") and hasattr(file, "seek"):
        try:
            pos = file.tell()
        except Exception:
            pos = None
    try:
        yield file
    finally:
        if pos is not None:
            try:
                file.seek(pos)
            except Exception:
                pass


try:
    from indexed_gzip import IndexedGzipFile as _IndexedGzipFile
except ImportError:
    _IndexedGzipFile = None

HAS_INDEXED_GZIP = _IndexedGzipFile is not None
"""Whether `indexed_gzip` is available to accelerate gzip seeking."""


def _open_gzip(file: tx.BinaryIO) -> tx.IO:
    """
    Open a gzip stream, preferring `indexed_gzip` when it is installed.

    `gzip.GzipFile` seeks by decompressing from the start of the stream
    every time, which makes random access into a large volume quadratic.
    `indexed_gzip` builds an index of decompression checkpoints instead,
    so seeking is fast -- which matters because array readers routinely
    seek to a voxel offset rather than reading front to back.

    Falls back to the standard library if `indexed_gzip` is missing, or
    refuses the stream (it needs a seekable one).
    """
    if _IndexedGzipFile is not None:
        try:
            return _IndexedGzipFile(fileobj=file)
        except Exception:
            pass
    return gzip.GzipFile(fileobj=file, mode="rb")


# Compressors are keyed by the magic bytes that start their streams.
# `bz2` and `lzma` are optional at build time (a Python compiled without
# libbz2 or liblzma simply lacks them), so each is registered only if it
# imported.
COMPRESSORS: tx.List[tx.Tuple[bytes, tx.Callable[[tx.BinaryIO], tx.IO]]] = [
    (b"\x1f\x8b", _open_gzip),
]
"""
Known compressed-stream formats, as `(magic, opener)` pairs.

Append to this list to teach `open_compressed` a new format; an opener
takes the compressed stream and returns a decompressing one.
"""

try:
    import bz2

    COMPRESSORS.append((b"BZh", lambda f: bz2.BZ2File(f, mode="rb")))
except ImportError:  # pragma: no cover - Python built without libbz2
    pass

try:
    import lzma

    COMPRESSORS.append(
        (b"\xfd7zXZ\x00", lambda f: lzma.LZMAFile(f, mode="rb"))
    )
except ImportError:  # pragma: no cover - Python built without liblzma
    pass


def open_compressed(file: tx.BinaryIO) -> tx.IO:
    """
    Wrap a stream in a decompressor if its content is compressed.

    Libraries generally decide whether to decompress from the *file
    name*, which a bare stream does not have -- so a `.nii.gz` handed
    over as a file object gets read as garbage. Sniffing the magic bytes
    works whatever the stream is called, or is not called.

    Detection starts from the stream's current position, so a compressed
    member embedded partway through a larger file is handled too, and the
    position is restored before wrapping.

    A stream that is not compressed, or cannot be inspected because it is
    not seekable, is returned unchanged.

    Parameters
    ----------
    file : BinaryIO
        The stream to inspect.

    Returns
    -------
    stream : IO
        A decompressing stream, or `file` itself.
    """
    width = max(len(magic) for magic, _ in COMPRESSORS)
    try:
        pos = file.tell()
        head = file.read(width)
        file.seek(pos)
    except Exception:
        # Not seekable: peeking would consume bytes we cannot put back.
        return file

    for magic, opener in COMPRESSORS:
        if head.startswith(magic):
            return opener(file)

    return file
