"""Utilities for reading streams without consuming them."""

# TODO: merge with _core.peek, which also looks ahead without consuming;
# the modules stay separate until the many callers of peek are updated.

__all__ = [
    "preserve_position",
    "open_compressed",
    "COMPRESSORS",
    "HAS_INDEXED_GZIP",
]

import gzip
from contextlib import contextmanager

import typing_extensions as tx


@contextmanager
def preserve_position(file: tx.IO) -> tx.Generator[tx.IO, None, None]:
    """Restore a stream to its current position when the context exits.

    Sniffers and parsers consume the streams that they read, and a
    dispatch may try several of them on one stream. The entry position is
    restored even after an exception, rather than zero, since the stream
    may be a sub-stream of a larger file. A stream that cannot seek is
    yielded untouched.

    Parameters
    ----------
    file : IO
        Stream to protect.

    Yields
    ------
    tx.IO
        The same stream.
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
"""Whether indexed_gzip can be imported, which speeds up seeking in gzip."""


def _open_gzip(file: tx.BinaryIO) -> tx.IO:
    """Open a gzip stream, preferably with indexed_gzip.

    The standard [`GzipFile`][gzip.GzipFile] seeks by decompressing again
    from the start, which makes random access into large volumes
    quadratic, whereas indexed_gzip keeps an index of checkpoints. The
    standard library is used when indexed_gzip is missing or refuses the
    stream.
    """
    if _IndexedGzipFile is not None:
        try:
            return _IndexedGzipFile(fileobj=file)
        except Exception:
            pass
    return gzip.GzipFile(fileobj=file, mode="rb")


COMPRESSORS: tx.List[tx.Tuple[bytes, tx.Callable[[tx.BinaryIO], tx.IO]]] = [
    (b"\x1f\x8b", _open_gzip),
]
"""Decompressors, as pairs of leading magic bytes and an opener.

An opener takes the compressed stream and returns a decompressing one.
Appending a pair teaches [`open_compressed`][] a new format. The bz2 and
lzma formats are registered only if their modules can be imported.
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
    """Wrap a binary stream in a decompressor if its magic bytes match.

    Libraries usually detect compression from the file name, which a bare
    stream lacks, so a `.nii.gz` file object would otherwise read as
    garbage. Detection starts at the current position, which is restored
    before the stream is wrapped.

    Parameters
    ----------
    file : BinaryIO
        Binary stream, read from its current position.

    Returns
    -------
    tx.IO
        A decompressing stream, or `file` itself when the data are not
        compressed or the stream cannot seek.
    """
    width = max(len(magic) for magic, _ in COMPRESSORS)
    try:
        pos = file.tell()
        head = file.read(width)
        file.seek(pos)
    except Exception:
        # Peeking at a stream that cannot seek would consume bytes that cannot
        # be recovered.
        return file

    for magic, opener in COMPRESSORS:
        if head.startswith(magic):
            return opener(file)

    return file
