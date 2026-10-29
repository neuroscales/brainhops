"""Utilities for reading streams without consuming them, and compression."""

# TODO: Merge this module with _core.peek, which also looks ahead without
# consuming its input. The modules stay separate until the many callers of
# peek are updated.

__all__ = [
    "preserve_position",
    "open_compressed",
    "open_decompressed",
    "is_seekable",
    "decompress",
    "compression_suffix",
    "open_compressing",
    "COMPRESSORS",
    "COMPRESSING",
    "HAS_INDEXED_GZIP",
]

import gzip
from contextlib import contextmanager
from io import BytesIO, TextIOWrapper

import typing_extensions as tx


@contextmanager
def preserve_position(file: tx.IO) -> tx.Generator[tx.IO, None, None]:
    """Restore a stream to its current position when the context exits.

    Format detectors and parsers consume the streams that they read, and
    the code that chooses a reader may try several of them on the same
    stream. The context manager therefore records the position of the
    stream on entry and seeks back to it on exit, even after an exception.
    The stream returns to the entry position rather than to the start
    because it may be a part of a larger file. A stream that cannot seek is
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


def open_gzip(file: tx.BinaryIO) -> tx.IO:
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
    (b"\x1f\x8b", open_gzip),
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

    Libraries usually detect compression from the file name. A bare stream
    has no file name, so a file object opened on a `.nii.gz` file would
    otherwise read as garbage. The magic bytes are read from the current
    position, which is restored before the stream is wrapped.

    Parameters
    ----------
    file : BinaryIO
        Binary stream, read from its current position.

    Returns
    -------
    tx.IO
        A decompressing stream, or `file` itself when the data are not
        compressed, when the stream cannot seek, or when it is a text
        stream.
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
    if not isinstance(head, (bytes, bytearray)):
        # A text stream has already been decoded, so it is not compressed.
        return file

    for magic, opener in COMPRESSORS:
        if head.startswith(magic):
            return opener(file)

    return file


@contextmanager
def open_decompressed(
    file: tx.Any, mode: str = "rb"
) -> tx.Generator[tx.IO, None, None]:
    """Open a file for reading, decompressed if its magic bytes say so.

    A reader opens a path with this function so that it reads a file
    compressed with one of the [`COMPRESSORS`][] as if it were not. The
    file is opened in binary mode with its own `open` method, which a
    `Path` of `bagof.paths` has, and decompressed with
    [`open_compressed`][]. In a text mode, such as `"r"` or `"rt"`, the
    stream is then decoded as the built-in `open` would decode it. A
    text format cannot decompress a stream that is already decoded,
    which is why a path is decompressed as it is opened.

    Parameters
    ----------
    file : Path
        The path of the file, an object with an `open` method.
    mode : str, default="rb"
        The mode to read in, binary or text.

    Yields
    ------
    tx.IO
        The decompressed stream.
    """
    with file.open("rb") as raw:
        stream = open_compressed(raw)
        try:
            if "b" in mode:
                yield stream
            else:
                with TextIOWrapper(stream) as text:
                    yield text
        finally:
            if stream is not raw:
                stream.close()


def is_seekable(file: tx.IO) -> bool:
    """Return whether a stream can seek.

    A stream without a `seekable` method, or whose method raises, cannot
    seek.
    """
    try:
        return bool(file.seekable())
    except Exception:  # noqa: BLE001
        return False


def decompress(content: bytes) -> bytes:
    """Return the decompressed content of a file, if its magic bytes match.

    Content that is not compressed with one of the [`COMPRESSORS`][] is
    returned as it is.

    Parameters
    ----------
    content : bytes
        The content of a file.

    Returns
    -------
    bytes
        The decompressed content.
    """
    stream = BytesIO(content)
    opened = open_compressed(stream)
    if opened is stream:
        return content
    return opened.read()


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


COMPRESSING: tx.Dict[str, tx.Callable[[tx.BinaryIO], tx.BinaryIO]] = {
    # The name of the file is left out of the gzip header, because a writer
    # may write a temporary file that is renamed afterwards.
    ".gz": lambda f: gzip.GzipFile(filename="", fileobj=f, mode="wb"),
}
"""Compressors, keyed by the file name suffix that calls for them.

A compressor takes a binary stream open for writing and returns a
stream that compresses what it receives into it. Closing the
compressing stream finishes the compressed data without closing the
stream that it wraps. The bz2 and lzma formats are registered only if
their modules can be imported.
"""

try:
    import bz2

    COMPRESSING[".bz2"] = lambda f: bz2.BZ2File(f, mode="wb")
except ImportError:  # pragma: no cover - Python built without libbz2
    pass

try:
    import lzma

    COMPRESSING[".xz"] = lambda f: lzma.LZMAFile(f, mode="wb")
except ImportError:  # pragma: no cover - Python built without liblzma
    pass


def compression_suffix(name: tx.Optional[str]) -> tx.Optional[str]:
    """Return the compression suffix that ends a file name, or `None`.

    The suffixes are the keys of [`COMPRESSING`][], such as `".gz"`, and
    they are compared without regard to case.
    """
    if not name:
        return None
    lower = name.lower()
    for suffix in COMPRESSING:
        if lower.endswith(suffix):
            return suffix
    return None


@contextmanager
def open_compressing(
    file: tx.BinaryIO, suffix: tx.Optional[str]
) -> tx.Generator[tx.BinaryIO, None, None]:
    """Compress what is written into a binary stream, as a suffix asks.

    The context yields a stream that compresses into `file` with the
    compressor of `suffix`, and finishes the compressed data when it
    exits, without closing `file`. Without a suffix, `file` itself is
    yielded.

    Parameters
    ----------
    file : BinaryIO
        The stream that receives the compressed data.
    suffix : str, optional
        A key of [`COMPRESSING`][], such as `".gz"`.

    Yields
    ------
    BinaryIO
        The stream to write the uncompressed content to.
    """
    if suffix is None:
        yield file
        return
    stream = COMPRESSING[suffix](file)
    try:
        yield stream
    finally:
        stream.close()
