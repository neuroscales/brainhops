"""Tests of the generic stream helpers."""

import bz2
import gzip
import io
import lzma

import pytest

from brainhops._core.streams import (
    COMPRESSORS,
    open_compressed,
    preserve_position,
)

PAYLOAD = b"the quick brown fox" * 16

CODECS = {
    "gzip": gzip.compress,
    "bz2": bz2.compress,
    "lzma": lzma.compress,
}


# ----------------------------------------------------------------------
#   preserve_position
# ----------------------------------------------------------------------


def test_preserve_position_restores_the_entry_offset() -> None:
    stream = io.BytesIO(b"0123456789")
    stream.seek(4)
    with preserve_position(stream):
        stream.read()
    assert stream.tell() == 4


def test_preserve_position_restores_when_the_body_raises() -> None:
    stream = io.BytesIO(b"0123456789")
    stream.seek(3)
    with pytest.raises(ValueError):
        with preserve_position(stream):
            stream.read()
            raise ValueError("boom")
    assert stream.tell() == 3


# ----------------------------------------------------------------------
#   open_compressed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CODECS))
def test_a_compressed_stream_is_decompressed(name: str) -> None:
    """Compression is detected from the magic bytes, not from a name."""
    stream = io.BytesIO(CODECS[name](PAYLOAD))
    assert open_compressed(stream).read() == PAYLOAD


def test_an_uncompressed_stream_is_returned_unchanged() -> None:
    stream = io.BytesIO(PAYLOAD)
    assert open_compressed(stream) is stream
    assert stream.read() == PAYLOAD


def test_detection_does_not_consume_the_stream() -> None:
    """Detection puts the peeked magic bytes back."""
    stream = io.BytesIO(PAYLOAD)
    open_compressed(stream)
    assert stream.tell() == 0
    assert stream.read() == PAYLOAD


@pytest.mark.parametrize("name", sorted(CODECS))
def test_a_compressed_member_at_a_non_zero_offset(name: str) -> None:
    """Detection starts at the current position, not at offset zero."""
    prefix = b"CONTAINER-HEADER"
    stream = io.BytesIO(prefix + CODECS[name](PAYLOAD))
    stream.seek(len(prefix))
    assert open_compressed(stream).read() == PAYLOAD


def test_a_non_seekable_stream_is_returned_unchanged() -> None:
    """Peeking would lose bytes that a non-seekable stream cannot give back."""

    class Pipe:
        def __init__(self, data: bytes) -> None:
            self._stream = io.BytesIO(data)

        def read(self, *args) -> bytes:  # noqa: ANN002
            return self._stream.read(*args)

    pipe = Pipe(gzip.compress(PAYLOAD))
    assert open_compressed(pipe) is pipe


def test_every_registered_compressor_round_trips() -> None:
    """Every registered compressor decodes the format it claims."""
    for magic, opener in COMPRESSORS:
        payload = {
            b"\x1f\x8b": gzip.compress,
            b"BZh": bz2.compress,
            b"\xfd7zXZ\x00": lzma.compress,
        }[magic](PAYLOAD)
        assert payload.startswith(magic), magic
        assert opener(io.BytesIO(payload)).read() == PAYLOAD


def test_gzip_uses_indexed_gzip_when_available() -> None:
    """`indexed_gzip` is preferred when installed; the output is the same."""
    from brainhops._core.streams import HAS_INDEXED_GZIP, _open_gzip

    stream = io.BytesIO(gzip.compress(PAYLOAD))
    opened = _open_gzip(stream)
    if HAS_INDEXED_GZIP:
        assert type(opened).__name__ == "IndexedGzipFile"
    else:
        assert isinstance(opened, gzip.GzipFile)
    assert opened.read() == PAYLOAD


def test_gzip_seeking_works_whichever_backend_is_used() -> None:
    """Random access, the reason to prefer `indexed_gzip`, works either way."""
    from brainhops._core.streams import _open_gzip

    body = bytes(range(256)) * 64
    opened = _open_gzip(io.BytesIO(gzip.compress(body)))
    for offset in (0, 100, 5000, 250, len(body) - 8):
        opened.seek(offset)
        assert opened.read(8) == body[offset : offset + 8]


def test_gzip_falls_back_when_the_accelerator_refuses(monkeypatch) -> None:  # noqa: ANN001
    """A refusal by `indexed_gzip` falls back to the standard library."""
    import brainhops._core.streams as streams

    def refuse(**kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("nope")

    monkeypatch.setattr(streams, "_IndexedGzipFile", refuse)
    opened = streams._open_gzip(io.BytesIO(gzip.compress(PAYLOAD)))
    assert isinstance(opened, gzip.GzipFile)
    assert opened.read() == PAYLOAD
