"""Unit tests of the parser and sniffer contracts in `io.base.parsers`."""

import io as _io

import pytest
import typing_extensions as tx

from brainhops._core.path import exists
from brainhops.io.base._base import (
    FileFormat,
    format_registry,
    register_format,
)
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    ParserExistsError,
    ParserNotImplementedError,
    ParserTypeError,
    SnifferContentError,
    SnifferNotImplementedError,
    TextFileReader,
    TextFileWriter,
    WriterError,
    _delegated_from_fileobj,
    _overrides_from_fileobj,
    disabled,
    preserve_position,
)


class Greeting(TextFileReader):
    """A one-line text format that exercises the base contracts."""

    EXTENSIONS = (".greet",)

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return (
            Confidence.CERTAIN if line.startswith("HELLO") else Confidence.NO
        )

    @classmethod
    def from_line(cls, line, **kwargs) -> tx.Any:  # noqa: ANN001
        return line.strip()


class WritableGreeting(TextFileReader, TextFileWriter):
    """The same format, writable."""

    def __init__(self, text: str) -> None:
        self.text = text

    def to_lines(self, **kwargs):  # noqa: ANN001, ANN201
        yield "HELLO " + self.text


# ----------------------------------------------------------------------
#   STREAM POSITION
# ----------------------------------------------------------------------


def test_preserve_position_restores_the_entry_offset() -> None:
    stream = _io.StringIO("0123456789")
    stream.seek(4)
    with preserve_position(stream):
        stream.read()
    assert stream.tell() == 4


def test_preserve_position_restores_even_when_the_body_raises() -> None:
    stream = _io.StringIO("0123456789")
    stream.seek(3)
    with pytest.raises(ValueError):
        with preserve_position(stream):
            stream.read()
            raise ValueError("boom")
    assert stream.tell() == 3


def test_preserve_position_tolerates_a_non_seekable_stream() -> None:
    class Pipe:
        def read(self, *args) -> str:  # noqa: ANN001, ANN002
            return "data"

    pipe = Pipe()
    with preserve_position(pipe) as f:
        assert f.read() == "data"


def test_sniffing_leaves_the_stream_where_it_found_it() -> None:
    stream = _io.StringIO("HELLO world\n")
    assert Greeting.sniff_fileobj(stream)
    assert stream.tell() == 0


def test_parsing_leaves_the_stream_where_it_found_it() -> None:
    stream = _io.StringIO("HELLO world\n")
    assert Greeting.from_fileobj(stream) == "HELLO world"
    assert stream.tell() == 0


def test_a_substream_is_restored_to_its_own_offset_not_to_zero() -> None:
    """A stream handed over at an offset is restored to it, not to zero."""
    stream = _io.StringIO("JUNK-PREFIX-HELLO world\n")
    stream.seek(12)
    assert Greeting.sniff_fileobj(stream) == Confidence.CERTAIN
    assert stream.tell() == 12
    assert Greeting.from_fileobj(stream) == "HELLO world"
    assert stream.tell() == 12


def test_sniffing_then_parsing_the_same_stream_both_succeed() -> None:
    stream = _io.StringIO("HELLO world\n")
    assert Greeting.sniff_fileobj(stream)
    assert Greeting.from_fileobj(stream) == "HELLO world"


def test_dispatch_works_over_a_non_seekable_stream() -> None:
    """A pipe cannot rewind, so dispatch buffers it."""

    @format_registry
    class Root(FileFormat):
        pass

    fmt = register_format(
        type(
            "Greet",
            (TextFileReader, Root),
            {
                "EXTENSIONS": (".greet",),
                "sniff_line": Greeting.__dict__["sniff_line"],
                "from_line": Greeting.__dict__["from_line"],
            },
        )
    )

    class Pipe:
        def __init__(self, data: str) -> None:
            self._stream = _io.StringIO(data)

        def read(self, *args) -> str:  # noqa: ANN001, ANN002
            return self._stream.read(*args)

    try:
        assert Root.from_fileobj(Pipe("HELLO world\n")) == "HELLO world"
    finally:
        FileFormat._REGISTRY.discard(fmt)


# ----------------------------------------------------------------------
#   READER / WRITER CONTRACTS
# ----------------------------------------------------------------------


def test_reading_a_missing_file_raises_rather_than_returning_false(
    tmp_path,  # noqa: ANN001
) -> None:
    """A missing file raises instead of returning False as the object."""
    with pytest.raises(ParserExistsError):
        Greeting.from_file(tmp_path / "absent.greet")


def test_sniffing_a_missing_file_scores_zero(tmp_path) -> None:  # noqa: ANN001
    assert Greeting.sniff_file(tmp_path / "absent.greet") == Confidence.NO


def test_a_name_too_long_to_look_up_is_a_missing_file() -> None:
    """Content passed as a path is a missing file, not an OSError."""
    content = "HELLO world\n" * 100
    assert not exists(content)
    assert Greeting.sniff_file(content) == Confidence.NO
    with pytest.raises(ParserExistsError):
        Greeting.from_file(content)


def test_a_text_sniffer_scores_binary_content_zero(tmp_path) -> None:  # noqa: ANN001
    """Binary content scores zero without leaking a UnicodeDecodeError."""
    binary = b"\x00\x01HELLO\x9a\xff"
    path = tmp_path / "binary.greet"
    path.write_bytes(binary)

    assert Greeting.sniff_bytes(binary) == Confidence.NO
    assert Greeting.sniff(path) == Confidence.NO
    with open(path, "rb") as f:
        assert Greeting.sniff(f) == Confidence.NO
    with pytest.raises(SnifferContentError):
        Greeting.sniff(path, error=True)
    with pytest.raises(SnifferContentError):
        Greeting.sniff_bytes(binary, error=True)


def test_writing_creates_a_file_that_did_not_exist(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "out.greet"
    WritableGreeting("world").save(target)
    assert target.read_text() == "HELLO world\n"


def test_writing_then_reading_round_trips(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "out.greet"
    WritableGreeting("world").save(target)
    assert Greeting.from_file(target) == "HELLO world"


# ----------------------------------------------------------------------
#   SCORES
# ----------------------------------------------------------------------


def test_a_boolean_sniffer_is_a_valid_scoring_sniffer() -> None:
    """A boolean sniffer is valid, with True and False scoring 1.0 and 0.0."""

    class Boolean(TextFileReader):
        @classmethod
        def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
            return line.startswith("X")

    assert Boolean.sniff_line("X") == Confidence.CERTAIN
    assert Boolean.sniff_line("y") == Confidence.NO
    assert float(Boolean.sniff_line("X")) == 1.0


def test_the_writer_entry_point_does_not_shadow_the_converter() -> None:
    """The writer entry point must not shadow Transformation.to(cls)."""
    from brainhops.datamodel.transformations import Transformation
    from brainhops.io.base.parsers import FileWriter, _WriterMixin
    from brainhops.io.transformations.base import (
        TransformationFormat,
    )

    class WritableTransformation(
        FileWriter, TransformationFormat, Transformation
    ):
        pass

    mro = WritableTransformation.__mro__
    assert next(c for c in mro if "to" in c.__dict__) is Transformation
    assert next(c for c in mro if "save" in c.__dict__) is _WriterMixin


# ----------------------------------------------------------------------
#   FROM_BYTES / FROM_FILEOBJ FALLBACKS
# ----------------------------------------------------------------------


class HeaderOnly(BinaryFileReader):
    """A binary format that reads only a 4-byte header from the stream."""

    def __init__(self, magic: bytes) -> None:
        self.magic = magic

    @classmethod
    def from_fileobj(cls, file, **kwargs) -> tx.Self:  # noqa: ANN001
        return cls(file.read(4), **kwargs)


class BytesOnly(BinaryFileReader):
    """A binary format with only from_bytes."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    @classmethod
    def from_bytes(cls, content, **kwargs) -> tx.Self:  # noqa: ANN001
        return cls(bytes(content), **kwargs)


class Neither(BinaryFileReader):
    """A binary format that implements neither entry point."""


class Forwarder(Neither):
    """An unmarked from_fileobj that only forwards to the default."""

    @classmethod
    def from_fileobj(cls, file, **kwargs) -> tx.Self:  # noqa: ANN001
        return super().from_fileobj(file, **kwargs)


def test_a_fileobj_only_parser_loads_from_bytes() -> None:
    assert HeaderOnly.from_bytes(b"MAGICrest").magic == b"MAGI"
    assert HeaderOnly.from_content(b"MAGICrest").magic == b"MAGI"


def test_a_subclass_of_a_fileobj_only_parser_loads_from_bytes() -> None:
    class Sub(HeaderOnly):
        pass

    assert Sub.from_bytes(b"ABCDEF").magic == b"ABCD"


def test_a_parser_with_neither_entry_point_raises_without_recursing() -> None:
    with pytest.raises(ParserNotImplementedError):
        Neither.from_bytes(b"data")
    with pytest.raises(ParserNotImplementedError):
        Neither.from_fileobj(_io.BytesIO(b"data"))


def test_an_unmarked_forwarding_fileobj_raises_without_recursing() -> None:
    with pytest.raises(ParserNotImplementedError):
        Forwarder.from_bytes(b"data")
    with pytest.raises(ParserNotImplementedError):
        Forwarder.from_fileobj(_io.BytesIO(b"data"))


def test_a_bytes_only_parser_still_loads_from_a_fileobj() -> None:
    assert BytesOnly.from_fileobj(_io.BytesIO(b"payload")).content == (
        b"payload"
    )


def test_a_dispatcher_mixin_does_not_count_as_a_fileobj_override() -> None:
    from brainhops.io.base._base import FileFormat

    class Concrete(Neither, FileFormat):
        pass

    with pytest.raises(ParserNotImplementedError):
        Concrete.from_bytes(b"data")

    class ConcreteHeader(HeaderOnly, FileFormat):
        pass

    assert ConcreteHeader.from_bytes(b"WXYZ!").magic == b"WXYZ"


class MagicSniffer(BinaryFileReader):
    """A binary format that scores only a stream, by its first bytes."""

    @classmethod
    def sniff_fileobj(cls, file, error=False, **kwargs) -> float:  # noqa: ANN001
        if file.read(4) == b"MAGI":
            return Confidence.CERTAIN
        return Confidence.NO


def test_a_fileobj_only_sniffer_scores_bytes() -> None:
    assert MagicSniffer.sniff_bytes(b"MAGICrest") == Confidence.CERTAIN
    assert MagicSniffer.sniff_bytes(b"other") == Confidence.NO
    assert MagicSniffer.sniff(b"MAGICrest") == Confidence.CERTAIN


def test_a_sniffer_with_neither_entry_point_raises_without_recursing() -> None:
    with pytest.raises(SnifferNotImplementedError):
        Neither.sniff_bytes(b"data")
    with pytest.raises(SnifferNotImplementedError):
        Neither.sniff_fileobj(_io.BytesIO(b"data"))


def test_text_parsers_still_decode_bytes_to_text() -> None:
    assert Greeting.from_bytes(b"HELLO world\n") == "HELLO world"
    assert Greeting.from_bytes(
        "HELLO \u00e9".encode("latin-1"), encoding="latin-1"
    ) == ("HELLO \u00e9")
    assert Greeting.from_fileobj(_io.BytesIO(b"HELLO there")) == (
        "HELLO there"
    )


def test_a_delegated_fileobj_counts_when_its_delegate_reads_streams() -> None:
    class Delegating(Neither):
        DELEGATE: tx.ClassVar[tx.Optional[type]] = None

        @classmethod
        @_delegated_from_fileobj(lambda cls: cls.DELEGATE)
        def from_fileobj(cls, file, **kwargs) -> tx.Any:  # noqa: ANN001
            if cls.DELEGATE is None:
                return super().from_fileobj(file, **kwargs)
            return cls.DELEGATE.from_fileobj(file, **kwargs)

    class ToHeader(Delegating):
        DELEGATE = HeaderOnly

    class ToBytes(Delegating):
        DELEGATE = BytesOnly

    # Without a delegate, the method is a passthrough.
    assert not _overrides_from_fileobj(Delegating)
    with pytest.raises(ParserNotImplementedError):
        Delegating.from_bytes(b"data")
    # The delegate decides whether the stream is read.
    assert _overrides_from_fileobj(ToHeader)
    assert ToHeader.from_bytes(b"WXYZ!").magic == b"WXYZ"
    assert not _overrides_from_fileobj(ToBytes)


# ----------------------------------------------------------------------
#   DISABLED METHODS
# ----------------------------------------------------------------------


class NoBytes(HeaderOnly):
    """A format that reads streams but refuses bytes."""

    from_bytes = disabled(ParserContentError, "Read a stream instead.")


class NoStream(BinaryFileReader):
    """A format that refuses streams, and therefore bytes as well."""

    from_fileobj = disabled(ParserTypeError, "Read a path instead.")


class NoBytesGreeting(WritableGreeting):
    """A writable format that refuses to be written to bytes."""

    to_bytes = disabled(WriterError, "Write to a path instead.")


def test_a_disabled_reader_raises_its_error_with_its_message() -> None:
    with pytest.raises(ParserContentError, match="Read a stream instead."):
        NoBytes.from_bytes(b"MAGIC")
    # The other entry points still read.
    assert NoBytes.from_fileobj(_io.BytesIO(b"MAGIC")).magic == b"MAGI"
    assert NoBytes.from_bytes.__name__ == "from_bytes"
    assert NoBytes.from_bytes.__doc__ == "Read a stream instead."


def test_a_disabled_fileobj_disables_the_bytes_that_fall_back_to_it() -> None:
    with pytest.raises(ParserTypeError, match="Read a path instead."):
        NoStream.from_fileobj(_io.BytesIO(b"data"))
    with pytest.raises(ParserTypeError, match="Read a path instead."):
        NoStream.from_bytes(b"data")


def test_a_disabled_writer_raises_on_an_instance() -> None:
    greeting = NoBytesGreeting("world")
    with pytest.raises(WriterError, match="Write to a path instead."):
        greeting.to_bytes()
    assert list(greeting.to_lines()) == ["HELLO world"]
