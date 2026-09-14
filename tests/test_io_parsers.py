"""Unit tests for the parser/sniffer contracts in `io.base.parsers`."""

import io as _io

import pytest
import typing_extensions as tx

from brainhops.io.base._base import (
    FileBasedObject,
    TextFileBasedObject,
    format_registry,
    register_format,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserExistsError,
    TextFileParser,
    TextFileParserWriter,
    preserve_position,
)


class Greeting(TextFileParser):
    """A one-line text format, used to exercise the base contracts."""

    EXTENSIONS = (".greet",)

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return (
            Confidence.CERTAIN if line.startswith("HELLO") else Confidence.NO
        )

    @classmethod
    def from_line(cls, line, **kwargs) -> tx.Any:  # noqa: ANN001
        return line.strip()


class WritableGreeting(TextFileParserWriter):
    """The same format, able to write itself back."""

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
    """
    A parser may be handed a stream positioned deliberately -- a NIfTI
    embedded in a larger container, say. Rewinding to zero would read
    the wrong bytes and corrupt the caller's position.
    """
    stream = _io.StringIO("JUNK-PREFIX-HELLO world\n")
    stream.seek(12)
    assert Greeting.sniff_fileobj(stream) == Confidence.CERTAIN
    assert stream.tell() == 12
    assert Greeting.from_fileobj(stream) == "HELLO world"
    assert stream.tell() == 12


def test_sniffing_then_parsing_the_same_stream_both_succeed() -> None:
    """Sniffing used to consume the stream, leaving nothing to parse."""
    stream = _io.StringIO("HELLO world\n")
    assert Greeting.sniff_fileobj(stream)
    assert Greeting.from_fileobj(stream) == "HELLO world"


def test_dispatch_works_over_a_non_seekable_stream() -> None:
    """A pipe cannot be rewound, so dispatch must buffer it."""

    @format_registry
    class Root(TextFileBasedObject):
        pass

    fmt = register_format(
        type(
            "Greet",
            (Root,),
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
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


# ----------------------------------------------------------------------
#   READER / WRITER CONTRACTS
# ----------------------------------------------------------------------


def test_reading_a_missing_file_raises_rather_than_returning_false(
    tmp_path,  # noqa: ANN001
) -> None:
    """
    `from_file` used to return `False` for a missing path, which dispatch
    then handed back to the caller as a successfully parsed object.
    """
    with pytest.raises(ParserExistsError):
        Greeting.from_file(tmp_path / "absent.greet")


def test_sniffing_a_missing_file_scores_zero(tmp_path) -> None:  # noqa: ANN001
    assert Greeting.sniff_file(tmp_path / "absent.greet") == Confidence.NO


def test_writing_creates_a_file_that_did_not_exist(tmp_path) -> None:  # noqa: ANN001
    """`to_file` used to refuse to write unless the target already
    existed, and to raise even after writing successfully."""
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
    """`True`/`False` already are `1.0`/`0.0`, so old sniffers keep
    working unchanged."""

    class Boolean(TextFileParser):
        @classmethod
        def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
            return line.startswith("X")

    assert Boolean.sniff_line("X") == Confidence.CERTAIN
    assert Boolean.sniff_line("y") == Confidence.NO
    assert float(Boolean.sniff_line("X")) == 1.0


def test_the_writer_entry_point_does_not_shadow_the_converter() -> None:
    """
    `Transformation.to(cls)` converts an object to another type, and is
    used throughout the datamodel. The writer's generic entry point used
    to be called `to` as well, and won the MRO on every writable
    transformation -- so writing one to a file was impossible.
    """
    from brainhops.datamodel.transformations import Transformation
    from brainhops.io.base.parsers import FileParserWriter
    from brainhops.io.transformations.base import (
        WritableFileBasedTransformation,
    )

    mro = WritableFileBasedTransformation.__mro__
    assert next(c for c in mro if "to" in c.__dict__) is Transformation
    assert next(c for c in mro if "save" in c.__dict__) is FileParserWriter
