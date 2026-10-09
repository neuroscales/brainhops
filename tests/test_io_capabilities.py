"""Independent I/O capabilities and public format registration."""

import io
from pathlib import Path

import pytest
import typing_extensions as tx

from brainhops.datamodel.base import DataModelBase
from brainhops.io.base import (
    FileBasedObject,
    Format,
    format_registry,
    register_format,
    save,
)
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    FileReader,
    FileWriter,
    ParserContentError,
    TextFileReader,
    TextFileWriter,
)


@pytest.mark.parametrize(
    "reader",
    [
        FileReader,
        TextFileReader,
        BinaryFileReader,
        FileBasedObject,
    ],
)
def test_readers_do_not_acquire_writing(reader: type) -> None:
    assert hasattr(reader, "load")
    assert hasattr(reader, "sniff")
    assert not hasattr(reader, "save")
    assert not hasattr(reader, "to_file")


@pytest.mark.parametrize(
    "writer", [FileWriter, TextFileWriter, BinaryFileWriter]
)
def test_writers_do_not_acquire_reading(writer: type) -> None:
    assert hasattr(writer, "save")
    assert not hasattr(writer, "load")
    assert not hasattr(writer, "from_file")
    assert not hasattr(writer, "sniff")


def test_format_membership_and_public_adapters_do_not_imply_parsing() -> None:
    assert hasattr(Format, "load")
    assert not hasattr(Format, "save")


@pytest.mark.parametrize(
    "dispatcher",
    [
        Format,
        FileBasedObject,
    ],
)
def test_dispatchers_do_not_inherit_the_reader_contract(
    dispatcher: type,
) -> None:
    assert not issubclass(dispatcher, FileReader)


@pytest.mark.parametrize(
    "method",
    [
        "sniff",
        "sniff_file",
        "sniff_filename",
        "sniff_fileobj",
        "sniff_content",
        "sniff_bytes",
        "sniff_text",
        "sniff_lines",
        "sniff_line",
    ],
)
@pytest.mark.parametrize("matches", [True, False])
def test_sniff_selects_a_class_on_dispatchers_and_scores_on_readers(
    method: str, matches: bool, tmp_path: Path
) -> None:
    class Native(TextFileReader):
        @classmethod
        def sniff_line(cls, line: str, **kwargs) -> float:
            return 0.75 if line == "HELLO" else 0.0

    @format_registry
    class Family(FileBasedObject):
        pass

    @register_format
    class Public(TextFileReader, Family):
        @classmethod
        def sniff_line(cls, line: str, **kwargs) -> float:
            return Native.sniff_line(line, **kwargs)

    content = "HELLO" if matches else "OTHER"
    filename = tmp_path / "content"
    filename.write_text(content)
    sources = {
        "sniff": filename,
        "sniff_file": filename,
        "sniff_filename": filename,
        "sniff_fileobj": io.StringIO(content),
        "sniff_content": content,
        "sniff_bytes": content.encode(),
        "sniff_text": content,
        "sniff_lines": [content],
        "sniff_line": content,
    }
    try:
        assert getattr(Native, method)(sources[method]) == (
            0.75 if matches else 0.0
        )
        assert getattr(Family, method)(sources[method]) is (
            Public if matches else None
        )
    finally:
        for base in Public.__mro__[1:]:
            if "_REGISTRY" in base.__dict__:
                base._REGISTRY.discard(Public)


@pytest.mark.parametrize(
    "reader,writer",
    [
        (FileReader, FileWriter),
        (TextFileReader, TextFileWriter),
        (BinaryFileReader, BinaryFileWriter),
    ],
)
def test_format_parsers_compose_both_routes(
    reader: type, writer: type
) -> None:
    class Record(reader, writer):
        @classmethod
        def from_line(cls, line: str, **kwargs) -> tx.Self:
            return cls()

        def to_line(self, **kwargs) -> str:
            return "record"

    assert issubclass(Record, FileReader)
    assert issubclass(Record, FileWriter)
    assert Record.from_text("record").to_text() == "record"


@pytest.mark.parametrize("binary", [False, True])
def test_write_only_adapters_preserve_borrowed_streams(
    binary: bool, tmp_path: Path
) -> None:
    base = BinaryFileWriter if binary else TextFileWriter

    class Record(base):
        def to_line(self, **kwargs) -> str:
            return "café"

        if binary:

            def to_bytes(self, **kwargs) -> bytes:
                return b"\x00\xff"

    obj = Record()
    expected = b"\x00\xff" if binary else "café\n"
    stream = io.BytesIO() if binary else io.StringIO()
    obj.save(stream)
    assert not stream.closed
    assert stream.getvalue() == expected
    target = tmp_path / "record"
    obj.save(target)
    assert (target.read_bytes() if binary else target.read_text()) == expected
    if not binary:
        assert obj.to_bytes(encoding="latin-1") == b"\x63\x61\x66\xe9"


class Note(DataModelBase):
    text: str = ""


@pytest.fixture
def public_formats() -> tx.Iterator[tx.Tuple[type, type, type]]:
    @register_format
    class Export(Note, FileBasedObject, TextFileWriter):
        EXTENSIONS = (".capability",)

        def to_line(self, **kwargs) -> str:
            return self.text

    @register_format
    class ReadOnly(Note, FileBasedObject, TextFileReader):
        # A longer extension must not hide the eligible writer on save.
        EXTENSIONS = (".readonly.capability",)

        @classmethod
        def from_line(cls, line: str, **kwargs) -> tx.Self:
            return cls(text=line)

    @register_format
    class ReadWrite(ReadOnly, TextFileWriter):
        EXTENSIONS = (".readwrite",)

        def to_line(self, **kwargs) -> str:
            return self.text

    yield Export, ReadOnly, ReadWrite
    for fmt in (Export, ReadOnly, ReadWrite):
        for base in fmt.__mro__[1:]:
            if "_REGISTRY" in base.__dict__:
                base._REGISTRY.discard(fmt)


def test_write_only_public_formats_are_not_load_candidates(
    public_formats: tx.Tuple[type, type, type],
) -> None:
    export, read_only, _ = public_formats
    assert export in FileBasedObject._REGISTRY
    assert export not in FileBasedObject._reader_formats()
    assert not issubclass(export, FileReader)
    assert read_only in FileBasedObject._REGISTRY
    assert not hasattr(read_only, "save")


def test_dispatch_ignores_export_only_formats_even_with_matching_names(
    tmp_path: Path,
) -> None:
    @format_registry
    class Family(Format):
        pass

    @register_format
    class Export(Family, TextFileWriter):
        EXTENSIONS = (".exportonly",)

        @classmethod
        def sniff_line(cls, line: str, **kwargs) -> float:
            pytest.fail("An export-only format must not be sniffed")

    filename = tmp_path / "file.exportonly"
    filename.write_text("content")
    assert Family.sniff(filename) is None
    assert Family.sniff_filename(filename) is None
    with pytest.raises(ParserContentError):
        Family.load(filename)


def test_save_selects_writer_capabilities(
    public_formats: tx.Tuple[type, type, type],
    tmp_path: Path,
) -> None:
    export, _, read_write = public_formats
    for fmt, suffix in (
        (export, ".readonly.capability"),
        (read_write, ".readwrite"),
    ):
        assert issubclass(fmt, FileWriter)
        target = tmp_path / ("note" + suffix)
        save(Note(text="converted"), target)
        assert target.read_text() == "converted\n"
        stream = io.StringIO()
        save(fmt(text="direct"), stream)
        assert stream.getvalue() == "direct\n"


def test_standalone_metadata_dispatch_does_not_join_public_formats() -> None:
    @format_registry
    class MetadataFormats(Format):
        pass

    @register_format
    class Metadata(MetadataFormats, TextFileReader, TextFileWriter):
        @classmethod
        def sniff_line(cls, line: str, **kwargs) -> float:
            return 1.0

        @classmethod
        def from_line(cls, line: str, **kwargs) -> tx.Self:
            return cls()

    assert isinstance(MetadataFormats.from_text("metadata"), Metadata)
    assert Metadata not in FileBasedObject._REGISTRY
