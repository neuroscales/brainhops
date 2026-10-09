"""Independent I/O capabilities and public format registration."""

import io
from pathlib import Path

import pytest
import typing_extensions as tx

from brainhops.datamodel.base import DataModelBase
from brainhops.io.base import (
    FileBasedObject,
    Format,
    FormatDispatcher,
    TextFileBasedObject,
    WritableFileBasedObject,
    format_registry,
    register_format,
    save,
)
from brainhops.io.base.parsers import (
    BinaryFileParser,
    BinaryFileParserWriter,
    BinaryFileReader,
    BinaryFileWriter,
    FileParser,
    FileParserWriter,
    FileReader,
    FileWriter,
    TextFileParser,
    TextFileParserWriter,
    TextFileReader,
    TextFileWriter,
)


@pytest.mark.parametrize(
    "reader",
    [
        FileReader,
        TextFileReader,
        BinaryFileReader,
        FileParser,
        TextFileParser,
        BinaryFileParser,
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
    assert not hasattr(Format, "load")
    assert not hasattr(Format, "save")
    assert not issubclass(FormatDispatcher, FileParser)
    assert not issubclass(FileBasedObject, FileParser)
    assert not issubclass(WritableFileBasedObject, FileParser)


@pytest.mark.parametrize(
    "base",
    [FileParserWriter, TextFileParserWriter, BinaryFileParserWriter],
)
def test_legacy_combinations_still_supply_both_routes(base: type) -> None:
    class Record(base):
        @classmethod
        def from_line(cls, line: str, **kwargs) -> tx.Self:
            return cls()

        def to_line(self, **kwargs) -> str:
            return "record"

    assert issubclass(Record, FileParser)
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
        assert obj.to_bytes(encoding="latin-1") == b"caf\xe9"


class Note(DataModelBase):
    text: str = ""


@pytest.fixture
def public_formats() -> tx.Iterator[tx.Tuple[type, type, type]]:
    @register_format
    class Export(Note, Format, TextFileWriter):
        EXTENSIONS = (".capability",)

        def to_line(self, **kwargs) -> str:
            return self.text

    @register_format
    class ReadOnly(Note, TextFileBasedObject):
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
    assert export in Format._REGISTRY
    assert export not in FileBasedObject._REGISTRY
    assert not hasattr(export, "load")
    assert read_only in FileBasedObject._REGISTRY
    assert not hasattr(read_only, "save")


def test_save_selects_capabilities_without_the_legacy_base(
    public_formats: tx.Tuple[type, type, type],
    tmp_path: Path,
) -> None:
    export, _, read_write = public_formats
    for fmt, suffix in (
        (export, ".readonly.capability"),
        (read_write, ".readwrite"),
    ):
        assert not issubclass(fmt, WritableFileBasedObject)
        target = tmp_path / ("note" + suffix)
        save(Note(text="converted"), target)
        assert target.read_text() == "converted\n"
        stream = io.StringIO()
        save(fmt(text="direct"), stream)
        assert stream.getvalue() == "direct\n"


def test_standalone_metadata_dispatch_does_not_join_public_formats() -> None:
    @format_registry
    class MetadataFormats(FormatDispatcher):
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
    assert Metadata not in Format._REGISTRY
    assert Metadata not in FileBasedObject._REGISTRY
