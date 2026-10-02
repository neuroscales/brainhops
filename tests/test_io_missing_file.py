"""
A `str` given where a file is expected is a path (issue #121).

`FileParser.load("missing.nii")` used to raise `ParserTypeError: Cannot
parse file of type str`, while `load(Path("missing.nii"))` raised
`FileNotFoundError`. Dispatchers such as `io.load` reported either
spelling as a list of parsers that each failed to read the file. A
missing file now raises `FileNotFoundError`, with the path in the
message, through every entry point and whichever way the path is spelled.
"""

import io as _io
from pathlib import Path

import numpy as np
import pytest
import typing_extensions as tx

nb = pytest.importorskip("nibabel")

import brainhops.io as io  # noqa: E402
from brainhops.cli._errors import CliError  # noqa: E402
from brainhops.cli._io import load_image, load_transform  # noqa: E402
from brainhops.io.base import ImageSpec  # noqa: E402
from brainhops.io.base._base import (  # noqa: E402
    FileBasedObject,
    TextFileBasedObject,
    format_registry,
    register_format,
)
from brainhops.io.base._dispatch import Source  # noqa: E402
from brainhops.io.base.parsers import (  # noqa: E402
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferExistsError,
    TextFileParser,
)
from brainhops.io.images import FileBasedImage  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations import FileBasedTransformation  # noqa: E402


class Greeting(TextFileParser):
    """A one-line text format, which can also be read from memory."""

    EXTENSIONS = (".greet",)

    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return (
            Confidence.CERTAIN if line.startswith("HELLO") else Confidence.NO
        )

    @classmethod
    def from_line(cls, line, **kwargs) -> tx.Any:  # noqa: ANN001
        return line.strip()


@pytest.fixture
def missing(tmp_path) -> Path:  # noqa: ANN001
    return tmp_path / "missing.nii"


def _spellings(missing: Path) -> tx.List[tx.Any]:
    return [str(missing), missing]


def _assert_missing(excinfo: pytest.ExceptionInfo, missing: Path) -> None:
    assert isinstance(excinfo.value, FileNotFoundError)
    assert str(missing) in str(excinfo.value)


# ----------------------------------------------------------------------
#   CONCRETE PARSERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize("as_str", [True, False])
def test_concrete_load_of_a_missing_file(missing, as_str) -> None:  # noqa: ANN001
    source = str(missing) if as_str else missing
    with pytest.raises(ParserExistsError) as excinfo:
        NiftiImage.load(source)
    _assert_missing(excinfo, missing)


@pytest.mark.parametrize("as_str", [True, False])
def test_generic_parser_load_of_a_missing_file(tmp_path, as_str) -> None:  # noqa: ANN001
    missing = tmp_path / "missing.greet"
    source = str(missing) if as_str else missing
    with pytest.raises(ParserExistsError) as excinfo:
        Greeting.load(source)
    _assert_missing(excinfo, missing)


def test_concrete_load_of_an_existing_file_given_as_str(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "hello.greet"
    target.write_text("HELLO world\n")
    assert Greeting.load(str(target)) == "HELLO world"


def test_in_memory_text_is_still_read_through_the_content_doors() -> None:
    assert Greeting.from_text("HELLO world") == "HELLO world"
    assert Greeting.from_content("HELLO world") == "HELLO world"
    assert Greeting.sniff_text("HELLO world") == Confidence.CERTAIN


# ----------------------------------------------------------------------
#   DISPATCHERS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [io.load, io.images.load, io.transformations.load],
    ids=["io.load", "images.load", "transformations.load"],
)
def test_dispatched_load_of_a_missing_file(missing, entry) -> None:  # noqa: ANN001
    for source in _spellings(missing):
        with pytest.raises(ParserExistsError) as excinfo:
            entry(source)
        _assert_missing(excinfo, missing)


@pytest.mark.parametrize(
    "dispatcher", [FileBasedObject, FileBasedImage, FileBasedTransformation]
)
def test_dispatched_from_file_of_a_missing_file(missing, dispatcher) -> None:  # noqa: ANN001
    for source in _spellings(missing):
        with pytest.raises(ParserExistsError) as excinfo:
            dispatcher.from_file(source)
        _assert_missing(excinfo, missing)


def test_dispatched_load_of_a_missing_spec(missing) -> None:  # noqa: ANN001
    with pytest.raises(ParserExistsError) as excinfo:
        io.images.load(ImageSpec(path=str(missing)))
    _assert_missing(excinfo, missing)


def test_dispatched_load_of_an_unknown_extension(tmp_path) -> None:  # noqa: ANN001
    # No parser claims the name, so nothing is even tried.
    missing = tmp_path / "missing.unknown"
    with pytest.raises(ParserExistsError) as excinfo:
        io.load(str(missing))
    _assert_missing(excinfo, missing)


def test_dispatched_load_of_an_existing_file_given_as_str(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "image.nii"
    nb.save(nb.Nifti1Image(np.zeros((2, 3, 4), "float32"), np.eye(4)), target)
    assert isinstance(io.images.load(str(target)), NiftiImage)


def test_unreadable_existing_file_is_not_reported_as_missing(tmp_path) -> None:  # noqa: ANN001
    target = tmp_path / "garbage.unknown"
    target.write_bytes(b"not a known format")
    with pytest.raises(ParserContentError) as excinfo:
        io.load(str(target))
    assert not isinstance(excinfo.value, FileNotFoundError)


# ----------------------------------------------------------------------
#   SNIFFING
# ----------------------------------------------------------------------


@pytest.mark.parametrize("as_str", [True, False])
def test_concrete_sniff_of_a_missing_file(missing, as_str) -> None:  # noqa: ANN001
    source = str(missing) if as_str else missing
    assert NiftiImage.sniff(source) == Confidence.NO
    with pytest.raises(SnifferExistsError) as excinfo:
        NiftiImage.sniff(source, error=True)
    _assert_missing(excinfo, missing)


@pytest.mark.parametrize(
    "entry",
    [FileBasedObject.sniff, FileBasedObject.sniff_file],
    ids=["sniff", "sniff_file"],
)
def test_dispatched_sniff_of_a_missing_file(tmp_path, entry) -> None:  # noqa: ANN001
    missing = tmp_path / "missing.unknown"
    for source in _spellings(missing):
        assert io.sniff(source) is None
        with pytest.raises(SnifferExistsError) as excinfo:
            entry(source, error=True)
        _assert_missing(excinfo, missing)


def test_dispatched_sniff_still_predicts_from_the_name(missing) -> None:  # noqa: ANN001
    # `sniff` is a prediction of what `load` would reach for, and a name
    # is evidence of that even before the file exists.
    assert io.images.sniff(str(missing)) is NiftiImage
    assert io.images.sniff(missing) is NiftiImage


# ----------------------------------------------------------------------
#   IN-MEMORY TEXT THROUGH A DISPATCHER
# ----------------------------------------------------------------------


@pytest.fixture
def greeting_root() -> tx.Iterator[type]:
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
    try:
        yield Root
    finally:
        FileBasedObject._REGISTRY.discard(fmt)
        TextFileBasedObject._REGISTRY.discard(fmt)


def test_dispatched_text_is_content_not_a_path(greeting_root) -> None:  # noqa: ANN001
    assert greeting_root.from_text("HELLO world") == "HELLO world"
    assert greeting_root.from_content("HELLO world") == "HELLO world"
    assert greeting_root.from_fileobj(_io.StringIO("HELLO x")) == "HELLO x"


def test_unreadable_dispatched_text_is_not_reported_as_missing(
    greeting_root,  # noqa: ANN001
) -> None:
    with pytest.raises(ParserContentError) as excinfo:
        greeting_root.from_text("GOODBYE world")
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_a_source_str_is_a_path_unless_wrapped_as_content(
    missing,  # noqa: ANN001
) -> None:
    assert str(Source(str(missing)).missing) == str(missing)
    assert str(Source(missing).missing) == str(missing)
    assert Source(str(missing.parent)).missing is None
    assert Source.content(str(missing)).missing is None


def test_text_content_is_not_matched_by_extension() -> None:
    # A `str` of content used to be read as a file name too, so text that
    # happened to end in ".nii" was described as a file and matched by
    # extension.
    assert Source("image.nii").name == "image.nii"
    assert Source.content("image.nii").name is None
    assert repr(Source.content("HELLO world")) == "input content"


# ----------------------------------------------------------------------
#   CLI
# ----------------------------------------------------------------------


def test_cli_reports_a_missing_image_as_not_found(missing) -> None:  # noqa: ANN001
    with pytest.raises(CliError, match="Input image not found"):
        load_image(str(missing))


def test_cli_reports_a_missing_transform_as_not_found(missing) -> None:  # noqa: ANN001
    with pytest.raises(CliError, match="Transformation not found"):
        load_transform(str(missing))
