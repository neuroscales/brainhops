"""Formats that are registered before their module is imported.

A stub declares the hints and the file names of a format, and is
registered with the import path of the format. Dispatch imports the
module of the format only when the format may read the file at hand (see
`brainhops.io.base._registry`). The toy formats of this module are
written to a temporary directory, so that a test can tell whether a
format module was imported.
"""

import importlib
import subprocess
import sys
import textwrap
import types
import uuid

import pytest
import typing_extensions as tx

from brainhops.io.base import _registry
from brainhops.io.base._base import (
    FileFormat,
    format_registry,
    register_format,
)
from brainhops.io.base._registry import is_stub
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    TextFileReader,
)

# The code of a toy format, which reads the files whose first line starts
# with its marker. `{stubs}` is the module that holds the stub, and
# `{body}` holds further class attributes.
_TOY = """
from {stubs} import Stub
from brainhops.io.base.parsers import (
    Confidence, ParserContentError, TextFileReader,
)


class Toy(TextFileReader, Stub):
    @classmethod
    def sniff_line(cls, line, error=False, **kwargs):
        return Confidence.CERTAIN if line.startswith("toy") else Confidence.NO

    @classmethod
    def from_line(cls, line, **kwargs):
        if not line.startswith("toy"):
            raise ParserContentError("not a toy")
        return {{"format": "toy", "line": line}}
{body}
"""


class _Toy(tx.NamedTuple):
    """A toy dispatcher, the stub of its lazy format, and their modules."""

    root: type
    stub: type
    module: str


def _toy(tmp_path, monkeypatch, body: str = "", **attrs: tx.Any) -> _Toy:  # noqa: ANN001
    """Register a stub, and write the module of its format, unimported."""
    uid = uuid.uuid4().hex[:8]
    stubs = types.ModuleType(f"toy_stubs_{uid}")
    module = f"toy_format_{uid}"

    @format_registry
    class Root(FileFormat):
        pass

    attrs.setdefault("EXTENSIONS", (".toy",))
    stub = type("Stub", (Root,), dict(attrs, __module__=stubs.__name__))
    register_format(f"{module}:Toy")(stub)
    stubs.Stub = stub
    monkeypatch.setitem(sys.modules, stubs.__name__, stubs)

    (tmp_path / f"{module}.py").write_text(
        _TOY.format(stubs=stubs.__name__, body=textwrap.indent(body, "    "))
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return _Toy(Root, stub, module)


@pytest.fixture
def cleanup() -> tx.Iterator[tx.List[type]]:
    """Remove the toy formats from the root registry after a test."""
    roots: tx.List[type] = []
    yield roots
    for root in roots:
        for entry in root._REGISTRY.entries():
            FileFormat._REGISTRY.discard(entry)


@pytest.fixture
def toy(tmp_path, monkeypatch, cleanup) -> _Toy:  # noqa: ANN001
    """A toy dispatcher with one format registered as a stub."""
    toy = _toy(tmp_path, monkeypatch)
    cleanup.append(toy.root)
    return toy


# ----------------------------------------------------------------------
#   REGISTRATION
# ----------------------------------------------------------------------


def test_registering_a_stub_does_not_import_its_format(toy: _Toy) -> None:
    assert toy.module not in sys.modules
    assert toy.stub in toy.root._REGISTRY.entries()
    assert is_stub(toy.stub)


def test_iterating_over_a_registry_imports_its_formats(toy: _Toy) -> None:
    (fmt,) = list(toy.root._REGISTRY)
    assert fmt.__module__ == toy.module
    assert issubclass(fmt, toy.stub)
    # The format takes the place of the stub in every registry.
    assert not is_stub(toy.stub)
    assert toy.root._REGISTRY.entries() == {fmt}
    assert fmt in FileFormat._REGISTRY.entries()


def test_a_registry_holds_the_format_of_a_stub(toy: _Toy) -> None:
    fmt = importlib.import_module(toy.module).Toy
    assert fmt in toy.root._REGISTRY


def test_the_import_path_must_name_a_module_and_a_class(
    toy: _Toy,
) -> None:
    with pytest.raises(ValueError, match="module:qualname"):
        register_format("no_qualname")(type("Other", (toy.root,), {}))


class _NotDerived:
    """A class that a stub may wrongly name as its format."""


def test_a_format_must_derive_from_its_stub(
    toy: _Toy,
    monkeypatch,  # noqa: ANN001
) -> None:
    *_, registries = _registry._PENDING[toy.stub]
    monkeypatch.setitem(
        _registry._PENDING, toy.stub, (__name__, "_NotDerived", registries)
    )
    with pytest.raises(TypeError, match="does not derive from it"):
        list(toy.root._REGISTRY)


def test_a_format_must_not_change_what_its_stub_declares(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
    cleanup,  # noqa: ANN001
) -> None:
    toy = _toy(tmp_path, monkeypatch, body='EXTENSIONS = (".other",)')
    cleanup.append(toy.root)
    with pytest.raises(TypeError, match="EXTENSIONS"):
        list(toy.root._REGISTRY)


# ----------------------------------------------------------------------
#   DISPATCH
# ----------------------------------------------------------------------


def test_a_file_with_the_extension_of_a_stub_imports_its_format(
    toy: _Toy,
    tmp_path,  # noqa: ANN001
) -> None:
    path = tmp_path / "a.toy"
    path.write_text("toy content\n")
    assert toy.root.load(path)["format"] == "toy"
    assert toy.module in sys.modules


def test_a_file_that_an_imported_format_reads_imports_no_stub(
    toy: _Toy,
    tmp_path,  # noqa: ANN001
) -> None:
    @classmethod
    def sniff_line(cls, line, error=False, **kwargs) -> float:  # noqa: ANN001
        return Confidence.CERTAIN if line.startswith("txt") else Confidence.NO

    @classmethod
    def from_line(cls, line, **kwargs) -> tx.Any:  # noqa: ANN001
        if not line.startswith("txt"):
            raise ParserContentError("not a txt")
        return {"format": "txt"}

    register_format(
        type(
            "Txt",
            (TextFileReader, toy.root),
            {
                "EXTENSIONS": (".txt",),
                "sniff_line": sniff_line,
                "from_line": from_line,
            },
        )
    )
    path = tmp_path / "a.txt"
    path.write_text("txt content\n")
    assert toy.root.load(path)["format"] == "txt"
    assert toy.root.sniff(path).__name__ == "Txt"
    assert toy.module not in sys.modules


def test_a_misnamed_file_is_read_by_a_stub_in_a_second_pass(
    toy: _Toy,
    tmp_path,  # noqa: ANN001
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("toy content\n")
    assert toy.root.load(path)["format"] == "toy"
    assert toy.root.sniff(path).__name__ == "Toy"


def test_a_hint_imports_the_formats_that_answer_to_it(
    tmp_path,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
    cleanup,  # noqa: ANN001
) -> None:
    toy = _toy(tmp_path, monkeypatch, HINTS=("toy",))
    other = _toy(tmp_path, monkeypatch, HINTS=("other",))
    cleanup.extend([toy.root, other.root])
    path = tmp_path / "a.txt"
    path.write_text("toy content\n")
    assert FileFormat.load(path, hint="toy")["format"] == "toy"
    assert toy.module in sys.modules
    assert other.module not in sys.modules


# ----------------------------------------------------------------------
#   FSL
# ----------------------------------------------------------------------


_FSL = """
import sys
import numpy as np
import brainhops.io as io

FLIRT = "brainhops.io.transformations.fsl.flirt._xform"
FNIRT = "brainhops.io.transformations.fsl.fnirt._base"
assert FLIRT not in sys.modules and FNIRT not in sys.modules
np.savetxt(sys.argv[1], np.eye(4))
assert type(io.load(sys.argv[1])).__name__ == "FlirtTransform"
assert FLIRT in sys.modules and FNIRT not in sys.modules
"""


def test_reading_a_flirt_matrix_imports_flirt_alone(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "src2ref.mat"
    subprocess.run([sys.executable, "-c", _FSL, str(path)], check=True)
