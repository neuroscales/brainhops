"""
Tests for method resolution where a format reader meets a dispatcher.

A concrete format like `NiftiImage` inherits its reading methods from two
directions at once: a format-specific mixin (`NiftiParser`) and the
registry machinery (`FileBasedObject`, via `FileBasedImage`). If the
machinery ever won *and* stopped delegating, the format-specific reader
would be silently skipped -- a NIfTI would still "load", just through the
generic byte path, losing the lazy nibabel handle.
"""

import inspect

import pytest

from brainhops.io.base._base import (
    FileBasedObject,
    format_registry,
)
from brainhops.io.base.parsers import (
    BinaryFileParser,
    FileParser,
    FileSniffer,
)

nb = pytest.importorskip("nibabel")

from brainhops.io.base.nifti import NiftiParser  # noqa: E402
from brainhops.io.images.nifti import NiftiImage  # noqa: E402
from brainhops.io.transformations.nifti import (  # noqa: E402
    NiftiRASCoordinatesField,
    NiftiVoxelToRAS,
)
from brainhops.io.transformations.spm.y import (  # noqa: E402
    SPMCoordinatesField,
)

NIFTI_FORMATS = [
    NiftiImage,
    NiftiVoxelToRAS,
    NiftiRASCoordinatesField,
    SPMCoordinatesField,
]

# The methods NiftiParser specializes. Anything else may legitimately
# come from the generic ladder, which re-dispatches back into these.
SPECIALIZED = [
    "from_file",
    "from_fileobj",
    "from_bytes",
    "sniff_fileobj",
    "sniff_bytes",
]


def _owner(cls: type, name: str) -> type:
    return next(c for c in cls.__mro__ if name in c.__dict__)


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", SPECIALIZED)
def test_the_format_specific_reader_wins(cls: type, method: str) -> None:
    assert _owner(cls, method) is NiftiParser


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
def test_binary_read_mode_survives_the_diamond(cls: type) -> None:
    """
    `FileSniffer._READ_MODE` is `"r"`. If it won, every NIfTI would be
    opened as text and every sniffer would fail on the first byte.
    """
    assert cls._READ_MODE == "rb"


@pytest.mark.parametrize("cls", NIFTI_FORMATS, ids=lambda c: c.__name__)
def test_concrete_formats_are_not_dispatchers(cls: type) -> None:
    """
    Only a dispatcher consults a registry. A concrete format that owned
    one would try every parser -- including itself -- on every read.
    """
    assert not cls._is_dispatcher()


def test_dispatcher_overrides_are_pass_throughs_for_concrete_formats() -> None:
    """
    Every reading method `FileBasedObject` overrides must hand straight
    back to `super()` when the class is not a dispatcher. That is what
    makes resolution independent of base order: whichever of the two
    directions wins, the other is still reached.
    """
    overridden = [
        name
        for name, value in vars(FileBasedObject).items()
        if (name.startswith(("from_", "sniff")) or name == "load")
        and isinstance(value, classmethod)
    ]
    assert overridden, "no reading methods found to check"
    for name in overridden:
        source = inspect.getsource(getattr(FileBasedObject, name).__func__)
        assert "_is_dispatcher()" in source, name
        assert "super()." + name in source, name


def test_resolution_does_not_depend_on_base_order() -> None:
    """
    The format mixin is reached whether it is listed before or after the
    registry machinery, because the machinery delegates rather than
    terminating the chain.
    """

    @format_registry
    class Root(FileBasedObject):
        pass

    class Special(BinaryFileParser):
        @classmethod
        def from_file(cls, file, **kwargs):  # noqa: ANN001, ANN206
            return "special-reader"

    class SpecialFirst(Special, Root):
        pass

    class RootFirst(Root, Special):
        pass

    assert _owner(SpecialFirst, "from_file") is Special
    assert _owner(RootFirst, "from_file") is FileBasedObject
    # ...and yet both reach the specialized reader
    assert SpecialFirst.from_file("x") == "special-reader"
    assert RootFirst.from_file("x") == "special-reader"


def test_the_generic_ladder_redispatches_through_cls() -> None:
    """
    Each rung of the generic ladder must call `cls.<next>`, never
    `super().<next>`: `super()` would walk past a subclass's override and
    read the file with the wrong implementation.
    """
    rungs = {
        (FileSniffer, "sniff_file"): "sniff_fileobj",
        (FileSniffer, "sniff_fileobj"): "sniff_content",
        (FileSniffer, "sniff_text"): "sniff_lines",
        (FileSniffer, "sniff_lines"): "sniff_line",
        (FileParser, "from_file"): "from_fileobj",
        (FileParser, "from_fileobj"): "from_content",
        (FileParser, "from_text"): "from_lines",
        (FileParser, "from_lines"): "from_line",
    }
    for (owner, name), nxt in rungs.items():
        source = inspect.getsource(getattr(owner, name).__func__)
        assert "cls." + nxt in source, (name, nxt)
        assert "super()." + nxt not in source, (name, nxt)


def test_loading_a_nifti_goes_through_the_nifti_reader(tmp_path) -> None:  # noqa: ANN001
    """
    The behavioural counterpart: `NiftiParser.from_file` hands the path
    to nibabel so it keeps the file handle and can read voxels lazily.
    Falling through to the generic reader would close the file first.
    """
    import numpy as np

    img = nb.Nifti1Image(np.zeros((3, 4, 5), "float32"), np.eye(4))
    target = tmp_path / "scan.nii"
    nb.save(img, str(target))

    loaded = NiftiImage.from_file(target)
    assert loaded.image is not None, "nibabel handle was not kept"
    assert loaded.shape == (3, 4, 5)
