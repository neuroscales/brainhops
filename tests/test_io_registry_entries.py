"""
The declared formats agree with their classes.

Dispatch decides which formats to import from what the packages of each
kind declare (`brainhops.io.images._entries`, ...), so a declaration that
drifts from its class makes `load` choose differently before and after
the class is imported, and a format registered without a declaration is
invisible to `load` until something imports it.
"""

import importlib
import pkgutil

import pytest

import brainhops.io.images
import brainhops.io.transformations
from brainhops.io.base import _registry
from brainhops.io.base._base import FileBasedObject, WritableFileBasedObject
from brainhops.io.base.specs import format_hints
from brainhops.io.images.base import FileBasedImage, WritableFileBasedImage
from brainhops.io.transformations.base import (
    FileBasedTransformation,
    WritableFileBasedTransformation,
)


def _dispatchers(cls: type) -> set:
    return {
        _registry.key_of(base)
        for base in cls.__mro__
        if "_REGISTRY" in base.__dict__
    }


def _expected(cls: type) -> str:
    """The declaration `cls` calls for, to copy into `_entries`."""
    return (
        f"{cls.__module__}:{cls.__qualname__}\n"
        f"  hints: {' '.join(sorted(format_hints(cls)))!r}\n"
        f"  extensions={cls.EXTENSIONS!r}, prefixes={cls.PREFIXES!r}, "
        f"priority={cls.PRIORITY!r}\n"
        f"  dispatchers: {sorted(_dispatchers(cls))}"
    )


def _import_all() -> None:
    """Import every module of every kind, so that all formats register."""
    for package in (brainhops.io.images, brainhops.io.transformations):
        prefix = package.__name__ + "."
        for info in pkgutil.walk_packages(package.__path__, prefix):
            try:
                importlib.import_module(info.name)
            except ImportError:  # an optional dependency is missing
                pass


_ENTRIES = [entry for entry in _registry.declared() if entry.missing is None]


@pytest.mark.parametrize("entry", _ENTRIES, ids=lambda entry: entry.key)
def test_declaration_matches_class(entry: _registry.FormatEntry) -> None:
    cls = entry.resolve()
    if cls is None and entry.check is not None and not entry.available:
        pytest.skip(f"{entry.key} is not available here")
    assert cls is not None, entry.error
    expected = _expected(cls)
    assert cls.__module__ == entry.module, expected
    assert entry._hints == format_hints(cls), expected
    assert entry._extensions == cls.EXTENSIONS, expected
    assert entry._prefixes == cls.PREFIXES, expected
    assert entry._priority == cls.PRIORITY, expected
    assert set(entry.dispatchers) == _dispatchers(cls), expected
    assert len(entry.dispatchers) == len(_dispatchers(cls)), expected
    assert "_REGISTRY" not in cls.__dict__


def test_every_registered_format_is_declared() -> None:
    _import_all()
    undeclared = [
        _expected(cls)
        for cls in FileBasedObject._REGISTRY
        if cls.__module__.startswith("brainhops.")
        and _registry.key_of(cls) not in _registry._ENTRIES
    ]
    assert not undeclared, "\n".join(undeclared)


@pytest.mark.parametrize(
    "key, dispatcher",
    [
        (_registry.OBJECT, FileBasedObject),
        (_registry.WRITABLE, WritableFileBasedObject),
        (_registry.IMAGE, FileBasedImage),
        (_registry.WRITABLE_IMAGE, WritableFileBasedImage),
        (_registry.TRANSFORMATION, FileBasedTransformation),
        (_registry.WRITABLE_TRANSFORMATION, WritableFileBasedTransformation),
    ],
)
def test_declared_dispatchers_are_the_classes(
    key: str, dispatcher: type
) -> None:
    # The entries are declared into the registry of the dispatcher's key:
    # a dispatcher that moved would silently lose its formats.
    assert _registry.key_of(dispatcher) == key
    assert _registry.registry_for(key) is dispatcher._REGISTRY
