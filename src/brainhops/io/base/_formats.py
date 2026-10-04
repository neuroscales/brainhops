"""
The formats of `brainhops`, imported when dispatch first needs them.

A format registers itself as its module is imported (`@register_format`),
but the packages of each kind (`brainhops.io.images`, ...) import their
format subpackages lazily, so that importing them is cheap. Dispatch
imports them all the first time it chooses a format: every format sniffs
the file, so every one is needed then.

Kept free of any import beyond the standard library and
`brainhops._core.dependencies`, so that the packages of each kind can
record their missing formats as they are imported.
"""

__all__ = ["import_formats", "register_missing_format"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import has_abczarr_driver

# The packages of each kind, whose `__all__` names their format
# subpackages, those whose dependencies are installed.
_KINDS = ("brainhops.io.images", "brainhops.io.transformations")
_IMPORTED = False

# Hints of formats that are not registered because an optional dependency
# is missing, with what to install: hint -> (package, extra).
MISSING_FORMATS: tx.Dict[str, tx.Tuple[str, str]] = {}


def import_formats() -> None:
    """Import every format subpackage of every kind, so that their
    formats register; only the first call does anything."""
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    for name in _KINDS:
        package = importlib.import_module(name)
        for attribute in package.__all__:
            # The Zarr formats need a driver as well as abczarr, which is
            # known only now, since telling may import abczarr.
            if attribute != "zarr" or has_abczarr_driver():
                getattr(package, attribute)


def register_missing_format(
    hints: tx.Iterable[str], package: str, extra: str
) -> None:
    """
    Record that the format answering to `hints` is not registered because
    `package` is not installed, so that asking for it by hint says what
    to install (`pip install brainhops[extra]`).
    """
    for hint in hints:
        MISSING_FORMATS[str(hint).lower()] = (package, extra)
