"""The ``brainhops`` command-line interface.

The interface exposes the library as subcommands of a single
``brainhops`` program. `main` is the entry point registered as the
console script and is also what ``python -m brainhops`` calls.
"""

from __future__ import annotations

__all__ = ["main"]

from ._main import main
