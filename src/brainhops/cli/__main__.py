"""Entry point for ``python -m brainhops.cli``."""

from __future__ import annotations

import sys

from ._main import main

if __name__ == "__main__":
    sys.exit(main())
