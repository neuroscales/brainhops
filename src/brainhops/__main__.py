"""Entry point for ``python -m brainhops``."""

from __future__ import annotations

import sys

from brainhops.cli import main

if __name__ == "__main__":
    sys.exit(main())
