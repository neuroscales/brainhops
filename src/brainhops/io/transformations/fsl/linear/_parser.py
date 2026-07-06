# stdlib
from os import PathLike
from pathlib import Path as LocalPath

import numpy as np
import typing_extensions as _tx

from brainhops._core.peek import peekable_lines
from brainhops.io.base.parsers import TextFileParser

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

_FileLike = _tx.Union[_tx.IO, PathLike, str]
_FileOrContentLike = _tx.Union[_FileLike, _tx.Iterable[str]]

# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------


class FslLinearTransformParser(TextFileParser):
    # ---------------------------------------------------------
    # Sniffing
    # ---------------------------------------------------------

    @classmethod
    def sniff_line(cls, line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        parts = line.split()
        try:
            [float(p) for p in parts]
            return True
        except ValueError:
            return False

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------
    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])

        if not rows:
            raise ValueError("No data found — empty input.")

        n_cols = len(rows[0])
        if any(len(r) != n_cols for r in rows):
            raise ValueError(
                f"Inconsistent row lengths: {[len(r) for r in rows]}. "
                f"All rows must have the same number of columns."
            )

        obj = cls()
        obj.matrix = np.array(rows, dtype=np.float64)
        return obj

    # ---------------------------------------------------------
    # Object
    # ---------------------------------------------------------

    def __init__(self):
        self.matrix = np.zeros((1, 1))

    # ---------------------------------------------------------
    # Writing
    # ---------------------------------------------------------

    def _fmt_float(self, x: float) -> str:
        return f"{x:.15g}"

    def to_lines(self) -> _tx.Iterator[str]:
        """
        Convert the object to an iterable over lines of an TFM file.

        Additional keyword arguments can be passed to the underlying
        field formatter.

        Returns
        -------
        Iterator[str]
            An iterable over lines of an TFM file representing the object.
        """
        for i in range(self.matrix.shape[0]):
            yield " ".join(self._fmt_float(j) for j in self.matrix[i, :])

    def to_text(self) -> str:
        """
        Convert the object to a string in TFM format.

        Returns
        -------
        str
            The string representation of the object in TFM format.
        """
        return "\n".join(self.to_lines())

    def to_file(self, fileobj: _tx.Union[_tx.IO, PathLike, str]) -> None:
        """
        Write the object to a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Output file.

        Returns
        -------
        None
        """
        text = self.to_text()
        if isinstance(fileobj, str):
            fileobj = LocalPath(fileobj)
        if isinstance(fileobj, PathLike):
            LocalPath(fileobj).write_text(text)
            return
        fileobj.write(text)
