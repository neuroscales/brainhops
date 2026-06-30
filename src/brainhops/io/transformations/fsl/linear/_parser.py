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
        if len(line.split(" ")) != 4:
            return False
        return True

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: _tx.Iterable[str]) -> _tx.Self:
        """
        Build an object from an iterable over lines on an TFM files.

        Parameters
        ----------
        lines : Iterable[str]
            Iterable content of an TFM file.

        Returns
        -------
        obj
            The parsed object.
        """
        if not isinstance(lines, peekable_lines):
            lines = peekable_lines(lines)

        obj = cls()
        for i, line in enumerate(lines):
            line = line.strip()
            for j, num in enumerate(line.split(" ")):
                obj.matrix[i, j] = float(num)

        return obj

    # ---------------------------------------------------------
    # Object
    # ---------------------------------------------------------

    def __init__(self):
        self.matrix = np.zeros((4, 4))

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
        for i in range(4):
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
