# stdlib
import math
import re
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path as LocalPath
import numpy as np
import typing_extensions as _tx
from brainhops._ext.struct import Struct
from brainhops.io.transformations.itk._utils import TransformBlock

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

_FileLike = _tx.Union[_tx.IO, PathLike, str]
_FileOrContentLike = _tx.Union[_FileLike, bytes, _tx.Iterable[str]]
_HEADER = "#Insight Transform File V1.0"
_TRANSFORM_RE = re.compile(r"^#Transform\s+(\d+)$")
_KEYVAL_RE = re.compile(r"^(\w+)\s*:\s*(.*)$")

# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------


class TxtTransformParser(Struct):
    VERSION = _HEADER
    _HAS_KEYS = True

    # ---------------------------------------------------------
    # Sniffing
    # ---------------------------------------------------------

    @classmethod
    def sniff(cls, other: _FileOrContentLike) -> bool:
        try:
            obj = cls.from_(other)
            return len(obj.transform_blocks) > 0
        except Exception:
            return False

    @classmethod
    def sniff_text(cls, text: str) -> bool:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            return line == _HEADER
        return False

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    @classmethod
    def from_(cls, other: _FileOrContentLike) -> _tx.Self:
        """
        Build an object from a file (path, file-like object or iterable
        of lines).

        Parameters
        ----------
        other : str | PathLike | IO | Iterable[str]
            Input file, or its content.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(other, str):
            p = LocalPath(other)
            if p.exists():
                return cls.from_file(p)
            return cls.from_text(other)
        if isinstance(other, PathLike):
            return cls.from_file(other)
        if hasattr(other, "read"):
            return cls.from_file(other)
        return cls.from_lines(other)

    @classmethod
    def from_file(cls, fileobj: _FileLike) -> _tx.Self:
        """
        Build an object from a file (path or file-like object).

        Parameters
        ----------
        fileobj : str | PathLike | IO
            Input file.

        Returns
        -------
        obj
            The parsed object.
        """
        if isinstance(fileobj, str):
            return cls.from_file(LocalPath(fileobj))
        if isinstance(fileobj, PathLike):
            return cls.from_text(LocalPath(fileobj).read_text())
        return cls.from_lines(fileobj)

    @classmethod
    def from_text(cls, text: str) -> _tx.Self:
        """
        Build an object from a string in tfm format.

        Parameters
        ----------
        text : str
            The content of an TFM file.

        Returns
        -------
        obj
            The parsed object.
        """
        return cls.from_lines(text.splitlines())

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
        if not isinstance(lines, _iterlines):
            lines = _iterlines(lines)

        obj = cls()
        first = next(lines, None)

        if first != _HEADER:
            raise ValueError(f"Expected '{_HEADER}'")

        current = None
        active_key = None  # <-- track multiline fields

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ---- new transform ----
            if _TRANSFORM_RE.match(line):
                if current:
                    obj.transform_blocks.append(current)
                current = TransformBlock()
                active_key = None
                continue

            if current is None:
                continue

            key, value = _read_key(line)

            # ---- new key encountered ----
            if key is not None:
                active_key = key

                if key == "TransformType":
                    current.TransformType = value

                elif key == "TransformParameters":
                    current.TransformParameters = _read_vector(value)

                elif key == "TransformFixedParameters":
                    current.TransformFixedParameters = _read_vector(value)

                continue

            # ---- continuation line (multiline data) ----
            if active_key == "TransformParameters":
                current.TransformParameters.extend(_read_vector(line))

            elif active_key == "TransformFixedParameters":
                current.TransformFixedParameters.extend(_read_vector(line))

        if current:
            obj.transform_blocks.append(current)

        return obj

    # ---------------------------------------------------------
    # Object
    # ---------------------------------------------------------

    def __init__(self):
        self.transform_blocks = []

    # ---------------------------------------------------------
    # Writing
    # ---------------------------------------------------------

    def _fmt_float(self, x: float) -> str:
        return f"{x:.15g}"

    def _block_lines(self, transform) -> _tx.Iterable[str]:
        yield f"Transform: {transform.TransformType}"
        yield "Parameters: " + " ".join(map(self._fmt_float, transform.TransformParameters))
        yield "FixedParameters: " + " ".join(map(self._fmt_float, transform.TransformFixedParameters))

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
        yield _HEADER
        yield ""
        for i, transform in enumerate(self.transform_blocks):
            yield f"#Transform {i}"
            yield from self._block_lines(transform)
            yield ""

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

# ---------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------


class _iterlines:
    EMPTY = object()

    def __init__(self, lines: _tx.Iterable[str]):
        self.lines = iter(lines)
        self.peeked = self.EMPTY

    def _clean(self, line) -> str:
        if line is None:
            return None
        line = line.rstrip("\r\n")
        if not line.strip():
            return None
        return line.strip()

    def _next_valid(self) -> _tx.Optional[str]:
        while True:
            line = next(self.lines, None)
            if line is None:
                return None
            line = self._clean(line)
            if line is not None:
                return line

    def peek(self) -> _tx.Optional[str]:
        if self.peeked is self.EMPTY:
            self.peeked = self._next_valid()
        return self.peeked

    def __next__(self) -> str:
        if self.peeked is not self.EMPTY:
            out = self.peeked
            self.peeked = self.EMPTY
        else:
            out = self._next_valid()
        if out is None:
            raise StopIteration
        return out

    def __iter__(self):
        while True:
            try:
                yield next(self)
            except StopIteration:
                return

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _read_key(line: str) -> _tx.Tuple[str]:
    m = _KEYVAL_RE.match(line)
    if not m:
        return None, None
    return (m.group(1), m.group(2).strip(),)


def _read_vector(text: _tx.Optional[str]) -> _tx.List[float]:
    if not text:
        return []
    return [float(v) for v in text.split()]
