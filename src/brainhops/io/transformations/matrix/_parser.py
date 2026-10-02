# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE, Magic

# core
from brainhops._core import path
from brainhops._core.peek import peekable_lines
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayLike

# io
from brainhops.io.base import arrays
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    SnifferContentError,
)

from . import _conventions

# Plain matrices are tiny. Sniffing reads at most this many bytes, and a
# larger file is not claimed -- so a multi-gigabyte image is never read
# whole just to be turned down.
_SNIFF_LIMIT = 1 << 20

# Extensions that name a plain text matrix and nothing more specific. A
# `.mat` is not one of them: FLIRT and ITK own that extension.
_TEXT_EXTENSIONS = (".txt", ".csv", ".tsv", ".dat")

# Keyword arguments that select or interpret the matrix.
_CONVENTIONS = (
    "input",
    "output",
    "source",
    "target",
    "index_base",
    "direction",
    "vector",
    "ndim",
)


class MatrixParser(Magic, BinaryFileParser, repr=HIDE_IF_NONE):
    """
    Reader for a bare affine matrix in a text, NumPy or MATLAB file.

    The container is recognized from the content: a NumPy `.npy` or
    `.npz` file or a MATLAB v5-v7 or v7.3 `.mat` file from its magic
    number, then delimited text, then a MATLAB v4 `.mat` file (which has
    no magic number).

    What the matrix means is not stored in the file, so it is given as
    keyword arguments when reading; see
    [`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine].
    """

    raw_matrix: tx.Optional[ArrayLike] = None
    """The matrix exactly as stored in the file, before any convention
    (transposition, inversion, index shift, image placement) is applied."""

    container: tx.Optional[str] = None
    """The container the matrix was read from: `"text"`, `"npy"`,
    `"npz"` or `"mat"`."""

    variable: tx.Optional[str] = None
    """The `.npz` key or `.mat` variable the matrix was read from."""

    vector: tx.Optional[str] = None
    """The vector convention the raw matrix was read with."""

    direction: tx.Optional[str] = None
    """The direction the raw matrix was read with."""

    index_base: tx.Optional[tx.Tuple[int, int]] = None
    """The `(input, output)` voxel index base the raw matrix was read
    with."""

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        filename = os.fspath(filename)
        if not os.path.isfile(filename):
            return _reject(error, f"Not a file: {filename}")
        return super().sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score an open file, reading at most 1 MiB of it.

        Plain matrices are tiny, so a larger file is turned down without
        being read whole.
        """
        with preserve_position(file):
            content = file.read(_SNIFF_LIMIT + 1)
        if len(content) > _SNIFF_LIMIT:
            return _reject(error, "Too large to be a plain matrix.")
        score = cls.sniff_content(content, error=error, **kwargs)
        name = getattr(file, "name", None)
        if (
            score
            and isinstance(name, str)
            and name.lower().endswith(_TEXT_EXTENSIONS)
            and _read(content, **_selection(kwargs))[0] == "text"
        ):
            # A text matrix whose extension says "plain matrix" is more
            # than plausible. LIKELY ties with FLIRT's content-only
            # score, and the matching extension then breaks the tie.
            return Confidence.LIKELY
        return score

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff(bytes(content), error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff(_as_lines(lines), error, **kwargs)

    @classmethod
    def _sniff(
        cls,
        content: tx.Union[bytes, tx.List[str]],
        error: tx.Union[bool, tx.Type[Exception]],
        **kwargs,
    ) -> float:
        """
        Score content as a plain matrix: `WEAK` at best.

        Any small numeric table can be read as a matrix, so this reader
        must lose to every format that recognizes the file positively --
        a FLIRT `.mat` is a text `(4, 4)`, an ITK `.mat` a MATLAB v4
        file, an ITK `.txt` a text file.
        """
        vector = kwargs.get("vector", "column")
        ndim = kwargs.get("ndim", None)
        try:
            _, _, raw = _read(content, **_selection(kwargs))
        except (arrays.ArrayContainerError, ParserContentError) as e:
            return _reject(error, str(e))
        except Exception as e:  # anything a container library raises
            return _reject(error, f"Not a plain matrix: {e}")
        if not _conventions.is_affine_matrix(raw, vector, ndim):
            return _reject(
                error, f"A {raw.shape} array is not an affine matrix."
            )
        return Confidence.WEAK

    # --- from ---------------------------------------------------------

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        return cls._build(bytes(content), **kwargs)

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        return cls._build(_as_lines(lines), **kwargs)

    @classmethod
    def _build(
        cls, content: tx.Union[bytes, tx.List[str]], **kwargs
    ) -> tx.Self:
        selection = _selection(kwargs, pop=True)
        conventions = {
            key: kwargs.pop(key) for key in _CONVENTIONS if key in kwargs
        }
        try:
            container, variable, raw = _read(content, **selection)
        except arrays.ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        vector = _conventions.check_vector(conventions.get("vector", "column"))
        direction = _conventions.check_direction(
            conventions.get("direction", "forward")
        )
        try:
            matrix, input, output, index_base = _conventions.to_affine(
                raw, **conventions
            )
        except ValueError as e:
            raise ParserContentError(str(e)) from e
        return cls(
            matrix=matrix,
            input=input,
            output=output,
            raw_matrix=raw,
            container=container,
            variable=variable,
            vector=vector,
            direction=direction,
            index_base=index_base,
            **kwargs,
        )


# ----------------------------------------------------------------------
#   UTILITIES
# ----------------------------------------------------------------------


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    """Return `Confidence.NO`, or raise if the caller asked for it."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO


def _as_lines(lines: tx.Iterable[str]) -> tx.List[str]:
    if isinstance(lines, peekable_lines):
        lines = list(lines)
    return [line for line in lines if isinstance(line, str)]


def _selection(kwargs: dict, pop: bool = False) -> tx.Dict[str, tx.Any]:
    """The keyword arguments that select the matrix in its container."""
    get = kwargs.pop if pop else kwargs.get
    variable = get("variable", None)
    key = get("key", None)
    return {
        "variable": variable if variable is not None else key,
        "encoding": get("encoding", "utf-8"),
    }


def _read(
    content: tx.Union[bytes, str, tx.List[str]],
    variable: tx.Optional[str] = None,
    encoding: str = "utf-8",
) -> tx.Tuple[str, tx.Optional[str], np.ndarray]:
    """
    Read the raw matrix from any supported container.

    Returns
    -------
    container : str
    variable : str | None
    matrix : ndarray
    """
    if isinstance(content, str):
        content = content.splitlines()
    if not isinstance(content, (bytes, bytearray)):
        return "text", None, arrays.read_text_array(content)

    kind = arrays.detect_container(content)
    if kind == "npy":
        raw = arrays.read_npy(content)
        if raw.ndim != 2 or not arrays.is_numeric_array(raw):
            raise arrays.ArrayContainerError(
                f"Expected a numeric 2-D array, got {raw.dtype} {raw.shape}."
            )
        return "npy", None, raw
    if kind == "npz":
        name, raw = arrays.select_array(
            arrays.read_npz(content), variable, _is_matrix, "npz array"
        )
        return "npz", name, _check_numeric(raw, name)
    if kind in ("mat5", "mat73", "hdf5"):
        name, raw = arrays.select_array(
            arrays.read_mat(content), variable, _is_matrix, "mat variable"
        )
        return "mat", name, _check_numeric(raw, name)

    # No magic number: text, else a MATLAB v4 file.
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        text = None
    if text is not None and "\x00" not in text:
        return "text", None, arrays.read_text_array(text.splitlines())
    name, raw = arrays.select_array(
        arrays.read_mat(content), variable, _is_matrix, "mat variable"
    )
    return "mat", name, _check_numeric(raw, name)


def _is_matrix(value: tx.Any) -> bool:
    return arrays.is_numeric_array(value) and value.ndim == 2


def _check_numeric(raw: tx.Any, name: str) -> np.ndarray:
    if not _is_matrix(raw):
        raise arrays.ArrayContainerError(
            f"{name!r} is not a numeric 2-D array."
        )
    return raw
