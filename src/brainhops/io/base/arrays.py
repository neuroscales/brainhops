"""
Format-independent readers for generic array containers.

Several formats store a bare numeric array in a generic container: a
delimited text file, a NumPy `.npy` / `.npz` file, or a MATLAB `.mat`
file. This module reads those containers into NumPy arrays and knows
nothing about what the arrays mean, so that every format built on one of
them (FLIRT matrices, plain affine matrices, and later the generic array
readers) shares one parser.

It has two layers:

- **Functions** (`read_text_array`, `read_npy`, `read_npz`, `read_mat`,
  ...) that parse the content of one container.
- **Parsers**, one per container, deriving from the abstract
  [`ArrayParser`][], which turns the array they read into an object
  through hooks that a format class implements (`accepts_array`,
  `array_confidence`, `from_array`). A format mixes one of them in per
  container, as in `class NpyMatrixAffine(NpyArrayParser, MatrixAffine)`.

  Text (built on `TextFileParser`):

  - [`TextArrayParser`][]: the abstract text base, any separator;
  - [`TxtArrayParser`][]: whitespace; `.txt`, `.dat`, `.1D`; hint
    `"txt"`;
  - [`CsvArrayParser`][]: commas; `.csv`; hint `"csv"`;
  - [`TsvArrayParser`][]: tabs; `.tsv`; hint `"tsv"`.

  Binary (built on `BinaryFileParser`):

  - [`NpyArrayParser`][]: `.npy`; hint `"npy"`;
  - [`NpzArrayParser`][]: `.npz`; hint `"npz"`;
  - [`MatArrayParser`][]: any MATLAB `.mat`; hint `"mat"`; dispatches
    to [`MatLegacyArrayParser`][] (v4, v5-v7, no extra hint) or
    [`Mat73ArrayParser`][] (v7.3; hints `"mat73"`, `"73"`, hence
    `"mat.73"`).

The text readers build on the shared text plumbing (decoding, and
declining content that does not decode) and differ by their separator:
whitespace, commas or tabs. AFNI `.1D` and generic `.dat` files are
whitespace-separated columns with `#` comments, so they are extensions
of [`TxtArrayParser`][] rather than formats of their own.
[`MatArrayParser`][] dispatches between the two unrelated MATLAB
containers: its score is the best of its variants', and it reads a file
with the variant whose container it is.

Every reader takes the whole file content as `bytes` (or text lines),
never a path, so that it works the same on files, streams and in-memory
content. Nothing here unpickles: NumPy containers are read with
`allow_pickle=False`, and MATLAB cell arrays or structs are skipped.
"""

__all__ = [
    "ArrayContainerError",
    "ArrayParser",
    "CsvArrayParser",
    "Mat73ArrayParser",
    "MatArrayParser",
    "MatLegacyArrayParser",
    "NpyArrayParser",
    "NpzArrayParser",
    "TextArrayParser",
    "TsvArrayParser",
    "TxtArrayParser",
    "detect_container",
    "is_numeric_array",
    "read_mat",
    "read_mat73",
    "read_npy",
    "read_npz",
    "read_text_array",
    "read_text_rows",
    "select_array",
]

# stdlib
import io
import os
import re

# dependencies
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import path
from brainhops._core.peek import peekable_lines
from brainhops._core.streams import preserve_position

# io
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    FileParser,
    ParserContentError,
    ParserNotImplementedError,
    SnifferContentError,
    TextFileParser,
    _not_text,
)


class ArrayContainerError(ValueError):
    """Raised when a container cannot be read as (numeric) arrays."""


# ----------------------------------------------------------------------
#   CONTAINER DETECTION
# ----------------------------------------------------------------------

_NPY_MAGIC = b"\x93NUMPY"
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_MAT5_MAGIC = b"MATLAB 5.0 MAT-file"
_MAT73_MAGIC = b"MATLAB 7.3 MAT-file"


def detect_container(content: bytes) -> tx.Optional[str]:
    """
    Identify a binary array container from its magic number.

    Parameters
    ----------
    content : bytes
        The file content, or at least its first 1 KiB.

    Returns
    -------
    kind : {"npy", "npz", "mat5", "mat73", "hdf5"} | None
        `"mat5"` is a MATLAB v5-v7 file, `"mat73"` a MATLAB v7.3 (HDF5)
        file and `"hdf5"` an HDF5 file without a MATLAB header. `None`
        means no magic number was recognized: the content may be text,
        a MATLAB v4 file (which has no magic number), or anything else.
    """
    head = bytes(content[:1024])
    if head.startswith(_NPY_MAGIC):
        return "npy"
    if head.startswith(_ZIP_MAGICS):
        return "npz"
    if head.startswith(_MAT73_MAGIC):
        return "mat73"
    if head.startswith(_MAT5_MAGIC):
        return "mat5"
    if head.startswith(_HDF5_MAGIC) or head[512:520] == _HDF5_MAGIC:
        return "hdf5"
    return None


# ----------------------------------------------------------------------
#   TEXT
# ----------------------------------------------------------------------

# Commas, semicolons, tabs and spaces all separate values.
_SEPARATORS = re.compile(r"[\s,;]+")


def read_text_rows(
    lines: tx.Iterable[str],
    comments: tx.Optional[tx.Union[str, tx.Sequence[str]]] = "#",
    separators: tx.Optional[str] = None,
) -> tx.List[tx.List[float]]:
    """
    Read rows of floats from delimited text.

    Blank lines (after removing comments) are skipped. Rows may have
    different lengths; use [`read_text_array`][] to require a rectangle.

    Parameters
    ----------
    lines : Iterable[str]
        The lines of text.
    comments : str | Sequence[str] | None
        Markers that start a comment running to the end of the line.
        `None` or an empty sequence disables comments.
    separators : str | None
        A regular expression matching value separators. The default
        accepts whitespace (spaces, tabs), commas and semicolons.

    Returns
    -------
    rows : list[list[float]]

    Raises
    ------
    ArrayContainerError
        If a value is not a number.
    """
    if isinstance(comments, str):
        comments = (comments,)
    comments = tuple(comments or ())
    split = re.compile(separators).split if separators else _SEPARATORS.split
    rows = []
    for line in lines:
        for marker in comments:
            line = line.split(marker, 1)[0]
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([float(v) for v in split(line) if v])
        except ValueError as e:
            raise ArrayContainerError(f"Not a numeric row: {line!r}") from e
    return rows


def read_text_array(
    lines: tx.Iterable[str],
    comments: tx.Optional[tx.Union[str, tx.Sequence[str]]] = "#",
    separators: tx.Optional[str] = None,
) -> np.ndarray:
    """
    Read a 2D array of floats from delimited text.

    See [`read_text_rows`][] for the accepted syntax.

    Raises
    ------
    ArrayContainerError
        If the text is empty, holds a non-numeric value, or its rows do
        not all have the same length.
    """
    rows = read_text_rows(lines, comments=comments, separators=separators)
    if not rows:
        raise ArrayContainerError("No numeric values in text.")
    widths = sorted({len(row) for row in rows})
    if len(widths) != 1:
        raise ArrayContainerError(
            f"Rows of a text array must have equal lengths, got {widths}."
        )
    return np.asarray(rows, dtype=np.float64)


# ----------------------------------------------------------------------
#   NUMPY
# ----------------------------------------------------------------------


def read_npy(content: bytes) -> np.ndarray:
    """Read the array stored in `.npy` content, without unpickling."""
    try:
        array = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as e:
        raise ArrayContainerError(f"Not a readable .npy file: {e}") from e
    if not isinstance(array, np.ndarray):
        raise ArrayContainerError("Not a .npy file.")
    return array


def read_npz(content: bytes) -> tx.Dict[str, np.ndarray]:
    """Read every array stored in `.npz` content, without unpickling."""
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as npz:
            return {key: npz[key] for key in npz.files}
    except Exception as e:
        raise ArrayContainerError(f"Not a readable .npz file: {e}") from e


# ----------------------------------------------------------------------
#   MATLAB
# ----------------------------------------------------------------------


def read_mat(content: bytes) -> tx.Dict[str, np.ndarray]:
    """
    Read the numeric arrays stored in MATLAB `.mat` content.

    MATLAB v4 and v5-v7 files are read with `scipy.io.loadmat`. MATLAB
    v7.3 files are HDF5 files and are read with `h5py`, imported only
    when needed. HDF5 stores MATLAB's column-major arrays with their axes
    reversed, so v7.3 arrays are transposed back to MATLAB's shape.

    Only top-level numeric (or logical) arrays are returned; structs,
    cells, strings and objects are skipped.

    Raises
    ------
    ArrayContainerError
        If the content is not a readable MATLAB file.
    ImportError
        If the file is v7.3 and `h5py` is not installed.
    """
    kind = detect_container(content)
    if kind in ("mat73", "hdf5"):
        return read_mat73(content)
    if kind not in ("mat5", None):
        raise ArrayContainerError(f"Not a MATLAB file (found {kind}).")
    import scipy.io

    try:
        variables = scipy.io.loadmat(
            io.BytesIO(content),
            squeeze_me=False,
            struct_as_record=True,
            mat_dtype=False,
        )
    except Exception as e:
        raise ArrayContainerError(f"Not a readable .mat file: {e}") from e
    return {
        name: value
        for name, value in variables.items()
        if not name.startswith("__") and is_numeric_array(value)
    }


_MAT73_NUMERIC = {
    b"double",
    b"single",
    b"int8",
    b"uint8",
    b"int16",
    b"uint16",
    b"int32",
    b"uint32",
    b"int64",
    b"uint64",
    b"logical",
}


def read_mat73(content: bytes) -> tx.Dict[str, np.ndarray]:
    """
    Read the numeric arrays stored in MATLAB v7.3 (HDF5) `.mat` content.

    The file is read with `h5py`, imported only now. HDF5 stores
    MATLAB's column-major arrays with their axes reversed, so arrays are
    transposed back to MATLAB's shape. Only top-level numeric (or
    logical) datasets are returned.

    Raises
    ------
    ArrayContainerError
        If the content is not a readable HDF5 file.
    ImportError
        If `h5py` is not installed.
    """
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - h5py is a test extra
        raise ImportError(
            "Reading a MATLAB v7.3 .mat file needs h5py. Install it with "
            "`pip install h5py` (or `brainhops[matrix]`)."
        ) from e
    arrays = {}
    try:
        with h5py.File(io.BytesIO(content), "r") as f:
            for name, item in f.items():
                if name.startswith("#") or not isinstance(item, h5py.Dataset):
                    continue
                cls = item.attrs.get("MATLAB_class", b"double")
                if isinstance(cls, str):
                    cls = cls.encode()
                if cls not in _MAT73_NUMERIC:
                    continue
                if item.attrs.get("MATLAB_empty", 0):
                    continue
                value = np.asarray(item[()])
                if not is_numeric_array(value):
                    continue
                # HDF5 holds MATLAB's column-major array with its axes
                # reversed; transposing restores MATLAB's (rows, cols).
                arrays[name] = value.T
    except ImportError:  # pragma: no cover
        raise
    except Exception as e:
        raise ArrayContainerError(f"Not a readable v7.3 .mat file: {e}") from e
    return arrays


# ----------------------------------------------------------------------
#   SELECTION
# ----------------------------------------------------------------------


def is_numeric_array(value: tx.Any) -> bool:
    """Whether `value` is a real numeric (or boolean) NumPy array."""
    return isinstance(value, np.ndarray) and value.dtype.kind in "biuf"


def select_array(
    arrays: tx.Mapping[str, np.ndarray],
    name: tx.Optional[str] = None,
    predicate: tx.Optional[tx.Callable[[np.ndarray], bool]] = None,
    what: str = "array",
) -> tx.Tuple[str, np.ndarray]:
    """
    Pick one array from a named collection.

    Parameters
    ----------
    arrays : Mapping[str, ndarray]
        Named arrays, e.g. the variables of a `.mat` or the keys of a
        `.npz`.
    name : str, optional
        The name to select. By default, the collection must hold exactly
        one array that satisfies `predicate`.
    predicate : callable, optional
        Which arrays may be selected by default (ignored when `name` is
        given). By default, any numeric array.
    what : str
        How to call the arrays in error messages.

    Returns
    -------
    name : str
    array : ndarray

    Raises
    ------
    ArrayContainerError
        If `name` is missing, or no single array can be chosen.
    """
    if name is not None:
        if name not in arrays:
            raise ArrayContainerError(
                f"No {what} named {name!r}; available: {sorted(arrays)}."
            )
        return name, arrays[name]
    predicate = predicate or is_numeric_array
    matches = [key for key, value in arrays.items() if predicate(value)]
    if len(matches) != 1:
        raise ArrayContainerError(
            f"Expected exactly one candidate {what}, found {len(matches)} "
            f"({sorted(matches)}) among {sorted(arrays)}. Select one by "
            f"name."
        )
    return matches[0], arrays[matches[0]]


# ----------------------------------------------------------------------
#   PARSERS
# ----------------------------------------------------------------------

_Content = tx.Union[bytes, tx.List[str]]
"""What a container reads: the file's bytes, or the lines of a text."""


class ArrayParser(FileParser):
    """
    Abstract base of the readers of one generic array container.

    A concrete subclass reads one container (`read_arrays`) and declares
    its `EXTENSIONS`, `HINTS` and `CONTAINER`. What the array *means* is
    left to a format class, which mixes the container parser in and
    implements three hooks:

    - `accepts_array`: which arrays it
      can read, used to pick one in a container that holds several;
    - `array_confidence`: how sure it
      is that the array is its own;
    - `from_array`: how to build the object.

    Abstract: it reads no container, and is never registered as a
    format.

    **Selecting an array**: in a container that holds several (`.npz`,
    `.mat`), `key=` (or its alias `variable=`) names the array to read.
    By default, the container must hold exactly one array that
    `accepts_array` accepts. Single-array containers ignore it.
    """

    CONTAINER: tx.ClassVar[str] = ""
    """A short name for the container, e.g. `"npy"`."""

    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    """The score of content whose container and array are both readable.

    A generic container says nothing about what its array means, so the
    default is `WEAK`: any format that recognizes the file positively
    wins. A format raises it in `array_confidence` when the array itself
    identifies it."""

    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = None
    """The largest file, in bytes, that sniffing reads. A larger file is
    turned down without being read whole. `None` means no limit."""

    # --- container ----------------------------------------------------

    @classmethod
    def read_arrays(
        cls, content: _Content, encoding: str = "utf-8"
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        """
        Read the arrays stored in the container.

        Parameters
        ----------
        content : bytes | list[str]
            The file content, or the lines of a text file.
        encoding : str
            How text content is encoded.

        Returns
        -------
        arrays : dict[str | None, ndarray]
            The named arrays of the container, or `{None: array}` for a
            container that stores a single, unnamed array.

        Raises
        ------
        ArrayContainerError
            If the content is not this container.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not read a container; use one of the "
            f"concrete array parsers."
        )

    @classmethod
    def read_array(
        cls,
        content: _Content,
        key: tx.Optional[str] = None,
        encoding: str = "utf-8",
    ) -> tx.Tuple[tx.Optional[str], np.ndarray]:
        """
        Read the one array of the container that this format reads.

        Parameters
        ----------
        content : bytes | list[str]
            The file content, or the lines of a text file.
        key : str, optional
            The name of the array, in a container that holds several.
        encoding : str
            How text content is encoded.

        Returns
        -------
        key : str | None
            The name of the array, or `None` in a single-array container.
        array : ndarray

        Raises
        ------
        ArrayContainerError
            If the content is not this container, or holds no single
            array that `accepts_array` accepts.
        """
        arrays = cls.read_arrays(content, encoding=encoding)
        if list(arrays) == [None]:
            name, array = None, arrays[None]
        else:
            name, array = select_array(
                arrays, key, cls.accepts_array, f"{cls.CONTAINER} array"
            )
        if not cls.accepts_array(array):
            what = "The array" if name is None else repr(name)
            raise ArrayContainerError(
                f"{what} cannot be read by {cls.__name__} "
                f"({array.dtype} {array.shape})."
            )
        return name, array

    # --- hooks ----------------------------------------------------------

    @classmethod
    def accepts_array(cls, array: np.ndarray) -> bool:
        """Whether this format can read `array`. Any numeric array by
        default."""
        return is_numeric_array(array)

    @classmethod
    def array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """How confident this format is that an accepted `array` is its
        own, given the reading options. `SNIFF_CONFIDENCE` by default."""
        return cls.SNIFF_CONFIDENCE

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """Build the object from the array read from the container.
        Implemented by the format class."""
        raise ParserNotImplementedError(
            f"from_array() is not available in parser of type {cls.__name__}"
        )

    # --- sniff ----------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        if isinstance(filename, (str, os.PathLike)) and os.path.isdir(
            os.fspath(filename)
        ):
            return _reject(error, f"Not a file: {filename}")
        return super().sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open file, reading at most `SNIFF_LIMIT` bytes."""
        limit = cls.SNIFF_LIMIT
        with preserve_position(file):
            content = file.read(-1 if limit is None else limit + 1)
        if limit is not None and len(content) > limit:
            return _reject(error, f"Larger than {limit} bytes.")
        return cls.sniff_content(content, error=error, **kwargs)

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
        content: _Content,
        error: tx.Union[bool, tx.Type[Exception]],
        **kwargs,
    ) -> float:
        key, encoding = _pop_selection(dict(kwargs))
        try:
            _, array = cls.read_array(content, key=key, encoding=encoding)
        except ArrayContainerError as e:
            return _reject(error, str(e))
        except Exception as e:  # anything a container library raises
            return _reject(error, f"Not a {cls.CONTAINER} file: {e}")
        score = cls.array_confidence(array, **kwargs)
        if not score:
            return _reject(
                error,
                f"A {array.dtype} {array.shape} array is not read by "
                f"{cls.__name__}.",
            )
        return score

    # --- from -----------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        return cls._parse(_as_lines(lines), **kwargs)

    @classmethod
    def _parse(cls, content: _Content, **kwargs) -> tx.Self:
        key, encoding = _pop_selection(kwargs)
        try:
            name, array = cls.read_array(content, key=key, encoding=encoding)
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        return cls.from_array(array, key=name, **kwargs)


class TextArrayParser(ArrayParser, TextFileParser):
    """
    Base of the readers of a 2-D array stored as delimited text.

    One row per line; `#` starts a comment; blank lines are skipped. All
    rows must have the same length. Decoding is the shared text
    plumbing of [`TextFileParser`][brainhops.io.base.parsers.TextFileParser]:
    content that does not decode (or holds a NUL byte) is not text, and
    scores `NO`.

    This base splits values on any run of whitespace, commas or
    semicolons, and declares no extension or hint: it is not a format.
    The concrete readers fix the separator and the file names:

    - [`TxtArrayParser`][]: whitespace (`.txt`, `.dat`, `.1D`);
    - [`CsvArrayParser`][]: commas (`.csv`);
    - [`TsvArrayParser`][]: tabs (`.tsv`).

    **Sniffing.** A file whose name ends in one of a reader's
    `EXTENSIONS` and whose content it reads scores at least
    `NAMED_CONFIDENCE`: the name says "plain numbers", which no content
    check can. Without such a name, a reader also requires its
    *signature* (`has_signature`), so that one content is claimed by one
    reader only: CSV needs a comma, TSV a tab, and whitespace text
    yields tab-separated content to TSV.
    """

    CONTAINER: tx.ClassVar[str] = "text"

    SEPARATORS: tx.ClassVar[tx.Optional[str]] = None
    """The regular expression that separates values (see
    [`read_text_rows`][]). `None` accepts whitespace, commas and
    semicolons."""

    NAMED_CONFIDENCE: tx.ClassVar[float] = Confidence.LIKELY
    """The least score of a text array whose file name has one of the
    reader's extensions."""

    @classmethod
    def read_arrays(
        cls, content: _Content, encoding: str = "utf-8"
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        if isinstance(content, (bytes, bytearray)):
            try:
                content = bytes(content).decode(encoding).splitlines()
            except UnicodeDecodeError as e:
                raise ArrayContainerError("Not text.") from e
        lines = list(content)
        if any("\x00" in line for line in lines):
            raise ArrayContainerError("Not text (NUL byte).")
        return {None: read_text_array(lines, separators=cls.SEPARATORS)}

    @classmethod
    def has_signature(cls, lines: tx.List[str]) -> bool:
        """Whether text that this reader reads is recognizably its own,
        whatever the file is called. Always, by default."""
        return True

    # --- sniff ----------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        name = getattr(file, "name", None)
        if isinstance(name, str) and name.lower().endswith(
            tuple(e.lower() for e in cls.EXTENSIONS)
        ):
            kwargs["_named"] = True
        try:
            return super().sniff_fileobj(file, error=error, **kwargs)
        except UnicodeDecodeError as e:  # a binary file opened as text
            return _not_text(cls, error, e)

    @classmethod
    def _sniff(
        cls,
        content: _Content,
        error: tx.Union[bool, tx.Type[Exception]],
        **kwargs,
    ) -> float:
        named = kwargs.pop("_named", False)
        score = super()._sniff(content, error, **kwargs)
        if not score:
            return score
        if named:
            return max(score, cls.NAMED_CONFIDENCE)
        if not cls.has_signature(_data_lines(content)):
            return _reject(
                error,
                f"Text without the signature of {cls.__name__}, in a file "
                f"not named {' / '.join(cls.EXTENSIONS)}.",
            )
        return score

    # --- from -----------------------------------------------------------

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        try:
            return super().from_fileobj(file, **kwargs)
        except UnicodeDecodeError as e:
            raise ParserContentError(f"Not text: {e}") from e

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        try:
            return super().from_bytes(content, **kwargs)
        except UnicodeDecodeError as e:
            raise ParserContentError(f"Not text: {e}") from e


class TxtArrayParser(TextArrayParser):
    """
    Reader for a 2-D array stored as whitespace-separated text (`.txt`).

    Values are separated by spaces or tabs, as `numpy.loadtxt` and
    `numpy.savetxt` do by default. This is also the layout of AFNI
    `.1D` files (whitespace-separated columns, `#` comments) and of the
    conventionless `.dat` files many tools write, so both are read
    here rather than by readers of their own.

    Its signature is anything but tab-separated values: unnamed
    content that [`TsvArrayParser`][] reads is left to it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt", ".dat", ".1D")
    HINTS = ("txt",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r"\s+"

    @classmethod
    def has_signature(cls, lines: tx.List[str]) -> bool:
        tsv = TsvArrayParser
        return not (tsv.has_signature(lines) and _reads(lines, tsv.SEPARATORS))


class CsvArrayParser(TextArrayParser):
    """
    Reader for a 2-D array stored as comma-separated values (`.csv`).

    Values are separated by commas, with optional spaces around them.
    Its signature is a comma.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".csv",)
    HINTS = ("csv",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r" *, *"

    @classmethod
    def has_signature(cls, lines: tx.List[str]) -> bool:
        return any("," in line for line in lines)


class TsvArrayParser(TextArrayParser):
    """
    Reader for a 2-D array stored as tab-separated values (`.tsv`).

    Values are separated by one tab, with optional spaces around it.
    Its signature is a tab.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tsv",)
    HINTS = ("tsv",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r" *\t *"

    @classmethod
    def has_signature(cls, lines: tx.List[str]) -> bool:
        return any("\t" in line for line in lines)


class _BinaryArrayParser(ArrayParser, BinaryFileParser):
    """An array container that is never text."""

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff(bytes(content), error, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        return cls._parse(bytes(content), **kwargs)

    @classmethod
    def read_arrays(
        cls, content: _Content, encoding: str = "utf-8"
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        if not isinstance(content, (bytes, bytearray)):
            raise ArrayContainerError(
                f"A {cls.CONTAINER} file is binary, not text."
            )
        return cls._read_binary(bytes(content))

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        raise NotImplementedError


class NpyArrayParser(_BinaryArrayParser):
    """Reader for the single array of a NumPy `.npy` file, without
    unpickling."""

    CONTAINER: tx.ClassVar[str] = "npy"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".npy",)
    HINTS = ("npy",)

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        if detect_container(content) != "npy":
            raise ArrayContainerError("Not a .npy file.")
        return {None: read_npy(content)}


class NpzArrayParser(_BinaryArrayParser):
    """Reader for one array of a NumPy `.npz` archive, selected by
    `key=`, without unpickling."""

    CONTAINER: tx.ClassVar[str] = "npz"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".npz",)
    HINTS = ("npz",)

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        if detect_container(content) != "npz":
            raise ArrayContainerError("Not a .npz file.")
        return dict(read_npz(content))


class MatArrayParser(_BinaryArrayParser):
    """
    Reader for one variable of a MATLAB `.mat` file of any version,
    selected by `key=` (or `variable=`).

    MATLAB has two unrelated containers behind one extension, each read
    by a subclass:

    - [`MatLegacyArrayParser`][]: v4 and v5-v7, read with
      `scipy.io.loadmat`;
    - [`Mat73ArrayParser`][]: v7.3, an HDF5 file read with `h5py`.

    This class dispatches between them: its score is the best of theirs,
    and it reads a file with the one whose container the content is. It
    declares the hint `"mat"`, which the subclasses inherit; the v7.3
    reader adds `"mat73"` and `"73"` (so `"mat.73"`).

    **Variants.** Only the class that lists `VARIANTS` in its own body
    dispatches; its subclasses inherit the attribute but not the role,
    and read their own container. A format built on this one (as in
    `MatMatrixAffine`) declares its own `VARIANTS`, so that dispatch
    returns objects of its own variant classes. Register the variants
    for dispatch, not the dispatcher: it would only ever tie with (and
    lose to) its own subclasses.
    """

    CONTAINER: tx.ClassVar[str] = "mat"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The version-specific readers this class dispatches to, when it
    declares them itself."""

    @classmethod
    def variants(cls) -> tx.Tuple[type, ...]:
        """The readers this class dispatches to; empty for a reader of
        one version."""
        return cls.__dict__.get("VARIANTS", ())

    @classmethod
    def is_container(cls, content: _Content) -> bool:
        """Whether `content` is the container this class reads (for the
        dispatcher, the container of any of its variants)."""
        return any(v.is_container(content) for v in cls.variants())

    @classmethod
    def variant(cls, content: _Content) -> type:
        """
        The class that reads `content`: this class itself for a reader of
        one version, else the variant whose container the content is.

        Raises
        ------
        ArrayContainerError
            If no variant reads this container.
        """
        variants = cls.variants()
        if not variants:
            return cls
        for variant in variants:
            if variant.is_container(content):
                return variant
        kind = (
            detect_container(content)
            if isinstance(content, (bytes, bytearray))
            else "text"
        )
        raise ArrayContainerError(f"Not a MATLAB file (found {kind}).")

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        return cls.variant(content)._read_binary(content)

    @classmethod
    def _sniff(
        cls,
        content: _Content,
        error: tx.Union[bool, tx.Type[Exception]],
        **kwargs,
    ) -> float:
        variants = cls.variants()
        if not variants:
            return super()._sniff(content, error, **kwargs)
        score = max(v._sniff(content, False, **kwargs) for v in variants)
        if not score:
            return _reject(error, f"Not read by {cls.__name__}.")
        return score

    @classmethod
    def _parse(cls, content: _Content, **kwargs) -> tx.Self:
        if not cls.variants():
            return super()._parse(content, **kwargs)
        try:
            variant = cls.variant(content)
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        return variant._parse(content, **kwargs)


class MatLegacyArrayParser(MatArrayParser):
    """
    Reader for one variable of a MATLAB v4 or v5-v7 `.mat` file,
    selected by `key=` (or `variable=`), read with `scipy.io.loadmat`.

    A v5-v7 file is recognized by its header; a v4 file has none, so it
    is any binary content that `scipy.io` reads. Only numeric variables
    are considered. It adds no hint to `"mat"`.
    """

    @classmethod
    def is_container(cls, content: _Content) -> bool:
        if not isinstance(content, (bytes, bytearray)):
            return False
        kind = detect_container(content)
        return kind == "mat5" or (kind is None and not _is_text(content))

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        kind = detect_container(content)
        if kind is None and _is_text(content):
            raise ArrayContainerError("Text, not a MATLAB v4 file.")
        if kind not in ("mat5", None):
            raise ArrayContainerError(
                f"Not a MATLAB v4-v7 file (found {kind})."
            )
        return dict(read_mat(content))


class Mat73ArrayParser(MatArrayParser):
    """
    Reader for one variable of a MATLAB v7.3 `.mat` file, selected by
    `key=` (or `variable=`).

    A v7.3 file is an HDF5 file with a MATLAB header. It is read with
    `h5py`, imported only then. HDF5 stores MATLAB's column-major arrays
    transposed, which is undone. Only numeric variables are considered.
    It adds the hints `"mat73"` and `"73"` to `"mat"`, so `"mat.73"`
    selects it.
    """

    CONTAINER: tx.ClassVar[str] = "mat73"
    HINTS = ("mat73", "73")

    @classmethod
    def is_container(cls, content: _Content) -> bool:
        return isinstance(content, (bytes, bytearray)) and detect_container(
            content
        ) in ("mat73", "hdf5")

    @classmethod
    def _read_binary(
        cls, content: bytes
    ) -> tx.Dict[tx.Optional[str], np.ndarray]:
        kind = detect_container(content)
        if kind not in ("mat73", "hdf5"):
            raise ArrayContainerError(
                f"Not a MATLAB v7.3 file (found {kind or 'no HDF5 header'})."
            )
        return dict(read_mat73(content))


MatArrayParser.VARIANTS = (MatLegacyArrayParser, Mat73ArrayParser)


# ----------------------------------------------------------------------
#   PARSER UTILITIES
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


def _pop_selection(
    kwargs: tx.Dict[str, tx.Any],
) -> tx.Tuple[tx.Optional[str], str]:
    """Pop the options that select the array: `key` (alias `variable`)
    and the text `encoding`."""
    variable = kwargs.pop("variable", None)
    key = kwargs.pop("key", None)
    encoding = kwargs.pop("encoding", "utf-8")
    return (variable if variable is not None else key), encoding


def _is_text(content: bytes, encoding: str = "utf-8") -> bool:
    try:
        return "\x00" not in content.decode(encoding)
    except UnicodeDecodeError:
        return False


def _data_lines(content: _Content) -> tx.List[str]:
    """The lines of a text that hold values: comments and blank lines
    removed."""
    if isinstance(content, (bytes, bytearray)):
        content = bytes(content).decode("utf-8", "replace").splitlines()
    lines = (line.split("#", 1)[0].strip() for line in content)
    return [line for line in lines if line]


def _reads(lines: tx.List[str], separators: tx.Optional[str]) -> bool:
    """Whether `lines` read as a text array with these separators."""
    try:
        read_text_array(lines, separators=separators)
    except ArrayContainerError:
        return False
    return True
