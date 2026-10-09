"""
Readers of generic array containers.

The containers are delimited text, NumPy `.npy` and `.npz` files, and
MATLAB `.mat` files. The readers produce NumPy arrays and know nothing
of what the arrays mean, so that FLIRT matrices, affine matrices and
other array-based formats share the same parsers.

The functions ([`read_text_array`][], [`read_npy`][], [`read_mat`][],
...) each parse one container. The parsers, one per container, derive
from [`ArrayParser`][] and hand the array to `from_array`, which a
format class implements. A format mixes in one parser per container,
for example:

```python
class NpyMatrixAffine(NpyArrayParser, MatrixAffine): ...
```

All readers take the whole content as bytes or lines, never a path, so
they work on files, streams and memory alike. Nothing is unpickled, and
MATLAB cells and structs are skipped.
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

import functools
import io
import os
import re

import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.peek import peekable_lines
from brainhops._core.streams import preserve_position
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    FileReader,
    ParserContentError,
    ParserNotImplementedError,
    SnifferContentError,
    TextFileReader,
    _not_text,
)


class ArrayContainerError(ValueError):
    """The content cannot be read as numeric arrays of this container."""


# ----------------------------------------------------------------------
#   CONTAINER DETECTION
# ----------------------------------------------------------------------

_NPY_MAGIC = b"\x93NUMPY"
_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06")
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_MAT5_MAGIC = b"MATLAB 5.0 MAT-file"
_MAT73_MAGIC = b"MATLAB 7.3 MAT-file"
_MAT73_KINDS = ("mat73", "hdf5")
"""The kinds that [`detect_container`][] reports for MATLAB v7.3."""


def detect_container(content: bytes) -> tx.Optional[str]:
    """
    Identify a binary array container from its first kilobyte.

    The kind is `"npy"`, `"npz"`, `"mat5"` (MATLAB v5 to v7), `"mat73"`
    (MATLAB v7.3), or `"hdf5"` (HDF5 without a MATLAB header). Unrecognised
    content, such as text or a MATLAB v4 file, which has no magic number,
    gives `None`.
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

_SEPARATORS = re.compile(r"[\s,;]+")


def read_text_rows(
    lines: tx.Iterable[str],
    comments: tx.Optional[tx.Union[str, tx.Sequence[str]]] = "#",
    separators: tx.Optional[str] = None,
) -> tx.List[tx.List[float]]:
    """
    Read rows of numbers from delimited text.

    Comments, which start at any of the `comments` markers, are removed and
    blank lines skipped. The `separators` regular expression defaults to
    runs of whitespace, commas and semicolons. Rows may differ in length,
    unlike in [`read_text_array`][].

    Raises
    ------
    ArrayContainerError
        If a value is not a number, or if a line holds a NUL character,
        which betrays binary content.
    """
    if isinstance(comments, str):
        comments = (comments,)
    comments = tuple(comments or ())
    split = re.compile(separators).split if separators else _SEPARATORS.split
    rows = []
    for line in lines:
        if "\x00" in line:
            raise ArrayContainerError("Not text (NUL byte).")
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
    Read a two-dimensional float array from delimited text.

    The syntax is that of [`read_text_rows`][].

    Raises
    ------
    ArrayContainerError
        If the text holds no numbers, holds a value that is not a number, or
        has rows of different lengths.
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
    """Read the array of `.npy` content, without unpickling."""
    if detect_container(content) != "npy":
        raise ArrayContainerError("Not a .npy file.")
    try:
        array = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as e:
        raise ArrayContainerError(f"Not a readable .npy file: {e}") from e
    if not isinstance(array, np.ndarray):
        raise ArrayContainerError("Not a .npy file.")
    return array


def read_npz(content: bytes) -> tx.Dict[str, np.ndarray]:
    """Read all arrays of `.npz` content, without unpickling."""
    if detect_container(content) != "npz":
        raise ArrayContainerError("Not a .npz file.")
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
    Read the top-level numeric and logical arrays of `.mat` content.

    MATLAB v4 to v7 files are read with `scipy.io.loadmat`, and v7.3 files
    with h5py (see [`read_mat73`][]). Structs, cells, strings and objects
    are skipped.

    Raises
    ------
    ArrayContainerError
        If the content is not a readable MATLAB file.
    ImportError
        If the content is a v7.3 file and h5py is not installed.
    """
    if detect_container(content) in _MAT73_KINDS:
        return read_mat73(content)
    return _read_mat_legacy(content)


def _read_mat_legacy(content: bytes) -> tx.Dict[str, np.ndarray]:
    """
    Read the numeric arrays of MATLAB v4 to v7 content.

    A v5 to v7 file is recognised by its header. A v4 file has none, so any
    binary content that `scipy.io` reads is accepted, but text is rejected.
    """
    kind = detect_container(content)
    if kind not in ("mat5", None):
        raise ArrayContainerError(f"Not a MATLAB v4-v7 file (found {kind}).")
    if kind is None:
        try:
            is_text = "\x00" not in bytes(content).decode("utf-8")
        except UnicodeDecodeError:
            is_text = False
        if is_text:
            raise ArrayContainerError("Text, not a MATLAB v4 file.")
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
    Read the numeric arrays of MATLAB v7.3 content with h5py.

    Only top-level numeric and logical datasets are returned. HDF5 stores
    the column-major MATLAB arrays with reversed axes, so each array is
    transposed back to its MATLAB shape.

    Raises
    ------
    ArrayContainerError
        If the content is not a readable HDF5 file.
    ImportError
        If h5py is not installed.
    """
    kind = detect_container(content)
    if kind not in _MAT73_KINDS:
        raise ArrayContainerError(
            f"Not a MATLAB v7.3 file (found {kind or 'no HDF5 header'})."
        )
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
    """Tell whether a value is a real numeric or boolean NumPy array."""
    return isinstance(value, np.ndarray) and value.dtype.kind in "biuf"


def select_array(
    arrays: tx.Mapping[str, np.ndarray],
    name: tx.Optional[str] = None,
    predicate: tx.Optional[tx.Callable[[np.ndarray], bool]] = None,
    what: str = "array",
) -> tx.Tuple[str, np.ndarray]:
    """
    Pick one array from a named collection, such as `.npz` keys or `.mat`
    variables, and return its name and value.

    The array is the one called `name` if given. Otherwise the collection
    must hold exactly one array that satisfies `predicate`, which by
    default accepts any numeric array. The noun `what` is used in error
    messages.

    Raises
    ------
    ArrayContainerError
        If no array has the given name, or if no single array can be chosen.
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

_Reader = tx.Callable[
    [tx.Any], tx.Union[np.ndarray, tx.Mapping[str, np.ndarray]]
]
"""A container reader, which returns one array or a mapping of named arrays."""


class ArrayParser(FileReader):
    """
    The abstract base of the readers of one generic array container.

    An array parser is never registered as a format. A subclass reads its
    container in `from_lines` and `sniff_lines` (text) or in `from_bytes`
    and `sniff_bytes` (binary), and declares `EXTENSIONS`, `HINTS` and
    [`CONTAINER`][]. The meaning of the array is left to the format class
    that mixes the parser in, which implements [`from_array`][] and may
    override `_accepts_array` and `_array_confidence`.

    In a container that holds several arrays (`.npz`, `.mat`), the `key`
    option, or its alias `variable`, names the array to read. By default,
    the container must hold exactly one array that the format accepts.
    Single-array containers ignore the option.
    """

    CONTAINER: tx.ClassVar[str] = ""
    """The short name of the container, such as `"npy"`."""

    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    """
    The score when the container and its array are both readable.

    The default is `WEAK`, so that any format which positively recognises
    the file wins. A format raises the score in `_array_confidence`
    when the array identifies it.
    """

    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = None
    """
    The size in bytes of the largest file that is sniffed, or `None` for no
    limit; larger files are turned down unread.
    """

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """
        Build an object from the array of the container.

        The format class implements this method.
        """
        raise ParserNotImplementedError(
            f"from_array() is not available in parser of type {cls.__name__}"
        )

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        """
        Tell whether the format can read an array (any numeric one by default).

        In a container of several arrays, this test also picks the one to read.
        """
        return is_numeric_array(array)

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """
        Return the confidence that an accepted array belongs to the format.

        The default is [`SNIFF_CONFIDENCE`][].
        """
        return cls.SNIFF_CONFIDENCE

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
    def _read_array(
        cls, read: _Reader, content: tx.Any, kwargs: tx.Dict[str, tx.Any]
    ) -> tx.Tuple[tx.Optional[str], np.ndarray]:
        """
        Read a container with `read` and pick the array of the format.

        The selection options (`key`, its alias `variable`, and the text
        `encoding`) are popped from `kwargs`. The name returned with the array
        is `None` for a single-array container.

        Raises
        ------
        ParserContentError
            If the content is not this container, or if it holds no single
            acceptable array.
        """
        variable = kwargs.pop("variable", None)
        key = kwargs.pop("key", None)
        kwargs.pop("encoding", None)
        key = variable if variable is not None else key
        try:
            arrays = read(content)
            if isinstance(arrays, np.ndarray):
                name, array = None, arrays
            else:
                name, array = select_array(
                    arrays, key, cls._accepts_array, f"{cls.CONTAINER} array"
                )
        except ArrayContainerError as e:
            raise ParserContentError(str(e)) from e
        if not cls._accepts_array(array):
            what = "The array" if name is None else repr(name)
            raise ParserContentError(
                f"{what} cannot be read by {cls.__name__} "
                f"({array.dtype} {array.shape})."
            )
        return name, array

    @classmethod
    def _sniff_array(
        cls,
        read: _Reader,
        content: tx.Any,
        error: tx.Union[bool, tx.Type[Exception]],
        **kwargs,
    ) -> float:
        """
        Score content by the confidence in the array that `_read_array`
        picks.
        """
        try:
            _, array = cls._read_array(read, content, kwargs)
        except ParserContentError as e:
            return _reject(error, str(e))
        except Exception as e:  # any library error means another container
            return _reject(error, f"Not a {cls.CONTAINER} file: {e}")
        score = cls._array_confidence(array, **kwargs)
        if not score:
            return _reject(
                error,
                f"A {array.dtype} {array.shape} array is not read by "
                f"{cls.__name__}.",
            )
        return score


class TextArrayParser(ArrayParser, TextFileReader):
    """
    The base of the readers of two-dimensional text arrays.

    The text holds one row per line, with `#` comments and blank lines
    skipped, and all rows must have the same length. Content that cannot be
    decoded, or that holds NUL characters, is not text. This base class
    declares no extension or hint, so it is not a format itself.

    A readable file named with an extension of a reader scores at least
    [`NAMED_CONFIDENCE`][]. Without such a name, the content must also bear
    the signature of the reader, so that only one reader claims it: a comma
    for CSV, a tab for TSV, and anything that TSV does not read for
    whitespace-separated text.
    """

    CONTAINER: tx.ClassVar[str] = "text"

    SEPARATORS: tx.ClassVar[tx.Optional[str]] = None
    """
    A regular expression matching the separators, or `None` for runs of
    whitespace, commas and semicolons.
    """

    NAMED_CONFIDENCE: tx.ClassVar[float] = Confidence.LIKELY
    """
    The minimum score of a text array whose file has an extension of this
    reader.
    """

    @classmethod
    def _has_signature(cls, lines: tx.List[str]) -> bool:
        """Tell whether text is this reader's, whatever its file name."""
        return True

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
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        named = kwargs.pop("_named", False)
        lines = _as_lines(lines)
        read = functools.partial(read_text_array, separators=cls.SEPARATORS)
        score = cls._sniff_array(read, lines, error, **kwargs)
        if not score:
            return score
        if named:
            return max(score, cls.NAMED_CONFIDENCE)
        if not cls._has_signature(_data_lines(lines)):
            return _reject(
                error,
                f"Text without the signature of {cls.__name__}, in a file "
                f"not named {' / '.join(cls.EXTENSIONS)}.",
            )
        return score

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        read = functools.partial(read_text_array, separators=cls.SEPARATORS)
        name, array = cls._read_array(read, _as_lines(lines), kwargs)
        return cls.from_array(array, key=name, **kwargs)

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
    The reader of whitespace-separated text arrays (`.txt`).

    This is the layout of `numpy.loadtxt` and `numpy.savetxt`, and also of
    AFNI `.1D` files and of the `.dat` files many tools write. Without a
    matching file name, content that [`TsvArrayParser`][] reads is left to
    that reader.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt", ".dat", ".1D")
    HINTS = ("txt",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r"\s+"

    @classmethod
    def _has_signature(cls, lines: tx.List[str]) -> bool:
        tsv = TsvArrayParser
        return not (
            tsv._has_signature(lines) and _reads(lines, tsv.SEPARATORS)
        )


class CsvArrayParser(TextArrayParser):
    """
    The reader of comma-separated text arrays (`.csv`).

    Spaces around the commas are allowed.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".csv",)
    HINTS = ("csv",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r" *, *"

    @classmethod
    def _has_signature(cls, lines: tx.List[str]) -> bool:
        return any("," in line for line in lines)


class TsvArrayParser(TextArrayParser):
    """
    The reader of tab-separated text arrays (`.tsv`).

    Each separator is one tab, with optional spaces around it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tsv",)
    HINTS = ("tsv",)
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r" *\t *"

    @classmethod
    def _has_signature(cls, lines: tx.List[str]) -> bool:
        return any("\t" in line for line in lines)


class _BinaryArrayParser(ArrayParser, BinaryFileReader):
    """An array container that is never text."""

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return _reject(error, f"A {cls.CONTAINER} file is binary, not text.")

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        raise ParserContentError(
            f"A {cls.CONTAINER} file is binary, not text."
        )


class NpyArrayParser(_BinaryArrayParser):
    """The reader of the single array of a `.npy` file."""

    CONTAINER: tx.ClassVar[str] = "npy"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".npy",)
    HINTS = ("npy",)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff_array(read_npy, bytes(content), error, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        name, array = cls._read_array(read_npy, bytes(content), kwargs)
        return cls.from_array(array, key=name, **kwargs)


class NpzArrayParser(_BinaryArrayParser):
    """The reader of one array of a `.npz` file, selected by `key`."""

    CONTAINER: tx.ClassVar[str] = "npz"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".npz",)
    HINTS = ("npz",)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff_array(read_npz, bytes(content), error, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        name, array = cls._read_array(read_npz, bytes(content), kwargs)
        return cls.from_array(array, key=name, **kwargs)


class MatArrayParser(_BinaryArrayParser):
    """
    The reader of one variable of a MATLAB `.mat` file of any version.

    Two unrelated containers hide behind the extension:
    [`MatLegacyArrayParser`][] reads versions 4 to 7 and
    [`Mat73ArrayParser`][] reads version 7.3, an HDF5 file. This class
    dispatches between its [`VARIANTS`][]: `sniff_bytes` returns their best
    score, and `from_bytes` reads with the variant whose container matches.

    A format built on this class, such as `MatMatrixAffine`, declares its
    own `VARIANTS` so that the dispatch returns its own classes. The
    variants are registered, not the dispatcher, which would only tie with
    them and lose.
    """

    CONTAINER: tx.ClassVar[str] = "mat"
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The readers dispatched to: MATLAB v4 to v7, then v7.3."""

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        score = max(v.sniff_bytes(content, **kwargs) for v in cls.VARIANTS)
        return score or _reject(error, f"Not read by {cls.__name__}.")

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        legacy, v73 = cls.VARIANTS
        is_v73 = detect_container(content) in _MAT73_KINDS
        return (v73 if is_v73 else legacy).from_bytes(content, **kwargs)


class MatLegacyArrayParser(MatArrayParser):
    """
    The reader of one variable of a MATLAB v4 to v7 `.mat` file.

    The file is read with `scipy.io.loadmat`, and only numeric variables
    are kept. The reader adds no hint to `"mat"`.
    """

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff_array(
            _read_mat_legacy, bytes(content), error, **kwargs
        )

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        name, array = cls._read_array(_read_mat_legacy, bytes(content), kwargs)
        return cls.from_array(array, key=name, **kwargs)


class Mat73ArrayParser(MatArrayParser):
    """
    The reader of one variable of a MATLAB v7.3 `.mat` file, with h5py.

    The hints `"mat73"` and `"73"` let `"mat.73"` select this reader.
    """

    CONTAINER: tx.ClassVar[str] = "mat73"
    HINTS = ("mat73", "73")

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        return cls._sniff_array(read_mat73, bytes(content), error, **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        name, array = cls._read_array(read_mat73, bytes(content), kwargs)
        return cls.from_array(array, key=name, **kwargs)


MatArrayParser.VARIANTS = (MatLegacyArrayParser, Mat73ArrayParser)


# ----------------------------------------------------------------------
#   PARSER UTILITIES
# ----------------------------------------------------------------------


def _reject(error: tx.Union[bool, tx.Type[Exception]], message: str) -> float:
    """Return `Confidence.NO`, or raise if the caller asks for it."""
    if error:
        if error is True:
            error = SnifferContentError
        raise error(message)
    return Confidence.NO


def _as_lines(lines: tx.Iterable[str]) -> tx.List[str]:
    if isinstance(lines, peekable_lines):
        lines = list(lines)
    return [line for line in lines if isinstance(line, str)]


def _data_lines(lines: tx.List[str]) -> tx.List[str]:
    """Keep the lines that hold values, without comments."""
    lines = (line.split("#", 1)[0].strip() for line in lines)
    return [line for line in lines if line]


def _reads(lines: tx.List[str], separators: tx.Optional[str]) -> bool:
    """Tell whether lines read as a text array with these separators."""
    try:
        read_text_array(lines, separators=separators)
    except ArrayContainerError:
        return False
    return True
