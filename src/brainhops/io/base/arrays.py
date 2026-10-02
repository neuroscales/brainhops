"""
Format-independent readers for generic array containers.

Several formats store a bare numeric array in a generic container: a
delimited text file, a NumPy `.npy` / `.npz` file, or a MATLAB `.mat`
file. The helpers here parse those containers into NumPy arrays and know
nothing about what the arrays mean, so that every format built on one of
them (FLIRT matrices, plain affine matrices, and later the generic array
readers) shares one parser.

Every reader takes the whole file content as `bytes` (or text lines),
never a path, so that it works the same on files, streams and in-memory
content. Nothing here unpickles: NumPy containers are read with
`allow_pickle=False`, and MATLAB cell arrays or structs are skipped.
"""

__all__ = [
    "ArrayContainerError",
    "detect_container",
    "is_numeric_array",
    "read_mat",
    "read_npy",
    "read_npz",
    "read_text_array",
    "read_text_rows",
    "select_array",
]

# stdlib
import io
import re

# dependencies
import numpy as np
import typing_extensions as tx


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
        return _read_mat73(content)
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


def _read_mat73(content: bytes) -> tx.Dict[str, np.ndarray]:
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
