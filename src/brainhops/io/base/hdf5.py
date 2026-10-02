"""
Shared plumbing for formats stored in HDF5 files.

Several transformation formats are HDF5 containers -- ITK's binary
transforms (`.h5`) and BIDS X5 (`.x5`) among them -- and they cannot be
told apart by their container: both are HDF5 files, and both even keep
their transforms under a root group named `TransformGroup`. What
differs is their *content*, so each format only says how to recognise
and decode an open [`h5py.File`][h5py.File], and the mixins here deal
with everything else: paths, open binary streams, bytes held in memory,
and lazily read datasets.

- [`Hdf5Parser`][] turns every input that `brainhops` accepts into an
  open `h5py.File`, and hands it to the format's `sniff_h5` and
  `from_h5`.
- [`Hdf5ParserWriter`][] does the same for writing, through the
  format's `to_h5`.
- [`DelayedH5Array`][] is a dataset that can still be read after its
  file was closed, by reopening it on demand.

`h5py` is an optional dependency, so this module is imported only by
the formats that need it, and only when it is installed.
"""

__all__ = [
    "DelayedH5Array",
    "H5Like",
    "Hdf5Parser",
    "Hdf5ParserWriter",
    "delayed_dataset",
    "read_string",
]

# stdlib
from io import BytesIO
from math import prod
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# core
from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol

# io
from brainhops.io.base.parsers import (
    BinaryFileParser,
    BinaryFileParserWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
)

H5Like = tx.Union[tx.BinaryIO, PathLike, str, h5py.File]
"""Anything an HDF5 format can be read from or written to."""


def _raise_or(
    error: tx.Union[bool, tx.Type[Exception]],
    default: tx.Type[Exception],
    message: str,
    cause: tx.Optional[BaseException] = None,
) -> float:
    """Raise `error` (or `default` if `error is True`), or return NO."""
    if error:
        if error is True:
            error = default
        raise error(message) from cause
    return Confidence.NO


class Hdf5Parser(BinaryFileParser):
    """
    Reads a format stored in an HDF5 file.

    A concrete format implements two class methods, both of which take an
    open `h5py.File`:

    - `sniff_h5(h5file, error=False) -> float`
    - `from_h5(h5file, keep_open=False, load=True, **kwargs) -> Self`

    and this mixin routes paths, streams and bytes to them. A path is
    handed to `h5py` by name rather than as a stream, so that a dataset
    read lazily (`load=False`) can reopen the file long after it was
    parsed.
    """

    # --- to implement -------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Score how confident the format is that an open HDF5 file is
        one of its files."""
        raise NotImplementedError

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> tx.Self:
        """Build an object from an open HDF5 file."""
        raise NotImplementedError

    # --- sniff --------------------------------------------------------

    @classmethod
    def sniff_file(
        cls,
        file: H5Like,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a path, open HDF5 file, or binary stream."""
        if isinstance(file, h5py.File):
            return cls.sniff_h5(file, error=error)
        if isinstance(file, (str, path.PathLike)):
            return cls.sniff_filename(file, error=error, **kwargs)
        return cls.sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_filename(
        cls,
        filename: tx.Union[str, PathLike],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the HDF5 file found at a path."""
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            return _raise_or(
                error, SnifferExistsError, f"No such file: {filename}"
            )
        try:
            is_hdf5 = h5py.is_hdf5(str(filename))
        except Exception:  # noqa: BLE001  (remote paths, ...)
            is_hdf5 = None
        if is_hdf5 is False:
            return _raise_or(
                error, SnifferContentError, f"Not an HDF5 file: {filename}"
            )
        if is_hdf5 is None:
            with filename.open("rb") as f:
                return cls.sniff_fileobj(f, error=error, **kwargs)
        with h5py.File(str(filename), "r") as f:
            return cls.sniff_h5(f, error=error)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score an open, seekable binary stream."""
        with preserve_position(file):
            try:
                with h5py.File(file, "r") as f:
                    return cls.sniff_h5(f, error=error)
            except Exception as e:
                return _raise_or(
                    error,
                    SnifferContentError,
                    f"Content is not a {cls.__name__} HDF5 file",
                    e,
                )

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score the bytes of an HDF5 file."""
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    # --- from ---------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        file: H5Like,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """
        Build an object from a file (path, file-like object, or HDF5 file).

        Parameters
        ----------
        file : str | PathLike | IO | h5py.File
            Input file.
        keep_open : bool, optional
            If True, keep the HDF5 file open after loading.
            If False, close the file after loading.
            If `load=False` and `keep_open=False`, the file is reopened
            every time a lazily read dataset is accessed.
        load : bool, optional
            If True, read large datasets into memory.
            If False, keep them on disk.
        """
        if isinstance(file, h5py.File):
            return cls.from_h5(file, keep_open=keep_open, load=load, **kwargs)
        if isinstance(file, (str, path.PathLike)):
            return cls.from_filename(
                file, keep_open=keep_open, load=load, **kwargs
            )
        return cls.from_fileobj(file, keep_open=keep_open, load=load, **kwargs)

    @classmethod
    def from_filename(
        cls,
        filename: tx.Union[str, PathLike],
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """Build an object from the HDF5 file found at a path."""
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        f = h5py.File(str(filename), "r")
        try:
            return cls.from_h5(f, keep_open=keep_open, load=load, **kwargs)
        finally:
            if not keep_open:
                f.close()

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """Build an object from an open, seekable binary stream."""
        with preserve_position(file):
            f = h5py.File(file, "r")
            try:
                return cls.from_h5(f, keep_open=keep_open, load=load, **kwargs)
            finally:
                if not keep_open:
                    f.close()

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build an object from the bytes of an HDF5 file."""
        return cls.from_fileobj(BytesIO(content), **kwargs)


class Hdf5ParserWriter(Hdf5Parser, BinaryFileParserWriter):
    """
    Reads and writes a format stored in an HDF5 file.

    On top of what [`Hdf5Parser`][] asks for, a concrete format
    implements `to_h5(self, h5file, **kwargs)`, which fills a new, empty
    HDF5 file open for writing.
    """

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write this object into an empty HDF5 file open for writing."""
        raise NotImplementedError

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        """
        The function that fills an HDF5 file with this object.

        It is asked for *before* the file is opened, so a format that
        encodes its content here refuses an object it cannot write
        without creating or truncating the file.
        """
        return lambda h5file: self.to_h5(h5file, **kwargs)

    def to_file(self, file: H5Like, **kwargs) -> None:
        """Write to a path, an open binary stream, or an HDF5 file."""
        if isinstance(file, h5py.File):
            return self._h5_writer(**kwargs)(file)
        return super().to_file(file, **kwargs)

    def to_filename(self, filename: tx.Union[str, PathLike], **kwargs) -> None:
        """Write to the file found at a path, replacing it."""
        writer = self._h5_writer(**kwargs)
        with h5py.File(str(filename), "w") as f:
            writer(f)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write to a binary stream open for writing."""
        file.write(self.to_bytes(**kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """The bytes of the HDF5 file that encodes this object."""
        writer = self._h5_writer(**kwargs)
        buffer = BytesIO()
        with h5py.File(buffer, "w") as f:
            writer(f)
        return buffer.getvalue()


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def read_string(value: tx.Any) -> tx.Optional[str]:
    """
    Decode a string stored in HDF5, whatever its storage.

    HDF5 strings come back from `h5py` as `str` (variable-length UTF-8
    attributes), as `bytes` (fixed-length or ASCII ones), as `numpy`
    scalars, or wrapped in a one-element array or dataset.
    """
    if value is None:
        return None
    if isinstance(value, h5py.Dataset):
        value = value[()]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Not a scalar string: shape {value.shape}")
        value = value.reshape(()).item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode()
    return str(value)


class DelayedH5Array:
    """
    An HDF5 dataset that can be read even after its file was closed,
    by reopening the file when needed.
    """

    def __init__(self, file: H5Like, path: str) -> None:
        self.file: H5Like = file
        self.path: str = path
        self._file: tx.Optional[h5py.File] = None
        self._shape: tx.Optional[tx.Tuple[int]] = None
        self._dtype: tx.Optional[np.dtype] = None
        self._chunks: tx.Optional[tx.Tuple[int]] = None

    def open(self) -> h5py.File:
        """Open (or reuse) the underlying HDF5 file and return it."""
        self.to_dataset(keep_open=True)
        return self._file

    def close(self) -> None:
        """Close the underlying HDF5 file, if this array opened it."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        """Close the underlying HDF5 file, if this array opened it."""
        self.close()

    def to_dataset(
        self, file: tx.Optional[H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        """Return the underlying `h5py.Dataset`, opening the file if
        needed."""

        if file is None:
            return self.to_dataset(self.file, keep_open=keep_open)

        if self._file is not None:
            file = self._file

        if isinstance(file, (str, PathLike)):
            if not keep_open:
                with h5py.File(file, "r") as f:
                    return self.to_dataset(f)
            else:
                file = self._file = h5py.File(file, "r")

        if isinstance(file, h5py.File):
            dataset = file[self.path]
            # cache info
            self._shape = dataset.shape
            self._dtype = dataset.dtype
            self._chunks = dataset.chunks
            return dataset

        raise ValueError("Invalid file type")

    def _read(self, index: tx.Any = ...) -> tx.Any:
        """
        Read `index` of the dataset, opening the file if needed.

        A file opened here is opened and closed within the call, and never
        stored on the object: `dask` reads chunks from several threads at
        once, and a handle shared between them could be closed by one
        while another reads from it.
        """
        file = self._file if self._file is not None else self.file
        if isinstance(file, (str, PathLike)):
            with h5py.File(file, "r") as f:
                return f[self.path][index]
        return self.to_dataset(file)[index]

    def to_array(self, **kwargs) -> np.ndarray:
        """Read the whole dataset into a `numpy` array."""
        return np.asarray(self._read(), **kwargs)

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        """Wrap the dataset as a `dask` array that reads chunks lazily."""
        import dask.array as da

        kwargs.setdefault("chunks", self.chunks or "auto")
        if keep_open:
            array_like = self.to_dataset(keep_open=True)
        else:
            array_like = self
        return da.from_array(array_like, **kwargs)

    def __getitem__(self, index: tx.Any) -> tx.Any:
        """Read the indexed chunk of the dataset, opening the file if
        needed."""
        return self._read(index)

    def __array__(
        self, dtype: np.dtype = None, copy: tx.Any = None
    ) -> np.ndarray:
        """Read the whole dataset into a `numpy` array."""
        return self.to_array(dtype=dtype)

    @property
    def shape(self) -> tuple:
        """The shape of the dataset."""
        if self._shape is None:
            self.to_dataset()
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the dataset."""
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def chunks(self) -> tuple:
        """The chunk shape of the dataset, or `None` if it is not
        chunked."""
        if self._chunks is None:
            self.to_dataset()
        return self._chunks

    @property
    def ndim(self) -> int:
        """The number of dimensions of the dataset."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The total number of elements in the dataset."""
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        """The size of the dataset in bytes."""
        return self.size * self.dtype.itemsize


def delayed_dataset(
    h5file: h5py.File, key: str, keep_open: bool
) -> tx.Union[DelayedH5Array, ArrayProtocol]:
    """
    A dataset of an open file, to be read later rather than now.

    With `keep_open`, the open file is held; otherwise the file is
    reopened by name when the data is read. The result is wrapped in a
    `dask` array when `dask` is installed.
    """
    from brainhops.backends import da

    if keep_open:
        fileish = h5file
    else:
        import os.path as op

        fileish = h5file.filename
        fileish = op.abspath(fileish) if fileish else None
    array = DelayedH5Array(fileish, key)
    if da:
        array = array.to_dask(keep_open=keep_open)
    return array
