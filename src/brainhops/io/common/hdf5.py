"""
Shared machinery for formats stored in HDF5 files.

ITK `.h5` transforms and BIDS X5 files both keep their transforms under
a root `TransformGroup`, so only their content tells them apart. A
format therefore only says how to recognise and decode an open
`h5py.File`, and the classes of this module handle paths, streams,
bytes and lazy datasets. h5py is optional, so this module is only
imported by the formats that need it.
"""

__all__ = [
    "DelayedH5Array",
    "H5Like",
    "Hdf5Parser",
    "Hdf5ParserWriter",
    "delayed_dataset",
    "read_string",
]

from io import BytesIO
from math import prod
from os import PathLike

import h5py
import numpy as np
import typing_extensions as tx

from brainhops._core import path
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
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
    """Raise `error`, or `default` if `error` is true, or return NO."""
    if error:
        if error is True:
            error = default
        raise error(message) from cause
    return Confidence.NO


class Hdf5Parser(BinaryFileParser):
    """
    A mixin that reads a format stored in HDF5.

    A concrete format implements the class methods `sniff_h5` and
    `from_h5`, which take an open `h5py.File`, and the mixin routes paths,
    streams and bytes to them. A path is handed to h5py by name, so that a
    lazily read dataset can reopen the file long after parsing.
    """

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is of this format."""
        raise NotImplementedError

    @classmethod
    def from_h5(cls, h5file: h5py.File, **kwargs) -> tx.Self:
        """Build an object from an open HDF5 file."""
        raise NotImplementedError

    @classmethod
    def sniff_file(
        cls,
        file: H5Like,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a path, an open HDF5 file or a binary stream."""
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
        """Score the HDF5 file at a path."""
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

    @classmethod
    def from_file(
        cls,
        file: H5Like,
        keep_open: bool = False,
        load: bool = True,
        **kwargs,
    ) -> tx.Self:
        """
        Build an object from a path, a binary stream or an open HDF5 file.

        With `load`, large datasets are read into memory; otherwise they stay
        on disk. With `keep_open`, the file stays open after loading; otherwise
        it is closed, and a lazy dataset reopens it on each access.
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
        """Build an object from the HDF5 file at a path."""
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


class Hdf5ParserWriter(Hdf5Parser, BinaryFileParserWriter):
    """
    A mixin that reads and writes a format stored in HDF5.

    The format also implements `to_h5`, which fills a new, empty HDF5 file.
    """

    def to_h5(self, h5file: h5py.File, **kwargs) -> None:
        """Write the object into an empty HDF5 file open for writing."""
        raise NotImplementedError

    def _h5_writer(self, **kwargs) -> tx.Callable[[h5py.File], None]:
        """
        Return a function that fills an HDF5 file with the object.

        The function is obtained before the file is opened, so that a format
        can refuse an unwritable object without truncating the file.
        """
        return lambda h5file: self.to_h5(h5file, **kwargs)

    def to_file(self, file: H5Like, **kwargs) -> None:
        """Write to a path, a binary stream or an open HDF5 file."""
        if isinstance(file, h5py.File):
            return self._h5_writer(**kwargs)(file)
        return super().to_file(file, **kwargs)

    def to_filename(self, filename: tx.Union[str, PathLike], **kwargs) -> None:
        """Write to the file at a path, replacing it."""
        writer = self._h5_writer(**kwargs)
        with h5py.File(str(filename), "w") as f:
            writer(f)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write to a binary stream open for writing."""
        file.write(self.to_bytes(**kwargs))

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of an HDF5 file that encodes the object."""
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
    Decode an HDF5 string, whatever its storage.

    The value may be `str`, `bytes`, a NumPy scalar, or a one-element array
    or dataset wrapping one of these. `None` is returned unchanged.

    Raises
    ------
    ValueError
        If the value is an array of more than one element.
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
    An HDF5 dataset that can be read after its file was closed, by
    reopening the file on demand.
    """

    def __init__(self, file: H5Like, path: str) -> None:
        self.file: H5Like = file
        self.path: str = path
        self._file: tx.Optional[h5py.File] = None
        self._shape: tx.Optional[tx.Tuple[int]] = None
        self._dtype: tx.Optional[np.dtype] = None
        self._chunks: tx.Optional[tx.Tuple[int]] = None

    def open(self) -> h5py.File:
        """Open the HDF5 file, or reuse the open one, and return it."""
        self.to_dataset(keep_open=True)
        return self._file

    def close(self) -> None:
        """Close the HDF5 file if this array opened it."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    def to_dataset(
        self, file: tx.Optional[H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        """
        Return the `h5py.Dataset`, opening the file if needed.

        With `keep_open`, a file opened by name stays open until [`close`][].
        """

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
            self._shape = dataset.shape
            self._dtype = dataset.dtype
            self._chunks = dataset.chunks
            return dataset

        raise ValueError("Invalid file type")

    def _read(self, index: tx.Any = ...) -> tx.Any:
        """
        Read an index of the dataset.

        A file opened here is closed within the call and never stored, because
        dask reads chunks from several threads and one could close a shared
        handle while another is reading.
        """
        file = self._file if self._file is not None else self.file
        if isinstance(file, (str, PathLike)):
            with h5py.File(file, "r") as f:
                return f[self.path][index]
        return self.to_dataset(file)[index]

    def to_array(self, **kwargs) -> np.ndarray:
        """Read the whole dataset into a NumPy array."""
        return np.asarray(self._read(), **kwargs)

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        """
        Wrap the dataset in a dask array that reads its chunks lazily.

        With `keep_open`, the chunks are read from the open dataset; otherwise
        the file is reopened for each chunk.
        """
        import dask.array as da

        kwargs.setdefault("chunks", self.chunks or "auto")
        if keep_open:
            array_like = self.to_dataset(keep_open=True)
        else:
            array_like = self
        return da.from_array(array_like, **kwargs)

    def __getitem__(self, index: tx.Any) -> tx.Any:
        """Read an index of the dataset, opening the file if needed."""
        return self._read(index)

    def __array__(
        self, dtype: np.dtype = None, copy: tx.Any = None
    ) -> np.ndarray:
        """Read the whole dataset into a NumPy array."""
        return self.to_array(dtype=dtype)

    @property
    def shape(self) -> tuple:
        """The shape of the dataset."""
        if self._shape is None:
            self.to_dataset()
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the dataset."""
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def chunks(self) -> tuple:
        """The shape of a chunk, or `None` if the dataset is not chunked."""
        if self._chunks is None:
            self.to_dataset()
        return self._chunks

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        """The size of the data in bytes."""
        return self.size * self.dtype.itemsize


def delayed_dataset(
    h5file: h5py.File, key: str, keep_open: bool
) -> tx.Union[DelayedH5Array, ArrayProtocol]:
    """
    Return a dataset of an open HDF5 file that is read later.

    With `keep_open`, the dataset holds the open file; otherwise the file
    is reopened by name on each read. The dataset is wrapped in a dask
    array if dask is installed.
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
