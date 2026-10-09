"""HDF5 datasets that can still be read after their file was
closed."""

# stdlib
from math import prod
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

# core
from brainhops._core.typing import ArrayProtocol

# this format
from ._parsers import H5Like


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
