"""HDF5 datasets that can still be read after their file was closed."""

# stdlib
import os.path as op
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx
from bagof.magic import NoInit, NoRepr

# core
from brainhops._core.backends import has_dask
from brainhops._core.proxy import StoredProxy
from brainhops._core.typing import ArrayProtocol

# this format
from ._parsers import H5Like

# typing
_T = tx.TypeVar("_T")
_Internal = NoInit[NoRepr[tx.Optional[_T]]]


class H5ArrayProxy(StoredProxy):
    """
    An HDF5 dataset that can be read after its file was closed, by
    reopening the file on demand.

    Indexing the array returns another lazy array (see
    [`StoredProxy`][brainhops._core.proxy.StoredProxy]), and
    converting that array reads only the selected part of the dataset.
    That lazy array reads through this one, so a file that this array
    keeps open is used by every index of it.
    """

    file: tx.Any
    """The HDF5 file, by its path, as an open file, or as bytes."""

    path: str
    """The path of the dataset in the file."""

    _handle: _Internal[tx.Any] = None
    """The file that the array opened and keeps, or `None`."""

    _shape: _Internal[tx.Tuple[int, ...]] = None
    """The shape of the dataset, once it was opened."""

    _dtype: _Internal[np.dtype] = None
    """The type of the dataset, once it was opened."""

    _chunks: _Internal[tx.Tuple[int, ...]] = None
    """The chunks of the dataset, once it was opened."""

    def open(self) -> h5py.File:
        """Open the HDF5 file, or reuse the open one, and return it."""
        self.to_dataset(keep_open=True)
        return self._handle

    def close(self) -> None:
        """Close the HDF5 file if this array opened it."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def to_dataset(
        self, file: tx.Optional[H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        """
        Return the whole `h5py.Dataset`, opening the file if needed.

        With `keep_open`, a file opened by name stays open until [`close`][].
        """
        if file is None:
            return self.to_dataset(self.file, keep_open=keep_open)

        if self._handle is not None:
            file = self._handle

        if isinstance(file, (str, PathLike)):
            if not keep_open:
                with h5py.File(file, "r") as f:
                    return self.to_dataset(f)
            else:
                file = self._handle = h5py.File(file, "r")

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
        file = self._handle if self._handle is not None else self.file
        if isinstance(file, (str, PathLike)):
            with h5py.File(file, "r") as f:
                return f[self.path][index]
        return self.to_dataset(file)[index]

    def to_array(self, **kwargs) -> np.ndarray:
        """Read the values of the array into a NumPy array."""
        return np.asarray(self, **kwargs)

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        """
        Wrap the array in a dask array that reads its chunks lazily.

        With `keep_open`, the chunks are read from the open dataset;
        otherwise the file is reopened for each chunk.
        """
        if keep_open:
            import dask.array as da

            kwargs.setdefault("chunks", self.chunks or "auto")
            return da.from_array(self.to_dataset(keep_open=True), **kwargs)
        return super().to_dask(**kwargs)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the dataset."""
        if self._shape is None:
            self.to_dataset()
        return tuple(self._shape)

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the dataset."""
        if self._dtype is None:
            self.to_dataset()
        return self._dtype

    @property
    def chunks(self) -> tx.Optional[tuple]:
        """The shape of a chunk, or `None` if the dataset is not chunked."""
        if self._shape is None:
            self.to_dataset()
        return self._chunks


def to_proxy(
    h5file: h5py.File, key: str, keep_open: bool
) -> tx.Union[H5ArrayProxy, ArrayProtocol]:
    """
    Return a dataset of an open HDF5 file that is read later.

    With `keep_open`, the dataset holds the open file; otherwise the file
    is reopened by name on each read. The dataset is wrapped in a dask
    array if dask is installed.
    """

    if keep_open:
        fileish = h5file
    else:
        fileish = h5file.filename
        fileish = op.abspath(fileish) if fileish else None
    array = H5ArrayProxy(fileish, key)
    if has_dask():
        array = array.to_dask(keep_open=keep_open)
    return array
