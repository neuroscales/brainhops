"""HDF5 datasets that can still be read after their file was
closed."""

# stdlib
from os import PathLike

# dependencies
import h5py
import numpy as np
import typing_extensions as tx

from brainhops._core.proxy import ArrayProxy

# core
from brainhops._core.typing import ArrayProtocol

# this format
from ._parsers import H5Like


class DelayedH5Array(ArrayProxy):
    """
    An HDF5 dataset that can be read after its file was closed, by
    reopening the file on demand.

    Indexing the array returns another lazy array (see
    [`ArrayProxy`][brainhops._core.proxy.ArrayProxy]), and
    converting that array reads only the selected part of the dataset.
    A view reads through the array that it was made from, and opening or
    closing a view opens or closes the file of that array.
    """

    def __init__(self, file: H5Like, path: str) -> None:
        self.file: H5Like = file
        self.path: str = path
        self._file: tx.Optional[h5py.File] = None
        self._shape: tx.Optional[tx.Tuple[int]] = None
        self._dtype: tx.Optional[np.dtype] = None
        self._chunks: tx.Optional[tx.Tuple[int]] = None

    @property
    def _owner(self) -> "DelayedH5Array":
        """The array that owns the file handle and the cached attributes."""
        return self._root if self._root is not None else self

    def open(self) -> h5py.File:
        """Open the HDF5 file, or reuse the open one, and return it."""
        owner = self._owner
        owner.to_dataset(keep_open=True)
        return owner._file

    def close(self) -> None:
        """Close the HDF5 file if this array opened it."""
        owner = self._owner
        if owner._file is not None:
            owner._file.close()
            owner._file = None

    def __del__(self) -> None:
        # A view does not own the file of the array it was made from.
        if self._root is None:
            self.close()

    def to_dataset(
        self, file: tx.Optional[H5Like] = None, keep_open: bool = False
    ) -> h5py.Dataset:
        """
        Return the whole `h5py.Dataset`, opening the file if needed.

        With `keep_open`, a file opened by name stays open until [`close`][].
        The dataset is the whole stored dataset, also for a view.
        """
        if self._root is not None:
            return self._root.to_dataset(file, keep_open=keep_open)

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
        """Read the values of the array into a NumPy array."""
        return np.asarray(self, **kwargs)

    def to_dask(self, *, keep_open: bool = False, **kwargs) -> ArrayProtocol:
        """
        Wrap the array in a dask array that reads its chunks lazily.

        With `keep_open`, the chunks of an array that is not a view are
        read from the open dataset; otherwise the file is reopened for each
        chunk.
        """
        if keep_open and not self.is_view:
            import dask.array as da

            kwargs.setdefault("chunks", self.chunks or "auto")
            return da.from_array(self.to_dataset(keep_open=True), **kwargs)
        return super().to_dask(**kwargs)

    @property
    def base_shape(self) -> tuple:
        """The shape of the dataset."""
        owner = self._owner
        if owner._shape is None:
            owner.to_dataset()
        return tuple(owner._shape)

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the dataset."""
        owner = self._owner
        if owner._dtype is None:
            owner.to_dataset()
        return owner._dtype

    @property
    def base_chunks(self) -> tx.Optional[tuple]:
        """The shape of a chunk, or `None` if the dataset is not chunked."""
        owner = self._owner
        if owner._shape is None:
            owner.to_dataset()
        return owner._chunks


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
