"""The lazy array of a Zarr array, and the writing of an array store."""

# stdlib
import os
import shutil

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr import create

# core
from brainhops._core import path
from brainhops._core.dependencies import da

# internals
from brainhops.io.base._utils_files import is_local, same_file

# this format
from ._raw import ZarrRaw, _document_keys, same_storage

# ----------------------------------------------------------------------
#   LAZY ARRAY
# ----------------------------------------------------------------------


class _ZarrProxy:
    """A lazy array that reads an opened Zarr array when it is indexed.

    The proxy holds the abczarr node of the array, the location of its
    store, and an optional permutation of its axes. Opening it reads
    nothing but the document of the array, and indexing it reads only the
    chunks that the selection needs. An empty selection is answered
    without reading the store, so that the probe of Dask reads nothing.

    The permutation `perm` presents the stored axes in another order:
    axis `i` of the proxy is axis `perm[i]` of the store, as in
    `numpy.transpose`. A plain Zarr array has none. The levels of an
    OME-Zarr pyramid use it to present their axes in the brainhops order,
    because the order depends on the record of the pyramid.

    Parameters
    ----------
    node : ZarrArray
        The opened array.
    location : str or PathLike or None
        The location of the store of the array, or `None` when it is not
        known.
    perm : sequence of int, optional
        The permutation from the stored axes to the axes of the proxy.
    """

    def __init__(
        self,
        node: tx.Any,
        location: tx.Any,
        perm: tx.Optional[tx.Sequence[int]] = None,
    ) -> None:
        self.node = node
        self.location = location
        self.perm = None if perm is None else tuple(int(p) for p in perm)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array, in the order of the proxy."""
        shape = tuple(int(n) for n in self.node.shape)
        if self.perm is None:
            return shape
        return tuple(shape[p] for p in self.perm)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        return np.dtype(self.node.dtype)

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self.node.shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The shape of a chunk, in the order of the proxy."""
        chunks = tuple(int(n) for n in self.node.chunks)
        if self.perm is None:
            return chunks
        return tuple(chunks[p] for p in self.perm)

    def __getitem__(self, index: tx.Any) -> np.ndarray:
        """Read a selection of the array.

        An empty selection of slices and integers is answered without
        reading the store. A selection with a negative step, or a
        selection of another kind, such as an array of indices, of a
        proxy that permutes its axes, reads the whole array first,
        because the drivers of abczarr read neither.
        """
        basic = _basic_index(index, len(self.shape))
        if basic is not None:
            selected = np.broadcast_to(np.empty((), bool), self.shape)[basic]
            if selected.size == 0:
                return np.empty(selected.shape, self.dtype)
        if basic is not None and not _reverses(basic):
            return self._read(basic)
        if self.perm is None and basic is None:
            return np.asarray(self.node[index])
        return self._read((slice(None),) * len(self.shape))[index]

    def _read(self, basic: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        """Read a selection of slices with a positive step and integers."""
        if self.perm is None:
            return np.asarray(self.node[basic])
        stored = [slice(None)] * len(self.perm)
        for axis, item in enumerate(basic):
            stored[self.perm[axis]] = item
        read = np.asarray(self.node[tuple(stored)])
        kept = [
            axis
            for axis, item in enumerate(stored)
            if not isinstance(item, (int, np.integer))
        ]
        order = [
            kept.index(self.perm[axis])
            for axis, item in enumerate(basic)
            if not isinstance(item, (int, np.integer))
        ]
        return np.transpose(read, order)

    def __array__(
        self, dtype: tx.Optional[np.dtype] = None, copy: tx.Any = None
    ) -> np.ndarray:
        """Read the whole array into a NumPy array."""
        data = self[...]
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        return data

    def to_dask(self) -> tx.Any:
        """Return a Dask array that reads the chunks of the store lazily.

        The blocks of the Dask array are the chunks of the store, and each
        block is read through the proxy, so a selection of the Dask array
        reads only the chunks that it needs.
        """
        return da.from_array(self, chunks=self.chunks, asarray=True)


def store_location(node: tx.Any) -> tx.Any:
    """Return the location of the store of an opened array, or `None`.

    The location is the store path of the node when a document of the
    array lies there. A node whose store path is wrong, as the
    zarr-python driver of abczarr 0.2.2 gives a node opened from a
    relative path, has no known location, so that a writer never copies
    the files of another directory.
    """
    location = node.store_path
    try:
        if any((location / key).exists() for key in _all_document_keys()):
            return location
    except Exception:  # noqa: BLE001
        pass
    return None


def reads_store(raw: tx.Any, location: tx.Any) -> bool:
    """Tell whether an array is the lazy array of a store at a location.

    A local location is compared once its symbolic links are resolved,
    and a remote location is compared by its URL.
    """
    if not isinstance(raw, _ZarrProxy) or raw.location is None:
        return False
    source = raw.location
    if _local(source) and _local(location):
        return same_file(_fspath(source), _fspath(location))
    if _local(source) or _local(location):
        return False
    return str(source).rstrip("/") == str(location).rstrip("/")


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def write_array(
    location: tx.Any,
    record: ZarrRaw,
    raw: tx.Any,
    driver: tx.Optional[str] = None,
) -> None:
    """Fill a store location with an array that a record describes.

    When `raw` is the lazy array of a local store whose chunks are stored
    as the record stores them, the files of that store are copied without
    being decoded, so that an array that did not change keeps the bytes
    of each chunk. Otherwise, the array is created from the document of
    the record with abczarr and `raw` is written into it. The documents
    are then written with the bytes that [`ZarrRaw.documents`][] gives,
    so that a document that did not change keeps its bytes. A remote
    location is written in place.

    Parameters
    ----------
    location : str or PathLike
        The location of the new store, which a local writer has just
        created empty.
    record : ZarrRaw
        The record of the consolidated array.
    raw : array_like
        The stored array of the consolidated array.
    driver : str, optional
        The abczarr driver that creates the array, or `None` to let
        abczarr choose one.
    """
    documents = record.documents()
    local = _local(location)
    if local and _copies_chunks(raw, record):
        _copy_chunks(_fspath(raw.location), _fspath(location))
    else:
        options = {"overwrite": True}
        if driver is not None:
            options["driver"] = driver
        node = create(location, record.document, **options)
        _store(node, raw)
    if local:
        directory = _fspath(location)
        for key in _all_document_keys():
            name = os.path.join(directory, key)
            if key not in documents and os.path.exists(name):
                os.remove(name)
        for key, content in documents.items():
            with open(os.path.join(directory, key), "wb") as file:
                file.write(content)
        return
    root = path.Path(location) if isinstance(location, str) else location
    for key, content in documents.items():
        (root / key).write_bytes(content)


def write_into(node: tx.Any, raw: tx.Any) -> None:
    """Write an array into an opened Zarr array of the same shape."""
    _store(node, raw)


def _store(node: tx.Any, raw: tx.Any) -> None:
    """Write an array, lazy or not, into an opened Zarr array."""
    if isinstance(raw, _ZarrProxy) and da is not None:
        raw = raw.to_dask()
    if da is not None and isinstance(raw, da.Array):
        node.store(raw)
        return
    node[...] = np.asarray(raw)


def _copies_chunks(raw: tx.Any, record: ZarrRaw) -> bool:
    """Tell whether the chunk files of the store of `raw` can be copied."""
    if not isinstance(raw, _ZarrProxy) or raw.perm is not None:
        return False
    if raw.location is None or not _local(raw.location):
        return False
    source = _fspath(raw.location)
    if not os.path.isdir(source):
        return False
    return same_storage(raw.node.metadata, record.document)


def _copy_chunks(source: str, target: str) -> None:
    """Copy every file of a store but its documents, keeping their bytes."""
    documents = set(_all_document_keys())
    for root, _, names in os.walk(source):
        relative = os.path.relpath(root, source)
        for name in names:
            if relative == os.curdir and name in documents:
                continue
            directory = os.path.join(target, relative)
            os.makedirs(directory, exist_ok=True)
            shutil.copyfile(
                os.path.join(root, name), os.path.join(directory, name)
            )


def _all_document_keys() -> tx.Tuple[str, ...]:
    """Return the names of the document files of an array in any version."""
    return _document_keys(3) + _document_keys(2)


def _basic_index(
    index: tx.Any, ndim: int
) -> tx.Optional[tx.Tuple[tx.Any, ...]]:
    """Expand a selection of slices, integers and an ellipsis to `ndim`.

    `None` is returned for a selection of another kind, such as an array
    of indices or a new axis.
    """
    if not isinstance(index, tuple):
        index = (index,)
    if any(
        not isinstance(item, (slice, int, np.integer, type(Ellipsis)))
        or isinstance(item, bool)
        for item in index
    ):
        return None
    ellipses = [i for i, item in enumerate(index) if item is Ellipsis]
    if len(ellipses) > 1:
        return None
    if ellipses:
        at = ellipses[0]
        fill = ndim - (len(index) - 1)
        if fill < 0:
            return None
        index = index[:at] + (slice(None),) * fill + index[at + 1 :]
    if len(index) > ndim:
        return None
    return tuple(index) + (slice(None),) * (ndim - len(index))


def _reverses(index: tx.Tuple[tx.Any, ...]) -> bool:
    """Tell whether a selection has a slice with a negative step."""
    return any(
        isinstance(item, slice) and item.step is not None and item.step < 0
        for item in index
    )


def _local(location: tx.Any) -> bool:
    """Tell whether a store location is a path on the local file system."""
    try:
        return is_local(location)
    except Exception:  # noqa: BLE001
        return False


def _fspath(location: tx.Any) -> str:
    """Return a local store location as a string."""
    return os.fspath(location)
