"""The lazy array of a Zarr array, and the writing of an array store."""

# stdlib
import os
import shutil

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr import create
from bagof.magic import ConvertTo, NoRepr

# core
from brainhops._core import backends, path

# internals
from brainhops._core.dependencies import da
from brainhops._core.path import is_same_file
from brainhops._core.properties import alias
from brainhops._core.proxy import ArrayProxy, StoredProxy

# this format
from ._raw import ZarrRaw, document_keys, same_storage

# ----------------------------------------------------------------------
#   LAZY ARRAY
# ----------------------------------------------------------------------


class ZarrProxy(StoredProxy):
    """A lazy array that reads an opened Zarr array when it is converted.

    The proxy holds the abczarr node of the array, the location of its
    store, and an optional permutation of its axes. Opening it reads
    nothing but the document of the array. Indexing it returns another
    lazy array (see [`StoredProxy`][]), and converting that array reads
    only the chunks that the selection needs.

    The permutation `perm` presents the stored axes in another order:
    axis `i` of the proxy is axis `perm[i]` of the store, as in
    `numpy.transpose`. A plain Zarr array has none. The levels of an
    OME-Zarr pyramid use it to present their axes in the brainhops order,
    because the order depends on the record of the pyramid.
    """

    # --- magic attributes ----

    node: NoRepr[tx.Any]
    """The opened array."""

    location: tx.Annotated[
        tx.Optional[tx.Union[path.Path, tx.Any]],
        ConvertTo(path.as_path_or_stream),
    ]
    """
    The location of the store of the array, or `None` when it is not known.
    A location given as a string or a path-like object is converted to a
    [`Path`][brainhops._core.path.Path], and a store object that is not
    a path is kept as it is.
    """

    perm: tx.Optional[tx.Tuple[int, ...]] = None
    """
    The permutation from the stored axes to the axes of the proxy, or
    `None` when the axes are those of the store."
    """

    # --- properties ----

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the array, in the order of the proxy."""
        shape = tuple(int(n) for n in self.node.shape)
        if self.perm is None:
            return shape
        return tuple(shape[p] for p in self.perm)

    dtype = alias("node.dtype", readonly=True)
    """The type of the values of the array."""

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        """The shape of a chunk, in the order of the proxy."""
        chunks = tuple(int(n) for n in self.node.chunks)
        if self.perm is None:
            return chunks
        return tuple(chunks[p] for p in self.perm)

    # --- methods ----

    def _read(self, index: tx.Tuple[tx.Any, ...]) -> np.ndarray:
        """Read a box of the array, through the permutation of the axes."""
        if self.perm is None:
            return np.asarray(self.node[index])
        stored = [slice(None)] * len(self.perm)
        for axis, item in enumerate(index):
            stored[self.perm[axis]] = item
        read = np.asarray(self.node[tuple(stored)])
        kept = [
            axis
            for axis, item in enumerate(stored)
            if not isinstance(item, (int, np.integer))
        ]
        order = [
            kept.index(self.perm[axis])
            for axis, item in enumerate(index)
            if not isinstance(item, (int, np.integer))
        ]
        return np.transpose(read, order)


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

    Two paths are compared as [`Path`][brainhops._core.path.Path]
    objects, so that two spellings of the same remote location, such as
    a trailing slash or a scheme alias, compare equal. A local location
    is compared once its symbolic links are resolved. A store object
    that is not a path is compared by its text.
    """
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(raw, ZarrProxy) or raw.location is None:
        return False
    source = raw.location
    location = path.as_path_or_stream(location)
    if not (isinstance(source, path.Path) and isinstance(location, path.Path)):
        return _text(source) == _text(location)
    if source.is_local() != location.is_local():
        return False
    if source.is_local():
        return is_same_file(source, location)
    return source == location


def reads_inside(raw: tx.Any, location: tx.Any) -> bool:
    """Tell whether an array is the lazy array of a store inside a location.

    The store of the array lies inside the location when it is the
    location itself or one of its members, such as a level of a group.
    A writer that replaces the group at the location must then read the
    array before the old group is removed. Two paths are compared as
    [`Path`][brainhops._core.path.Path] objects, and a local location is
    compared once its symbolic links are resolved. A store object that
    is not a path is compared by its text.
    """
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(raw, ZarrProxy) or raw.location is None:
        return False
    source = raw.location
    location = path.as_path_or_stream(location)
    if not (isinstance(source, path.Path) and isinstance(location, path.Path)):
        source, target = _text(source), _text(location)
        return source == target or source.startswith(target + "/")
    if source.is_local() != location.is_local():
        return False
    if source.is_local():
        source, location = source.resolve(), location.resolve()
    return source == location or location in source.parents


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
    location = path.as_path_or_stream(location)
    documents = record.documents()
    local = _local(location)
    if local and _copies_chunks(raw, record):
        _copy_chunks(raw.location, location)
    else:
        options = {"overwrite": True}
        if driver is not None:
            options["driver"] = driver
        node = create(location, record.document, **options)
        _store(node, raw)
    if local:
        for key in _all_document_keys():
            if key not in documents:
                (location / key).unlink(missing_ok=True)
    for key, content in documents.items():
        (location / key).write_bytes(content)


def write_into(node: tx.Any, raw: tx.Any) -> None:
    """Write an array into an opened Zarr array of the same shape."""
    _store(node, raw)


def _store(node: tx.Any, raw: tx.Any) -> None:
    """Write an array, lazy or not, into an opened Zarr array.

    A lazy array of a file and a Dask array are written block by block
    when Dask is installed, so that they are never read whole into
    memory. A Dask array whose blocks lie on a GPU has them copied to
    memory one at a time. Any other array is copied to memory and
    written at once.
    """
    if da is not None and isinstance(raw, (ArrayProxy, da.Array)):
        node.store(backends.to_backend(raw, "dask"))
        return
    node[...] = backends.to_backend(raw, "numpy")


def store_chunks(raw: tx.Any) -> tx.Optional[tx.Tuple[int, ...]]:
    """Return the chunk shape with which an array is best stored, or `None`.

    A lazy array of a file prefers the blocks of its storage, such as
    the chunks of a Zarr array. The blocks of a Dask array are given by
    the first block along each axis, since all the blocks but the last
    ones usually have that shape. When a Zarr array is created with
    these chunks, each block of the source is written into chunks of
    its own, and the new store keeps the chunks of the source. A NumPy
    or CuPy array has no preference, and `None` is returned.
    """
    if isinstance(raw, ArrayProxy):
        return raw.chunks
    if da is not None and isinstance(raw, da.Array):
        return tuple(max(1, int(sizes[0])) for sizes in raw.chunks)
    return None


def _copies_chunks(raw: tx.Any, record: ZarrRaw) -> bool:
    """Tell whether the chunk files of the store of `raw` can be copied."""
    if not isinstance(raw, ZarrProxy):
        return False
    if raw.perm is not None:
        return False
    if not _local(raw.location) or not raw.location.is_dir():
        return False
    return same_storage(raw.node.metadata, record.document)


def _copy_chunks(source: path.Path, target: path.Path) -> None:
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
    return document_keys(3) + document_keys(2)


def _local(location: tx.Any) -> bool:
    """Tell whether a store location is a path on the local file system."""
    return isinstance(location, path.Path) and location.is_local()


def _text(location: tx.Any) -> str:
    """Return the text of a store location, without a trailing slash."""
    return str(location).rstrip("/")
